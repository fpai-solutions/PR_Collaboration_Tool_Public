from flask import Flask, request, jsonify, render_template, session, Response, stream_template
from flask_cors import CORS
import re
import requests
import os
from uuid import uuid4
import google.generativeai as genai
import tldextract
import json
import time
import random
import threading
from queue import Queue
import signal
from functools import wraps
import logging
import sys

app = Flask(__name__, template_folder='templates')
CORS(app)
app.secret_key = 'supersecretkey'  # Needed for session

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Azure App Service configuration
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Check if running on Azure App Service
def is_azure_app_service():
    """Check if running on Azure App Service"""
    return os.environ.get('WEBSITE_SITE_NAME') is not None

# Log Azure environment detection
if is_azure_app_service():
    logger.info("Running on Azure App Service")
else:
    logger.info("Running in local/development environment")

# Configure API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Configure SERP API
SERP_API_KEY = os.getenv("SERP_API_KEY", "your-serp-api-key-here")
SERP_ENGINE = os.getenv("SERP_ENGINE", "google")

# Global variable to store progress updates and real-time results
progress_updates = {}
real_time_results = {}

# Timeout configuration for Azure
REQUEST_TIMEOUT = 240  # 4 minutes max for Azure App Service
BROWSER_TIMEOUT = 30000  # 30 seconds for browser operations
PAGE_LOAD_TIMEOUT = 15000  # 15 seconds for page loads

# Global timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def with_timeout(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutException:
                print(f"[DEBUG] Function {func.__name__} timed out after {timeout_seconds} seconds")
                return None
            finally:
                # Restore old handler
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
        return wrapper
    return decorator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress/<session_id>')
def progress_stream(session_id):
    def generate():
        while True:
            if session_id in progress_updates:
                update = progress_updates[session_id]
                yield f"data: {json.dumps(update)}\n\n"
                if update.get('status') == 'complete' or update.get('status') == 'error':
                    break
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/realtime/<session_id>')
def realtime_stream(session_id):
    def generate():
        while True:
            if session_id in real_time_results:
                results = real_time_results[session_id]
                if results:
                    # Send the latest result and remove it from queue
                    result = results.pop(0)
                    yield f"data: {json.dumps(result)}\n\n"
                elif progress_updates.get(session_id, {}).get('status') in ['complete', 'error']:
                    break
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

def send_progress_update(session_id, message, emoji="🔄"):
    progress_updates[session_id] = {
        'message': message,
        'emoji': emoji,
        'timestamp': time.time()
    }

def send_real_time_result(session_id, lead):
    if session_id not in real_time_results:
        real_time_results[session_id] = []
    real_time_results[session_id].append(lead)

def search_with_serp_api(query: str, location: str = None, num_results: int = 10, page: int = 0) -> list:
    """
    Search using SERP API for business lead generation.
    This provides reliable search results for finding business contacts.
    """
    try:
        # SERP API endpoint
        url = "https://serpapi.com/search"
        
        params = {
            "api_key": SERP_API_KEY,
            "engine": SERP_ENGINE,
            "q": query,
            "num": num_results,
            "gl": "us",  # Country
            "hl": "en",  # Language
            "start": page * num_results  # Pagination
        }
        
        if location:
            params["location"] = location
        
        print(f"[DEBUG] Searching with SERP API: {query}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract organic results
        results = []
        if "organic_results" in data:
            for result in data["organic_results"]:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0)
                })
        
        print(f"[DEBUG] Found {len(results)} results from SERP API")
        return results
        
    except Exception as e:
        print(f"[DEBUG] SERP API search failed: {e}")
        return []

# Legacy browser functions removed - now using SERP API

def extract_email(text):
    # More robust regex for emails - only capture valid business emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    print(f"[DEBUG] Raw emails found by regex: {emails}")
    
    # Filter out invalid emails
    valid_emails = []
    for email in emails:
        print(f"[DEBUG] Checking email: {email}")
        # Skip emails with suspicious patterns
        if any(skip in email.lower() for skip in [
            '.webp', '.jpeg', '.jpg', '.png', '.gif', '.svg', '.ico', '.bmp',
            '.pdf', '.doc', '.docx', '.txt', '.csv', '.xls', '.xlsx',
            'sentry', 'netvlies', 'example', 'test', 'noreply', 'no-reply',
            'mailer', 'postmaster', 'webmaster', 'admin@localhost'
        ]):
            print(f"[DEBUG] Skipping email due to suspicious pattern: {email}")
            continue
        
        # Skip emails with unusual domain patterns
        domain = email.split('@')[1] if '@' in email else ''
        if any(skip in domain for skip in [
            'localhost', '127.0.0.1', 'test', 'example', 'invalid'
        ]):
            print(f"[DEBUG] Skipping email due to invalid domain: {email}")
            continue
        
        # Skip emails that are too short or too long
        if len(email) < 8 or len(email) > 100:
            print(f"[DEBUG] Skipping email due to length: {email}")
            continue
            
        # Skip emails with consecutive dots or unusual characters
        if '..' in email or email.count('@') != 1:
            print(f"[DEBUG] Skipping email due to format: {email}")
            continue
            
        print(f"[DEBUG] Valid email found: {email}")
        valid_emails.append(email)
    
    print(f"[DEBUG] Final valid emails: {valid_emails}")
    return valid_emails

def extract_business_name_from_email(email):
    """Extract business name from email domain and use AI to separate words"""
    try:
        print(f"[DEBUG] Extracting business name from email: {email}")
        # Extract domain from email (between @ and .)
        if '@' in email:
            domain = email.split('@')[1].split('.')[0]
            print(f"[DEBUG] Extracted domain: {domain}")
            
            # Use AI to separate words in the domain
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Convert this phrase with no spaces into proper words by separating and capitalizing them: '{domain}'. Only return the name, nothing else. For example: 'goodmangallery' should become 'Goodman Gallery'"
            response = model.generate_content(prompt)
            business_name = response.text.strip()
            print(f"[DEBUG] AI generated business name: '{business_name}'")
            
            # Fallback if AI fails
            if not business_name or len(business_name) < 3:
                print(f"[DEBUG] AI failed, using fallback for domain: {domain}")
                # Simple fallback: replace common separators and capitalize
                domain = domain.replace('-', ' ').replace('_', ' ')
                business_name = ' '.join(word.capitalize() for word in domain.split())
                print(f"[DEBUG] Fallback business name: '{business_name}'")
            
            return business_name
        else:
            print(f"[DEBUG] No @ found in email: {email}")
            return ''
    except Exception as e:
        print(f"[DEBUG] Error extracting business name from email {email}: {e}")
        # Fallback to simple domain extraction
        try:
            if '@' in email:
                domain = email.split('@')[1].split('.')[0]
                domain = domain.replace('-', ' ').replace('_', ' ')
                fallback_name = ' '.join(word.capitalize() for word in domain.split())
                print(f"[DEBUG] Error fallback business name: '{fallback_name}'")
                return fallback_name
        except:
            pass
        return ''

def get_website_url_from_email(email):
    """Extract website URL from email"""
    try:
        if '@' in email:
            domain = email.split('@')[1]
            return f"https://www.{domain}"
        return ''
    except:
        return ''

# Legacy function removed - now using requests for website scraping

def generate_business_description(desc):
    if not desc or not desc.strip():
        return ''
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = f"Given this business description: '{desc}', generate a concise, two-sentence summary of what the business does. Do not use brackets."
        response = model.generate_content(prompt)
        return response.text.strip().replace('[', '').replace(']', '')
    except Exception:
        return desc

def generate_gemini_message(business_name, outreach_goal, business_description, pr_brand):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    prompt = (
        f"Write a concise, direct, and ready-to-send outreach email to {business_name}. "
        f"The goal is: {outreach_goal}. "
        f"About our PR brand: {pr_brand}. "
        f"Business description: {business_description}. "
        f"Mention the business name. Do not use brackets. Do not include subject line just email. Do not sign email with name if not included. Keep it short and persuasive."
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip().replace('[', '').replace(']', '')
    except Exception as e:
        return f"Error generating message: {e}"

def process_lead_async(lead, outreach_goal, pr_brand, session_id):
    """Process a single lead asynchronously and send real-time result"""
    try:
        # Generate message
        lead['outreach_message'] = generate_gemini_message(
            lead['business_name'], outreach_goal, lead.get('business_description', ''), pr_brand
        )
        
        # Send real-time result
        send_real_time_result(session_id, lead)
        
        send_progress_update(session_id, f"Generated email for {lead['business_name']}", "💌")
        
    except Exception as e:
        print(f"[DEBUG] Error processing lead {lead['business_name']}: {e}")

@app.route('/scrape', methods=['POST'])
@with_timeout(REQUEST_TIMEOUT)
def scrape():
    try:
        logger.info("Starting scrape request")
        data = request.json
        company_types = data.get('company_types', [])
        pr_brand = data.get('pr_brand', '')
        outreach_goal = data.get('outreach_goal')
        location = data.get('location')
        start = int(data.get('start', 0))
        count = int(data.get('count', 5))
        
        logger.info(f"Scrape parameters - Company types: {company_types}, Location: {location}, Count: {count}")
    except Exception as e:
        logger.error(f"Error parsing request data: {e}")
        return jsonify({'error': 'Invalid request data', 'leads': [], 'has_more': False, 'session_id': None})

    # If company_types is a string, split it into a list
    if isinstance(company_types, str):
        company_types = [t.strip() for t in company_types.split(',') if t.strip()]
    
    # If no company types provided, use a default
    if not company_types:
        company_types = ['business']

    print(f"[DEBUG] Company types: {company_types}")
    print(f"[DEBUG] PR Brand: {pr_brand}")
    print(f"[DEBUG] Outreach goal: {outreach_goal}")
    print(f"[DEBUG] Location: {location}")

    # Create a unique session ID for this multi-query search
    search_key = f"{','.join(company_types)}_{outreach_goal}_{location}"
    session_id = session.get('scrape_session_id')
    if not session_id or session.get('scrape_query') != search_key:
        session_id = str(uuid4())
        session['scrape_session_id'] = session_id
        session['scrape_query'] = search_key
        session['scrape_results'] = []
        session['scrape_page'] = 0
        session['processed_domains'] = []
        session['company_types'] = company_types

    all_leads = session.get('scrape_results', [])
    page_num = session.get('scrape_page', 0)
    processed_domains = set(session.get('processed_domains', []))
    new_domains_this_batch = 0
    max_results_per_page = 5  # Reduced for Azure timeout limits

    # Initialize real-time results for this session
    real_time_results[session_id] = []

    # Send initial progress
    send_progress_update(session_id, "Starting up the search engine...", "🚀")

    # Run queries for each company type using SERP API
    try:
        send_progress_update(session_id, "Starting SERP API search...", "🌐")
        
        # Process each company type
        total_company_types = len(company_types)
        for company_type_index, business_type in enumerate(company_types):
            query = f"{business_type} {location} contact OR about OR email"
            
            send_progress_update(
                session_id, 
                f"Searching for {business_type}...", 
                "🔍"
            )
            
            print(f"[DEBUG] Processing company type {company_type_index + 1}/{len(company_types)}: {business_type}")
            print(f"[DEBUG] SERP API query: {query}")
            
            # Search using SERP API
            results = search_with_serp_api(query, location=location, num_results=10, page=page_num)
            
            if not results:
                print(f"[DEBUG] No results found for {business_type}, continuing...")
                continue
                
            print(f"[DEBUG] Found {len(results)} results for {business_type}")
            
            send_progress_update(
                session_id, 
                f"Found {len(results)} results for {business_type}", 
                "📊"
            )
            
            leads_found_for_this_type = 0
            for idx, result in enumerate(results):
                send_progress_update(
                    session_id,
                    f"Analyzing result {idx + 1} of {len(results)} for {business_type}...",
                    "🔎"
                )
                
                business_name = result.get('title', '')
                link = result.get('link', '')
                snippet = result.get('snippet', '')
                
                domain = ''
                if link:
                    ext = tldextract.extract(link)
                    domain = ext.domain.lower()
                    
                if domain and domain in processed_domains:
                    print(f"[DEBUG] Skipping duplicate domain: {domain}")
                    continue
                    
                # Only process up to 10 unique domains per page
                if new_domains_this_batch >= max_results_per_page:
                    print(f"[DEBUG] Reached max unique domains for this batch: {max_results_per_page}")
                    break
                    
                print(f"[DEBUG] Result {idx+1}: business_name='{business_name}', link='{link}'")
                
                # Extract emails from snippet
                emails = extract_email(snippet) if snippet else []
                print(f"[DEBUG] Emails found in snippet: {emails}")
                business_description = generate_business_description(snippet) if snippet else ''
                
                if emails:
                    # Extract business name from email domain
                    business_name_from_email = extract_business_name_from_email(emails[0])
                    website_url = get_website_url_from_email(emails[0])
                    
                    print(f"[DEBUG] Adding lead from snippet: {emails[0]}")
                    lead = {
                        'business_name': business_name_from_email,
                        'email': emails[0],
                        'business_description': business_description,
                        'outreach_message': '',
                        'website_url': website_url,
                    }
                    all_leads.append(lead)
                    
                    # Process lead asynchronously for real-time results
                    threading.Thread(
                        target=process_lead_async,
                        args=(lead, outreach_goal, pr_brand, session_id)
                    ).start()
                    
                    if domain:
                        processed_domains.add(domain)
                        new_domains_this_batch += 1
                    leads_found_for_this_type += 1
                    print(f"[DEBUG] Added lead {leads_found_for_this_type} for {business_type}: {emails[0]}")
                elif link:
                    print(f"[DEBUG] Searching website for email: {link}")
                    send_progress_update(
                        session_id,
                        f"Visiting website to find email...",
                        "🌍"
                    )
                    
                    # Use requests to get page content instead of Playwright
                    try:
                        response = requests.get(link, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        })
                        if response.status_code == 200:
                            emails2 = extract_email(response.text)
                            print(f"[DEBUG] Emails found on website: {emails2}")
                            
                            if emails2:
                                # Extract business name from email domain
                                business_name_from_email = extract_business_name_from_email(emails2[0])
                                website_url = get_website_url_from_email(emails2[0])
                                
                                print(f"[DEBUG] Added lead from website: {emails2[0]}")
                                lead = {
                                    'business_name': business_name_from_email,
                                    'email': emails2[0],
                                    'business_description': business_description,
                                    'outreach_message': '',
                                    'website_url': website_url,
                                }
                                all_leads.append(lead)
                                
                                # Process lead asynchronously for real-time results
                                threading.Thread(
                                    target=process_lead_async,
                                    args=(lead, outreach_goal, pr_brand, session_id)
                                ).start()
                                
                                if domain:
                                    processed_domains.add(domain)
                                    new_domains_this_batch += 1
                                leads_found_for_this_type += 1
                                print(f"[DEBUG] Added lead {leads_found_for_this_type} for {business_type}: {emails2[0]}")
                    except Exception as e:
                        print(f"[DEBUG] Error fetching website {link}: {e}")
                        continue
                
                # Small delay between results
                time.sleep(random.uniform(0.1, 0.3))
            
            print(f"[DEBUG] Total leads found for {business_type}: {leads_found_for_this_type}")
            
            # Small delay between company types
            time.sleep(random.uniform(0.5, 1.0))
            
            # Only break if we have enough results AND have processed all categories
            if len(all_leads) >= start + count and company_type_index == len(company_types) - 1:
                print(f"[DEBUG] Reached target number of leads ({start + count}) and processed all categories, stopping.")
                break
        
        print(f"[DEBUG] Total leads found across all company types: {len(all_leads)}")
    
    except Exception as e:
        print(f"[DEBUG] SERP API search failed: {e}")
        send_progress_update(session_id, f"Error during search: {str(e)}", "❌")
        return jsonify({'error': 'Search operation failed', 'leads': [], 'has_more': False, 'session_id': session_id})
    
    send_progress_update(session_id, f"Complete! Found {len(all_leads)} leads", "🎉")
    
    session['scrape_results'] = all_leads
    session['scrape_page'] = page_num + 1
    session['processed_domains'] = list(processed_domains)

    # Prepare the batch: just return the new leads from this page
    batch = all_leads[start:len(all_leads)]
    
    has_more = True  # Always allow loading more unless no results found
    return jsonify({'leads': batch, 'has_more': has_more, 'session_id': session_id})

@app.route('/health')
def health_check():
    """Health check endpoint for Azure App Service"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'environment': 'azure' if is_azure_app_service() else 'local'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors gracefully"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'leads': [], 'has_more': False, 'session_id': None}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    # Get port from environment or default
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False)
