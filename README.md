# PR Collaboration Tool

A Flask-based lead generation and outreach tool that helps find business contacts and generate personalized outreach messages using AI.

## Features

- **Lead Generation**: Search for business contacts using SERP API
- **AI-Powered Outreach**: Generate personalized outreach messages using Google Gemini
- **Real-time Results**: Stream results as they're found
- **Email Extraction**: Automatically extract business emails from search results
- **Business Intelligence**: Generate business descriptions and contact information

## Installation

1. **Clone and navigate to the project:**
   ```bash
   cd PR_Colloboration_Tool
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your browser and go to `http://localhost:8000`

## Environment Variables

Create a `.env` file with the following variables:

```env
# Google Gemini API Key for AI analysis and message generation
GEMINI_API_KEY=your-gemini-api-key-here

# SERP API Key for search functionality
SERP_API_KEY=your-serp-api-key-here

# SERP Engine (google, bing, etc.)
SERP_ENGINE=google

# Flask configuration
FLASK_DEBUG=false
PORT=8000
```

## Usage

1. **Enter Company Types**: Specify the types of businesses you want to find
2. **Set Location**: Enter the target location for your search
3. **Define Outreach Goal**: Describe what you want to achieve
4. **Enter PR Brand**: Provide information about your brand
5. **Start Search**: The tool will search for leads and generate outreach messages

## Technical Details

### Architecture

- **Backend**: Flask web framework
- **Search API**: SERP API for reliable search results
- **AI Integration**: Google Gemini for message generation
- **Real-time Updates**: Server-sent events for live results

### API Endpoints

- `GET /` - Main application interface
- `POST /scrape` - Start lead generation
- `GET /progress/<session_id>` - Get progress updates
- `GET /realtime/<session_id>` - Get real-time results
- `GET /health` - Health check endpoint

## Requirements

- Python 3.7+
- Flask
- Google Gemini API key
- SERP API key

## License

This project is for educational purposes. Please respect API Terms of Service when using search functionality.
