# AI Meeting Assistant - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Set Up API Keys

#### Groq API (Required)
1. Go to [Groq Console](https://console.groq.com/keys)
2. Create a free account and get your API key
3. Set the environment variable:

```bash
# Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"

# Windows
set GROQ_API_KEY=your_groq_api_key_here
```

#### Gemini API (Optional - Fallback)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Get your API key
3. Set the environment variable:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Run the Application
```bash
python3 demo.py
```

The server will start on `http://localhost:8002`

## 🔧 Features

- **Real-time AI Suggestions**: Powered by Groq's Llama 3.1 8B Instant
- **File Upload & Analysis**: Upload PDFs, docs for AI analysis
- **Live Transcript**: Real-time meeting transcription
- **WebSocket Communication**: Real-time updates
- **Beautiful UI**: Modern Co-pilot Nexus interface

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 test_groq_integration.py
python3 diagnose.py
```

## 🐛 Troubleshooting

### Debug UI
Open `debug_ui.html` in your browser for interactive debugging.

### Common Issues
1. **API Key Not Working**: Check environment variables
2. **WebSocket Issues**: Try manual connection via "Start Meeting" button
3. **Port Conflicts**: Kill existing processes on port 8002

## 📁 Project Structure

```
├── demo.py                 # Main application
├── services/
│   ├── asr/                # Audio-to-text service
│   └── nlu/                # Natural language processing
├── test_*.py              # Test scripts
├── debug_ui.html          # Debug interface
└── uploads/               # File upload directory
```

## 🔐 Security

- API keys are loaded from environment variables
- No hardcoded secrets in the codebase
- File uploads are stored locally in `uploads/` directory
