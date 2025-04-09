# Sentiment Analyzer Streamlit Deployment Instructions

This document provides instructions for deploying the Comment Sentiment Analyzer Streamlit application.

## Files Overview

- `app.py` - The main Streamlit application file
- `requirements.txt` - List of Python dependencies
- `cleaned_comments.csv` - Sample data file (your actual data file)

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps to Run Locally

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the application**:

   ```bash
   streamlit run app.py
   ```
3. **Access the application**:
   The application will be available at http://localhost:8501

## Production Deployment Options

### Option 1: Streamlit Cloud (Recommended for Easy Deployment)

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository containing the app
3. Select the repository, branch, and file path to your app.py
4. Click "Deploy"

### Option 2: Deploy on a Server

1. Set up a server with Python installed
2. Clone your repository or upload your files
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app with nohup to keep it running after you log out:
   ```bash
   nohup streamlit run app.py --server.port 8501 &
   ```
5. Set up a reverse proxy (like Nginx) to forward traffic to your Streamlit app

### Option 3: Docker Deployment

1. Create a Dockerfile:

   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run the Docker container:

   ```bash
   docker build -t sentiment-analyzer .
   docker run -p 8501:8501 sentiment-analyzer
   ```

## Configuration Options

Streamlit provides several configuration options that can be set in a `.streamlit/config.toml` file:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Troubleshooting

- If you encounter NLTK resource errors, make sure the required NLTK packages are downloaded:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```
- For memory issues with large datasets, consider using data chunking or sampling techniques

## Security Considerations

- Ensure your server has appropriate security measures in place
- Consider adding authentication if your app contains sensitive data
- Keep all packages updated to avoid security vulnerabilities
