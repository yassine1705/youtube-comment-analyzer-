# YouTube Comment Analyzer

An AI-powered web application that analyzes YouTube comments and classifies their sentiment as positive, neutral, or negative.

## Live Demo

Try the application online: **[Open the YouTube Comment Analyzer](YOUR_STREAMLIT_APP_URL)**

## Overview

The YouTube Comment Analyzer uses a pretrained Transformer model to analyze comments and present sentiment statistics through an interactive Streamlit dashboard.

The project demonstrates the integration of Natural Language Processing (NLP), a pretrained machine learning model, the YouTube Data API, and a web application.

## Features

- Retrieve YouTube video information and comments.
- Classify comments as positive, neutral, or negative.
- Display model confidence scores.
- Flag predictions with confidence scores below 70% for manual review.
- Visualize sentiment counts and percentages.
- Filter comments by sentiment and review status.
- Export analysis results to CSV.

## Technologies Used

- **Python** — application logic and data processing
- **Streamlit** — interactive web interface
- **Hugging Face Transformers** — sentiment analysis pipeline
- **CardiffNLP RoBERTa** — pretrained sentiment classification model
- **YouTube Data API v3** — retrieve video information and comments
- **Pandas** — data processing and CSV export

## How It Works

1. The user enters a YouTube video URL and a YouTube Data API key.
2. The application retrieves the video's information and available comments.
3. The pretrained model predicts the sentiment of each comment.
4. The application displays sentiment statistics, confidence scores, and review flags.
5. The user can filter the results and download them as a CSV file.

## Run Locally

### Requirements

- Python 3.10 or a compatible Python version
- A Google account and a YouTube Data API key
- An internet connection

### Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd youtube-comment-analyzer
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Enter your YouTube Data API key in the application's sidebar.

## Model Limitations

The application uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, a pretrained model developed for sentiment analysis of social media text.

- The model was trained on tweets, so its performance on YouTube comments may vary.
- Sarcasm, slang, irony, emojis, and ambiguous comments can lead to incorrect predictions.
- The model analyzes comments individually and does not currently incorporate the video's title or description into each prediction.
- Confidence scores are not guarantees of correctness.
- The 70% review threshold is a configurable heuristic, not a validated measure of prediction reliability.
- Comments may be unavailable when a video disables comments or API access is restricted.

## Evaluation

**Formal model evaluation is pending.**

Accuracy and other evaluation metrics should be reported only after comparing model predictions with a manually labeled test dataset. Future evaluation can include precision, recall, F1-score, and a confusion matrix.

## Security

Never commit API keys, passwords, or other secrets to a public repository. Users should provide their own YouTube Data API key. Protect credentials and respect YouTube API quotas and applicable terms.

## Future Improvements

- Evaluate model performance on manually labeled YouTube comments.
- Improve handling of sarcasm and platform-specific language.
- Add more detailed visualizations and filtering options.
- Explore context-aware sentiment analysis.

## Author

Developed as a personal project to explore NLP, pretrained Transformer models, API integration, and AI application development.
