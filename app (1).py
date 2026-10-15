import re
import pandas as pd
import streamlit as st

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline


# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="📊",
    layout="wide"
)


st.title("📊 YouTube Comment Analyzer")

st.markdown("""
### Understand what people think about a YouTube video

This application uses a Transformer-based AI model
to classify YouTube comments into three categories:

- 😊 Positive
- 😐 Neutral
- 😠 Negative

**Features**
- Retrieve comments using the YouTube Data API
- Analyze sentiment and confidence scores
- Identify predictions that may need review
- Explore sentiment statistics and percentages
- Filter comments and export results to CSV

**Model:** CardiffNLP RoBERTa sentiment model
""")



# -----------------------------
# 2. Sidebar settings
# -----------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "YouTube Data API Key",
    type="password"
)

video_url = st.sidebar.text_input(
    "YouTube Video URL",
    placeholder="Paste a YouTube video URL here"
)

max_comments = st.sidebar.number_input(
    "Maximum comments to analyze",
    min_value=1,
    max_value=500,
    value=100,
    step=10
)


# -----------------------------
# 3. Extract video ID
# -----------------------------
def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# -----------------------------
# 4. Fetch video information
# -----------------------------
def fetch_video_context(api_key, video_id):
    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    items = response.get("items", [])

    if not items:
        raise ValueError(
            "Video not found or unavailable."
        )

    snippet = items[0]["snippet"]

    return {
        "title": snippet.get("title", "Unknown title"),
        "description": snippet.get("description", "")
    }


# -----------------------------
# 5. Fetch YouTube comments
# -----------------------------
def fetch_comments(api_key, video_id, limit):
    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    comments = []
    next_page_token = None

    while len(comments) < limit:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, limit - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance"
        )

        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"][
                "topLevelComment"
            ]["snippet"]

            comments.append({
                "author": comment.get(
                    "authorDisplayName", "Unknown"
                ),
                "comment": comment.get(
                    "textDisplay", ""
                ),
                "published_at": comment.get(
                    "publishedAt", ""
                ),
                "like_count": comment.get(
                    "likeCount", 0
                )
            })

            if len(comments) >= limit:
                break

        next_page_token = response.get(
            "nextPageToken"
        )

        if not next_page_token:
            break

    return pd.DataFrame(comments)


# -----------------------------
# 6. Load AI sentiment model
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


# -----------------------------
# 7. Analyze sentiment
# -----------------------------
def analyze_sentiment(comments):
    model = load_sentiment_model()

    predictions = []

    for start in range(0, len(comments), 32):
        batch = comments[start:start + 32]

        results = model(
            batch,
            truncation=True,
            max_length=512
        )

        predictions.extend(results)

    labels = []
    scores = []

    for result in predictions:
        labels.append(
            result["label"].lower()
        )
        scores.append(
            float(result["score"])
        )

    return labels, scores


# -----------------------------
# 8. Analyze button
# -----------------------------
if st.sidebar.button(
    "Analyze Comments",
    type="primary"
):
    if not api_key.strip():
        st.error(
            "Please enter your YouTube Data API key."
        )

    elif not video_url.strip():
        st.error(
            "Please enter a YouTube video URL."
        )

    else:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error(
                "Invalid YouTube URL. Please check it."
            )

        else:
            try:
                with st.spinner(
                    "Fetching video information..."
                ):
                    video_context = fetch_video_context(
                        api_key, video_id
                    )

                with st.spinner(
                    "Fetching YouTube comments..."
                ):
                    df = fetch_comments(
                        api_key,
                        video_id,
                        int(max_comments)
                    )

                if df.empty:
                    st.warning(
                        "No comments were found. "
                        "Comments may be disabled."
                    )

                else:
                    with st.spinner(
                        "AI is analyzing the comments..."
                    ):
                        labels, scores = analyze_sentiment(
                            df["comment"].tolist()
                        )

                    df["sentiment"] = labels
                    df["sentiment_score"] = scores

                    df["needs_review"] = (
                        df["sentiment_score"] < 0.70
                    )

                    st.session_state["analysis_df"] = df
                    st.session_state[
                        "video_context"
                    ] = video_context

                    st.success(
                        f"Analyzed {len(df)} comments successfully!"
                    )

            except HttpError as error:
                st.error(
                    "YouTube API request failed. "
                    "Check your API key, quota, video ID, "
                    "and comment availability."
                )
                st.caption(str(error))

            except Exception as error:
                st.error(
                    "An error occurred during analysis."
                )
                st.caption(str(error))


# -----------------------------
# 9. Display results
# -----------------------------
if "analysis_df" in st.session_state:
    df = st.session_state["analysis_df"].copy()

    video_context = st.session_state.get(
        "video_context", {}
    )

    st.divider()
    st.header("Video Information")

    st.subheader(
        video_context.get("title", "Unknown title")
    )

    description = video_context.get(
        "description", ""
    )

    with st.expander("Video description"):
        if description.strip():
            st.write(description)
        else:
            st.write("No description available.")

    st.divider()
    st.header("Sentiment Dashboard")

    # Original AI sentiment predictions
    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reindex(
            ["positive", "neutral", "negative"],
            fill_value=0
        )
    )

    total = len(df)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Comments", total)

    col2.metric(
        "Positive",
        int(sentiment_counts["positive"])
    )

    col3.metric(
        "Neutral",
        int(sentiment_counts["neutral"])
    )

    col4.metric(
        "Negative",
        int(sentiment_counts["negative"])
    )

    # Sentiment percentages
    st.subheader("Sentiment Distribution (%)")

    if total > 0:
        sentiment_percentages = (
            sentiment_counts / total * 100
        ).round(1)

        st.bar_chart(sentiment_percentages)

        percentage_df = (
            sentiment_percentages
            .rename("Percentage (%)")
            .reset_index()
        )

        percentage_df.columns = [
            "Sentiment",
            "Percentage (%)"
        ]

        st.dataframe(
            percentage_df,
            use_container_width=True,
            hide_index=True
        )

    # Review warning
    review_count = int(df["needs_review"].sum())

    st.info(
        f"{review_count} comments need review "
        "because their AI confidence score is below 70%."
    )

    # -----------------------------
    # 10. Filter comments
    # -----------------------------
    st.divider()
    st.header("Explore Comments")

    col1, col2 = st.columns(2)

    with col1:
        selected_sentiments = st.multiselect(
            "Filter by sentiment",
            options=[
                "positive",
                "neutral",
                "negative"
            ],
            default=[
                "positive",
                "neutral",
                "negative"
            ]
        )

    with col2:
        review_filter = st.selectbox(
            "Review status",
            options=[
                "All comments",
                "Needs review only",
                "No review needed"
            ]
        )

    filtered_df = df[
        df["sentiment"].isin(selected_sentiments)
    ]

    if review_filter == "Needs review only":
        filtered_df = filtered_df[
            filtered_df["needs_review"]
        ]

    elif review_filter == "No review needed":
        filtered_df = filtered_df[
            ~filtered_df["needs_review"]
        ]

    st.write(
        f"Showing {len(filtered_df)} comments."
    )

    display_columns = [
        "author",
        "comment",
        "sentiment",
        "sentiment_score",
        "needs_review",
        "like_count",
        "published_at"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # 11. Export CSV
    # -----------------------------
    st.divider()
    st.header("Export Results")

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download All Comments as CSV",
        data=csv_data,
        file_name="youtube_sentiment_analysis.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Enter your API key and video URL in the sidebar, "
        "then click Analyze Comments to begin."
    )
