import numpy as np
import streamlit as st
import requests
import datetime
from textblob import TextBlob
from nltk.corpus import stopwords
import string
import plotly.express as px
from collections import defaultdict
import pandas as pd

# ========== SETUP ==========
STOPWORDS = set(stopwords.words("english"))
API_KEY = "65675dec54ea4305a56417cd6a64880b"

# ========== FUNCTION DEFINITIONS ==========

def fetch_articles(category, from_date, to_date):
    url = f"https://newsapi.org/v2/everything?q={category}&from={from_date}&to={to_date}&language=en&sortBy=popularity&pageSize=100&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["articles"]
    else:
        st.error("Failed to fetch news data.")
        return []

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_daily_sentiment_trends(articles):
    daily_sentiments = defaultdict(list)
    for article in articles:
        try:
            date = datetime.datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00")).date().isoformat()
        except Exception:
            continue

        content = article.get("content") or article.get("description") or ""
        clean = clean_text(content)
        if clean:
            sentiment = analyze_sentiment(clean)
            daily_sentiments[date].append(sentiment)

    daily_data = []
    for date, sentiments in daily_sentiments.items():
        pos = sentiments.count("Positive")
        neu = sentiments.count("Neutral")
        neg = sentiments.count("Negative")
        total = len(sentiments)
        daily_data.append({
            "date": date,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "total": total
        })

    df = pd.DataFrame(daily_data).sort_values("date")
    return df

def get_top_articles(articles, sentiment_type, top_n=5):
    scored_articles = []
    for article in articles:
        content = article.get("content") or article.get("description") or ""
        clean = clean_text(content)
        if clean:
            blob = TextBlob(clean)
            polarity = blob.sentiment.polarity
            if (sentiment_type == "Positive" and polarity > 0.1) or (sentiment_type == "Negative" and polarity < -0.1):
                scored_articles.append((article["title"], polarity, article.get("url", "#")))

    if sentiment_type == "Negative":
        scored_articles.sort(key=lambda x: x[1])
    else:
        scored_articles.sort(key=lambda x: x[1], reverse=True)

    return scored_articles[:top_n]

def plot_regression(df, sentiment_col):
    df["date_num"] = pd.to_datetime(df["date"]).map(datetime.datetime.toordinal)
    x = df["date_num"]
    y = df[sentiment_col]
    m, c = np.polyfit(x, y, 1)
    df[f"{sentiment_col}_trend"] = m * x + c

    fig = px.scatter(df, x="date", y=sentiment_col)
    fig.add_traces(px.line(df, x="date", y=f"{sentiment_col}_trend").data)
    st.plotly_chart(fig, use_container_width=True)


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="News Lens", layout="wide")
st.title("🔍 News Lens")
st.markdown("---")
# Store all trends in a list
all_trends = []

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    categories = st.multiselect("Select News Categories", 
        ["general", "business", "entertainment", "health", "science", "sports", "technology"])
    
    start_date = st.date_input("From Date", value=datetime.date.today() - datetime.timedelta(days=14))
    end_date = st.date_input("To Date", value=datetime.date.today())
    st.markdown("---")

# ========== MAIN LOGIC ==========
if categories and start_date and end_date:
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    for category in categories:
        col_head, col_spin = st.columns([1, 1])

        with col_head:
            st.header(f"📂 {category.title()} News Analysis")

        with col_spin:
            with st.spinner(f"Fetching and analyzing {category} news..."):
                articles = fetch_articles(category, from_date, to_date)
                st.info(f"Total articles fetched: {len(articles)}")

        if not articles:
            st.warning(f"No articles found for {category} in selected date range.")
            continue

        full_texts = []
        sentiments = []
        for article in articles:
            content = article.get("content") or article.get("description") or ""
            clean = clean_text(content)
            if clean:
                full_texts.append(clean)
                sentiments.append(analyze_sentiment(clean))

        if full_texts:
            col1, col2 = st.columns(2)
            trend_df = get_daily_sentiment_trends(articles)
            trend_df["category"] = category  # Add this line to label the data
            all_trends.append(trend_df)
            counts = {
                "Positive": sentiments.count("Positive"),
                "Neutral": sentiments.count("Neutral"),
                "Negative": sentiments.count("Negative")
            }

            with col1:
                st.subheader("⚪ Sentiment Distribution")
                pie_data = pd.DataFrame({
                    "Sentiment": list(counts.keys()),
                    "Count": list(counts.values())
                })
                fig_pie = px.pie(pie_data, values="Count", names="Sentiment", 
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive": "#25b045",
                                    "Neutral": "#5484bc",
                                    "Negative": "#c15554"
                                })
                st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("📈 Linear Regression for Positive Sentiment")
                plot_regression(trend_df, "positive")

            with col2:
                st.subheader("📊 Article Count Per Day")
                if not trend_df.empty:
                    fig3 = px.bar(
                        trend_df,
                        x="date",
                        y="total",
                        labels={"total": "Article Count", "date": "Date"},
                        text="total",
                        color="total",
                        color_continuous_scale="Blues"
                    )
                    fig3.update_traces(textposition="outside")
                    fig3.update_layout(
                        xaxis_tickangle=90,
                        yaxis_title="Article Count",
                        xaxis_title="Date"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No data for article count.")

                st.subheader("📈 Linear Regression for Negative Sentiment")
                plot_regression(trend_df, "negative")

            # Daily sentiment line chart
            st.subheader("📈 Daily Sentiment Trend")
            if not trend_df.empty:
                fig_line = px.line(
                    trend_df,
                    x="date",
                    y=["neutral", "positive", "negative"],
                    labels={"value": "Sentiment Count", "variable": "Sentiment Type", "date": "Date"}
                )
                fig_line.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Articles",
                    legend_title="Sentiment",
                    template="plotly_white"
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Not enough trend data to display.")

            # Side-by-side columns for positive and negative news
            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.subheader("🌟 Top 5 Positive Headlines")
                top_pos = get_top_articles(articles, "Positive")
                for i, (title, score, url) in enumerate(top_pos, 1):
                    st.markdown(f"**{i}.** [{title}]({url})")

            with col_neg:
                st.subheader("⚠️ Top 5 Negative Headlines")
                top_neg = get_top_articles(articles, "Negative")
                for i, (title, score, url) in enumerate(top_neg, 1):
                    st.markdown(f"**{i}.** [{title}]({url})")

        else:
            st.info("Not enough valid content to analyze.")
        st.markdown("---")

    # Combine all trends into a single DataFrame for final display
    if all_trends:
        combined_df = pd.concat(all_trends)
        
        st.subheader("📊 Cross-Category Sentiment Trend")
        
        # Create a container for the chart
        chart_container = st.container()
        
        # Initialize session state for sentiment selection
        if 'selected_sentiment' not in st.session_state:
            st.session_state.selected_sentiment = 'positive'
        
        # Create columns for radio buttons
        _, col1, col2, col3, col4 = st.columns([0.5, 1, 1, 1, 1])
        
        # Use individual buttons instead of radio
        with col1:
            if st.button('Positive', key='pos_btn'):
                st.session_state.selected_sentiment = 'positive'
        with col2:
            if st.button('Negative', key='neg_btn'):
                st.session_state.selected_sentiment = 'negative'
        with col3:
            if st.button('Neutral', key='neu_btn'):
                st.session_state.selected_sentiment = 'neutral'
        with col4:
            if st.button('Total', key='total_btn'):
                st.session_state.selected_sentiment = 'total'
        
        # Display the chart using the selected sentiment from session state
        with chart_container:
            st.markdown(
                f"""<div style='text-align: center; font-size: 20px; font-weight: bold;'>
                    {st.session_state.selected_sentiment.title()} Trend Across {len(categories)} Categories
                </div>""", unsafe_allow_html=True
            )

            fig_multi = px.line(
                combined_df,
                x="date",
                y=st.session_state.selected_sentiment,
                color="category",
                labels={
                    "date": "Date",
                    st.session_state.selected_sentiment: f"{st.session_state.selected_sentiment.title()} Count",
                    "category": "News Category"
                })
            st.plotly_chart(fig_multi, use_container_width=True)


else:
    st.markdown(
        """
            <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh;'>
                <div style='text-align: center; font-size: 24px; font-weight: 800;'>Select categories and dates to analyze news.</div>
                <br>
                <div style='text-align: center; font-size: 20px;'>Make sure to select at least one category.</div>
            </div>
        """,
        unsafe_allow_html=True
    )