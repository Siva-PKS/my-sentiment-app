import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Fix torch Streamlit bug
try:
    del torch._classes
except AttributeError:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# Session state setup
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# Handle data input
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(sample_data_path):
        st.success("✅ Using 'sample_data.csv' from directory.")
        df = pd.read_csv(sample_data_path)
    else:
        st.error("❌ 'sample_data.csv' not found. Please upload a CSV.")
        st.stop()

# Validation
if df.empty or "Review_text" not in df.columns:
    st.error("❌ Missing or empty 'Review_text' column.")
    st.stop()

# Limit rows for demo speed
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting to first {MAX_ROWS} rows for demo.")
    df = df.head(MAX_ROWS)

# Load models
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

# Initialize models
sentiment_pipeline = load_sentiment_pipeline()
tokenizer, model = load_llm_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Sentiment analysis with confidence score
def analyze_all_sentiments(texts):
    results = sentiment_pipeline([t[:512] for t in texts], return_all_scores=True)
    labels, confidences = [], []
    for res in results:
        top = max(res, key=lambda x: x['score'])
        label = label_map.get(top['label'], "Unknown")
        confidence = round(top['score'], 2)
        labels.append(label)
        confidences.append(confidence)
    return labels, confidences

# LLM response generation
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."
    prompt = (
        "You are a polite and helpful customer support agent. "
        "Write a short, professional reply to this negative customer review:\n"
        f"Review: {review}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=150)
    llm_reply = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return f"Thank you for your review. We will look into the issue. {llm_reply.rstrip('.!?')}."

# Processing step
if not st.session_state.processed:
    progress_bar = st.progress(0)
    sentiments, confidences = analyze_all_sentiments(df["Review_text"].tolist())
    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    df["Email"] = df["Sentiment"].apply(lambda x: "Yes" if x == "Negative" else "---")
    responses = []
    for i, row in df.iterrows():
        sentiment = row["Sentiment"]
        review = row["Review_text"]
        responses.append(generate_response(sentiment, review))
        progress_bar.progress((i + 1) / len(df))
    df["Response"] = responses
    st.session_state.df_processed = df.copy()
    st.session_state.processed = True

# Use cached processed data
df = st.session_state.df_processed

st.success("✅ Processing complete!")

# 📋 Custom Preview Table with Buttons (for Negative Reviews)
st.subheader("📋 Preview with Email Trigger Buttons")

display_columns = ["Unique_ID", "Category", "Review_text", "Sentiment", "Confidence", "Response", "Email"]

# Table Header
header_cols = st.columns(len(display_columns))
for i, col in enumerate(display_columns):
    header_cols[i].markdown(f"**{col}**")

# Table Rows
for i, row in df.iterrows():
    cols = st.columns(len(display_columns))
    for j, col_name in enumerate(display_columns):
        cell_val = row[col_name] if col_name in row else ""
        if isinstance(cell_val, str) and len(cell_val) > 150:
            cell_val = cell_val[:150] + "..."

        style = ""
        if col_name in ["Unique_ID", "Sentiment", "Confidence"]:
            style = "max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
        elif col_name == "Response":
            style = "max-width: 300px;"
        else:
            style = "max-width: 200px;"

        if col_name == "Email" and row["Sentiment"] == "Negative":
            if cols[j].button("📧 Send", key=f"send_{i}"):
                st.success(f"✅ Email triggered for Row {i+1}")
        else:
            bg_color = "#ffe6e6" if row["Sentiment"] == "Negative" else "transparent"
            cols[j].markdown(
                f"<div style='background-color:{bg_color}; padding:4px; {style}'>{cell_val}</div>",
                unsafe_allow_html=True
            )

# 📊 Sentiment Breakdown
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

# 📈 Sentiment by Category
if "Category" in df.columns:
    st.subheader("📈 Sentiment by Category")
    grouped = df.groupby(["Category", "Sentiment"]).size().reset_index(name="Count")
    fig2 = px.bar(grouped, x="Category", y="Count", color="Sentiment", barmode="group",
                  color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig2, use_container_width=True)

# 🧮 Sortable Full Table for Exploration
st.subheader("🔍 Sortable Full Table")
st.dataframe(df[display_columns], use_container_width=True)

# ⬇️ CSV Download
st.download_button(
    label="⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
