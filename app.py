import logging
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import os
import warnings

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Fix torch Streamlit bug
try:
    del torch._classes
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("\U0001F310 Multilingual Sentiment Analyzer & Auto-Responder")

try:
    for key in ["processed", "last_uploaded_filename"]:
        if key not in st.session_state:
            st.session_state[key] = None if key == "last_uploaded_filename" else False

    uploaded_file = st.file_uploader("\U0001F4C1 Upload CSV with 'Review_text' column", type="csv")
    sample_data_path = "sample_data.csv"

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

    if df.empty or "Review_text" not in df.columns:
        st.error("❌ Missing or empty 'Review_text' column.")
        st.stop()

    MAX_ROWS = 3  # Smaller for safety
    df = df.dropna(subset=["Review_text"]).head(MAX_ROWS)

    @st.cache_resource
    def load_sentiment_model():
        try:
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            st.error("❌ Sentiment model loading failed")
            st.exception(e)
            st.stop()

    @st.cache_resource
    def load_llm_model():
        try:
            model_name = "sshleifer/distilbart-cnn-12-6"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            st.error("❌ LLM model loading failed")
            st.exception(e)
            st.stop()

    sentiment_tokenizer, sentiment_model = load_sentiment_model()
    llm_tokenizer, llm_model = load_llm_model()

    def map_star_to_label(star_rating):
        if star_rating in [1, 2]:
            return "Negative"
        elif star_rating == 3:
            return "Neutral"
        else:
            return "Positive"

    def analyze_all_sentiments(texts, batch_size=4):
        sentiment_model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = sentiment_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                outputs = sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                labels = torch.argmax(probs, dim=1).tolist()
                results.extend([map_star_to_label(l + 1) for l in labels])
        return results

    @st.cache_data(show_spinner=False)
    def generate_response_cached(sentiment, review):
        if sentiment != "Negative":
            return "No response needed."
        prompt = f"Write a polite short reply to this complaint: {review}"
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        output = llm_model.generate(**inputs, max_new_tokens=40)
        reply = llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return f"Thank you for your review. We will look into the issue. {reply.rstrip('.!?')}"

    if not st.session_state.processed:
        with st.spinner("Analyzing sentiments and generating responses..."):
            df["Sentiment"] = analyze_all_sentiments(df["Review_text"].tolist())
            responses = [generate_response_cached(s, r) for s, r in zip(df["Sentiment"], df["Review_text"])]
            df["Response"] = responses
            st.session_state.df_processed = df.copy()
            st.session_state.processed = True

    df = st.session_state.df_processed

    st.success("✅ Processing complete!")
    st.subheader("\U0001F4CB Preview")
    cols_to_show = [col for col in ["Unique_ID", "Category", "Review_text", "Sentiment", "Response"] if col in df.columns]
    st.dataframe(df[cols_to_show], use_container_width=True)

    st.subheader("\U0001F4CA Sentiment Breakdown")
    chart_data = df["Sentiment"].value_counts().reset_index()
    chart_data.columns = ["Sentiment", "Count"]
    fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_responses.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"🔥 App crashed with error: {e}")
    import traceback
    st.text(traceback.format_exc())
    st.stop()
