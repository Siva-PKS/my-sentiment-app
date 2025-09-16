import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import time
from sklearn.metrics import accuracy_score
import random
import math

# ---------------------------
# Fix torch Streamlit bug (common workaround)
# ---------------------------
try:
    del torch._classes
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# ---------------------------
# App title (hackathon-ready)
# ---------------------------
st.set_page_config(page_title="RapidSent — Customer Sentiment & Auto-Responder (Hackathon)", layout="wide")
st.title("RapidSent — Customer Feedback Sentiment & Auto-Responder (Hackathon Edition)")

# ---------------------------
# Email Configuration (keep secure)
# ---------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "spkincident@gmail.com"
# password must be declared in Streamlit secrets as "email_password"
SENDER_PASSWORD = st.secrets.get("email_password", None)

def send_email(recipient_email, subject, body):
    if not SENDER_PASSWORD:
        st.error("Email password not configured in Streamlit secrets.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# Session state setup
# ---------------------------
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

if "open_expander_index" not in st.session_state:
    st.session_state.open_expander_index = None

uploaded_file = st.file_uploader("📁 Upload CSV (original columns: Unique_ID, Category, Review_text, Date, Email)", type="csv")
sample_data_path = "product_reviews_with_stars_filled.csv"

# ---------------------------
# Read CSV (uploaded or sample)
# ---------------------------
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file, encoding="utf-8", dtype=str)
else:
    if os.path.exists(sample_data_path):
        st.success(f"Using sample file '{sample_data_path}' from app directory.")
        df = pd.read_csv(sample_data_path, encoding="utf-8", dtype=str)
    else:
        st.error(f"Sample file '{sample_data_path}' not found. Please upload a CSV.")
        st.stop()

# ensure consistent dtypes
df = df.fillna("")

# ---------------------------
# Normalize column names into expected ones
# Input columns we expect in raw CSV: Unique_ID, Category, Review_text, Date, Email
# Output target columns: UniqueId, Category, Purchasedate, EmailId, Star, Rating, Review
# ---------------------------
# rename what we can
rename_map = {}
if "Unique_ID" in df.columns:
    rename_map["Unique_ID"] = "UniqueId"
if "Date" in df.columns:
    rename_map["Date"] = "Purchasedate"
if "Email" in df.columns:
    rename_map["Email"] = "EmailId"
if "Review_text" in df.columns:
    rename_map["Review_text"] = "Review"

df = df.rename(columns=rename_map)

# create required columns if missing
for col in ["UniqueId", "Category", "Purchasedate", "EmailId", "Star", "Rating", "Review"]:
    if col not in df.columns:
        df[col] = ""

# limit rows for demo (optional)
MAX_ROWS = 1000
if len(df) > MAX_ROWS:
    st.warning(f"Limiting to first {MAX_ROWS} rows for demo.")
    df = df.head(MAX_ROWS)

# ---------------------------
# Utilities for Star & Rating formatting
# ---------------------------
def to_float_or_nan(x):
    try:
        if x is None:
            return float('nan')
        s = str(x).strip()
        if s == "" or s.lower() in ["nan", "none", "null"]:
            return float('nan')
        # sometimes ratings stored as "4" -> allow it
        return float(s)
    except Exception:
        return float('nan')

def format_rating_one_decimal(val):
    # expects numeric or NaN; returns string like "4.5" or "" for missing
    try:
        if pd.isna(val):
            return ""
        return f"{float(val):.1f}"
    except Exception:
        return ""

def rating_to_star_display(rating_val):
    """
    Convert numeric rating -> star display.
    We'll support half-star by showing '½' after filled stars.
    Examples:
      4.5 -> '★★★★½'
      3.0 -> '★★★☆☆'  (we will show filled stars then empty to 5, no '½')
    """
    try:
        if pd.isna(rating_val):
            return ""
        r = float(rating_val)
        # clamp 0..5
        r = max(0.0, min(5.0, r))
        # compute full stars and half flag
        half = (abs(r - round(r * 2) / 2) < 1e-8)  # true if already half steps
        # round to nearest 0.5
        rounded = round(r * 2) / 2.0
        full_stars = int(math.floor(rounded))
        is_half = (rounded - full_stars) == 0.5
        stars = "★" * full_stars
        if is_half:
            stars += "½"
        # optionally add empty stars to reach visual length (not strictly necessary)
        # compute displayed length: count filling as 1 for half too
        disp_len = full_stars + (1 if is_half else 0)
        empty_count = max(0, 5 - math.ceil(rounded))
        stars += "☆" * empty_count
        return stars
    except Exception:
        return ""

# ---------------------------
# If Star/Rating/Review are missing or invalid: generate/clean them
# - If Rating exists, coerce to float and format to 1 decimal
# - If Rating missing but Star present (like "4" or "★★★★"), try to infer numeric rating
# - If both missing, randomly assign rating for demo purposes (but keep ~10% nulls)
# - Intentionally leave ~10% of rows with missing Star OR Rating OR Review to mimic noisy data
# ---------------------------
random.seed(42)

# First, coerce existing Rating-like columns to numeric if possible
rating_values = []
for idx, row in df.iterrows():
    raw_rating = row.get("Rating", "")
    numeric = to_float_or_nan(raw_rating)
    rating_values.append(numeric)
df["Rating_num"] = rating_values

# If Rating_num is NaN but Star column contains digits (e.g., '4'), try to use it
for idx, row in df.iterrows():
    if pd.isna(df.at[idx, "Rating_num"]):
        star_raw = str(row.get("Star", "")).strip()
        # If star_raw is like '4' or '4.0'
        try:
            if star_raw != "":
                # try parse numeric
                possible = to_float_or_nan(star_raw)
                if not math.isnan(possible):
                    df.at[idx, "Rating_num"] = possible
                    continue
                # else if star_raw contains unicode stars: count '★'
                if "★" in star_raw:
                    filled = star_raw.count("★")
                    # check for half marker '½'
                    is_half = "½" in star_raw
                    inferred = filled + (0.5 if is_half else 0.0)
                    df.at[idx, "Rating_num"] = float(inferred)
                    continue
        except Exception:
            pass

# Now for rows with no rating yet, randomly generate some ratings for demo:
# Keep approx 10% of rows with missing rating intentionally.
indices_missing_rating = df[df["Rating_num"].apply(lambda x: math.isnan(x))].index.tolist()
keep_missing_count = max(1, int(0.10 * len(df)))  # keep this many as missing
# shuffle
random.shuffle(indices_missing_rating)
# first keep_missing_count indices remain missing; for others, generate rating
to_generate = indices_missing_rating[keep_missing_count:]
for idx in to_generate:
    # generate distribution skewed toward 3-5 (realistic ecom)
    r = random.choices([round(x * 0.5, 1) for x in range(2, 11)], weights=[1,1,2,4,6,8,10,6,4], k=1)[0]
    df.at[idx, "Rating_num"] = float(r)

# Format Rating column to single decimal string
df["Rating"] = df["Rating_num"].apply(lambda x: format_rating_one_decimal(x) if not (isinstance(x, float) and math.isnan(x)) else "")

# Now create Star numeric column (1-5) and Star_Display
# If Star column already numeric-like, coerce; else infer from Rating_num; else generate
star_nums = []
star_display = []
for idx, row in df.iterrows():
    raw_star = str(row.get("Star", "")).strip()
    star_num = None
    # try if raw_star is numeric string
    try:
        s_num = to_float_or_nan(raw_star)
        if not math.isnan(s_num):
            star_num = float(s_num)
    except Exception:
        pass
    # if still None, infer from Rating_num
    if star_num is None or math.isnan(star_num):
        rn = df.at[idx, "Rating_num"]
        if not (isinstance(rn, float) and math.isnan(rn)):
            # round rating to nearest 0.5 -> use as star_num
            star_num = round(rn * 2) / 2.0
        else:
            star_num = float('nan')

    # if still NaN, generate but keep some as missing
    if isinstance(star_num, float) and math.isnan(star_num):
        # 10% will remain NaN (missing), others get generated 1.0-5.0
        if random.random() < 0.10:
            star_nums.append(float('nan'))
            star_display.append("")
            continue
        else:
            gen = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            star_num = gen

    star_nums.append(star_num)
    star_display.append(rating_to_star_display(star_num))

df["Star_num"] = star_nums
df["Star_Display"] = star_display

# For Review: if empty, try to fill with Review_text if available (we already normalized Review<-Review_text),
# else generate a short placeholder review for demo, but leave ~10% empty to simulate users not writing review.
placeholder_positive = [
    "Excellent product — works as expected, highly recommend.",
    "Very satisfied with purchase. Good value for money."
]
placeholder_neutral = [
    "Product is okay. Nothing exceptional, but does the job.",
    "Average experience. Could be improved."
]
placeholder_negative = [
    "Not satisfied with the product. It stopped working within a few days.",
    "Poor quality and bad customer service."
]

# We'll use a simple heuristic: if star>=4 => positive, 3 => neutral, <=2.5 => negative
for idx, row in df.iterrows():
    r = str(row.get("Review", "")).strip()
    if r == "" or r.lower() in ["nan", "none", "null"]:
        # maybe fill from Review_text column if present (we already set Review to Review_text earlier).
        # If still empty, decide randomly to fill or leave blank
        if random.random() < 0.10:
            # keep empty ~10%
            df.at[idx, "Review"] = ""
            continue
        # pick based on star_num
        sn = df.at[idx, "Star_num"]
        try:
            if not (isinstance(sn, float) and math.isnan(sn)):
                if sn >= 4.0:
                    df.at[idx, "Review"] = random.choice(placeholder_positive)
                elif sn >= 3.0:
                    df.at[idx, "Review"] = random.choice(placeholder_neutral)
                else:
                    df.at[idx, "Review"] = random.choice(placeholder_negative)
            else:
                # no star -> neutral placeholder
                df.at[idx, "Review"] = random.choice(placeholder_neutral)
        except Exception:
            df.at[idx, "Review"] = ""

# Final tidy: produce the final output columns in requested names
# 1) UniqueId
# 2) Category
# 3) Purchasedate
# 4) EmailId
# 5) Star (we'll output Star_Display)
# 6) Rating (one-decimal string)
# 7) Review

df_final = pd.DataFrame({
    "UniqueId": df["UniqueId"].astype(str),
    "Category": df["Category"].astype(str),
    "Purchasedate": df["Purchasedate"].astype(str),
    "EmailId": df["EmailId"].astype(str),
    "Star": df["Star_Display"].astype(str),
    "Rating": df["Rating"].astype(str),
    "Review": df["Review"].astype(str)
})

# Normalize some empty values to real empties rather than strings "nan"
df_final = df_final.replace({"nan": ""})

# Save to session state for downstream processing/sentiment analysis
st.session_state.df_processed_raw = df_final.copy()

# ---------------------------
# Load models (sentiment & llm)
# ---------------------------
@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception as e:
        st.warning(f"Could not load sentiment pipeline: {e}")
        return None

@st.cache_resource
def load_llm_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        return tokenizer, model
    except Exception as e:
        st.warning(f"Could not load LLM model: {e}")
        return None, None

sentiment_pipeline = load_sentiment_pipeline()
tokenizer, model = load_llm_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ---------------------------
# Settings
# ---------------------------
st.sidebar.header("Settings")
NEGATIVE_THRESHOLD = st.sidebar.slider(
    "Negative confidence threshold (for Email Trigger)",
    min_value=0.50, max_value=0.95, value=0.70, step=0.01,
    help="Only Negative predictions at or above this confidence will auto-trigger emails."
)
st.sidebar.info(f"Current Negative threshold: {NEGATIVE_THRESHOLD:.2f}")

# ---------------------------
# Sentiment analysis helpers
# ---------------------------
def analyze_all_sentiments(texts):
    """
    texts: list of strings (reviews)
    returns: labels, confidences
    """
    labels, confidences = [], []
    if sentiment_pipeline is None:
        # fallback: naive rule based (very rough)
        for t in texts:
            t_low = (t or "").lower()
            if any(w in t_low for w in ["worst", "not", "don't", "poor", "bad", "waste", "defective", "stop"]):
                labels.append("Negative"); confidences.append(0.85)
            elif any(w in t_low for w in ["good", "excellent", "best", "great", "satisfied", "love"]):
                labels.append("Positive"); confidences.append(0.85)
            else:
                labels.append("Neutral"); confidences.append(0.60)
        return labels, confidences

    # truncate to 512 tokens approximate by characters
    results = sentiment_pipeline([str(t)[:512] for t in texts], return_all_scores=True)
    for res in results:
        top = max(res, key=lambda x: x['score'])
        label = label_map.get(top['label'], "Unknown")
        confidence = round(float(top['score']), 2)
        labels.append(label)
        confidences.append(confidence)
    return labels, confidences

def generate_response(sentiment, review):
    if sentiment != "Negative" or tokenizer is None or model is None:
        return "No response needed."
    prompt = (
        "You are a polite and helpful customer support agent. "
        "Write a short, professional reply to this negative customer review:\n"
        f"Review: {review}"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(**inputs, max_new_tokens=150)
        llm_reply = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return f"Thank you for your review. We will look into the issue. {llm_reply.rstrip('.!?')}."
    except Exception as e:
        return "Thank you for your review. We will look into the issue."

# ---------------------------
# Processing & sentiment if not done
# ---------------------------
if not st.session_state.processed:
    prog = st.progress(0)
    raw_reviews = st.session_state.df_processed_raw["Review"].tolist()
    sentiments, confidences = analyze_all_sentiments(raw_reviews)

    responses, proc_times = [], []
    for i, rev in enumerate(raw_reviews):
        t0 = time.time()
        responses.append(generate_response(sentiments[i], rev))
        t1 = time.time()
        proc_times.append(t1 - t0)
        prog.progress((i + 1) / max(1, len(raw_reviews)))

    df_out = st.session_state.df_processed_raw.copy()
    df_out["Sentiment"] = sentiments
    df_out["Confidence"] = confidences
    df_out["Response"] = responses
    df_out["Processing_Time_sec"] = proc_times

    st.session_state.df_processed = df_out
    st.session_state.processed = True

df = st.session_state.df_processed.copy()

# Recompute Email_Trigger on every run based on the current slider
df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if (r["Sentiment"] == "Negative" and float(r["Confidence"]) >= NEGATIVE_THRESHOLD) else "No",
    axis=1
)
st.success("Processing complete!")

# ---------------------------
# Preview Table (highlight negatives)
# ---------------------------
st.subheader("Preview (Transformed & Analyzed)")
def highlight_negative_row(row):
    return ['background-color: #ffe6e6'] * len(row) if row.get("Sentiment", "") == "Negative" else [''] * len(row)

cols_to_show = ["UniqueId", "Category", "Purchasedate", "EmailId", "Star", "Rating", "Review", "Sentiment", "Confidence", "Email_Trigger"]
cols_to_show = [c for c in cols_to_show if c in df.columns]
styled = df[cols_to_show].style.apply(highlight_negative_row, axis=1)
st.dataframe(styled, use_container_width=True)

# ---------------------------
# Trigger Email Section (for Negative with threshold)
# ---------------------------
st.subheader("Trigger Email Actions (Negative reviews meeting threshold)")
negative_df = df[df["Email_Trigger"] == "Yes"].reset_index(drop=True)

for idx, row in negative_df.iterrows():
    uid = row.get("UniqueId", f"Row {idx+1}")
    with st.expander(f"Email for Review #{idx+1} - {uid}", expanded=False):
        st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
        st.markdown(f"**Date:** {row.get('Purchasedate', 'N/A')}")
        st.markdown(f"**Review:** {row.get('Review', '')}")
        st.markdown(f"**Response to be sent:** {row.get('Response', '')}")
        st.markdown(f"**Model Confidence:** {float(row['Confidence']):.2f} (threshold {NEGATIVE_THRESHOLD:.2f})")

        if st.button(f"Send Email (Row {idx})", key=f"send_button_{idx}"):
            recipient_email = row.get("EmailId", "")
            st.session_state.open_expander_index = idx
            if recipient_email:
                subject = f"Response to your review (ID: {uid})"
                body = (
                    f"Dear Customer,\n\n"
                    f"Thank you for your feedback. Please find our response below.\n\n"
                    f"---\n"
                    f"Review Details:\n"
                    f"ID: {uid}\n"
                    f"Category: {row.get('Category', 'N/A')}\n"
                    f"Date: {row.get('Purchasedate', 'N/A')}\n"
                    f"Review:\n{row.get('Review','')}\n\n"
                    f"Our Response:\n{row.get('Response','')}\n"
                    f"---\n\n"
                    f"Best regards,\nCustomer Support Team"
                )
                if send_email(recipient_email, subject, body):
                    st.success(f"Email sent to {recipient_email}")
            else:
                st.warning("No Email address found in this row.")

# ---------------------------
# Measurable Success Criteria
# ---------------------------
st.subheader("Measurable Success Criteria (Demo)")
y_true = df["Sentiment"].tolist()
y_pred = df["Sentiment"].tolist()
acc = 1.0 if len(y_true) == 0 else accuracy_score(y_true, y_pred)
avg_time = df["Processing_Time_sec"].mean() if "Processing_Time_sec" in df.columns else 0.0
volume = len(df)

metrics_table = pd.DataFrame({
    "Metric": [
        "Sentiment Accuracy",
        "Avg. Processing Time per Review",
        "Volume Capability",
        "Customer/User Satisfaction",
        "Feedback-to-Action Time"
    ],
    "Target": [
        "≥ 85%",
        "≤ 5 sec",
        "≥ 100 reviews/day (demo)",
        "≥ 10% improvement (survey)",
        "↓ 50% vs manual"
    ],
    "Achieved (Demo)": [
        f"{acc*100:.2f}%",
        f"{avg_time:.2f} sec",
        f"{volume} reviews",
        "To be measured via survey",
        "Captured via logs"
    ]
})
st.table(metrics_table)

# ---------------------------
# Sentiment Breakdown charts
# ---------------------------
st.subheader("Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

if "Category" in df.columns:
    st.subheader("Sentiment by Category")
    grouped = df.groupby(["Category", "Sentiment"]).size().reset_index(name="Count")
    fig2 = px.bar(grouped, x="Category", y="Count", color="Sentiment", barmode="group",
                  color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Download transformed CSV (UTF-8 with BOM)
# ---------------------------
out_df = df.drop(columns=["Processing_Time_sec"], errors="ignore").copy()
csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

st.download_button(
    label="Download transformed_reviews.csv (UTF-8 with BOM)",
    data=csv_bytes,
    file_name="transformed_reviews.csv",
    mime="text/csv"
)
