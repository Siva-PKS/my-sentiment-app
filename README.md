![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Siva-PKS.my-sentiment-app&left_color=black&right_color=blue&left_text=Visitors)
![GitHub all releases](https://img.shields.io/github/downloads/Siva-PKS/my-sentiment-app/total.svg)
![Dependabot alerts](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Siva-PKS/my-sentiment-app/main/badges/dependabot.json)


## Clone Stats

<!-- CLONE-STATS:START -->
Last updated: **2026-02-10 03:28:00.482 UTC**

- **Total clones (14 days):** `0`
- **Unique cloners (14 days):** `0`

<details><summary>Daily breakdown</summary>

- (unable to fetch traffic data right now)

</details>
<!-- CLONE-STATS:END -->


## Top Referrers (last 14 days)
<!-- REFERRERS:START -->
Last updated: **2026-02-10 03:31:43.386 UTC**

- (no data in the last 14 days)
<!-- REFERRERS:END -->


# Customer Review Sentiment Analyzer & Auto‑Responder

A Streamlit app that ingests customer reviews, detects sentiment with a transformer model, and **auto‑generates polite, professional replies to negative feedback** using a local LLM (FLAN‑T5). It helps support teams triage reviews and improve customer experience with minimal manual effort. It can also **send the generated replies via email** (SMTP), one review at a time.

---

##  Features

-- **Sentiment Detection**
Classifies each review as Positive / Neutral / Negative using cardiffnlp/twitter-roberta-base-sentiment.

-- **LLM Response Generation (local)**
For Negative reviews, creates a short, empathetic reply via google/flan-t5-small (runs locally; no external API keys needed).

-- **Email Sending (SMTP)**
One‑click Send Email button (per negative review) that emails the generated reply to the address in your CSV.

-- **CSV Upload & Preview**
Upload your own CSV (must contain a Review_text column) or use sample_data.csv in the repo. Preview table highlights negative rows.

-- **Progress & Caching**
Visual progress bar while processing; models are cached with @st.cache_resource to speed up subsequent runs.

-- **Metrics Dashboard**
Shows demo metrics (accuracy placeholder, avg. processing time, volume estimate, satisfaction & action‑time placeholders).

-- **Visual Breakdown**
Interactive Plotly bar charts for overall sentiment distribution and per‑category breakdown (if Category column exists).

-- **Downloadable Results**
Export processed data (including Sentiment, Confidence, Response, Email_Trigger) as CSV.

---

## File Structure

 sentiment-auto-responder/
┣ 📄 app.py # Main Streamlit application
┣ 📄 sample_data.csv # Sample input data
┣ 📄 README.md # This file


##  Requirements

Python 3.9–3.11

Install packages:
pip install streamlit pandas plotly transformers torch scikit-learn
- The first run will download Hugging Face models (~ a few hundred MB). They are cached for later runs.

## How to Run

streamlit run app.py
By default, it uses sample_data.csv in the same folder. You can also upload your own CSV file.

Required CSV Format
Your input CSV must have at least one column named:
Review_text
Optional additional columns like Unique_ID, Category will be retained and shown in the output.

## Models Used

* Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment
* LLM for Reply Generation: google/flan-t5-small

Both models are downloaded and cached on first run.

## Sample Output

Unique_ID	  Category	         Review_text	                   Sentiment	              Response
12345	      Billing	     "Your system double-charged me."     	Negative	     Thank you for your review. We will look into the issue. We're sorry to hear about the billing issue...
12346	      UX	         "App is easy to use and smooth."     	Positive	     No response needed.

## Sentiment Breakdown

The app shows a dynamic bar chart like:

✅ Positive: 60%
😐 Neutral: 25%
❌ Negative: 15%

##  Export

You can download the full dataset with generated columns (Sentiment, Response) as a CSV using the Download CSV button.

## Known Limits & Notes

**Demo Row Cap:** App limits to the first 100 rows to keep the demo snappy. Adjust MAX_ROWS in app.py as needed.

**LLM Scope:** flan-t5-small keeps things local but is small; you can swap in a larger T5 or other models if you have resources/GPU.

**Torch + Streamlit:** The app includes a small workaround for a historical torch._classes issue.

**Accuracy Metric:** As noted, the default accuracy is a placeholder until you supply ground truth labels.

##  License
MIT License © 2025

## Contributions
Got improvements? PRs are welcome!

##  Contact
For questions or support, contact: [Your Email or GitHub Handle]





