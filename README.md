# 📊 Customer Review Sentiment Analyzer & Auto-Responder

This **Streamlit** app analyzes customer reviews, detects sentiment using a transformer model, and auto-generates polite, professional responses to **negative feedback** using a local LLM (FLAN-T5). It helps support teams triage reviews and improve customer experience with minimal manual effort.

---

## 🚀 Features

- ✅ **Sentiment Detection**: Classifies reviews as Positive, Neutral, or Negative using `twitter-roberta-base-sentiment`.
- 🧠 **LLM Response Generation**: Auto-generates a short, helpful reply for each negative review using `google/flan-t5-small`.
- 📦 **CSV Upload & Preview**: Upload your own CSV (with `Review_text` column) or use the provided sample.
- 📊 **Visual Breakdown**: Displays sentiment distribution via interactive bar chart.
- ⬇️ **Downloadable Results**: Export processed reviews, sentiments, and responses to CSV.

---

## 📁 File Structure

📦 sentiment-auto-responder/
┣ 📄 app.py # Main Streamlit application
┣ 📄 sample_data.csv # Sample input data
┣ 📄 README.md # This file


## 🛠️ Requirements

Install required packages:
pip install streamlit pandas plotly transformers torch

## 🧪 How to Run

streamlit run app.py
By default, it uses sample_data.csv in the same folder. You can also upload your own CSV file.

Required CSV Format
Your input CSV must have at least one column named:
Review_text
Optional additional columns like Unique_ID, Category will be retained and shown in the output.

## 🤖 Models Used

* Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment
* LLM for Reply Generation: google/flan-t5-small

Both models are downloaded and cached on first run.

## 📷 Sample Output

Unique_ID	  Category	         Review_text	                   Sentiment	              Response
12345	      Billing	     "Your system double-charged me."     	Negative	     Thank you for your review. We will look into the issue. We're sorry to hear about the billing issue...
12346	      UX	         "App is easy to use and smooth."     	Positive	     No response needed.

## 📊 Sentiment Breakdown

The app shows a dynamic bar chart like:

✅ Positive: 60%
😐 Neutral: 25%
❌ Negative: 15%

## ⬇️ Export

You can download the full dataset with generated columns (Sentiment, Response) as a CSV using the Download CSV button.

## 🛡️ License
MIT License © 2025

## 🤝 Contributions
Got improvements? PRs are welcome!

## 📬 Contact
For questions or support, contact: [Your Email or GitHub Handle]





