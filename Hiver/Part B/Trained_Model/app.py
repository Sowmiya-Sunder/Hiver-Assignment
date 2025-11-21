import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📧 Email Tagging + Sentiment Analysis (Hiver Assignment)")

df = pd.read_excel("Data.xlsx")   
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")


sentiment_labels = []
for text in df["body"]:
    text = text.lower()

    if any(word in text for word in [
        "error", "fail", "issue", "not working", "stopped working",
        "can't", "cannot", "unable", "bug", "crash",
        "problem", "incorrect", "charged", "duplicate",
        "disappeared", "stuck", "delay", "missing", "failing",
        "authorization required", "slow", "not loading"
    ]):
        sentiment_labels.append("negative")

    elif any(word in text for word in [
        "thanks", "thank you", "appreciate", "great", "good", "awesome"
    ]):
        sentiment_labels.append("positive")

    else:
        sentiment_labels.append("neutral")

df["sentiment"] = sentiment_labels
customer_to_tags = df.groupby("customer_id")["tag"].unique().to_dict()

tag_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
X_tag = tag_vectorizer.fit_transform(df["text"])
y_tag = df["tag"]

tag_clf = LogisticRegression(max_iter=2000)
tag_clf.fit(X_tag, y_tag)

def predict_tag(text, customer_id):
    Xv = tag_vectorizer.transform([text])
    probs = tag_clf.predict_proba(Xv)[0]

    all_tags = tag_clf.classes_
    allowed_tags = customer_to_tags[customer_id]

    filtered = {tag: probs[list(all_tags).index(tag)] for tag in allowed_tags}

    final_tag = max(filtered, key=filtered.get)
    confidence = filtered[final_tag]

    return final_tag, round(float(confidence), 3)

sent_vec = TfidfVectorizer(max_features=2000)
X_sent = sent_vec.fit_transform(df["text"])
y_sent = df["sentiment"]

sent_clf = MultinomialNB()
sent_clf.fit(X_sent, y_sent)


def predict_sentiment(text):
    Xv = sent_vec.transform([text])
    pred = sent_clf.predict(Xv)[0]
    conf = sent_clf.predict_proba(Xv).max()
    return pred, round(float(conf), 3)


st.header("🔍 Enter Email Details")

customer_id = st.selectbox("Select Customer ID", df["customer_id"].unique())
subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

combined_text = subject + " " + body

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏷️ Predict Tag")
    if st.button("Predict Tag"):
        tag, conf = predict_tag(combined_text, customer_id)
        st.success(f"Predicted Tag: **{tag}**")
        st.info(f"Confidence: **{conf}**")

with col2:
    st.subheader("🙂 Sentiment Analysis")
    if st.button("Predict Sentiment"):
        sentiment, conf = predict_sentiment(combined_text)

        if sentiment == "positive":
            st.success(f"Sentiment: **{sentiment}**")
        elif sentiment == "negative":
            st.error(f"Sentiment: **{sentiment}**")
        else:
            st.warning(f"Sentiment: **{sentiment}**")

        st.info(f"Confidence: **{conf}**")


