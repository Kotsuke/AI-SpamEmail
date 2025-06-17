import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load dataset global (spam.csv)
df1 = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df1.columns = ["label", "text"]

# Load dataset lokal Indonesia (email_spam_indo.csv)
df2 = pd.read_csv("email_spam_indo.csv")[["Kategori", "Pesan"]]
df2.columns = ["label", "text"]

# Gabung dua dataset
df = pd.concat([df1, df2], ignore_index=True)

# Normalisasi label
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})

# Drop baris yang label-nya bukan spam/ham (kalau ada null)
df = df.dropna(subset=["label", "text"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF + Model
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# menyimpan model dan vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model dan vectorizer berhasil disimpan.")