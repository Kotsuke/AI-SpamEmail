import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# 1. Load Indo dataset
df = pd.read_csv("email_spam_indo.csv")[["Kategori", "Pesan"]]
df.columns = ["label", "text"]
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
df = df.dropna()

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Vectorizer + Model
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 4. Evaluasi
y_pred = model.predict(X_test_vec)
print("📊 Evaluasi Model:")
print(classification_report(y_test, y_pred))

# 5. Simpan model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
