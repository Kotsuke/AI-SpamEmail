import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

# =====================
# 1. Load dua dataset
# =====================

# Dataset global (bahasa Inggris)
df1 = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df1.columns = ["label", "text"]

# Dataset lokal (bahasa Indonesia)
df2 = pd.read_csv("email_spam_indo.csv")[["Kategori", "Pesan"]]
df2.columns = ["label", "text"]

# Gabungkan dataset
df = pd.concat([df1, df2], ignore_index=True)

# =====================
# 2. Preprocessing
# =====================

# Normalisasi label: ham -> 0, spam -> 1
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})

# Buang baris yang kosong/null
df = df.dropna(subset=["label", "text"])

# =====================
# 3. Oversampling Spam
# =====================

# Pisahkan ham dan spam
df_ham = df[df["label"] == 0]
df_spam = df[df["label"] == 1]

# Oversample data spam agar seimbang dengan ham
df_spam_upsampled = resample(df_spam,
                             replace=True,
                             n_samples=len(df_ham),
                             random_state=42)

# Gabungkan kembali dan acak
df_balanced = pd.concat([df_ham, df_spam_upsampled]).sample(frac=1, random_state=42)

# =====================
# 4. Split & Vectorize
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"], df_balanced["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# =====================
# 5. Train Model
# =====================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =====================
# 6. Save Model & Vectorizer
# =====================
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model dan vectorizer berhasil disimpan.")
from sklearn.metrics import classification_report

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("\n📊 Evaluasi Model:")
print(classification_report(y_test, y_pred))
