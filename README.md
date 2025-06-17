# 📧 Deteksi Spam Email dengan AI (Naive Bayes + Streamlit)

Aplikasi web sederhana berbasis **Streamlit** untuk mendeteksi apakah isi sebuah email tergolong **spam** atau **normal** (bukan spam). Model ini menggunakan algoritma **Multinomial Naive Bayes** dan dilatih dari kombinasi **dataset internasional** dan **dataset bahasa Indonesia**.

---

## 🚀 Fitur Utama

- ✉️ Prediksi spam atau bukan dari teks email yang dimasukkan
- 🔎 Input teks via antarmuka web (Streamlit)
- 🧠 Model ringan dan cepat menggunakan Naive Bayes
- 🇮🇩 Mendukung email berbahasa Indonesia dan Inggris

---

## 📁 Struktur Proyek

├── app.py # Aplikasi Streamlit
├── train.py # Script training model
├── spam.csv # Dataset global (Inggris)
├── email_spam_indo.csv # Dataset lokal (Bahasa Indonesia)
├── spam_model.pkl # Model AI hasil training
├── vectorizer.pkl # TF-IDF vectorizer
├── requirement.txt # Daftar library yang dibutuhkan
└── README.md # Dokumentasi ini

asd
---

## 🧪 Cara Cepat Menjalankan Aplikasi

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector


python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirement.txt

streamlit run app.py

## 🔄 Training Ulang Model
Jika kamu ingin melatih ulang model dari awal (misalnya setelah mengedit dataset):

bash
Copy
Edit
python train.py
