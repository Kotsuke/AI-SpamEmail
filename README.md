# 📧 Deteksi Spam Email dengan AI (Naive Bayes + Streamlit)

Aplikasi web sederhana berbasis **Streamlit** untuk mendeteksi apakah isi sebuah email tergolong **spam** atau **normal (ham)**. Model ini menggunakan algoritma **Multinomial Naive Bayes** dan dilatih dari kombinasi dataset **internasional (spam.csv)** dan **lokal Bahasa Indonesia (email_spam_indo.csv)**.

---

## 🚀 Fitur Utama

- ✅ Prediksi apakah email termasuk spam atau tidak
- 🌐 Bisa input teks langsung lewat web app
- ⚡ Model ringan & cepat, cocok untuk demo/edukasi
- 🇮🇩 Mendukung email Bahasa Indonesia & Inggris

---

## 📂 Struktur Proyek
- app.py # Aplikasi utama (Streamlit)
- train.py # Script training model AI
- spam.csv # Dataset global (Bahasa Inggris)
- email_spam_indo.csv # Dataset lokal (Bahasa Indonesia)
- spam_model.pkl # Model AI hasil training
- vectorizer.pkl # TF-IDF vectorizer
- requirement.txt # Library yang dibutuhkan
- README.md # Dokumentasi ini
