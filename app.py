# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- Konfigurasi UI ---
st.set_page_config(page_title="📧 Deteksi Spam Email", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📧 Deteksi Spam Email</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Menggunakan model IndoBERT yang sudah dilatih</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan isi email di bawah ini untuk memeriksa apakah itu SPAM atau bukan.</p>", unsafe_allow_html=True)

# --- Load model & tokenizer ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("indobert_spam_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("indobert_spam_model")
    return tokenizer, model

tokenizer, model = load_model()

# --- Input email ---
with st.form(key="form_email"):
    input_email = st.text_area(
        "✉️ Tulis email kamu di sini:",
        placeholder="Contoh: Selamat! Anda mendapatkan hadiah uang tunai 10 juta rupiah...",
        height=200
    )
    submitted = st.form_submit_button("🔍 Deteksi Sekarang")

# --- Deteksi Spam ---
if submitted:
    if not input_email.strip():
        st.warning("⚠️ Silakan masukkan isi email terlebih dahulu.")
    else:
        with st.spinner("🔍 Menganalisis email..."):
            # Tokenisasi input
            inputs = tokenizer(input_email, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)

            # Hitung probabilitas spam
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            label = int(np.argmax(probs))
            confidence = probs[label]

            # Output hasil
            if label == 1:
                st.error(f"💥 Hasil: Ini adalah SPAM ❗️\n🧠 Probabilitas: {confidence:.2f}")
            else:
                st.success(f"✅ Hasil: Ini adalah email NORMAL 👍\n🧠 Probabilitas: {confidence:.2f}")


# --- Footer Info ---
st.markdown("""
<hr style="margin-top:50px;">
<div class="footer">
    Model: <code>IndoBERT (indobenchmark/indobert-base-p1)</code> | Dibuat oleh 
    <a href="https://github.com/YusufDW1" target="_blank">Yusuf Dwi Saputra</a> dan 
    <a href="https://github.com/Kotsuke" target="_blank">Subandrio</a><br>
</div>
""", unsafe_allow_html=True)

