import streamlit as st
import joblib

# Judul aplikasi
st.set_page_config(page_title="Deteksi Spam Email", layout="centered")
st.title("📧 Deteksi Spam Email dengan AI")

# Load model dan vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Input dari pengguna
st.write("Masukkan isi email untuk memeriksa apakah itu spam atau bukan.")
input_email = st.text_area("Teks Email", height=200)

# Tombol Prediksi
if st.button("🔍 Deteksi"):
    if input_email.strip() == "":
        st.warning("⚠️ Silakan masukkan teks email terlebih dahulu.")
    else:
        input_vector = vectorizer.transform([input_email])
        result = model.predict(input_vector)

        if result[0] == 1:
            st.error("💥 Hasil: Ini adalah SPAM!")
        else:
            st.success("✅ Hasil: Ini adalah email NORMAL.")
