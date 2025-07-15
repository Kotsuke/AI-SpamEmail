# 📧 Deteksi Spam Email dengan AI (IndoBERT + Streamlit)

Aplikasi web berbasis **Streamlit** untuk mendeteksi apakah sebuah email tergolong **spam** atau **normal (ham)**.  
Model menggunakan **IndoBERT (Transformer Bahasa Indonesia)** yang sudah di-finetune menggunakan dataset kombinasi **lokal dan mirip spam**.

---

## 🚀 Fitur Utama

- ✅ Prediksi email spam/ham dengan akurasi tinggi (berbasis IndoBERT)
- 📚 Mendukung Bahasa Indonesia
- 🌐 Aplikasi berbasis web (Streamlit)
- 🧠 Model transformer finetuned dengan data lokal
- ⚡ Bisa jalan di CPU atau GPU

---

## 🧪 Cara Menjalankan Aplikasi

### 1. Clone Project

```bash
git clone https://github.com/Kotsuke/AI-SpamEmail.git
cd AI-SpamEmail
```

### 2. (Opsional) Buat Virtual Environment
Hapus venv yang lama dan buat baru karena direktori akan menyangkut ke yang lama

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```
Jika kalian memiliki GPU Nvidia kalian bisa menggunakan code dibawah ini untuk mempercepat proses instalasi kalau tidak akan menggunakan CPU.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Retrain Model IndoBERT nya
```bash
python train.py
```

### 5. Jalankan Aplikasinya
```bash
streamlit run app.py
```
