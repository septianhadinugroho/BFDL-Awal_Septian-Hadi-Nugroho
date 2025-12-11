# 🚀 Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini bertujuan untuk melakukan analisis sentimen (Sentiment Analysis) terhadap ulasan pengguna aplikasi **Gojek** di Google Play Store. Proyek ini mencakup seluruh pipeline *machine learning*, mulai dari pengumpulan data (*scraping*), pemrosesan teks (*preprocessing*), pelatihan model (*training*), hingga penggunaan model untuk prediksi (*inference*).

Model terbaik yang dihasilkan menggunakan **DistilBERT** yang telah di-*fine-tune*, mencapai akurasi **\~92%** pada data uji.

## 📂 Struktur Proyek

```
📁 project-root/
│
├── 📜 scraping_gojek.py         # Script untuk mengambil data ulasan dari Google Play Store
├── 📓 submission_gojek.ipynb    # Notebook utama (EDA, Preprocessing, Training Model, Evaluasi)
├── 📜 inference_model.py        # Script untuk mencoba model (prediksi kalimat baru)
├── 📜 requirements.txt          # Daftar library yang dibutuhkan
├── 📊 gojek_reviews_*.csv       # Dataset hasil scraping (Raw Data)
└── 📁 sentiment_model_distilbert/ # Folder output model hasil training (dibuat otomatis oleh notebook)
```

## 🛠️ Instalasi

Pastikan kamu sudah menginstall Python. Disarankan menggunakan virtual environment (venv atau conda).

1.  **Clone repository ini** (jika menggunakan git) atau download file-filenya.
2.  **Install library yang dibutuhkan:**

<!-- end list -->

```bash
pip install -r requirements.txt
```

*Catatan: Untuk pelatihan model Deep Learning (DistilBERT), disarankan menggunakan GPU (CUDA) agar proses lebih cepat.*

## 🚀 Cara Penggunaan

### 1\. Data Scraping (Opsional)

Jika kamu ingin mengambil data ulasan terbaru dari Google Play Store, jalankan script ini. Script ini akan mengambil sekitar 10.000+ ulasan dan menyimpannya dalam format CSV dan JSON.

```bash
python scraping_gojek.py
```

*Output: File `gojek_reviews_yyyymmdd_hhmmss.csv`.*

### 2\. Eksplorasi Data & Pelatihan Model

Buka file Jupyter Notebook `submission_gojek.ipynb`. Notebook ini berisi langkah-langkah:

1.  **Load Data**: Membaca hasil scraping.
2.  **Balancing**: Menyeimbangkan jumlah data positif, netral, dan negatif (Undersampling/Oversampling).
3.  **Preprocessing**: Cleaning, Stopword Removal (Sastrawi), Stemming.
4.  **Modeling**: Membandingkan 3 algoritma:
      * Random Forest + TF-IDF
      * SVM + TF-IDF
      * **DistilBERT (Fine-tuned Transformer)** - *Best Model*
5.  **Saving**: Menyimpan model terbaik ke folder `sentiment_model_distilbert`.

### 3\. Inference (Prediksi Sentimen)

Setelah menjalankan notebook dan model tersimpan, kamu bisa menggunakan script ini untuk memprediksi sentimen dari teks inputan sendiri secara interaktif.

```bash
python inference_model.py
```

**Contoh Penggunaan di Terminal:**

```text
📝 Review: Drivernya ramah banget dan pengantaran cepat
🎯 Sentimen: POSITIF (Confidence: 99.4%)

📝 Review: Aplikasi sering error pas mau order
🎯 Sentimen: NEGATIF (Confidence: 97.4%)
```

## 📊 Hasil Evaluasi Model

Berdasarkan eksperimen yang dilakukan dalam notebook, berikut adalah perbandingan performa model:

| Model | Feature Extraction | Data Split | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | TF-IDF (1-2 gram) | 80/20 | 91.51% | ✅ Good |
| SVM (Linear) | TF-IDF (1-3 gram) | 70/30 | 89.19% | ✅ Good |
| **DistilBERT** | **Transformer Embeddings** | **85/15** | **92.24%** | ⭐ **Best** |

## 🧠 Detail Teknis

  * **Labeling Otomatis**: Dilakukan saat scraping berdasarkan rating bintang.
      * ⭐ 1-2 : Negatif
      * ⭐ 3 : Netral
      * ⭐ 4-5 : Positif
  * **Preprocessing**: Menggunakan library **Sastrawi** untuk Bahasa Indonesia. Stopword removal dimodifikasi agar tidak menghapus kata negasi (seperti: "tidak", "bukan", "jangan") karena penting untuk konteks sentimen.
  * **Base Model**: `distilbert-base-multilingual-cased`.

## 👤 Author

**Septian Hadi Nugroho**

  * Submission Awal — Belajar Fundamental Deep Learning (BFDL)
