# StressTracker AI

Aplikasi Monitoring Karakteristik Gaya Hidup dan Inferensi Model Deep Learning untuk Prediksi Level Stres Real-Time.

Aplikasi ini merupakan proyek capstone bertema Healthy Lives & Well-being dengan ID Tim CC26-PSU292 dalam program Coding Camp 2026 powered by DBS Foundation.

Link Live Dashboard: [https://stresstracker-dashboard.streamlit.app/](https://www.google.com/search?q=https://stresstracker-dashboard.streamlit.app/)

## Deskripsi Proyek

Tingkat stres pada masyarakat modern, khususnya di kalangan mahasiswa dan pekerja, terus melonjak seiring dengan menurunnya kualitas tidur akibat pola hidup yang tidak teratur, penggunaan gawai (screen time) berlebihan, serta tinggi dan padatnya tekanan aktivitas harian. Penelitian menunjukkan bahwa gangguan tidur berkontribusi terhadap peningkatan stres dan kecemasan secara dua arah.

StressTracker AI hadir sebagai solusi praktis berbasis data science untuk mendeteksi, memetakan, dan memprediksi level stres harian (Rendah, Sedang, Tinggi) berdasarkan 20 parameter metrik gaya hidup dan kualitas tidur menggunakan arsitektur Deep Learning. Sistem ini diintegrasikan langsung ke dalam dashboard monitoring internal berbasis web yang interaktif dan dinamis untuk membantu proses deteksi dini serta meningkatkan kesadaran terhadap kesehatan mental.

## Anggota Tim (CC26-PSU292)

* Nova Wijaya (CFCC848D6Y2793) - Fullstack Developer
* Fawwas Shelgi Rajabani (CFCC312D6Y0227) - Fullstack Developer
* R Dicky Indra Barata Wijaya (CDCC244D6Y1907) - Data Scientist
* Suroso Aditya Wibowo (CDCC184D6Y1024) - Data Scientist
* Satrio Sanjaya (CACC940D6Y0880) - AI Engineer

## Fitur Utama Dashboard

* Korelasi Antar Feature Terhadap Target: Visualisasi horizontal bar chart terintegrasi yang menunjukkan arah pengaruh positif (memperberat stres) atau negatif (meredam stres) dari setiap variabel gaya hidup terhadap variabel target stress_level.
* Matriks Heatmap Korelasi Linear Penuh: Sajian visualisasi peta panas (full-width view) interaktif berskala warna RdYlGn untuk mengeksplorasi nilai koefisien korelasi linear antar-seluruh fitur numerik dan kategorikal codes secara jelas.
* Analisis Karakteristik Kelompok:
* Grouped Bar Analysis: Membedah proporsi kasus stres secara dinamis berdasarkan dropdown pilihan variabel kategorikal biner seperti mimpi buruk, sering terbangun malam, lembur, dan merasa kesepian.
* Mean Analysis: Membandingkan nilai rata-rata variabel numerik kontinu seperti durasi tidur, screen time, jam kerja, dan usia lintas tingkatan kategori stres.


* Simulasi Interaktif Prediksi Model Real-Time: Formulir input terpadu 3 kolom yang terbagi rata atas parameter demografi, kuantitas istirahat/kerja, dan indikator psikososial untuk menghitung persentase probabilitas hasil prediksi model cerdas secara instan.

## Spesifikasi Teknologi & Sumber Daya

* Data Science & Analytics: Python, Pandas, NumPy, Plotly Express, Scikit-learn, Streamlit.
* AI & Machine Learning: TensorFlow, Keras (best_model.keras terkompilasi dengan arsitektur Fully Connected Dense Layers dan mencatatkan akurasi pengujian final sebesar 96.73%).

## Struktur Direktori Repositori

```text
├── .devcontainer/                    # Konfigurasi container untuk lingkungan pengembangan
├── Data Bersih/
│   └── StressTracker_Clean.csv       # Dataset bersih hasil wrangling (30.177 baris)
├── Data Kotor/                       # Repositori penyimpanan dataset mentah sebelum pembersihan
├── modelAPI/                         # Sistem backend berbasis REST API untuk model inferece
├── .gitignore                        # Berkas konfigurasi pengabaian pelacakan Git
├── Prosesing and Modelling Data.ipynb # Notebook dokumentasi end-to-end data pipeline & modeling
├── README.md                         # Dokumentasi utama proyek
├── app.py                            # Kode utama aplikasi dashboard Streamlit
├── best_model.keras                  # Berkas bobot model Deep Learning terkompilasi
└── requirements.txt                  # Daftar dependensi dan pustaka Python proyek

```

## Panduan Instalasi Lengkap di Laptop

Ikuti langkah-langkah di bawah ini untuk memasang dan menjalankan aplikasi ini secara lokal di laptop Anda:

### 1. Unduh dan Instal Python

Pastikan laptop Anda sudah terinstal Python (disarankan menggunakan Python versi 3.10, 3.11, atau 3.12 untuk stabilitas pustaka TensorFlow). Anda dapat mengunduhnya melalui situs resmi python.org. Pastikan untuk mencentang pilihan "Add Python to PATH" saat proses instalasi berlangsung.

### 2. Kloning Repositori Proyek

Buka Terminal (Mac/Linux) atau Command Prompt/PowerShell (Windows), lalu kloning repositori ini ke penyimpanan lokal laptop Anda:

```bash
git clone https://github.com/username/StressTracker-AI.git
cd StressTracker-AI

```

### 3. Buat dan Aktifkan Virtual Environment (Opsional - Disarankan)

Langkah ini sangat disarankan agar paket pustaka proyek tidak bertabrakan dengan dependensi global di sistem laptop Anda:

* Windows:
```bash
python -m venv venv
.\venv\Scripts\activate

```


* Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate

```



### 4. Instalasi Dependensi / Pustaka yang Diperlukan

Jalankan perintah berikut untuk memasang seluruh paket pustaka utama yang digunakan di dalam sistem dashboard:

```bash
pip install streamlit pandas plotly tensorflow numpy scikit-learn

```

### 5. Jalankan Aplikasi Dashboard Streamlit

Setelah seluruh pustaka berhasil terinstal, jalankan perintah ini di terminal laptop Anda:

```bash
streamlit run app.py

```

Aplikasi secara otomatis akan berjalan dan langsung membuka jendela peramban web (browser) Anda pada alamat lokal: http://localhost:8501.

## Hasil Evaluasi Akhir Model AI

Berdasarkan pengujian ketat pada data independen (testing set) yang dilakukan oleh divisi AI Engineer dan Data Science, model Deep Learning berhasil mengenali dan memetakan pola stres dengan performa sebagai berikut:

* Akurasi Pengujian Final: 96.73% (Melampaui target minimal evaluasi proyek awal sebesar 85%).
* Nilai Precision, Recall, & F1-Score: Stabil di rentang 0.95 hingga 0.98 untuk ketiga kelas target (Rendah, Sedang, Tinggi), menunjukkan kemampuan generalisasi model yang sangat baik terhadap data baru.
  
## Tautan Model ML

Model Deep Learning yang digunakan dapat diunduh melalui tautan berikut:

| File | Deskripsi | Tautan |
|------|-----------|--------|
| best_model.keras | Model utama dashboard Streamlit | (https://drive.google.com/file/d/1wk8PygVp1b4Kjn3kRC7mUpWuZAds_XyT/view?usp=sharing)) |
| best_model_fixed2.keras | Model untuk REST API inference |(https://drive.google.com/file/d/1m1of-oHfweTqUJeCVAzjA1vKvEoF5AbV/view?usp=sharing) |
