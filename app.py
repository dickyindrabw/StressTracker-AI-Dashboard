import streamlit as st
import pandas as pd
import os
import plotly.express as px
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="StressTracker AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dashboard Interaktif - StressTracker AI")
st.caption("Aplikasi monitoring internal untuk menampilkan visualisasi data, analisis insight, dan inferensi model secara real-time.")

CSV_PATH = os.path.join("Data Bersih", "StressTracker_Clean.csv")
MODEL_PATH = "best_model.keras"

if not os.path.exists(CSV_PATH):
    st.error("Data source 'StressTracker_Clean.csv' tidak ditemukan di folder 'Data Bersih'. Process halted.")
else:
    df = pd.read_csv(CSV_PATH)
    
    st.subheader("Indikator Utama & Metrik Kesehatan Gaya Hidup")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        avg_sleep = df['durasi_tidur_menit'].mean() / 60 if 'durasi_tidur_menit' in df.columns else 0
        st.metric(label="Rata-rata Durasi Tidur", value=f"{avg_sleep:.1f} Jam")
    with m2:
        avg_screen = df['screen_sebelum_tidur'].mean() if 'screen_sebelum_tidur' in df.columns else 0
        st.metric(label="Rata-rata Screen Time", value=f"{avg_screen:.0f} Menit")
    with m3:
        mimpi_buruk_pct = (df['mimpi_buruk'].astype(str).str.lower().str.strip() == 'ya').sum() / len(df) * 100 if 'mimpi_buruk' in df.columns else 0
        st.metric(label="Prevalensi Mimpi Buruk", value=f"{mimpi_buruk_pct:.1f}%")
    with m4:
        stres_tinggi_pct = (df['stress_level'] == 3).sum() / len(df) * 100 if 'stress_level' in df.columns else 0
        st.metric(label="Proporsi Stres Tinggi", value=f"{stres_tinggi_pct:.1f}%")

    st.subheader("Visualisasi Insight (Exploratory Data Analysis)")
    grafik_kolom1, grafik_kolom2 = st.columns(2)

    with grafik_kolom1:
        if 'stress_level' in df.columns:
            df_stress = df['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}).value_counts().reset_index()
            df_stress.columns = ['Tingkat Stres', 'Jumlah']
            
            fig_stress = px.bar(
                df_stress, 
                x='Tingkat Stres', 
                y='Jumlah',
                color='Tingkat Stres',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title="Distribusi Tingkat Stres Target"
            )
            fig_stress.update_layout(showlegend=False, xaxis_title="Kategori Stres", yaxis_title="Jumlah Sampel")
            st.plotly_chart(fig_stress, use_container_width=True)

    with grafik_kolom2:
        if 'pekerjaan' in df.columns:
            df_job = df['pekerjaan'].astype(str).str.title().str.strip().value_counts().reset_index()
            df_job.columns = ['Jenis Pekerjaan', 'Jumlah']
            
            fig_job = px.bar(
                df_job, 
                x='Jumlah', 
                y='Jenis Pekerjaan',
                orientation='h',
                color='Jenis Pekerjaan',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Sebaran Populasi Berdasarkan Jenis Pekerjaan"
            )
            fig_job.update_layout(showlegend=False, xaxis_title="Jumlah Sampel", yaxis_title="Pekerjaan")
            st.plotly_chart(fig_job, use_container_width=True)

    # Validasi visualisasi faktor numerik utama pendukung insight kesimpulan
    st.write("#### Analisis Faktor Risiko Utama (Faktor Numerik)")
    grafik_num1, grafik_num2 = st.columns(2)

    with grafik_num1:
        if 'durasi_tidur_menit' in df.columns and 'stress_level' in df.columns:
            df_box_sleep = df.copy()
            df_box_sleep['stress_level'] = df_box_sleep['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            df_box_sleep['Durasi Tidur (Jam)'] = df_box_sleep['durasi_tidur_menit'] / 60
            
            fig_sleep = px.box(
                df_box_sleep,
                x='stress_level',
                y='Durasi Tidur (Jam)',
                color='stress_level',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title="Pengaruh Durasi Tidur terhadap Tingkat Stres"
            )
            fig_sleep.update_layout(showlegend=False, xaxis_title="Tingkat Stres")
            st.plotly_chart(fig_sleep, use_container_width=True)

    with grafik_num2:
        if 'screen_sebelum_tidur' in df.columns and 'stress_level' in df.columns:
            df_box_screen = df.copy()
            df_box_screen['stress_level'] = df_box_screen['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            
            fig_screen = px.box(
                df_box_screen,
                x='stress_level',
                y='screen_sebelum_tidur',
                color='stress_level',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title="Pengaruh Screen Time Sebelum Tidur terhadap Tingkat Stres"
            )
            fig_screen.update_layout(showlegend=False, xaxis_title="Tingkat Stres", yaxis_title="Screen Time (Menit)")
            st.plotly_chart(fig_screen, use_container_width=True)

    st.subheader("Analisis Hubungan Fitur Interaktif")
    fitur_pilihan = st.selectbox(
        "Pilih variabel gaya hidup untuk menganalisis korelasinya dengan tingkat stres:",
        options=[
            "jenis_kelamin", "pekerjaan", "sering_terbangun_malam", "mimpi_buruk", 
            "minum_kopi_hari_ini", "merokok", "konsumsi_alkohol", "deadline_hari_ini", 
            "lembur", "aktivitas_hobi", "suasana_hati", "konflik_interpersonal", 
            "merasa_kesepian", "meditasi", "konsentrasi", "interaksi_sosial"
        ]
    )
    
    if fitur_pilihan in df.columns and 'stress_level' in df.columns:
        df_filtered = df.copy()
        df_filtered['stress_level'] = df_filtered['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
        df_group = df_filtered.groupby([fitur_pilihan, 'stress_level']).size().reset_index(name='Jumlah')
        
        fig_interaktif = px.bar(
            df_group,
            x=fitur_pilihan,
            y='Jumlah',
            color='stress_level',
            barmode='group',
            color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
            title=f"Analisis Perbandingan Fitur {fitur_pilihan.replace('_', ' ').title()} terhadap Tingkat Stres Target"
        )
        st.plotly_chart(fig_interaktif, use_container_width=True)

    st.info(
        "Kesimpulan Analisis Data & Model: \n"
        "1) Faktor Risiko Utama: Hasil EDA menunjukkan korelasi kuat antara durasi tidur yang tidak ideal (di bawah 7 jam) "
        "serta tingginya screen time sebelum tidur terhadap peningkatan risiko stres tingkat tinggi. Pola ini diperkuat oleh "
        "tingginya prevalensi gangguan tidur berupa mimpi buruk pada kelompok stres kronis. \n"
        "2) Faktor Protektif: Aktivitas luar ruangan (outdoor time) dan manajemen waktu istirahat teratur terbukti secara "
        "signifikan menjaga stabilitas emosional subjek pada tingkat stres rendah. \n"
        "3) Solusi Prevensi (Model AI): Berdasarkan pola-pola historis tersebut, model Deep Learning yang diintegrasikan di bawah "
        "berhasil mempelajari interaksi kompleks dari 20 fitur gaya hidup ini secara simultan. Model mampu menghasilkan kalkulasi "
        "probabilitas stres secara real-time sebagai landasan ilmiah untuk fitur intervensi dini pada sistem Tracker."
    )

    st.subheader("Simulasi Interaktif Prediksi Model Real-Time")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("Model 'best_model.keras' tidak ditemukan. Fitur simulasi prediksi dinonaktifkan.")
    else:
        # Cache resource agar model tidak dimuat ulang setiap interaksi widget
        @st.cache_resource
        def load_internal_model():
            return tf.keras.models.load_model(MODEL_PATH)
            
        try:
            model = load_internal_model()
            st.success("Model (.keras) sukses dimuat. Masukkan parameter input di bawah ini:")
        except Exception as e:
            st.error(f"Error loading model graph: {e}")
            model = None

        if model is not None:
            CATEGORICAL_FEATURES = [
                'jenis_kelamin', 'pekerjaan', 'sering_terbangun_malam', 'mimpi_buruk', 
                'minum_kopi_hari_ini', 'merokok', 'konsumsi_alkohol', 'deadline_hari_ini', 
                'lembur', 'aktivitas_hobi', 'suasana_hati', 'konflik_interpersonal', 
                'merasa_kesepian', 'meditasi', 'konsentrasi', 'interaksi_sosial'
            ]
            
            NUMERIC_FEATURES = [
                'usia', 'durasi_tidur_menit', 'screen_sebelum_tidur', 'waktu_outdoor'
            ]
            
            FINAL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
            
            with st.form("input_form"):
                kol1, kol2, kol3 = st.columns(3)
                
                with kol1:
                    input_usia = st.number_input("Usia", min_value=15, max_value=80, value=25)
                    input_jk = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
                    input_job = st.selectbox("Pekerjaan", options=["Karyawan", "Guru", "Mahasiswa", "Irt", "Dokter", "Polisi", "Pekerja Seni", "Sales", "Pilot", "Arsitek"])
                    input_tidur = st.number_input("Durasi Tidur (Menit)", min_value=60, max_value=600, value=420)
                    input_terbangun = st.selectbox("Sering Terbangun Malam", options=["Tidak", "Ya"])
                    input_mimpi = st.selectbox("Mimpi Buruk", options=["Tidak", "Ya"])
                    
                with kol2:
                    input_screen = st.number_input("Screen Time Sebelum Tidur (Menit)", min_value=0, max_value=300, value=45)
                    input_outdoor = st.number_input("Waktu Outdoor (Menit)", min_value=0, max_value=240, value=30)
                    input_kopi = st.selectbox("Minum Kopi Hari Ini", options=["Tidak", "Ya"])
                    input_rokok = st.selectbox("Merokok", options=["Tidak", "Ya"])
                    input_alkohol = st.selectbox("Konsumsi Alkohol", options=["Tidak", "Ya"])
                    input_deadline = st.selectbox("Deadline Hari Ini", options=["Tidak", "Ya"])
                    
                with kol3:
                    input_lembur = st.selectbox("Lembur", options=["Tidak", "Ya"])
                    input_hobi = st.selectbox("Aktivitas Hobi", options=["Ya", "Tidak"])
                    input_mood = st.selectbox("Suasana Hati", options=["Netral", "Negatif", "Positif"])
                    input_konflik = st.selectbox("Konflik Interpersonal", options=["Tidak", "Ya"])
                    input_kesepian = st.selectbox("Merasa Kesepian", options=["Tidak", "Ya"])
                    input_meditasi = st.selectbox("Meditasi", options=["Tidak", "Ya"])
                    input_konsentrasi = st.selectbox("Tingkat Konsentrasi (1-5)", options=[1, 2, 3, 4, 5], index=2)
                    input_interaksi = st.selectbox("Tingkat Interaksi Sosial (1-5)", options=[1, 2, 3, 4, 5], index=2)
                    
                submitted = st.form_submit_button("Hitung Prediksi Tingkat Stres")
                
            if submitted:
                raw_input_df = pd.DataFrame([{
                    'usia': input_usia,
                    'jenis_kelamin': input_jk,
                    'pekerjaan': input_job,
                    'durasi_tidur_menit': float(input_tidur),
                    'sering_terbangun_malam': input_terbangun,
                    'mimpi_buruk': input_mimpi,
                    'screen_sebelum_tidur': float(input_screen),
                    'minum_kopi_hari_ini': input_kopi,
                    'merokok': input_rokok,
                    'konsumsi_alkohol': input_alkohol,
                    'deadline_hari_ini': input_deadline,
                    'lembur': input_lembur,
                    'waktu_outdoor': float(input_outdoor),
                    'aktivitas_hobi': input_hobi,
                    'suasana_hati': input_mood,
                    'konflik_interpersonal': input_konflik,
                    'merasa_kesepian': input_kesepian,
                    'meditasi': input_meditasi,
                    'konsentrasi': int(input_konsentrasi),
                    'interaksi_sosial': int(input_interaksi)
                }])
                
                input_filtered_df = raw_input_df[FINAL_FEATURES].copy()
                
                # Casting tipe data spesifik untuk mencegah error tensor graph pas inferensi
                for col in CATEGORICAL_FEATURES:
                    if col in ['konsentrasi', 'interaksi_sosial']:
                        input_filtered_df[col] = input_filtered_df[col].astype('int64')
                    else:
                        input_filtered_df[col] = input_filtered_df[col].astype(str)
                    
                for col in NUMERIC_FEATURES:
                    if col == 'usia':
                        input_filtered_df[col] = input_filtered_df[col].astype('int64')
                    else:
                        input_filtered_df[col] = input_filtered_df[col].astype('float32')

                try:
                    # Konversi ke format tensor pipeline tf.data
                    input_dict = {col: input_filtered_df[col].values for col in input_filtered_df.columns}
                    input_dataset = tf.data.Dataset.from_tensor_slices(input_dict).batch(1)
                    
                    raw_prediction = model.predict(input_dataset, verbose=0)
                    probabilitas = raw_prediction[0]
                    idx_tertinggi = np.argmax(raw_prediction, axis=-1)[0]
                    
                    peta_stres = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
                    hasil_stres = peta_stres[idx_tertinggi]
                    
                    st.write("Persentase Probabilitas Hasil Model AI:")
                    prob_kolom1, prob_kolom2, prob_kolom3 = st.columns(3)
                    with prob_kolom1:
                        st.metric(label="Probabilitas Rendah", value=f"{probabilitas[0]*100:.2f}%")
                    with prob_kolom2:
                        st.metric(label="Probabilitas Sedang", value=f"{probabilitas[1]*100:.2f}%")
                    with prob_kolom3:
                        st.metric(label="Probabilitas Tinggi", value=f"{probabilitas[2]*100:.2f}%")
                    
                    if hasil_stres == "Tinggi":
                        st.error(f"Kesimpulan Model: Tingkat Stres {hasil_stres}")
                        st.warning("Rekomendasi: Terdeteksi beban kerja/stres tinggi yang konsisten. Disarankan membatasi jam lembur dan meluangkan waktu istirahat.")
                    elif hasil_stres == "Rendah":
                        st.success(f"Kesimpulan Model: Tingkat Stres {hasil_stres}")
                        st.info("Rekomendasi: Kondisi kesehatan mental terpantau stabil. Pertahankan ritme pola tidur dan batas waktu penggunaan device harian.")
                    else:
                        st.warning(f"Kesimpulan Model: Tingkat Stres {hasil_stres}")
                        st.info("Rekomendasi: Nilai stress berada pada batas ambang normal wajar. Jaga keseimbangan antara istirahat dan aktivitas.")
                        
                except Exception as e:
                    st.error("Gagal mengeksekusi inferensi real-time pada model internal.")
                    st.caption(f"Log Error: {str(e)}")