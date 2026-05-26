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

st.title("StressTracker AI Dashboard")
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
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        if 'stress_level' in df.columns:
            df_stress = df['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}).value_counts().sort_index().reset_index()
            df_stress.columns = ['Tingkat Stres', 'Jumlah']
            
            fig_stress = px.bar(
                df_stress, x='Tingkat Stres', y='Jumlah', color='Tingkat Stres',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title="Distribusi Tingkat Stres Target"
            )
            fig_stress.update_layout(showlegend=False, xaxis_title="Kategori Stres", yaxis_title="Jumlah Sampel")
            st.plotly_chart(fig_stress, use_container_width=True)
            
    with col_dist2:
        target_num = st.selectbox(
            "Pilih Variabel Numerik untuk Melihat Sebaran & Nilai Tengah:",
            options=["durasi_tidur_menit", "screen_sebelum_tidur", "jam_kerja_menit", "waktu_outdoor", "tingkat_kecemasan"]
        )
        if target_num in df.columns:
            mean_val = df[target_num].mean()
            median_val = df[target_num].median()
            
            fig_hist = px.histogram(
                df, x=target_num, color_discrete_sequence=['#3498db'],
                title=f"Histogram Distribusi {target_num.replace('_', ' ').title()}"
            )
            fig_hist.update_traces(opacity=0.85)
            fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val:.1f}")
            fig_hist.add_vline(x=median_val, line_dash="dash", line_color="orange", annotation_text=f"Median: {median_val:.1f}")
            fig_hist.update_layout(xaxis_title=target_num.replace('_', ' ').title(), yaxis_title="Jumlah Sampel")
            st.plotly_chart(fig_hist, use_container_width=True)

    col_bin_corr1, col_bin_corr2 = st.columns(2)
    
    with col_bin_corr1:
        target_bin = st.selectbox(
            "Pilih Fitur Gaya Hidup untuk Cek Frekuensi Jawaban:",
            options=['sering_terbangun_malam', 'mimpi_buruk', 'minum_kopi_hari_ini', 'merokok', 
                     'konsumsi_alkohol', 'deadline_hari_ini', 'lembur', 'aktivitas_hobi', 
                     'konflik_interpersonal', 'merasa_kesepian', 'meditasi']
        )
        if target_bin in df.columns:
            df_bin_counts = df[target_bin].value_counts().reset_index()
            df_bin_counts.columns = [target_bin, 'Jumlah']
            fig_bin = px.bar(
                df_bin_counts, x=target_bin, y='Jumlah', color=target_bin,
                color_discrete_map={'Tidak': '#2ecc71', 'Ya': '#e74c3c'},
                title=f"Distribusi Jawaban Variabel Biner - {target_bin.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_bin, use_container_width=True)
            
    with col_bin_corr2:
        df_corr = df.copy()
        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = df_corr[col].astype('category').cat.codes
            
        korelasi_target = df_corr.corr()['stress_level'].drop('stress_level').sort_values(ascending=False).reset_index()
        korelasi_target.columns = ['Fitur', 'Nilai Korelasi']
        korelasi_target['Arah Pengaruh'] = korelasi_target['Nilai Korelasi'].apply(lambda x: 'Memperberat Stres (+)' if x > 0 else 'Meredam Stres (-)')
        
        fig_corr = px.bar(
            korelasi_target, x='Nilai Korelasi', y='Fitur', orientation='h', color='Arah Pengaruh',
            color_discrete_map={'Memperberat Stres (+)': '#e74c3c', 'Meredam Stres (-)': '#3498db'},
            title="Pengaruh dan Nilai Korelasi Setiap Fitur terhadap Tingkat Stres"
        )
        fig_corr.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Koefisien Korelasi", yaxis_title="Fitur")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.write("#### Analisis Karakteristik Pola Hidup vs Kategori Stres (Extended EDA)")
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        if 'durasi_tidur_menit' in df.columns and 'stress_level' in df.columns:
            df_sleep_avg = df.groupby('stress_level')['durasi_tidur_menit'].mean().reset_index()
            df_sleep_avg['stress_level'] = df_sleep_avg['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            
            fig_sleep_bar = px.bar(
                df_sleep_avg, x='stress_level', y='durasi_tidur_menit', color='stress_level',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title="Rata-rata Durasi Tidur (Menit) di Tiap Tingkat Stres"
            )
            fig_sleep_bar.update_layout(showlegend=False, xaxis_title="Tingkat Stres", yaxis_title="Rata-rata Durasi Tidur (Menit)")
            st.plotly_chart(fig_sleep_bar, use_container_width=True)
            
    with col_ex2:
        exda_select = st.selectbox(
            "Pilih Parameter Dampak untuk Grouped Bar Analysis:",
            options=["sering_terbangun_malam", "mimpi_buruk", "merasa_kesepian", "merokok", "konsumsi_alkohol", "meditasi", "aktivitas_hobi"]
        )
        if exda_select in df.columns:
            df_exda_group = df.groupby([exda_select, 'stress_level']).size().reset_index(name='Jumlah')
            df_exda_group['stress_level'] = df_exda_group['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            
            fig_exda_grouped = px.bar(
                df_exda_group, x='stress_level', y='Jumlah', color=exda_select, barmode='group',
                color_discrete_map={'Tidak': '#2ecc71', 'Ya': '#e74c3c'},
                title=f"Proporsi Tingkat Stres Berdasarkan Status {exda_select.replace('_', ' ').title()}"
            )
            fig_exda_grouped.update_layout(xaxis_title="Tingkat Stres", yaxis_title="Jumlah Sampel")
            st.plotly_chart(fig_exda_grouped, use_container_width=True)

    st.info(
        "Kesimpulan Analisis Data & Model: \n"
        "1) Faktor Risiko Utama: Hasil eksplorasi menunjukkan adanya pengaruh kuat dari durasi tidur yang rendah serta tingginya screen time sebelum tidur terhadap peningkatan level stres. Pola gangguan tidur ini juga diperkuat oleh data karakteristik kelompok harian, di mana subjek dengan indikator mimpi buruk dan sering terbangun malam memiliki kecenderungan masuk ke dalam kategori stres tingkat tinggi. \n"
        "2) Faktor Protektif & Gaya Hidup: Berdasarkan analisis korelasi dan distribusi variabel penunjang, aktivitas positif seperti meditasi dan manajemen waktu outdoor terbukti memiliki koefisien korelasi yang bernilai negatif terhadap stress_level, yang menandakan bahwa faktor-faktor tersebut bekerja sebagai peredam atau penahan laju risiko stres. \n"
        "3) Solusi Prevensi (Model AI): Melalui pemetaan interaksi seluruh fitur gaya hidup tersebut, model Deep Learning yang ditanamkan pada sistem berhasil memberikan kalkulasi prediksi secara real-time dengan tingkat akurasi pengujian sebesar 96.73%. Hasil probabilitas ini dapat dijadikan landasan ilmiah untuk langkah intervensi dini."
    )

    st.subheader("Simulasi Interaktif Prediksi Model Real-Time")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("Model 'best_model.keras' tidak ditemukan. Fitur simulasi prediksi dinonaktifkan.")
    else:
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
            NUMERIC_FEATURES = ['usia', 'durasi_tidur_menit', 'screen_sebelum_tidur', 'waktu_outdoor']
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
                    'usia': input_usia, 'jenis_kelamin': input_jk, 'pekerjaan': input_job,
                    'durasi_tidur_menit': float(input_tidur), 'sering_terbangun_malam': input_terbangun,
                    'mimpi_buruk': input_mimpi, 'screen_sebelum_tidur': float(input_screen),
                    'minum_kopi_hari_ini': input_kopi, 'merokok': input_rokok, 'konsumsi_alkohol': input_alkohol,
                    'deadline_hari_ini': input_deadline, 'lembur': input_lembur, 'waktu_outdoor': float(input_outdoor),
                    'aktivitas_hobi': input_hobi, 'suasana_hati': input_mood, 'konflik_interpersonal': input_konflik,
                    'merasa_kesepian': input_kesepian, 'meditasi': input_meditasi, 'konsentrasi': int(input_konsentrasi),
                    'interaksi_sosial': int(input_interaksi)
                }])
                
                input_filtered_df = raw_input_df[FINAL_FEATURES].copy()
                
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