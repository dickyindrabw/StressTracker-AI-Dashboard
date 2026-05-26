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

    st.subheader("Exploratory Data Analysis (EDA)")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        if 'stress_level' in df.columns:
            df_stress = df['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}).value_counts(normalize=True).reset_index()
            df_stress.columns = ['Tingkat Stres', 'Persentase']
            df_stress['Persentase'] = df_stress['Persentase'] * 100
            
            fig_stress = px.bar(
                df_stress, x='Tingkat Stres', y='Persentase', color='Tingkat Stres',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                text=df_stress['Persentase'].apply(lambda x: f"{x:.2f}%"),
                title="Distribusi Tingkat Stres Target (Persentase)"
            )
            fig_stress.update_layout(showlegend=False, xaxis_title="Kategori Stres", yaxis_title="Persentase (%)")
            fig_stress.update_traces(textposition='outside')
            st.plotly_chart(fig_stress, use_container_width=True)
            
    with col_eda2:
        chosen_num_eda = st.selectbox(
            "Pilih Fitur untuk Melihat Sebaran Data Kontinu:",
            options=["usia", "durasi_tidur_menit", "screen_sebelum_tidur", "jam_kerja_menit", "waktu_outdoor", "tingkat_kecemasan"]
        )
        if chosen_num_eda in df.columns:
            fig_hist = px.histogram(
                df, x=chosen_num_eda, color_discrete_sequence=['#3498db'],
                title=f"Distribusi Feature Numerikal - {chosen_num_eda.replace('_', ' ').title()}"
            )
            fig_hist.update_traces(opacity=0.85)
            st.plotly_chart(fig_hist, use_container_width=True)

    col_eda3, col_eda4 = st.columns(2)
    
    with col_eda3:
        chosen_bin_eda = st.selectbox(
            "Pilih Fitur untuk Melihat Frekuensi Data Kategorikal:",
            options=['sering_terbangun_malam', 'mimpi_buruk', 'merasa_kesepian', 'meditasi',
                     'minum_kopi_hari_ini', 'merokok', 'konsumsi_alkohol', 'deadline_hari_ini', 
                     'lembur', 'aktivitas_hobi', 'jenis_kelamin', 'pekerjaan', 'suasana_hati']
        )
        if chosen_bin_eda in df.columns:
            df_bin_counts = df[chosen_bin_eda].value_counts().reset_index()
            df_bin_counts.columns = [chosen_bin_eda, 'Jumlah']
            fig_bin = px.bar(
                df_bin_counts, x=chosen_bin_eda, y='Jumlah', color=chosen_bin_eda,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title=f"Distribusi Data Biner/Kategorikal - {chosen_bin_eda.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_bin, use_container_width=True)

    with col_eda4:
        df_corr_target = df.copy()
        for col in df_corr_target.select_dtypes(include='object').columns:
            df_corr_target[col] = df_corr_target[col].astype('category').cat.codes
            
        korelasi_target = df_corr_target.corr()['stress_level'].drop('stress_level').sort_values(ascending=False).reset_index()
        korelasi_target.columns = ['Fitur', 'Nilai Korelasi']
        korelasi_target['Arah Pengaruh'] = korelasi_target['Nilai Korelasi'].apply(lambda x: 'Memperberat Stres (+)' if x > 0 else 'Meredam Stres (-)')
        
        fig_corr = px.bar(
            korelasi_target, x='Nilai Korelasi', y='Fitur', orientation='h', color='Arah Pengaruh',
            color_discrete_map={'Memperberat Stres (+)': '#e74c3c', 'Meredam Stres (-)': '#3498db'},
            title="Korelasi Antar Feature Terhadap Target (stress_level)"
        )
        fig_corr.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Koefisien Korelasi", yaxis_title="Fitur")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.write("#### Matriks Heatmap Korelasi Linear Penuh (EDA)")
    df_corr_full = df.copy()
    for col in df_corr_full.select_dtypes(include='object').columns:
        df_corr_full[col] = df_corr_full[col].astype('category').cat.codes
        
    corr_matrix = df_corr_full.corr()
    fig_heatmap = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Heatmap Korelasi Seluruh Fitur Numerik & Kategorikal Codes"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    st.subheader("Explanatory Data Analysis (ExDA)")
    
    col_exda1, col_exda2 = st.columns(2)
    
    with col_exda1:
        chosen_kat_exda = st.selectbox(
            "Pilih Faktor Dampak Gaya Hidup (Explanatory Grouped Bar Analysis):",
            options=['sering_terbangun_malam', 'mimpi_buruk', 'merasa_kesepian', 'meditasi',
                     'minum_kopi_hari_ini', 'merokok', 'konsumsi_alkohol', 'deadline_hari_ini', 
                     'lembur', 'aktivitas_hobi', 'jenis_kelamin', 'pekerjaan', 'suasana_hati']
        )
        
        if chosen_kat_exda in df.columns:
            df_kat_group = df.groupby([chosen_kat_exda, 'stress_level']).size().reset_index(name='Jumlah')
            df_kat_group['stress_level'] = df_kat_group['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            
            fig_kat_grouped = px.bar(
                df_kat_group, x='stress_level', y='Jumlah', color=chosen_kat_exda, barmode='group',
                color_discrete_map={'Tidak': '#2ecc71', 'Ya': '#e74c3c', 'Laki-laki': '#3498db', 'Perempuan': '#e84393', 'Netral': '#f39c12', 'Negatif': '#e74c3c', 'Positif': '#2ecc71'},
                title=f"Proporsi Kasus Stres Berdasarkan Variabel: {chosen_kat_exda.replace('_', ' ').title()}"
            )
            fig_kat_grouped.update_layout(xaxis_title="Tingkat Stres", yaxis_title="Jumlah Sampel")
            st.plotly_chart(fig_kat_grouped, use_container_width=True)
            
    with col_exda2:
        chosen_num_exda = st.selectbox(
            "Pilih Faktor Numerik Kontinu (Explanatory Mean Analysis):",
            options=["durasi_tidur_menit", "screen_sebelum_tidur", "jam_kerja_menit", "waktu_outdoor", "tingkat_kecemasan", "usia"]
        )
        if chosen_num_exda in df.columns:
            df_num_avg = df.groupby('stress_level')[chosen_num_exda].mean().reset_index()
            df_num_avg['stress_level'] = df_num_avg['stress_level'].map({1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'})
            
            fig_num_bar = px.bar(
                df_num_avg, x='stress_level', y=chosen_num_exda, color='stress_level',
                color_discrete_map={'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'},
                title=f"Rata-rata {chosen_num_exda.replace('_', ' ').title()} Berdasarkan Kategori Stres"
            )
            fig_num_bar.update_layout(showlegend=False, xaxis_title="Tingkat Stres", yaxis_title=f"Rata-rata {chosen_num_exda.replace('_', ' ').title()}")
            st.plotly_chart(fig_num_bar, use_container_width=True)

    st.info(
        "Kesimpulan Analisis Data & Model: \n"
        "1) Faktor Risiko Utama: Hasil eksplorasi menunjukkan adanya pengaruh kuat dari durasi tidur yang rendah serta tingginya screen time sebelum tidur terhadap peningkatan level stres. Pola gangguan tidur ini juga diperkuat oleh data karakteristik kelompok harian, di mana subjek dengan indikator mimpi buruk dan sering terbangun malam memiliki kecenderungan masuk ke dalam kategori stres tingkat tinggi. \n"
        "2) Faktor Protektif & Gaya Hidup: Berdasarkan analisis korelasi dan distribusi variabel penunjang, aktivitas positif seperti meditasi dan manajemen waktu outdoor terbukti memiliki koefisien korelasi yang bernilai negatif terhadap stress_level, yang mendeklarasikan bahwa faktor-faktor tersebut bekerja sebagai peredam atau penahan laju risiko stres. \n"
        "3) Solusi Prevensi (Model AI): Melalui pemetaan interaksi seluruh fitur gaya hidup tersebut, model Deep Learning yang ditanamkan pada sistem berhasil memberikan kalkulasi prediksi secara real-time dengan tingkat akurasi pengujian sebesar 96.73%. Hasil probabilitas ini dapat dijadikan landasan ilmiah untuk langkah intervensi dini."
    )

    st.markdown("---")
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
                    elif hasil_stres == "Rendah":
                        st.success(f"Kesimpulan Model: Tingkat Stres {hasil_stres}")
                    else:
                        st.warning(f"Kesimpulan Model: Tingkat Stres {hasil_stres}")
                        
                except Exception as e:
                    st.error("Gagal mengeksekusi inferensi real-time pada model internal.")