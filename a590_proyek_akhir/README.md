# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut membutuhkan cara yang lebih cepat untuk memahami faktor-faktor yang berpengaruh terhadap risiko dropout mahasiswa dan melakukan intervensi lebih awal. Selama ini evaluasi sering terlambat, sehingga tindakan pencegahan tidak selalu tepat waktu.

Pada proyek ini, data performa siswa digunakan untuk:
1. Membuat dashboard bisnis yang memudahkan monitoring kondisi mahasiswa.
2. Membangun prototype machine learning untuk memprediksi peluang mahasiswa dropout.
3. Menyusun action items berbasis data agar pengambilan keputusan lebih terarah.

### Permasalahan Bisnis
1. Faktor apa saja yang paling terkait dengan status Dropout, Graduate, atau Enrolled?
2. Bagaimana cara mengidentifikasi mahasiswa berisiko dropout lebih cepat?
3. Intervensi apa yang paling relevan untuk menurunkan angka dropout?

### Cakupan Proyek
1. Data preparation dari dataset student performance.
2. Exploratory Data Analysis (EDA) untuk menemukan pola dan insight utama.
3. Definisi target bisnis: prediksi risiko dropout (binary).
4. Pembuatan model klasifikasi untuk memprediksi probabilitas dropout.
5. Pembuatan dashboard bisnis untuk monitoring.
6. Pembuatan prototype aplikasi Streamlit untuk prediksi individual siswa.
7. Penyusunan rekomendasi action items berdasarkan hasil analisis.

### Persiapan
Sumber data:
1. Dataset utama: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv
2. Raw CSV: https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv

Setup environment lokal (Windows PowerShell):
```bash
cd a590_proyek_akhir
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Training model:
```bash
python train_model.py
```

## Business Dashboard
Dashboard dibuat untuk membantu Jaya Jaya Institut memonitor:
1. Distribusi status mahasiswa (Dropout, Graduate, Enrolled).
2. Tren dropout berdasarkan admission grade dan age at enrollment.
3. Perbandingan dropout berdasarkan tuition_fees_up_to_date, debtor, dan scholarship_holder.

Tool dashboard:
1. Google Looker Studio

Link dashboard:
1. Looker Studio: (https://lookerstudio.google.com/reporting/8d3dea9a-132a-4bed-97e9-c084270e2d96)

## Menjalankan Sistem Machine Learning
Prototype machine learning dibuat menggunakan Streamlit pada file app.py.

Fitur input model:
1. Marital_status
2. Application_mode
3. Course
4. Daytime_evening_attendance
5. Previous_qualification_grade
6. Admission_grade
7. Debtor
8. Tuition_fees_up_to_date
9. Gender
10. Scholarship_holder
11. Age_at_enrollment
12. Curricular_units_1st_sem_approved
13. Curricular_units_1st_sem_grade
14. Curricular_units_2nd_sem_approved
15. Curricular_units_2nd_sem_grade

Output model:
1. Prediksi label: Risiko Dropout Tinggi atau Risiko Dropout Rendah
2. Probabilitas Dropout
3. Rekomendasi intervensi awal berbasis hasil prediksi

Jalankan aplikasi secara lokal:
```bash
cd a590_proyek_akhir
streamlit run app.py
```

Deployment Streamlit Community Cloud:
1. Push repository ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository dan set entry point ke app.py.
4. Deploy dengan requirements.txt.

Link prototype Streamlit:
1. ISI_LINK_STREAMLIT_DI_SINI

## Conclusion
Berdasarkan analisis data dan prototype yang dibangun, risiko dropout mahasiswa berkaitan kuat dengan indikator akademik semester awal, status administrasi biaya kuliah, serta indikator ekonomi/dukungan seperti scholarship holder. Dengan model prediksi, institusi dapat melakukan screening awal terhadap mahasiswa yang berpotensi dropout sebelum kondisi memburuk.

Pendekatan dashboard + machine learning memberikan dua manfaat sekaligus:
1. Monitoring risiko dropout secara agregat melalui visualisasi.
2. Prediksi individual untuk mendukung intervensi yang lebih personal.

### Rekomendasi Action Items
1. Bangun sistem early warning berbasis probabilitas prediksi untuk menandai mahasiswa berisiko tiap awal semester.
2. Fokuskan dukungan akademik tambahan (mentoring/remedial) pada mahasiswa dengan performa rendah di semester pertama dan kedua.
3. Terapkan intervensi administratif proaktif untuk mahasiswa dengan status tuition belum terbayar atau debtor.
4. Lakukan evaluasi dampak intervensi bulanan menggunakan dashboard agar kebijakan dapat disesuaikan cepat.
5. Integrasikan proses prediksi ke alur operasional akademik agar keputusan intervensi tidak menunggu nilai akhir.
