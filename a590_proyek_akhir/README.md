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
1.(https://a590-belajar-penerapan-data-science-f5v3gmkbcug8gvrceefbrn.streamlit.app/)

## Conclusion
### 1) Kesimpulan Analisis Faktor Dropout (EDA + Dashboard)
Berdasarkan EDA pada notebook dan visualisasi dashboard, mahasiswa dengan risiko dropout cenderung memiliki karakteristik berikut:
1. Performa akademik awal yang lebih rendah, terutama pada nilai dan jumlah mata kuliah lulus di semester 1 dan semester 2.
2. Hambatan administrasi dan finansial, terutama pada mahasiswa yang belum up-to-date pembayaran tuition fees dan memiliki status debtor.
3. Variasi risiko antar kelompok mahasiswa (misalnya berdasarkan pola pendaftaran, program studi, serta dukungan beasiswa) yang mengindikasikan perlunya intervensi yang lebih tersegmentasi.

Kesimpulan ini menegaskan bahwa dropout bukan hanya isu akademik, tetapi juga berkaitan dengan kondisi administratif dan ekonomi mahasiswa.

### 2) Kesimpulan Performa Model dan Fitur Penting
Model klasifikasi dilatih ulang dengan data berlabel final saja (Status = Dropout atau Graduate), sedangkan data Enrolled tidak digunakan sebagai data latih agar target prediksi tetap valid.

Hasil evaluasi kuantitatif model:
1. Accuracy: 0.9063
2. Precision: 0.8600
3. Recall: 0.9085
4. F1-score: 0.8836
5. ROC-AUC: 0.9569

Fitur yang paling berpengaruh (berdasarkan koefisien absolut Logistic Regression setelah encoding) antara lain:
1. Variabel terkait program studi (contoh: `Course_171`, `Course_9853`, `Course_9119`).
2. Variabel terkait mode pendaftaran (contoh: `Application_mode_15`, `Application_mode_26`, `Application_mode_42`).
3. Variabel status pembayaran kuliah (`Tuition_fees_up_to_date_0` dan `Tuition_fees_up_to_date_1`).

Dengan performa ini, model sudah memadai untuk early warning system dan dapat digunakan untuk membantu prioritisasi intervensi pada mahasiswa berisiko tinggi.

### Rekomendasi Action Items
1. Bangun sistem early warning berbasis probabilitas prediksi untuk menandai mahasiswa berisiko tiap awal semester.
2. Fokuskan dukungan akademik tambahan (mentoring/remedial) pada mahasiswa dengan performa rendah di semester pertama dan kedua.
3. Terapkan intervensi administratif proaktif untuk mahasiswa dengan status tuition belum terbayar atau debtor.
4. Lakukan evaluasi dampak intervensi bulanan menggunakan dashboard agar kebijakan dapat disesuaikan cepat.
5. Integrasikan proses prediksi ke alur operasional akademik agar keputusan intervensi tidak menunggu nilai akhir.
