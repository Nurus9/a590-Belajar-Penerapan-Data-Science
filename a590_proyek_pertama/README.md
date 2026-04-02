# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional besar yang telah berdiri sejak tahun 2000 dan memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. Walaupun telah menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas pada tingginya attrition rate (rasio jumlah karyawan yang keluar berbanding dengan total karyawan keseluruhan) yang saat ini mencapai lebih dari 10%.

### Permasalahan Bisnis

1. Apa saja faktor-faktor internal maupun eksternal yang paling signifikan dalam mempengaruhi tingginya attrition rate karyawan di Jaya Jaya Maju?

2. Bagaimana cara memonitor faktor-faktor pemicu tersebut secara real-time dan interaktif agar departemen HR dapat mengambil tindakan preventif?

### Cakupan Proyek

1. Data Preparation & Preprocessing: Membersihkan dataset dari missing values, menghapus kolom yang tidak relevan (seperti EmployeeId), dan melakukan encoding pada data kategorikal agar siap dianalisis.

2. Exploratory Data Analysis (EDA): Menganalisis distribusi data dan mencari korelasi antara berbagai fitur (seperti gaji, lembur, dan departemen) terhadap status attrition.

3. Machine Learning Modeling: Membangun, melatih, dan mengevaluasi model prediktif (Logistic Regression) untuk mendeteksi potensi karyawan yang akan resign.

4. Data Visualization: Membuat Business Dashboard interaktif menggunakan Looker Studio untuk memonitor metrik HR secara komprehensif.

### Persiapan

Sumber data:
https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

Persiapan Proyek (Lengkap dan Sistematis):

1. Clone repository lalu masuk ke folder proyek. (jika dari github)

```bash
git clone <url-repository>
cd a590_proyek_pertama
```

2. Buat virtual environment.( Jika download dari gdrive, mulai dari sini)

```bash
python -m venv .venv
```

3. Aktifkan virtual environment.

Untuk Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Untuk Windows (Command Prompt):

```bat
.venv\Scripts\activate.bat
```

Untuk Linux/macOS:

```bash
source .venv/bin/activate
```

4. Install semua dependensi.

```bash
pip install -r requirements.txt
```

5. Jalankan proyek. Menjalankan Notebook (utama):

File utama analisis pada proyek ini adalah `notebook.ipynb`. Tekan Run All untuk menjalankan notebook.

## Business Dashboard

Business dashboard telah dibangun menggunakan Looker Studio untuk memberikan antarmuka visual yang interaktif kepada Manajer HR.
Dashboard ini dilengkapi dengan filter dinamis (Gender dan Department) dan menyajikan metrik dalam dua fokus utama:

1. Menampilkan metrik krusial seperti Total Karyawan (1.058), Rata-rata Masa Kerja (7,11 Tahun), Total Karyawan Resign (179), dan Tingkat Resign/Attrition Rate (16,92%).

2. Core Drivers Analysis: Menampilkan visualisasi faktor penyebab utama resign, meliputi dampak lembur (OverTime), perbandingan rata-rata gaji bulanan (Monthly Income), distribusi posisi pekerjaan (Job Role) yang paling rentan, serta distribusi tingkat kepuasan kerja (Job Satisfaction).

Tautan Dashboard: [https://lookerstudio.google.com/reporting/29a3c669-9899-49f4-8eef-7da8fc516ae9]

## Conclusion

Berdasarkan analisis data dan visualisasi pada business dashboard, tingginya attrition rate (16,92%) di Jaya Jaya Maju sangat dipengaruhi oleh faktor-faktor berikut:

1. Beban Kerja Berlebih (Lembur): Terdapat tren yang sangat jelas bahwa jumlah karyawan yang resign jauh lebih tinggi pada kelompok karyawan yang sering melakukan kerja lembur (Yes) dibandingkan yang tidak.

2. Kompensasi Finansial: Rata-rata gaji bulanan (Monthly Income) karyawan yang resign (Ya) terbukti berada di posisi paling rendah dibandingkan dengan karyawan yang memilih untuk bertahan (Tidak).

3. Kerentanan Posisi: Turnover tertinggi didominasi oleh posisi operasional dan spesialis, dengan tiga posisi teratas yang paling sering ditinggalkan adalah Laboratory Technician, Sales Executive, dan Research Scientist.

4. Faktor Pendorong Lain: Analisis kepuasan kerja menunjukkan bahwa mayoritas karyawan yang resign sebenarnya berada di tingkat kepuasan 3 (Puas) sebesar 34,6%. Hal ini mengonfirmasi bahwa alasan utama mereka keluar bukanlah lingkungan kerja yang buruk, melainkan faktor eksternal seperti kelelahan akibat lembur atau tawaran gaji yang lebih baik di tempat lain.
### Rekomendasi Action Items (Optional)

Berikut adalah beberapa rekomendasi tindakan preventif yang dapat dilakukan oleh departemen HR Jaya Jaya Maju:

1. Mengevaluasi Kebijakan Lembur: Melakukan pemerataan beban kerja atau menambah alokasi SDM pada departemen dengan intensitas lembur tertinggi untuk mencegah burnout berkepanjangan.

2. Tinjauan Kompensasi Berkala (Salary Adjustment): Melakukan riset standar gaji pasar secara spesifik untuk posisi Laboratory Technician, Sales Executive, dan Research Scientist, lalu melakukan penyesuaian kompensasi jika berada di bawah rata-rata industri.

3. Merancang Program Retensi Bertarget: Memfokuskan anggaran program kesejahteraan karyawan (seperti bonus performa, fleksibilitas kerja, atau training) ke departemen Sales dan Research & Development yang menyumbang angka resign tertinggi.
