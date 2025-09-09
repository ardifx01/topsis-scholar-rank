# 🎓 Seleksi Penerima Beasiswa dengan TOPSIS

Aplikasi berbasis **Streamlit** yang digunakan untuk melakukan seleksi calon penerima beasiswa menggunakan metode **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**.  
Aplikasi ini mempermudah proses pengambilan keputusan dengan perankingan alternatif berdasarkan kriteria yang telah ditentukan.

## ✨ Fitur Utama
- 📄 **Import Data** dari file CSV (contoh: `DATA.csv`)
- 🗃️ **Penyimpanan Data** menggunakan SQLite (data input tetap tersimpan meski aplikasi ditutup)
- 🧮 **Perhitungan TOPSIS** otomatis dengan kriteria:
  - C1: Jumlah Tanggungan  
  - C2: Penghasilan Orang Tua  
  - C3: Status Anak (yatim/piatu/lengkap)  
  - C4: Jarak ke Sekolah  
- 📊 **Visualisasi Interaktif**: ranking alternatif, grafik skor preferensi, detail per nama siswa
- 💾 **Ekspor Hasil** ke CSV/Excel
