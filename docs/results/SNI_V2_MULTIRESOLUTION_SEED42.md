# SNI v2 multiresolution screening seed 42

**Status: VALIDATION SCREENING FAIL -- STOP, TEST TERKUNCI.**

Eksperimen membandingkan dua model pada manifest SNI v2 dengan 15 kelas visual,
inverse-square-root weighted sampling hanya pada train, dan split grouped yang
tidak diubah:

- `S2G`: EfficientNetV2-B0, final-stage GAP, linear classifier;
- `S2MR`: EfficientNetV2-B0, stages 1--4 diproyeksikan dan difusikan sebelum
  linear classifier.

Hasil validation seed 42:

| Metrik | S2G GAP | S2MR multiresolusi | Delta |
|---|---:|---:|---:|
| Macro-F1 | 88,53% | 87,85% | -0,68 |
| Hard-F1 | 83,97% | 83,55% | -0,41 |
| Bottom-three F1 | 70,22% | 64,12% | -6,10 |
| Worst-F1 | 64,00% | 42,11% | -21,89 |

Ketiga kriteria fail-fast gagal: Macro-F1 tidak meningkat, Hard-F1 tidak
meningkat, dan bottom-three F1 tidak terjaga. Seed 123/2026 tidak dijalankan
dan test tidak dibuka.

## Audit unit independen

Audit post-hoc tanpa training menunjukkan bahwa hasil crop-level di atas tidak
boleh dibaca seolah-olah setiap crop adalah observasi independen:

| Domain validation | Crop | Foto sumber/group | Kelas |
|---|---:|---:|---:|
| Adrian detection | 4.462 | **8** | 15 |
| Faruq segmentation | 507 | 186 | 14 |

Delapan foto Adrian masing-masing menghasilkan 260--769 crop dan memuat 7--10
kelas. Ini bukan kesalahan penyatuan nama: identitas terbesar berupa nama foto
bertimestamp yang berbeda. Pembentukan grup berhasil mencegah satu foto sumber
masuk ke lebih dari satu split, tetapi allocator terlalu mengutamakan jumlah
crop. Akibatnya, ribuan crop berkorelasi dari sedikit foto padat mendominasi
validation.

Ketika probabilitas dirata-ratakan per kombinasi `foto sumber x kelas`, hasilnya
menjadi:

| Unit evaluasi | S2G GAP | S2MR multiresolusi | Delta S2MR |
|---|---:|---:|---:|
| Accuracy crop | 95,19% | 94,65% | -0,54 |
| Macro-F1 crop | 88,53% | 87,85% | -0,68 |
| Accuracy group/class | 92,28% | 92,65% | +0,37 |
| Macro-F1 group/class | 85,89% | 85,69% | -0,20 |

Outcome crop-level adalah 4.656 keduanya benar, 192 keduanya salah, 74 hanya
GAP benar, dan 47 hanya multiresolusi benar. Kegagalan lower-tail terutama
terkonsentrasi pada `biji_muda`: 58 crop validation hanya berasal dari empat
foto Adrian; F1 turun dari 64,00% menjadi 42,11%.

Jadi keputusan fail-fast **tetap STOP** karena multiresolusi tidak menjadi
unggul setelah unit evaluasi dikoreksi. Namun, besar penurunan Worst-F1
crop-level teramplifikasi oleh korelasi dalam foto. Protokol ini tidak cukup
kuat untuk klaim umum tentang keunggulan arsitektur sebelum split
menyeimbangkan jumlah foto sumber per dataset dan per kelas, dan interval
ketidakpastian dihitung dengan cluster bootstrap pada foto sumber.

Hasil ini tidak bertentangan dengan screening SNI-MRENet v1 21 kelas. SNI v2
mengubah target menjadi 15 kelas visual, menggabungkan label ukuran yang tidak
terkalibrasi, dan menggunakan weighted sampling. Manfaat multiresolusi pada
protokol lama tidak berpindah ke formulasi v2 yang lebih aman.

Ringkasan ini ditranskripsi dari keluaran Colab. Artefak sumber persisten:
`/content/drive/MyDrive/sni-v2-multiresolution-v1/val_reports/screen_seed42.json`.
