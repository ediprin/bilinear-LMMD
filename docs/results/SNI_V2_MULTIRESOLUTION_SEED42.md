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

Hasil ini tidak bertentangan dengan screening SNI-MRENet v1 21 kelas. SNI v2
mengubah target menjadi 15 kelas visual, menggabungkan label ukuran yang tidak
terkalibrasi, dan menggunakan weighted sampling. Manfaat multiresolusi pada
protokol lama tidak berpindah ke formulasi v2 yang lebih aman.

Ringkasan ini ditranskripsi dari keluaran Colab. Artefak sumber persisten:
`/content/drive/MyDrive/sni-v2-multiresolution-v1/val_reports/screen_seed42.json`.

