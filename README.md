# 🌾 Prediksi Harga Beras Indonesia

Aplikasi web prediksi harga beras menggunakan **Regresi Linier** dan **Random Forest Regressor**.

## 📁 Struktur Project

```
rice_price_app/
├── app.py              # Aplikasi Streamlit utama
├── model.py            # Logic model ML (preprocessing, training, prediksi)
├── requirements.txt    # Dependensi Python
├── data/
│   └── harga_beras.csv # Dataset harga beras
└── README.md
```

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone / download project
# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Push project ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo → set `app.py` sebagai main file
4. Klik **Deploy**

## 📊 Fitur Aplikasi

| Tab | Isi |
|-----|-----|
| Dashboard | Tren harga, distribusi, rata-rata bulanan |
| Prediksi | Forecast N minggu ke depan per jenis beras |
| Evaluasi Model | MAE, RMSE, R², MAPE, feature importance |
| Data | Preview & filter dataset |

## 🔧 Format Data

CSV dengan kolom:
```
tanggal,jenis_beras,harga
2024-01-01,Medium,19000
2024-01-01,Premium,22000
```

## 📐 Metodologi

### Feature Engineering
- Lag features (lag-1, lag-2, lag-4 minggu)
- Rolling mean & std (4 minggu)
- Fitur waktu: tahun, bulan, minggu, day of year
- Harga diff (selisih dari periode sebelumnya)

### Model
- **Linear Regression** — baseline
- **Random Forest Regressor** — model utama (200 trees, max_depth=10)

### Evaluasi
- Train/test split berbasis waktu (bukan random)
- Metrik: MAE, RMSE, R², MAPE
