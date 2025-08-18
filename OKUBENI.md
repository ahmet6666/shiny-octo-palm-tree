# BTC/USDT 1saatlik Regresyon Hattı

Bu depo, Binance Spot piyasasından alınan 1 saatlik mum verileriyle bir sonraki saatin log getirisini tahmin etmek için uçtan uca bir akış sunar. Veri indirme, ön işleme, teknik indikatörler, LightGBM eğitimi ve basit bir Streamlit arayüzü içerir.

## Hızlı Başlangıç

```bash
python scripts/make_dirs.py --config configs/config.yaml
python scripts/run_all.py --config configs/config.yaml
```

Streamlit arayüzü:

```bash
streamlit run src/ui_app.py
```

Duman testi (son 100 saat):

```bash
python scripts/smoke_test.py --config configs/config.yaml
```

## Veri Politikası
- Tüm zamanlar **UTC** ve sıkı saatlik ızgaraya hizalanır.
- Eksik mumlar atılır; isteğe bağlı olarak en uzun ardışık segment seçilir.
- Ham ve işlenmiş veriler `veriler/` altında tutulur.

## Özellikler ve Hedef
- RSI, MACD, EMA(12/26/50/200), Bollinger Bantları, Stokastik, ATR.
- Ekstra: log fiyat, 1 saatlik log getiri.
- Hedef: `next_1h_log_return = ln(close[t+1]/close[t])`.

## MLflow
Deney takibi varsayılan olarak `mlruns` klasörüne yazılır.

## Sorun Giderme
- Binance API limitleri: tekrar deneyin veya API anahtarlarını `.env` ile sağlayın.
- Zaman dilimi hataları: sistem saatinin UTC olduğundan emin olun.
- Eksik veri: `data_check` adımı kapsamı doğrular.

## Lisans
MIT
