# BTC 1H Regresyon Projesi

Bu proje, Binance spot piyasasından alınan 1 saatlik BTC/USDT mum verileri ile bir saat sonrasının log getirisini tahmin etmeyi amaçlar.

## Kurulum
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Komutlar
```
python scripts/data_check.py --config configs/config.yaml
python scripts/download_binance.py --config configs/config.yaml
python scripts/preprocess.py --config configs/config.yaml
python scripts/train_lgbm.py --config configs/config.yaml
python scripts/evaluate.py --config configs/config.yaml
```

Dashboard için:
```
streamlit run scripts/ui_app.py
```

## Notlar
- Veri indirmede Binance rate limitlerine dikkat edin.
- Zaman damgaları UTC olmalıdır.
- Eksik mumlar veri setinden çıkarılır.

## Lisans
MIT
