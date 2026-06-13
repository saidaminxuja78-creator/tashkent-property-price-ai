# Student Performance ML

## Fayllar

- `app.py` — Streamlit interfeysi
- `ml_pipeline.py` — data loading, preprocessing, Nested CV va modellar
- `requirements.txt` — Python paketlari
- `.streamlit/config.toml` — Streamlit tema sozlamasi

## Dataset

Repository ildiziga `student-mat.csv` faylini joylashtiring. Quyidagi yo'l ham ishlaydi:

```text
data/student-mat.csv
```

Agar datasetda `target` ustuni bo'lmasa, kod uni avtomatik yaratadi:

```python
target = (G3 >= 10).astype(int)
```

Modelning early-warning rejimida `G1`, `G2`, `G3` feature sifatida ishlatilmaydi.

## Lokal ishga tushirish

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

GitHub repository ichida kamida quyidagilar bo'lishi kerak:

```text
app.py
ml_pipeline.py
requirements.txt
student-mat.csv
.streamlit/config.toml
```

Main file path sifatida `app.py` ni tanlang.
