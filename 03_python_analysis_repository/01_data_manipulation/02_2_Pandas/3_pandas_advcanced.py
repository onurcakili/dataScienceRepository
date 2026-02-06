"""
=================================================
Pandas - İleri Seviye Eğitim Notları
=================================================

Bu bölümde Pandas'ın en ileri özelliklerini ve Data Science için 
veri ön işleme tekniklerini öğreneceksiniz.

Konu Başlıkları:
1. MultiIndex ve Hierarchical Indexing
2. Reshaping (Melt, Stack, Unstack)
3. Window Functions (Rolling, Expanding, EWM)
4. VERİ ÖN İŞLEME - Aykırı Değer Analizi (Outlier Detection)
5. VERİ ÖN İŞLEME - Veri Standardizasyonu ve Normalizasyon
6. VERİ ÖN İŞLEME - Feature Engineering
7. Performance Optimization
8. Best Practices ve İleri Teknikler
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. MultiIndex ve Hierarchical Indexing")
print("="*60)

# MultiIndex oluşturma
index = pd.MultiIndex.from_tuples([
    ('İstanbul', 2024), ('İstanbul', 2023),
    ('Ankara', 2024), ('Ankara', 2023),
    ('İzmir', 2024), ('İzmir', 2023)
], names=['Şehir', 'Yıl'])

df = pd.DataFrame({
    'Satış': [1000, 950, 800, 750, 600, 580],
    'Maliyet': [700, 680, 550, 520, 400, 390]
}, index=index)

print("MultiIndex DataFrame:")
print(df)
print()

# MultiIndex seçim
print("İstanbul verileri:")
print(df.loc['İstanbul'])
print()

print("2024 yılı (xs kullanarak):")
print(df.xs(2024, level='Yıl'))
print()

# Swap levels
df_swapped = df.swaplevel('Şehir', 'Yıl')
print("Seviyeler değiştirildi:")
print(df_swapped.head())
print()

print("="*60)
print("2. Reshaping (Melt, Stack, Unstack)")
print("="*60)

# Wide to Long (Melt)
df_wide = pd.DataFrame({
    'Şehir': ['İstanbul', 'Ankara', 'İzmir'],
    '2022': [100, 80, 60],
    '2023': [120, 90, 70],
    '2024': [150, 110, 85]
})

print("Wide Format:")
print(df_wide)
print()

df_long = pd.melt(df_wide, id_vars=['Şehir'], var_name='Yıl', value_name='Satış')
print("Long Format (Melt):")
print(df_long)
print()

# Long to Wide (Pivot)
df_wide_again = df_long.pivot(index='Şehir', columns='Yıl', values='Satış')
print("Wide Format'a geri:")
print(df_wide_again)
print()

# Stack/Unstack
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['X', 'Y', 'Z'])

print("Orijinal:")
print(df)
print("\nStack:")
print(df.stack())
print("\nUnstack:")
print(df.stack().unstack())
print()

print("="*60)
print("3. Window Functions (Rolling, Expanding, EWM)")
print("="*60)

dates = pd.date_range('2024-01-01', periods=30, freq='D')
df = pd.DataFrame({
    'Tarih': dates,
    'Satış': np.random.randint(100, 200, 30) + np.arange(30) * 2
})
df = df.set_index('Tarih')

print("İlk 10 satır:")
print(df.head(10))
print()

# Rolling
df['MA_3'] = df['Satış'].rolling(window=3).mean()
df['MA_7'] = df['Satış'].rolling(window=7).mean()
print("Hareketli Ortalamalar:")
print(df.head(10))
print()

# Expanding
df['Cumulative_Mean'] = df['Satış'].expanding().mean()
df['Cumulative_Sum'] = df['Satış'].expanding().sum()
print("Kümülatif İstatistikler:")
print(df[['Satış', 'Cumulative_Mean', 'Cumulative_Sum']].head(10))
print()

# EWM
df['EWM'] = df['Satış'].ewm(span=7).mean()
print("Exponential Weighted Mean:")
print(df[['Satış', 'EWM']].head(10))
print()

# Shift
df['Önceki'] = df['Satış'].shift(1)
df['Değişim'] = df['Satış'] - df['Önceki']
print("Günlük Değişim:")
print(df[['Satış', 'Önceki', 'Değişim']].head(10))
print()

print("="*60)
print("4. VERİ ÖN İŞLEME - Aykırı Değer Analizi")
print("="*60)

print("""
AYKIRI DEĞER ANALİZİ (OUTLIER DETECTION)

Aykırı Değer Nedir?
- Verideki genel eğilimin dışına çıkan gözlemlerdir
- Modelin performansını olumsuz etkileyebilir
- Genellenebilirliği azaltır

Tespit Yöntemleri:
1. Sektör Bilgisi (Domain Knowledge)
2. Standart Sapma Yöntemi
3. Z-Score Yöntemi  
4. IQR (Interquartile Range) - EN YAYGIN

Çözüm Yöntemleri:
1. Silme (Deletion)
2. Ortalama/Medyan ile Doldurma
3. Baskılama (Capping/Flooring)
""")

# Örnek veri oluştur
np.random.seed(42)
normal_data = np.random.randn(100) * 10 + 50  # Normal dağılım
outliers = np.array([120, 125, 130, -20, -15])  # Aykırı değerler
data = np.concatenate([normal_data, outliers])

df = pd.DataFrame({'Değer': data})

print("\n--- VERİ ÖZETİ ---")
print(df['Değer'].describe())
print()

# IQR Yöntemi
print("--- IQR YÖNTEMİ İLE AYKIRI DEĞER TESPİTİ ---")
Q1 = df['Değer'].quantile(0.25)
Q3 = df['Değer'].quantile(0.75)
IQR = Q3 - Q1

alt_sinir = Q1 - (1.5 * IQR)
ust_sinir = Q3 + (1.5 * IQR)

print(f"Q1 (25. yüzdelik): {Q1:.2f}")
print(f"Q3 (75. yüzdelik): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Alt Sınır: {alt_sinir:.2f}")
print(f"Üst Sınır: {ust_sinir:.2f}")
print()

# Aykırı değerleri tespit et
aykiri_mask = (df['Değer'] < alt_sinir) | (df['Değer'] > ust_sinir)
aykiri_degerler = df[aykiri_mask]

print(f"Toplam gözlem: {len(df)}")
print(f"Aykırı değer sayısı: {len(aykiri_degerler)}")
print(f"Aykırı oran: %{(len(aykiri_degerler)/len(df)*100):.2f}")
print()

print("Aykırı değerler:")
print(aykiri_degerler.sort_values('Değer'))
print()

# ÇÖZÜM 1: Silme
print("--- ÇÖZÜM 1: SİLME YÖNTEMİ ---")
df_cleaned = df[~aykiri_mask].copy()
print(f"Orijinal: {len(df)} satır")
print(f"Temizlenmiş: {len(df_cleaned)} satır")
print(f"Silinen: {len(df) - len(df_cleaned)} satır")
print(f"Yeni ortalama: {df_cleaned['Değer'].mean():.2f}")
print()

# ÇÖZÜM 2: Ortalama ile Doldurma
print("--- ÇÖZÜM 2: ORTALAMA İLE DOLDURMA ---")
df_filled = df.copy()
ortalama = df[~aykiri_mask]['Değer'].mean()
df_filled.loc[aykiri_mask, 'Değer'] = ortalama

print(f"Aykırı değerler {ortalama:.2f} ile dolduruldu")
print(f"Yeni ortalama: {df_filled['Değer'].mean():.2f}")
print()

# ÇÖZÜM 3: Baskılama (Capping)
print("--- ÇÖZÜM 3: BASKILAMA (Capping) ---")
df_capped = df.copy()
df_capped.loc[df['Değer'] < alt_sinir, 'Değer'] = alt_sinir
df_capped.loc[df['Değer'] > ust_sinir, 'Değer'] = ust_sinir

print(f"Alt sınırın altı → {alt_sinir:.2f}")
print(f"Üst sınırın üstü → {ust_sinir:.2f}")
print(f"Min: {df_capped['Değer'].min():.2f}, Max: {df_capped['Değer'].max():.2f}")
print()

# Fonksiyon hali
print("--- FONKSİYONLAŞTIRMA ---")

def detect_outliers_iqr(data, column, multiplier=1.5):
    """IQR yöntemi ile aykırı değer tespiti"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (multiplier * IQR)
    upper = Q3 + (multiplier * IQR)
    outliers = (data[column] < lower) | (data[column] > upper)
    return outliers, lower, upper

def handle_outliers_capping(data, column, lower, upper):
    """Baskılama ile aykırı değer çözümü"""
    data_copy = data.copy()
    data_copy.loc[data_copy[column] < lower, column] = lower
    data_copy.loc[data_copy[column] > upper, column] = upper
    return data_copy

# Kullanım
outliers, lower, upper = detect_outliers_iqr(df, 'Değer')
print(f"Tespit edilen aykırı: {outliers.sum()}")

df_handled = handle_outliers_capping(df, 'Değer', lower, upper)
print("Baskılama tamamlandı")
print()

print("="*60)
print("5. VERİ ÖN İŞLEME - Standardizasyon ve Normalizasyon")
print("="*60)

print("""
STANDARDİZASYON VE NORMALİZASYON

Neden Gerekli?
- Farklı ölçeklerdeki değişkenleri karşılaştırılabilir yapar
- Gradient-based algoritmalarda yakınsama hızını artırır
- Distance-based algoritmalarda (KNN, K-Means) gereklidir

Yöntemler:
1. Min-Max Normalizasyon (0-1 arası)
2. Z-Score Standardizasyonu (mean=0, std=1)
3. Robust Scaling (aykırılara dirençli)
4. Log Transformation (çarpık dağılımlar için)
""")

df = pd.DataFrame({
    'Yaş': [25, 30, 35, 40, 45, 50],
    'Maaş': [3000, 4000, 5000, 6000, 7000, 8000],
    'Deneyim': [2, 5, 8, 12, 15, 20]
})

print("\n--- ORİJİNAL VERİ ---")
print(df)
print()

# 1. Min-Max Normalizasyon
print("--- 1. MIN-MAX NORMALİZASYON (0-1) ---")
df_minmax = df.copy()

for col in df.columns:
    min_val = df[col].min()
    max_val = df[col].max()
    df_minmax[col] = (df[col] - min_val) / (max_val - min_val)

print("Min-Max (Manuel):")
print(df_minmax)
print()

# Sklearn ile
try:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_minmax_sk = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    print("Min-Max (Sklearn):")
    print(df_minmax_sk)
    print()
except ImportError:
    print("Sklearn kurulu değil, manuel yöntem kullanıldı\n")

# 2. Z-Score Standardizasyonu
print("--- 2. Z-SCORE STANDARDİZASYONU ---")
df_zscore = df.copy()

for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    df_zscore[col] = (df[col] - mean) / std

print("Z-Score (Manuel):")
print(df_zscore)
print()
print("İstatistikler (mean≈0, std≈1):")
print(df_zscore.describe())
print()

# 3. Log Transformation
print("--- 3. LOG TRANSFORMATION ---")
df_log = df.copy()

for col in df.columns:
    df_log[col] = np.log1p(df[col])  # log(1 + x)

print("Log Transform:")
print(df_log)
print()

print("="*60)
print("6. VERİ ÖN İŞLEME - Feature Engineering")
print("="*60)

print("""
FEATURE ENGINEERING

Nedir?
- Ham veriden yeni, anlamlı özellikler türetmektir
- Model performansını artırmanın en etkili yoludur
- Domain bilgisi gerektirir

Teknikler:
1. Kategorik Encoding (One-Hot, Label)
2. Tarih/Zaman Özellikleri
3. Matematiksel Dönüşümler
4. Aggregation Features
5. Interaction Features
""")

df = pd.DataFrame({
    'Tarih': pd.date_range('2024-01-01', periods=10),
    'Kategori': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Fiyat': [100, 150, 120, 200, 180, 110, 190, 160, 105, 210],
    'Miktar': [5, 3, 7, 2, 4, 6, 3, 5, 8, 2]
})

print("\n--- ORİJİNAL VERİ ---")
print(df)
print()

# 1. Kategorik Encoding
print("--- 1. KATEGORİK ENCODING ---")

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Kategori'], prefix='Kat')
print("One-Hot Encoding:")
print(df_encoded.head())
print()

# Label Encoding (manuel)
kategori_map = {'A': 0, 'B': 1, 'C': 2}
df['Kat_Label'] = df['Kategori'].map(kategori_map)
print("Label Encoding:")
print(df[['Kategori', 'Kat_Label']].head())
print()

# 2. Tarih Özellikleri
print("--- 2. TARİH ÖZELLİKLERİ ---")
df['Yıl'] = df['Tarih'].dt.year
df['Ay'] = df['Tarih'].dt.month
df['Gün'] = df['Tarih'].dt.day
df['Haftanın_Günü'] = df['Tarih'].dt.dayofweek
df['Hafta_Sonu'] = (df['Haftanın_Günü'] >= 5).astype(int)

print("Tarih özellikleri:")
print(df[['Tarih', 'Ay', 'Gün', 'Haftanın_Günü', 'Hafta_Sonu']].head())
print()

# 3. Matematiksel Özellikler
print("--- 3. MATEMATİKSEL ÖZELLİKLER ---")
df['Toplam_Gelir'] = df['Fiyat'] * df['Miktar']
df['Ortalama_Birim_Fiyat'] = df['Toplam_Gelir'] / df['Miktar']
df['Fiyat_Miktar_Oranı'] = df['Fiyat'] / (df['Miktar'] + 1)  # Sıfıra bölme önleme

print("Türetilen özellikler:")
print(df[['Fiyat', 'Miktar', 'Toplam_Gelir', 'Ortalama_Birim_Fiyat']].head())
print()

# 4. Aggregation Features
print("--- 4. AGGREGATION FEATURES ---")
kategori_stats = df.groupby('Kategori').agg({
    'Fiyat': ['mean', 'max', 'min'],
    'Miktar': ['sum', 'mean']
})

print("Kategori bazlı aggregation:")
print(kategori_stats)
print()

# DataFrame'e ekleme
df = df.merge(
    df.groupby('Kategori')['Fiyat'].mean().rename('Kat_Ort_Fiyat'),
    on='Kategori',
    how='left'
)

print("Kategori ortalama fiyatı eklendi:")
print(df[['Kategori', 'Fiyat', 'Kat_Ort_Fiyat']].head())
print()

# 5. Polynomial Features
print("--- 5. POLYNOMIAL FEATURES ---")
df['Fiyat_Kare'] = df['Fiyat'] ** 2
df['Miktar_Kare'] = df['Miktar'] ** 2
df['Fiyat_Miktar_Çarpım'] = df['Fiyat'] * df['Miktar']

print("Polynomial features:")
print(df[['Fiyat', 'Miktar', 'Fiyat_Kare', 'Fiyat_Miktar_Çarpım']].head())
print()

print("="*60)
print("7. Performance Optimization")
print("="*60)

print("""
PANDAS PERFORMANS OPTİMİZASYONU

1. Doğru Veri Tipi Seçimi
2. Vectorization Kullanımı
3. Inplace İşlemler
4. Kategorik Veri Kullanımı
5. Chunking (Büyük Dosyalar İçin)
""")

# 1. Veri Tipi Optimizasyonu
print("--- 1. VERİ TİPİ OPTİMİZASYONU ---")
df_original = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 10000),
    'float_col': np.random.rand(10000),
    'category_col': np.random.choice(['A', 'B', 'C'], 10000)
})

print("Orijinal memory kullanımı:")
print(df_original.memory_usage(deep=True))
print(f"Toplam: {df_original.memory_usage(deep=True).sum() / 1024:.2f} KB")
print()

# Optimize et
df_optimized = df_original.copy()
df_optimized['int_col'] = df_optimized['int_col'].astype('int8')
df_optimized['float_col'] = df_optimized['float_col'].astype('float32')
df_optimized['category_col'] = df_optimized['category_col'].astype('category')

print("Optimize edilmiş memory:")
print(df_optimized.memory_usage(deep=True))
print(f"Toplam: {df_optimized.memory_usage(deep=True).sum() / 1024:.2f} KB")
print()

reduction = (1 - df_optimized.memory_usage(deep=True).sum() / df_original.memory_usage(deep=True).sum()) * 100
print(f"Memory tasarrufu: %{reduction:.2f}")
print()

# 2. Vectorization vs Loop
print("--- 2. VECTORIZATION ---")
import time

df_test = pd.DataFrame({'A': range(100000), 'B': range(100000)})

# Loop yöntemi (Yavaş)
start = time.time()
result_loop = []
for i in range(len(df_test)):
    result_loop.append(df_test.loc[i, 'A'] + df_test.loc[i, 'B'])
time_loop = time.time() - start

# Vectorized yöntem (Hızlı)
start = time.time()
result_vec = df_test['A'] + df_test['B']
time_vec = time.time() - start

print(f"Loop süresi: {time_loop:.4f}s")
print(f"Vectorized süresi: {time_vec:.4f}s")
print(f"Hızlanma: {time_loop/time_vec:.2f}x")
print()

# 3. Eval ve Query
print("--- 3. EVAL ve QUERY ---")
df = pd.DataFrame({
    'A': np.random.rand(10000),
    'B': np.random.rand(10000),
    'C': np.random.rand(10000)
})

# Normal yöntem
start = time.time()
result1 = df['A'] + df['B'] * df['C']
time1 = time.time() - start

# Eval ile
start = time.time()
result2 = df.eval('A + B * C')
time2 = time.time() - start

print(f"Normal: {time1:.6f}s")
print(f"Eval: {time2:.6f}s")
print()

print("="*60)
print("8. Best Practices")
print("="*60)

print("""
PANDAS BEST PRACTICES

1. CHAIN İŞLEMLERİ KULLAN
   df.pipe(func1).pipe(func2).pipe(func3)

2. METHOD CHAINING İLE TEMİZ KOD
   (df
    .query('Yaş > 30')
    .groupby('Şehir')
    .agg({'Maaş': 'mean'})
   )

3. COPY vs VIEW DİKKAT ET
   - df[df['A'] > 5] → View (bazen)
   - df.copy() → Copy (her zaman)

4. INPLACE KULLANIMI
   - df.drop(columns=['A'], inplace=True)
   - Ama genelde inplace=False tercih et (daha güvenli)

5. DÜZENLİ MEMORY KONTROLÜ
   - df.info(memory_usage='deep')
   - df.memory_usage(deep=True)

6. KATEGORIK VERİ KULLAN
   - Tekrar eden string'ler için
   - Memory tasarrufu
   - Hız artışı

7. VECTORIZATION
   - Apply yerine vectorized işlemler
   - Loop yerine pandas operasyonları

8. CHUNK KULLANIMI (Büyük Dosyalar)
   - pd.read_csv('file.csv', chunksize=10000)
   - Bellek sorunlarını önler
""")

# Method Chaining Örneği
df = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali'],
    'Yaş': [25, 30, 35, 28, 32],
    'Şehir': ['İstanbul', 'Ankara', 'İstanbul', 'İzmir', 'Ankara'],
    'Maaş': [5000, 6000, 7000, 5500, 6500]
})

print("\n--- METHOD CHAINING ÖRNEĞİ ---")
result = (df
    .query('Yaş > 27')
    .groupby('Şehir')
    .agg({'Maaş': ['mean', 'count']})
    .round(2)
)

print("Yaşı 27'den büyük olanların şehre göre maaş ortalaması:")
print(result)
print()

print("="*60)
print("İLERİ SEVİYE TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ MultiIndex ve hierarchical indexing")
print("✓ Reshaping (melt, stack, unstack)")
print("✓ Window functions (rolling, expanding, ewm)")
print("✓ VERİ ÖN İŞLEME:")
print("  - Aykırı değer analizi (IQR, Z-Score)")
print("  - Standardizasyon ve normalizasyon")
print("  - Feature engineering")
print("✓ Performance optimization")
print("✓ Best practices")
print()
print("Tüm Pandas eğitimi tamamlandı!")
print("NumPy ve Pandas'ta artık ileri seviyesiniz!")
