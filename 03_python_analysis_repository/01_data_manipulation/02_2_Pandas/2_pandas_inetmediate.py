"""
=================================================
Pandas - Orta Seviye Eğitim Notları
=================================================

Bu bölümde Pandas'ın daha ileri özelliklerini öğreneceksiniz.

Konu Başlıkları:
1. Eksik Veri (Missing Data) Yönetimi
2. GroupBy İşlemleri
3. Pivot Tables ve Cross Tabulation
4. Birleştirme İşlemleri (Merge, Join, Concat)
5. Apply, Map ve Applymap Fonksiyonları
6. String İşlemleri
7. Datetime İşlemleri
8. Kategorik Veriler
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. Eksik Veri (Missing Data) Yönetimi")
print("="*60)

# Eksik veri içeren DataFrame oluştur
df = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', None, 'Ali', 'Fatma'],
    'Yaş': [25, np.nan, 35, 28, 32, np.nan],
    'Maaş': [5000, 6000, np.nan, 5500, np.nan, 5800],
    'Şehir': ['İstanbul', 'Ankara', None, 'Bursa', 'Antalya', 'İzmir']
})

print("Eksik Veri İçeren DataFrame:")
print(df)
print()

# Eksik veri kontrolü
print("--- EKSİK VERİ KONTROLÜ ---")
print("\nEksik veri var mı? (isnull):")
print(df.isnull())
print()

print("Eksik veri sayısı:")
print(df.isnull().sum())
print()

print("Eksik veri yüzdesi:")
print((df.isnull().sum() / len(df) * 100).round(2))
print()

# Eksik veri dolu mu?
print("Veri dolu mu? (notnull):")
print(df.notnull().sum())
print()

# Eksik veri içeren satırları bulma
print("--- EKSİK VERİ İÇEREN SATIRLAR ---")
print("\nHerhangi bir sütunda eksik veri olan satırlar:")
print(df[df.isnull().any(axis=1)])
print()

# Eksik veri silme
print("--- EKSİK VERİ SİLME ---")
print("\nEksik veri içeren satırları sil (dropna):")
df_dropped = df.dropna()
print(df_dropped)
print()

print("Belirli sütundaki eksik verileri sil:")
df_dropped_col = df.dropna(subset=['Yaş'])
print(df_dropped_col)
print()

print("Tüm değerleri NaN olan satırları sil:")
df_test = df.copy()
df_test.loc[1] = np.nan  # 1. satırı tamamen NaN yap
print(df_test.dropna(how='all'))
print()

# Eksik veri doldurma
print("--- EKSİK VERİ DOLDURMA ---")
print("\nSabit değerle doldur (0):")
print(df.fillna(0))
print()

print("Her sütun için farklı değer:")
fill_values = {'Ad': 'Bilinmeyen', 'Yaş': df['Yaş'].mean(), 'Maaş': df['Maaş'].median()}
print(df.fillna(fill_values))
print()

print("İleri doldurma (forward fill):")
print(df.fillna(method='ffill'))
print()

print("Geri doldurma (backward fill):")
print(df.fillna(method='bfill'))
print()

# İnterpolasyon
print("--- İNTERPOLASYON ---")
df_numeric = pd.DataFrame({
    'Değer': [1, 2, np.nan, np.nan, 5, 6]
})
print("\nOrijinal:")
print(df_numeric)
print("\nLinear interpolasyon:")
print(df_numeric.interpolate())
print()

print("="*60)
print("2. GroupBy İşlemleri")
print("="*60)

df = pd.DataFrame({
    'Departman': ['IT', 'HR', 'IT', 'HR', 'IT', 'Finance', 'HR', 'Finance'],
    'Çalışan': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali', 'Zeynep', 'Can', 'Deniz'],
    'Maaş': [5000, 6000, 7000, 5500, 6500, 8000, 5800, 7500],
    'Yaş': [25, 30, 35, 28, 32, 40, 27, 38]
})

print("Örnek DataFrame:")
print(df)
print()

# Basit GroupBy
print("--- BASIT GROUPBY ---")
print("\nDepartmana göre ortalama maaş:")
print(df.groupby('Departman')['Maaş'].mean())
print()

print("Departmana göre toplam maaş:")
print(df.groupby('Departman')['Maaş'].sum())
print()

print("Departmana göre çalışan sayısı:")
print(df.groupby('Departman').size())
print()

# Çoklu aggregation
print("--- ÇOKLU AGGREGATION ---")
print("\nDepartmana göre çoklu istatistik:")
print(df.groupby('Departman')['Maaş'].agg(['mean', 'min', 'max', 'count']))
print()

print("Farklı sütunlara farklı fonksiyonlar:")
print(df.groupby('Departman').agg({
    'Maaş': ['mean', 'sum'],
    'Yaş': ['min', 'max']
}))
print()

# Multiple columns groupby
print("--- ÇOKLU SÜTUN GROUPBY ---")
df2 = pd.DataFrame({
    'Departman': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR'],
    'Şehir': ['İstanbul', 'İstanbul', 'Ankara', 'Ankara', 'İstanbul', 'Ankara'],
    'Maaş': [5000, 6000, 7000, 5500, 6500, 5800]
})

print("\nOrijinal:")
print(df2)
print()

print("Departman ve Şehre göre ortalama:")
print(df2.groupby(['Departman', 'Şehir'])['Maaş'].mean())
print()

# Transform
print("--- TRANSFORM ---")
print("\nHer çalışanın departman ortalamasına göre farkı:")
df['Maaş Farkı'] = df['Maaş'] - df.groupby('Departman')['Maaş'].transform('mean')
print(df[['Çalışan', 'Departman', 'Maaş', 'Maaş Farkı']])
print()

# Filter
print("--- FILTER ---")
print("\nOrtalama maaşı 6000'den fazla olan departmanlar:")
filtered = df.groupby('Departman').filter(lambda x: x['Maaş'].mean() > 6000)
print(filtered)
print()

print("="*60)
print("3. Pivot Tables ve Cross Tabulation")
print("="*60)

df = pd.DataFrame({
    'Tarih': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-01', '2024-02'],
    'Kategori': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Bölge': ['Doğu', 'Batı', 'Doğu', 'Batı', 'Batı', 'Doğu'],
    'Satış': [100, 150, 200, 180, 120, 190]
})

print("Örnek DataFrame:")
print(df)
print()

# Pivot table
print("--- PIVOT TABLE ---")
print("\nKategori ve Bölgeye göre satış pivotu:")
pivot = df.pivot_table(values='Satış', index='Kategori', columns='Bölge', aggfunc='sum')
print(pivot)
print()

print("Çoklu aggregation:")
pivot2 = df.pivot_table(values='Satış', index='Tarih', columns='Kategori', 
                         aggfunc=['sum', 'mean'])
print(pivot2)
print()

# Cross tabulation
df_crosstab = pd.DataFrame({
    'Cinsiyet': ['E', 'K', 'E', 'K', 'E', 'K', 'E', 'K'],
    'Departman': ['IT', 'IT', 'HR', 'HR', 'IT', 'Finance', 'HR', 'Finance'],
    'Terfi': ['Evet', 'Hayır', 'Evet', 'Evet', 'Hayır', 'Evet', 'Evet', 'Hayır']
})

print("--- CROSS TABULATION ---")
print("\nCinsiyet ve Departman çapraz tablosu:")
ct = pd.crosstab(df_crosstab['Cinsiyet'], df_crosstab['Departman'])
print(ct)
print()

print("Terfi durumu ile cinsiyet:")
ct2 = pd.crosstab(df_crosstab['Cinsiyet'], df_crosstab['Terfi'], normalize='index')
print(ct2)
print()

print("="*60)
print("4. Birleştirme İşlemleri (Merge, Join, Concat)")
print("="*60)

# Concat
print("--- CONCAT (SATIR EKLEME) ---")
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

print("\nSatır bazında birleştirme:")
concat_result = pd.concat([df1, df2], ignore_index=True)
print(concat_result)
print()

# Sütun bazında concat
print("--- CONCAT (SÜTUN EKLEME) ---")
df3 = pd.DataFrame({'C': [13, 14, 15]})
concat_col = pd.concat([df1, df3], axis=1)
print(concat_col)
print()

# Merge (SQL JOIN benzeri)
print("--- MERGE ---")
df_musteri = pd.DataFrame({
    'MüşteriID': [1, 2, 3, 4],
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma']
})

df_siparis = pd.DataFrame({
    'SiparişID': [101, 102, 103, 104],
    'MüşteriID': [1, 2, 1, 3],
    'Tutar': [100, 200, 150, 300]
})

print("Müşteriler:")
print(df_musteri)
print("\nSiparişler:")
print(df_siparis)

print("\nInner Join (sadece eşleşenler):")
inner = pd.merge(df_musteri, df_siparis, on='MüşteriID', how='inner')
print(inner)
print()

print("Left Join (sol tablo tamamen):")
left = pd.merge(df_musteri, df_siparis, on='MüşteriID', how='left')
print(left)
print()

print("Right Join (sağ tablo tamamen):")
right = pd.merge(df_musteri, df_siparis, on='MüşteriID', how='right')
print(right)
print()

print("Outer Join (her iki tablo tamamen):")
outer = pd.merge(df_musteri, df_siparis, on='MüşteriID', how='outer')
print(outer)
print()

# Farklı isimli sütunlarda merge
df_urun = pd.DataFrame({
    'ÜrünKodu': [1, 2, 3],
    'Ürün': ['Laptop', 'Mouse', 'Klavye']
})

df_stok = pd.DataFrame({
    'Kod': [1, 2, 3],
    'Miktar': [50, 100, 75]
})

print("Farklı sütun isimleriyle merge:")
merged = pd.merge(df_urun, df_stok, left_on='ÜrünKodu', right_on='Kod')
print(merged)
print()

print("="*60)
print("5. Apply, Map ve Applymap Fonksiyonları")
print("="*60)

df = pd.DataFrame({
    'Ad': ['ahmet', 'ayşe', 'mehmet', 'fatma'],
    'Maaş': [5000, 6000, 7000, 5500],
    'Bonus': [500, 600, 700, 550]
})

print("Orijinal DataFrame:")
print(df)
print()

# Apply (Series)
print("--- APPLY (Sütun Bazında) ---")
df['Ad Büyük'] = df['Ad'].apply(lambda x: x.upper())
print("\nAd sütunu büyük harfe çevrildi:")
print(df)
print()

# Apply (DataFrame) - satır bazında
print("--- APPLY (Satır Bazında) ---")
df['Toplam Gelir'] = df.apply(lambda row: row['Maaş'] + row['Bonus'], axis=1)
print("\nToplam gelir eklendi:")
print(df[['Ad', 'Maaş', 'Bonus', 'Toplam Gelir']])
print()

# Map
print("--- MAP ---")
cinsiyet_map = {'ahmet': 'E', 'mehmet': 'E', 'ayşe': 'K', 'fatma': 'K'}
df['Cinsiyet'] = df['Ad'].map(cinsiyet_map.get)
print("\nCinsiyet sütunu eklendi (map ile):")
print(df[['Ad', 'Cinsiyet']])
print()

# Applymap (deprecated, use map instead)
print("--- MAP (Tüm DataFrame) ---")
df_numeric = pd.DataFrame({
    'A': [1.1, 2.2, 3.3],
    'B': [4.4, 5.5, 6.6]
})

print("Orijinal:")
print(df_numeric)
print("\nTüm değerleri 2 ile çarp:")
df_mapped = df_numeric.map(lambda x: x * 2)
print(df_mapped)
print()

# Custom fonksiyon ile apply
print("--- CUSTOM FONKSİYON İLE APPLY ---")
def categorize_salary(salary):
    if salary < 6000:
        return 'Düşük'
    elif salary < 7000:
        return 'Orta'
    else:
        return 'Yüksek'

df['Maaş Kategorisi'] = df['Maaş'].apply(categorize_salary)
print("\nMaaş kategorisi eklendi:")
print(df[['Ad', 'Maaş', 'Maaş Kategorisi']])
print()

print("="*60)
print("6. String İşlemleri")
print("="*60)

df = pd.DataFrame({
    'Ad': ['  Ahmet YILMAZ  ', 'ayşe Kaya', 'MEHMET demir', 'Fatma ŞAHİN'],
    'Email': ['ahmet@example.com', 'ayse@TEST.com', 'mehmet@example.COM', 'fatma@test.com'],
    'Telefon': ['0532-111-2233', '(543) 444 5566', '555.777.8899', '0-505-999-0011']
})

print("Orijinal DataFrame:")
print(df)
print()

# String temizleme
print("--- STRING TEMİZLEME ---")
print("\nBoşlukları temizle (strip):")
df['Ad Temiz'] = df['Ad'].str.strip()
print(df[['Ad', 'Ad Temiz']])
print()

# Case dönüşümleri
print("--- CASE DÖNÜŞÜMLERİ ---")
print("\nBüyük harf:")
print(df['Ad'].str.upper())
print()

print("Küçük harf:")
print(df['Email'].str.lower())
print()

print("Title case (Her kelimenin ilk harfi büyük):")
print(df['Ad'].str.title())
print()

# String kontrolleri
print("--- STRING KONTROLLERI ---")
print("\n'MEHMET' içeren satırlar:")
print(df[df['Ad'].str.contains('MEHMET', case=False)])
print()

print("'example.com' ile biten email:")
print(df[df['Email'].str.endswith('example.com', case=False)])
print()

print("'A' ile başlayan isimler:")
print(df[df['Ad'].str.strip().str.startswith('A', na=False)])
print()

# String değiştirme
print("--- STRING DEĞİŞTİRME ---")
print("\nTelefon formatını düzelt:")
df['Telefon Düzgün'] = df['Telefon'].str.replace(r'[^0-9]', '', regex=True)
print(df[['Telefon', 'Telefon Düzgün']])
print()

# String bölme
print("--- STRING BÖLME ---")
print("\nAd ve Soyad ayır:")
df_split = df['Ad'].str.strip().str.split(expand=True)
df_split.columns = ['İsim', 'Soyisim']
print(df_split)
print()

# String uzunluğu
print("--- STRING UZUNLUĞU ---")
print("\nEmail uzunlukları:")
df['Email Uzunluk'] = df['Email'].str.len()
print(df[['Email', 'Email Uzunluk']])
print()

# Extract (Regex ile çıkarma)
print("--- EXTRACT (REGEX) ---")
print("\nEmail domain çıkar:")
df['Domain'] = df['Email'].str.extract(r'@(.+)\.com')
print(df[['Email', 'Domain']])
print()

print("="*60)
print("7. Datetime İşlemleri")
print("="*60)

# Datetime oluşturma
print("--- DATETIME OLUŞTURMA ---")
df = pd.DataFrame({
    'Tarih Str': ['2024-01-15', '2024-02-20', '2024-03-25', '2024-04-30'],
    'Saat Str': ['14:30:00', '09:15:00', '16:45:00', '11:00:00']
})

print("Orijinal:")
print(df)
print()

# String'i datetime'a çevir
df['Tarih'] = pd.to_datetime(df['Tarih Str'])
print("Datetime'a çevrildi:")
print(df[['Tarih Str', 'Tarih']])
print(df['Tarih'].dtype)
print()

# Datetime bileşenleri çıkarma
print("--- DATETIME BİLEŞENLERİ ---")
df['Yıl'] = df['Tarih'].dt.year
df['Ay'] = df['Tarih'].dt.month
df['Gün'] = df['Tarih'].dt.day
df['Gün Adı'] = df['Tarih'].dt.day_name()
df['Hafta'] = df['Tarih'].dt.isocalendar().week

print("\nDatetime bileşenleri:")
print(df[['Tarih', 'Yıl', 'Ay', 'Gün', 'Gün Adı', 'Hafta']])
print()

# Tarih aralığı oluşturma
print("--- TARİH ARALIĞI ---")
date_range = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print("\n10 günlük tarih aralığı:")
print(date_range)
print()

# İş günleri
business_days = pd.bdate_range(start='2024-01-01', end='2024-01-15')
print("İş günleri (hafta sonu hariç):")
print(business_days)
print()

# Tarih farkı
print("--- TARİH FARKI ---")
df['Bugün'] = pd.Timestamp.now()
df['Gün Farkı'] = (df['Bugün'] - df['Tarih']).dt.days
print("\nBugüne kadar geçen gün sayısı:")
print(df[['Tarih', 'Gün Farkı']])
print()

# Tarih şekillendirme
print("--- TARİH ŞEKİLLENDİRME ---")
df['Tarih Formatı'] = df['Tarih'].dt.strftime('%d/%m/%Y')
print("\nFarklı tarih formatı:")
print(df[['Tarih', 'Tarih Formatı']])
print()

# Periyot (Period)
print("--- PERİYOT ---")
df['Ay Periyot'] = df['Tarih'].dt.to_period('M')
print("\nAy bazında periyot:")
print(df[['Tarih', 'Ay Periyot']])
print()

print("="*60)
print("8. Kategorik Veriler")
print("="*60)

# Kategorik veri oluşturma
df = pd.DataFrame({
    'Şehir': ['İstanbul', 'Ankara', 'İzmir', 'İstanbul', 'Ankara', 'İzmir', 'İstanbul'],
    'Eğitim': ['Lisans', 'Yüksek Lisans', 'Lise', 'Doktora', 'Lisans', 'Yüksek Lisans', 'Lise']
})

print("Orijinal DataFrame:")
print(df)
print(df.info())
print()

# Kategorik'e çevir
print("--- KATEGORİK'E ÇEVİR ---")
df['Şehir'] = df['Şehir'].astype('category')
df['Eğitim'] = df['Eğitim'].astype('category')

print("\nKategorik'e çevrildi:")
print(df.info())
print()

# Memory tasarrufu
print("--- MEMORY TASARRUFU ---")
df_str = pd.DataFrame({'Şehir': ['İstanbul'] * 10000})
df_cat = pd.DataFrame({'Şehir': pd.Categorical(['İstanbul'] * 10000)})

print(f"String column: {df_str.memory_usage(deep=True)['Şehir'] / 1024:.2f} KB")
print(f"Categorical column: {df_cat.memory_usage(deep=True)['Şehir'] / 1024:.2f} KB")
print()

# Sıralı kategorik
print("--- SIRALI KATEGORİK ---")
egitim_sirali = pd.Categorical(
    df['Eğitim'],
    categories=['Lise', 'Lisans', 'Yüksek Lisans', 'Doktora'],
    ordered=True
)
df['Eğitim Sıralı'] = egitim_sirali

print("\nSıralı kategorik:")
print(df[['Eğitim Sıralı']].head())
print()

# Sıralı kategorikle filtreleme
print("Lisans ve üzeri:")
print(df[df['Eğitim Sıralı'] >= 'Lisans'])
print()

# Kategori kodları
print("--- KATEGORİ KODLARI ---")
print("\nŞehir kodları:")
print(df['Şehir'].cat.codes)
print()

# Kategorileri görüntüleme
print("Tüm kategoriler:")
print(df['Şehir'].cat.categories)
print()

print("="*60)
print("ORTA SEVİYE TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ Eksik veri yönetimi")
print("✓ GroupBy ve aggregation işlemleri")
print("✓ Pivot tables ve cross tabulation")
print("✓ Birleştirme işlemleri (merge, join, concat)")
print("✓ Apply, map fonksiyonları")
print("✓ String işlemleri")
print("✓ Datetime işlemleri")
print("✓ Kategorik veriler")
print()
print("Sıradaki: pandas_ileri.py")
