"""
=================================================
Pandas - Giriş Seviyesi Eğitim Notları
=================================================

Pandas, Python'da veri analizi ve manipülasyonu için en önemli kütüphanedir.
Excel benzeri tablo (DataFrame) yapısı ile çalışmayı sağlar.

Konu Başlıkları:
1. Pandas'a Giriş ve Veri Yapıları
2. DataFrame ve Series Oluşturma
3. Veri Okuma ve Yazma
4. Temel İndeksleme ve Seçim İşlemleri
5. Veri Filtreleme
6. Sütun ve Satır Ekleme/Silme
7. Temel İstatistiksel İşlemler
8. Sıralama ve Sırayla İlgili İşlemler
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. Pandas'a Giriş ve Veri Yapıları")
print("="*60)

print(f"Pandas Versiyonu: {pd.__version__}")
print()

print("Pandas'ın 2 Temel Veri Yapısı:")
print("1. Series: Tek boyutlu, etiketli array (sütun)")
print("2. DataFrame: İki boyutlu, etiketli tablo (Excel benzeri)")
print()

print("Pandas Neden Kullanılır?")
print("- Excel benzeri veri tabloları")
print("- Eksik veri (NaN) ile çalışma")
print("- Etiketli indeksler")
print("- SQL benzeri birleştirme işlemleri")
print("- Grup işlemleri (groupby)")
print("- Zaman serisi analizi")
print()

print("="*60)
print("2. DataFrame ve Series Oluşturma")
print("="*60)

# Series oluşturma
print("--- SERIES OLUŞTURMA ---")
print()

# Liste'den Series
s1 = pd.Series([10, 20, 30, 40, 50])
print("Liste'den Series:")
print(s1)
print()

# İndeks ile Series
s2 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print("İndeksli Series:")
print(s2)
print()

# Dictionary'den Series
data_dict = {'Ahmet': 85, 'Ayşe': 92, 'Mehmet': 78, 'Fatma': 95}
s3 = pd.Series(data_dict)
print("Dictionary'den Series:")
print(s3)
print()

# Series özellikleri
print(f"Değerler: {s3.values}")
print(f"İndeksler: {s3.index}")
print(f"Boyut: {s3.shape}")
print(f"Veri tipi: {s3.dtype}")
print()

print("--- DATAFRAME OLUŞTURMA ---")
print()

# Dictionary'den DataFrame
data = {
    'İsim': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali'],
    'Yaş': [25, 30, 35, 28, 32],
    'Şehir': ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya'],
    'Maaş': [5000, 6000, 7000, 5500, 6500]
}

df = pd.DataFrame(data)
print("Dictionary'den DataFrame:")
print(df)
print()

# Liste listelerinden DataFrame
data_list = [
    ['Ahmet', 25, 'İstanbul', 5000],
    ['Ayşe', 30, 'Ankara', 6000],
    ['Mehmet', 35, 'İzmir', 7000]
]
df2 = pd.DataFrame(data_list, columns=['İsim', 'Yaş', 'Şehir', 'Maaş'])
print("Liste listelerinden DataFrame:")
print(df2)
print()

# NumPy array'den DataFrame
arr = np.random.randint(1, 100, size=(5, 4))
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
print("NumPy array'den DataFrame:")
print(df3)
print()

# DataFrame özellikleri
print("DataFrame Özellikleri:")
print(f"Boyut (shape): {df.shape}")
print(f"Satır sayısı: {len(df)}")
print(f"Sütun sayısı: {len(df.columns)}")
print(f"Toplam eleman: {df.size}")
print(f"Sütun isimleri: {df.columns.tolist()}")
print(f"İndeks: {df.index.tolist()}")
print()

# Veri tipleri
print("Veri Tipleri:")
print(df.dtypes)
print()

# Temel bilgiler
print("Genel Bilgi:")
print(df.info())
print()

print("="*60)
print("3. Veri Okuma ve Yazma")
print("="*60)

# CSV dosyası oluşturma ve okuma
print("--- CSV İŞLEMLERİ ---")

# Örnek veri oluştur ve kaydet
df_save = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali'],
    'Soyad': ['Yılmaz', 'Kaya', 'Demir', 'Şahin', 'Çelik'],
    'Not': [85, 92, 78, 95, 88],
    'Sınıf': ['A', 'B', 'A', 'C', 'B']
})

# CSV'ye kaydet
df_save.to_csv('/home/claude/ogrenciler.csv', index=False)
print("CSV dosyası kaydedildi: ogrenciler.csv")

# CSV'den oku
df_read = pd.read_csv('/home/claude/ogrenciler.csv')
print("\nCSV dosyası okundu:")
print(df_read)
print()

# Excel işlemleri
print("--- EXCEL İŞLEMLERİ ---")

# Excel'e kaydet
df_save.to_excel('/home/claude/ogrenciler.xlsx', index=False, sheet_name='Notlar')
print("Excel dosyası kaydedildi: ogrenciler.xlsx")

# Excel'den oku
df_excel = pd.read_excel('/home/claude/ogrenciler.xlsx', sheet_name='Notlar')
print("\nExcel dosyası okundu:")
print(df_excel)
print()

# JSON işlemleri
print("--- JSON İŞLEMLERİ ---")

# JSON'a kaydet
df_save.to_json('/home/claude/ogrenciler.json', orient='records', indent=2)
print("JSON dosyası kaydedildi")

# JSON'dan oku
df_json = pd.read_json('/home/claude/ogrenciler.json')
print("\nJSON dosyası okundu:")
print(df_json)
print()

print("="*60)
print("4. Temel İndeksleme ve Seçim İşlemleri")
print("="*60)

df = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali', 'Zeynep'],
    'Yaş': [25, 30, 35, 28, 32, 27],
    'Şehir': ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'İstanbul'],
    'Maaş': [5000, 6000, 7000, 5500, 6500, 5800]
})

print("Örnek DataFrame:")
print(df)
print()

# Sütun seçimi
print("--- SÜTUN SEÇİMİ ---")
print("\nTek sütun (Series döner):")
print(df['Ad'])
print()

print("Çoklu sütun (DataFrame döner):")
print(df[['Ad', 'Maaş']])
print()

# İlk ve son satırlar
print("--- İLK VE SON SATIRLAR ---")
print("\nİlk 3 satır:")
print(df.head(3))
print()

print("Son 2 satır:")
print(df.tail(2))
print()

# loc - etiket bazlı seçim
print("--- LOC (Etiket Bazlı Seçim) ---")
print("\nTek satır:")
print(df.loc[0])
print()

print("Satır aralığı:")
print(df.loc[1:3])
print()

print("Belirli satır ve sütunlar:")
print(df.loc[0:2, ['Ad', 'Şehir']])
print()

# iloc - pozisyon bazlı seçim
print("--- ILOC (Pozisyon Bazlı Seçim) ---")
print("\nTek satır (2. satır):")
print(df.iloc[1])
print()

print("Satır ve sütun aralığı:")
print(df.iloc[1:4, 0:2])
print()

print("Belirli pozisyonlar:")
print(df.iloc[[0, 2, 4], [0, 3]])
print()

# Tek hücre seçimi
print("--- TEK HÜCRE SEÇİMİ ---")
print(f"\nat ile: {df.at[0, 'Ad']}")
print(f"iat ile: {df.iat[0, 0]}")
print()

print("="*60)
print("5. Veri Filtreleme")
print("="*60)

print("Örnek DataFrame:")
print(df)
print()

# Basit filtreleme
print("--- BASIT FİLTRELEME ---")
print("\nYaşı 30'dan büyük olanlar:")
filtre1 = df[df['Yaş'] > 30]
print(filtre1)
print()

print("Şehri İstanbul olanlar:")
filtre2 = df[df['Şehir'] == 'İstanbul']
print(filtre2)
print()

# Çoklu koşul (VE - &)
print("--- ÇOKLU KOŞUL (VE) ---")
print("\nYaşı 28'den büyük VE maaşı 6000'den fazla:")
filtre3 = df[(df['Yaş'] > 28) & (df['Maaş'] > 6000)]
print(filtre3)
print()

# Çoklu koşul (VEYA - |)
print("--- ÇOKLU KOŞUL (VEYA) ---")
print("\nŞehri İstanbul VEYA Ankara olanlar:")
filtre4 = df[(df['Şehir'] == 'İstanbul') | (df['Şehir'] == 'Ankara')]
print(filtre4)
print()

# isin kullanımı
print("--- ISIN KULLANIMI ---")
print("\nŞehri İstanbul, Ankara veya İzmir olanlar:")
filtre5 = df[df['Şehir'].isin(['İstanbul', 'Ankara', 'İzmir'])]
print(filtre5)
print()

# String içerme kontrolü
print("--- STRING FİLTRELEME ---")
print("\nİsmi 'A' ile başlayanlar:")
filtre6 = df[df['Ad'].str.startswith('A')]
print(filtre6)
print()

print("İsminde 'e' geçenler:")
filtre7 = df[df['Ad'].str.contains('e')]
print(filtre7)
print()

print("="*60)
print("6. Sütun ve Satır Ekleme/Silme")
print("="*60)

df = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet'],
    'Yaş': [25, 30, 35],
    'Maaş': [5000, 6000, 7000]
})

print("Orijinal DataFrame:")
print(df)
print()

# Yeni sütun ekleme
print("--- SÜTUN EKLEME ---")
df['Şehir'] = ['İstanbul', 'Ankara', 'İzmir']
print("\nŞehir sütunu eklendi:")
print(df)
print()

# Hesaplanmış sütun
df['Yıllık Maaş'] = df['Maaş'] * 12
print("Yıllık Maaş sütunu eklendi:")
print(df)
print()

# Sütun silme
print("--- SÜTUN SİLME ---")
df_dropped = df.drop('Yıllık Maaş', axis=1)
print("\nYıllık Maaş sütunu silindi:")
print(df_dropped)
print()

# Çoklu sütun silme
df_dropped2 = df.drop(['Şehir', 'Yıllık Maaş'], axis=1)
print("Şehir ve Yıllık Maaş silindi:")
print(df_dropped2)
print()

# Yeni satır ekleme
print("--- SATIR EKLEME ---")
yeni_satir = pd.DataFrame({
    'Ad': ['Fatma'],
    'Yaş': [28],
    'Maaş': [5500],
    'Şehir': ['Bursa'],
    'Yıllık Maaş': [66000]
})

df_concat = pd.concat([df, yeni_satir], ignore_index=True)
print("\nYeni satır eklendi:")
print(df_concat)
print()

# Satır silme
print("--- SATIR SİLME ---")
df_row_dropped = df.drop(0, axis=0)  # 0. indeksi sil
print("\n0. satır silindi:")
print(df_row_dropped)
print()

print("="*60)
print("7. Temel İstatistiksel İşlemler")
print("="*60)

df = pd.DataFrame({
    'Matematik': [85, 92, 78, 95, 88, 76, 90],
    'Fizik': [78, 85, 82, 90, 79, 88, 85],
    'Kimya': [90, 88, 75, 92, 85, 80, 87]
})

print("Örnek DataFrame (Notlar):")
print(df)
print()

# Temel istatistikler
print("--- TEMEL İSTATİSTİKLER ---")
print("\nOrtalamalar:")
print(df.mean())
print()

print("Medyan:")
print(df.median())
print()

print("Standart sapma:")
print(df.std())
print()

print("Minimum:")
print(df.min())
print()

print("Maksimum:")
print(df.max())
print()

print("Toplam:")
print(df.sum())
print()

# Özet istatistikler
print("--- ÖZET İSTATİSTİKLER ---")
print("\ndescribe():")
print(df.describe())
print()

# Satır bazında işlemler
print("--- SATIR BAZINDA İŞLEMLER ---")
print("\nHer öğrencinin ortalaması:")
df['Ortalama'] = df.mean(axis=1)
print(df)
print()

# Korelasyon
print("--- KORELASYON ---")
print("\nKorelasyon matrisi:")
print(df[['Matematik', 'Fizik', 'Kimya']].corr())
print()

# Value counts
df_sehir = pd.DataFrame({
    'Şehir': ['İstanbul', 'Ankara', 'İzmir', 'İstanbul', 'Ankara', 'İstanbul', 'Bursa']
})

print("--- VALUE_COUNTS ---")
print("\nŞehir dağılımı:")
print(df_sehir['Şehir'].value_counts())
print()

print("="*60)
print("8. Sıralama ve Sırayla İlgili İşlemler")
print("="*60)

df = pd.DataFrame({
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali'],
    'Yaş': [25, 30, 35, 28, 32],
    'Maaş': [5000, 6000, 7000, 5500, 6500]
})

print("Orijinal DataFrame:")
print(df)
print()

# Tek sütuna göre sıralama
print("--- TEK SÜTUN SIRALAMA ---")
print("\nYaşa göre artan sıralama:")
df_sorted = df.sort_values('Yaş')
print(df_sorted)
print()

print("Maaşa göre azalan sıralama:")
df_sorted_desc = df.sort_values('Maaş', ascending=False)
print(df_sorted_desc)
print()

# Çoklu sütuna göre sıralama
print("--- ÇOKLU SÜTUN SIRALAMA ---")
df2 = pd.DataFrame({
    'Departman': ['IT', 'HR', 'IT', 'HR', 'IT'],
    'Ad': ['Ahmet', 'Ayşe', 'Mehmet', 'Fatma', 'Ali'],
    'Maaş': [5000, 6000, 7000, 5500, 6500]
})

print("\nOrijinal:")
print(df2)
print()

print("Departman ve Maaşa göre sıralama:")
df2_sorted = df2.sort_values(['Departman', 'Maaş'], ascending=[True, False])
print(df2_sorted)
print()

# İndekse göre sıralama
print("--- İNDEKS SIRALAMA ---")
df_index_sorted = df.sort_index()
print("\nİndekse göre sıralama:")
print(df_index_sorted)
print()

# Ranking
print("--- RANKING ---")
df['Maaş Sırası'] = df['Maaş'].rank(ascending=False)
print("\nMaaş sıralaması eklendi:")
print(df)
print()

print("="*60)
print("GİRİŞ SEVİYESİ TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ Series ve DataFrame oluşturma")
print("✓ Veri okuma ve yazma (CSV, Excel, JSON)")
print("✓ İndeksleme ve seçim (loc, iloc)")
print("✓ Veri filtreleme")
print("✓ Sütun ve satır ekleme/silme")
print("✓ Temel istatistiksel işlemler")
print("✓ Sıralama işlemleri")
print()
print("Sıradaki: pandas_orta.py")
