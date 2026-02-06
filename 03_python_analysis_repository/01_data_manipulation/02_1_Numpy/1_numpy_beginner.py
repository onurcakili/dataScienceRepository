"""
=================================================
NumPy - Giriş Seviyesi Eğitim Notları
=================================================

NumPy (Numerical Python), Python'da bilimsel hesaplama için temel kütüphanedir.
Çok boyutlu diziler ve matrisler üzerinde yüksek performanslı işlemler yapmayı sağlar.

Konu Başlıkları:
1. NumPy'a Giriş ve Kurulum
2. Array (Dizi) Oluşturma
3. Array Özellikleri ve Bilgi Alma
4. Temel İndeksleme ve Dilimleme
5. Temel Matematiksel İşlemler
6. Array Birleştirme ve Ayırma
7. Array Şekil Değiştirme
8. Basit İstatistiksel Fonksiyonlar
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. NumPy'a Giriş ve Kurulum")
print("="*60)

# NumPy versiyonu
print(f"NumPy Versiyonu: {np.__version__}")
print()

# NumPy neden kullanılır?
print("NumPy Avantajları:")
print("- Python listelerinden çok daha hızlı")
print("- Daha az bellek kullanır")
print("- Vektörize işlemler yapar")
print("- Bilimsel hesaplama için optimize edilmiştir")
print()

print("="*60)
print("2. Array (Dizi) Oluşturma")
print("="*60)

# Python listesinden array oluşturma
liste = [1, 2, 3, 4, 5]
array1 = np.array(liste)
print(f"Python listesinden array: {array1}")
print(f"Type: {type(array1)}")
print()

# Doğrudan array oluşturma
array2 = np.array([10, 20, 30, 40, 50])
print(f"Doğrudan array: {array2}")
print()

# 2 boyutlu array (matris)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2 Boyutlu Array:")
print(array_2d)
print()

# 3 boyutlu array
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3 Boyutlu Array:")
print(array_3d)
print()

# Özel array oluşturma fonksiyonları
print("Sıfırlardan oluşan array:")
zeros = np.zeros(5)
print(zeros)
print()

print("Birlerden oluşan array:")
ones = np.ones((3, 4))  # 3x4 boyutunda
print(ones)
print()

print("Boş array (başlatılmamış):")
empty = np.empty((2, 3))
print(empty)
print()

print("Belirli bir değerle dolu array:")
full = np.full((2, 3), 7)
print(full)
print()

# Aralık fonksiyonları
print("arange ile array (0'dan 10'a kadar, 2'şer artarak):")
arange_array = np.arange(0, 10, 2)
print(arange_array)
print()

print("linspace ile array (0 ile 1 arası 5 eşit parça):")
linspace_array = np.linspace(0, 1, 5)
print(linspace_array)
print()

# Birim matris (Identity matrix)
print("Birim matris (3x3):")
identity = np.eye(3)
print(identity)
print()

# Rastgele sayılar
print("Rastgele sayılar (0-1 arası):")
random_array = np.random.rand(3, 3)
print(random_array)
print()

print("Rastgele tam sayılar (1-100 arası):")
random_int = np.random.randint(1, 100, size=(3, 3))
print(random_int)
print()

print("="*60)
print("3. Array Özellikleri ve Bilgi Alma")
print("="*60)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Örnek Array:")
print(arr)
print()

# Temel özellikler
print(f"Boyut (ndim): {arr.ndim}")  # Kaç boyutlu
print(f"Şekil (shape): {arr.shape}")  # Her boyuttaki eleman sayısı
print(f"Toplam eleman sayısı (size): {arr.size}")
print(f"Veri tipi (dtype): {arr.dtype}")
print(f"Her elemanın byte boyutu (itemsize): {arr.itemsize}")
print(f"Toplam byte boyutu (nbytes): {arr.nbytes}")
print()

# Veri tipi dönüşümü
print("Veri Tipi Dönüşümü:")
float_array = arr.astype(float)
print(f"Float'a dönüştürülmüş: {float_array.dtype}")
print(float_array)
print()

print("="*60)
print("4. Temel İndeksleme ve Dilimleme")
print("="*60)

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Array: {arr}")
print()

# Tek elemana erişim
print(f"İlk eleman (index 0): {arr[0]}")
print(f"Son eleman (index -1): {arr[-1]}")
print(f"3. eleman (index 2): {arr[2]}")
print()

# Dilimleme (Slicing)
print(f"İlk 5 eleman: {arr[:5]}")
print(f"Son 3 eleman: {arr[-3:]}")
print(f"2. elemandan 7. elemana kadar: {arr[2:7]}")
print(f"2'şer atlayarak: {arr[::2]}")
print(f"Tersten yazdırma: {arr[::-1]}")
print()

# 2D array indeksleme
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:")
print(arr_2d)
print()

print(f"1. satır, 2. sütun: {arr_2d[0, 1]}")
print(f"2. satır: {arr_2d[1, :]}")
print(f"3. sütun: {arr_2d[:, 2]}")
print()

# Koşullu indeksleme (Boolean Indexing)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")
print(f"5'ten büyük elemanlar: {arr[arr > 5]}")
print(f"Çift sayılar: {arr[arr % 2 == 0]}")
print()

print("="*60)
print("5. Temel Matematiksel İşlemler")
print("="*60)

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print()

# Temel işlemler
print(f"Toplama: {arr1 + arr2}")
print(f"Çıkarma: {arr2 - arr1}")
print(f"Çarpma: {arr1 * arr2}")
print(f"Bölme: {arr2 / arr1}")
print(f"Üs alma: {arr1 ** 2}")
print()

# Skaler işlemler
print(f"Her elemana 10 ekleme: {arr1 + 10}")
print(f"Her elemanı 2 ile çarpma: {arr1 * 2}")
print()

# Matematiksel fonksiyonlar
arr = np.array([1, 4, 9, 16, 25])
print(f"Array: {arr}")
print(f"Karekök: {np.sqrt(arr)}")
print(f"Üstel (e^x): {np.exp(arr[:3])}")  # İlk 3 eleman
print(f"Logaritma: {np.log(arr)}")
print()

# Trigonometrik fonksiyonlar
angles = np.array([0, 30, 45, 60, 90])
radians = np.deg2rad(angles)  # Dereceyi radyana çevir
print(f"Açılar (derece): {angles}")
print(f"Sinüs değerleri: {np.sin(radians)}")
print(f"Kosinüs değerleri: {np.cos(radians)}")
print()

print("="*60)
print("6. Array Birleştirme ve Ayırma")
print("="*60)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate (birleştirme)
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Birleştirme (concatenate): {np.concatenate([arr1, arr2])}")
print()

# 2D array birleştirme
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])

print("Array 1 (2D):")
print(arr1_2d)
print("\nArray 2 (2D):")
print(arr2_2d)
print()

print("Dikey birleştirme (vstack):")
print(np.vstack([arr1_2d, arr2_2d]))
print()

print("Yatay birleştirme (hstack):")
print(np.hstack([arr1_2d, arr2_2d]))
print()

# Array ayırma (split)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Orijinal array: {arr}")
parcalar = np.split(arr, 3)  # 3 eşit parçaya böl
print(f"3 parçaya bölme: {parcalar}")
print()

print("="*60)
print("7. Array Şekil Değiştirme")
print("="*60)

arr = np.arange(1, 13)
print(f"Orijinal array: {arr}")
print(f"Shape: {arr.shape}")
print()

# Reshape
reshaped = arr.reshape(3, 4)
print("3x4 matrise dönüştürme (reshape):")
print(reshaped)
print()

reshaped2 = arr.reshape(4, 3)
print("4x3 matrise dönüştürme:")
print(reshaped2)
print()

# Flatten (düzleştirme)
print("Tekrar 1D array'e dönüştürme (flatten):")
print(reshaped.flatten())
print()

# Transpose (devrik alma)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Orijinal matris:")
print(matrix)
print(f"Shape: {matrix.shape}")
print()

print("Transpose (devrik):")
print(matrix.T)
print(f"Shape: {matrix.T.shape}")
print()

# Ravel (düzleştirme - flatten'a benzer)
print("Ravel ile düzleştirme:")
print(matrix.ravel())
print()

print("="*60)
print("8. Basit İstatistiksel Fonksiyonlar")
print("="*60)

data = np.array([12, 15, 18, 21, 24, 27, 30, 33, 36, 39])
print(f"Veri: {data}")
print()

# Temel istatistikler
print(f"Toplam (sum): {np.sum(data)}")
print(f"Ortalama (mean): {np.mean(data)}")
print(f"Medyan (median): {np.median(data)}")
print(f"Minimum (min): {np.min(data)}")
print(f"Maksimum (max): {np.max(data)}")
print(f"Standart sapma (std): {np.std(data)}")
print(f"Varyans (var): {np.var(data)}")
print()

# Min/Max indeks
print(f"Min değerin indeksi (argmin): {np.argmin(data)}")
print(f"Max değerin indeksi (argmax): {np.argmax(data)}")
print()

# Kümülatif işlemler
print(f"Kümülatif toplam (cumsum): {np.cumsum(data[:5])}")
print(f"Kümülatif çarpım (cumprod): {np.cumprod(data[:4])}")
print()

# 2D array için istatistikler
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:")
print(matrix)
print()

print(f"Tüm elemanların toplamı: {np.sum(matrix)}")
print(f"Sütun bazında toplam (axis=0): {np.sum(matrix, axis=0)}")
print(f"Satır bazında toplam (axis=1): {np.sum(matrix, axis=1)}")
print()

print(f"Sütun bazında ortalama: {np.mean(matrix, axis=0)}")
print(f"Satır bazında ortalama: {np.mean(matrix, axis=1)}")
print()

# Yüzdelik dilimler (percentile)
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Veri: {data}")
print(f"25. yüzdelik (Q1): {np.percentile(data, 25)}")
print(f"50. yüzdelik (Medyan): {np.percentile(data, 50)}")
print(f"75. yüzdelik (Q3): {np.percentile(data, 75)}")
print()

print("="*60)
print("GİRİŞ SEVİYESİ TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ NumPy array oluşturma yöntemleri")
print("✓ Array özellikleri ve bilgi alma")
print("✓ İndeksleme ve dilimleme")
print("✓ Temel matematiksel işlemler")
print("✓ Array birleştirme ve ayırma")
print("✓ Şekil değiştirme işlemleri")
print("✓ Temel istatistiksel fonksiyonlar")
print()
print("Sıradaki: numpy_orta.py")
