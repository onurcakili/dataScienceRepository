"""
=================================================
NumPy - Orta Seviyesi Eğitim Notları
=================================================

Bu bölümde NumPy'ın daha ileri özelliklerini öğreneceksiniz.

Konu Başlıkları:
1. Broadcasting (Yayılım)
2. Fancy Indexing (Gelişmiş İndeksleme)
3. Linear Algebra (Doğrusal Cebir)
4. Dosya İşlemleri
5. Gelişmiş Array Manipülasyonu
6. Sorting ve Searching
7. Set İşlemleri
8. Random Modülü
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. Broadcasting (Yayılım)")
print("="*60)

# Broadcasting, farklı şekillerdeki array'lerin aritmetik işlemlerde
# otomatik olarak uyumlu hale getirilmesidir

print("Broadcasting Örnek 1: Skaler ile array")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Array + 10: {arr + 10}")  # 10 tüm elemanlara eklenir
print()

print("Broadcasting Örnek 2: 1D array ile 2D array")
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_1d = np.array([10, 20, 30])

print("2D Array:")
print(arr_2d)
print(f"\n1D Array: {arr_1d}")
print("\n2D + 1D:")
print(arr_2d + arr_1d)  # 1D array her satıra eklenir
print()

print("Broadcasting Örnek 3: Sütun işlemleri")
column = np.array([[1], [2], [3]])
print("Column array:")
print(column)
print("\n2D Array:")
print(arr_2d)
print("\n2D + Column:")
print(arr_2d + column)
print()

# Broadcasting kuralları
print("Broadcasting Kuralları:")
print("1. Trailing boyutlar (en sağdaki) aynı olmalı")
print("2. Boyutlardan biri 1 olmalı")
print("3. Eksik boyutlar 1 olarak kabul edilir")
print()

# Pratik örnekler
print("Pratik Örnek: Merkezi normalizasyon")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Orijinal veri:")
print(data)
mean = np.mean(data, axis=0)  # Sütun ortalamaları
print(f"\nSütun ortalamaları: {mean}")
normalized = data - mean  # Broadcasting ile çıkarma
print("\nNormalize edilmiş veri:")
print(normalized)
print()

print("="*60)
print("2. Fancy Indexing (Gelişmiş İndeksleme)")
print("="*60)

# Integer array indexing
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Array: {arr}")

indices = [1, 3, 5, 7]
print(f"İndeksler {indices}: {arr[indices]}")
print()

# 2D fancy indexing
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

print("2D Array:")
print(arr_2d)
print()

# Belirli satır ve sütunları seçme
rows = [0, 2, 3]
cols = [1, 2, 3]
print(f"Satırlar {rows}, Sütunlar {cols}:")
print(arr_2d[rows, cols])  # (0,1), (2,2), (3,3) elemanları
print()

# Boolean masking ile fancy indexing
print("Boolean Masking Örnekleri:")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

# Çoklu koşul
mask = (arr > 3) & (arr < 8)
print(f"3'ten büyük VE 8'den küçük: {arr[mask]}")

mask = (arr < 3) | (arr > 8)
print(f"3'ten küçük VEYA 8'den büyük: {arr[mask]}")
print()

# np.where kullanımı
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

# 5'ten büyükse 'büyük', değilse 'küçük'
result = np.where(arr > 5, 'büyük', 'küçük')
print(f"Koşullu değiştirme: {result}")

# 5'ten büyük olanları 100 yap, diğerlerini koru
result = np.where(arr > 5, 100, arr)
print(f"5'ten büyük olanları 100 yap: {result}")
print()

# np.argwhere - koşulu sağlayan indeksleri bul
arr = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])
print(f"Array: {arr}")
indices = np.argwhere(arr > 5)
print(f"5'ten büyük olanların indeksleri:\n{indices.flatten()}")
print()

print("="*60)
print("3. Linear Algebra (Doğrusal Cebir)")
print("="*60)

# Matris çarpımı
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matris A:")
print(A)
print("\nMatris B:")
print(B)
print()

# Element-wise çarpım
print("Element-wise çarpım (A * B):")
print(A * B)
print()

# Matris çarpımı (dot product)
print("Matris çarpımı (A @ B veya np.dot(A, B)):")
print(A @ B)
print()

# Determinant
print(f"A'nın determinantı: {np.linalg.det(A)}")
print()

# Ters matris (Inverse)
print("A'nın tersi (inverse):")
A_inv = np.linalg.inv(A)
print(A_inv)
print("\nDoğrulama (A × A⁻¹ = I):")
print(np.round(A @ A_inv))  # Birim matris olmalı
print()

# Özdeğer ve özvektörler
print("Özdeğerler ve özvektörler:")
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Özdeğerler: {eigenvalues}")
print("Özvektörler:")
print(eigenvectors)
print()

# Matris ranki
print(f"A'nın rankı: {np.linalg.matrix_rank(A)}")
print()

# İz (Trace) - köşegen elemanların toplamı
print(f"A'nın izi (trace): {np.trace(A)}")
print()

# Norm hesaplama
vector = np.array([3, 4])
print(f"Vektör: {vector}")
print(f"L2 normu (Euclidean): {np.linalg.norm(vector)}")
print(f"L1 normu: {np.linalg.norm(vector, 1)}")
print()

# Lineer denklem çözme: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
print("Denklem sistemi: Ax = b")
print(f"A:\n{A}")
print(f"b: {b}")
x = np.linalg.solve(A, b)
print(f"Çözüm (x): {x}")
print(f"Doğrulama (Ax): {A @ x}")
print()

print("="*60)
print("4. Dosya İşlemleri")
print("="*60)

# NumPy binary format (.npy, .npz)
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Kaydedilecek array:")
print(arr)
print()

# Tek array kaydetme
np.save('/home/claude/test_array.npy', arr)
print("Array 'test_array.npy' olarak kaydedildi")

# Yükleme
loaded = np.load('/home/claude/test_array.npy')
print("Yüklenen array:")
print(loaded)
print()

# Çoklu array kaydetme (compressed)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.savez('/home/claude/test_arrays.npz', array1=arr1, array2=arr2)
print("Çoklu array 'test_arrays.npz' olarak kaydedildi")

# Yükleme
loaded_arrays = np.load('/home/claude/test_arrays.npz')
print(f"Array isimleri: {loaded_arrays.files}")
print(f"array1: {loaded_arrays['array1']}")
print(f"array2: {loaded_arrays['array2']}")
print()

# Text dosyası olarak kaydetme
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt('/home/claude/test_data.txt', data, delimiter=',', fmt='%d')
print("Array text dosyası olarak kaydedildi (virgülle ayrılmış)")

# Text dosyasından yükleme
loaded_text = np.loadtxt('/home/claude/test_data.txt', delimiter=',')
print("Text dosyasından yüklenen array:")
print(loaded_text)
print()

print("="*60)
print("5. Gelişmiş Array Manipülasyonu")
print("="*60)

# Repeat - elemanları tekrarlama
arr = np.array([1, 2, 3])
print(f"Orijinal: {arr}")
print(f"Repeat (her eleman 3 kez): {np.repeat(arr, 3)}")
print()

# Tile - array'i tekrarlama
print(f"Tile (tüm array 3 kez): {np.tile(arr, 3)}")
print()

# Meshgrid - koordinat matrisleri oluşturma
x = np.array([1, 2, 3])
y = np.array([10, 20])
print(f"x: {x}")
print(f"y: {y}")

X, Y = np.meshgrid(x, y)
print("\nMeshgrid X:")
print(X)
print("Meshgrid Y:")
print(Y)
print()

# Stack işlemleri
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

print(f"Array a: {a}")
print(f"Array b: {b}")
print(f"Array c: {c}")
print()

print("Stack (yeni boyutta birleştirme):")
stacked = np.stack([a, b, c])
print(stacked)
print(f"Shape: {stacked.shape}")
print()

print("Column_stack (sütun olarak birleştirme):")
col_stacked = np.column_stack([a, b, c])
print(col_stacked)
print()

print("Row_stack (satır olarak birleştirme):")
row_stacked = np.row_stack([a, b, c])
print(row_stacked)
print()

# Expand dims - boyut ekleme
arr = np.array([1, 2, 3, 4, 5])
print(f"Orijinal array: {arr}, shape: {arr.shape}")

expanded_0 = np.expand_dims(arr, axis=0)
print(f"Expand axis=0: {expanded_0}, shape: {expanded_0.shape}")

expanded_1 = np.expand_dims(arr, axis=1)
print(f"Expand axis=1:\n{expanded_1}, shape: {expanded_1.shape}")
print()

# Squeeze - tekli boyutları kaldırma
arr = np.array([[[1, 2, 3]]])
print(f"Orijinal shape: {arr.shape}")
squeezed = np.squeeze(arr)
print(f"Squeeze sonrası: {squeezed}, shape: {squeezed.shape}")
print()

print("="*60)
print("6. Sorting ve Searching")
print("="*60)

# Basit sıralama
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
print(f"Orijinal: {arr}")
print(f"Sıralanmış (sort): {np.sort(arr)}")
print(f"Orijinal (değişmedi): {arr}")
print()

# In-place sıralama
arr_copy = arr.copy()
arr_copy.sort()
print(f"In-place sort: {arr_copy}")
print()

# Argsort - sıralı indeksleri döndür
indices = np.argsort(arr)
print(f"Sıralı indeksler (argsort): {indices}")
print(f"İndekslerle erişim: {arr[indices]}")
print()

# 2D sıralama
arr_2d = np.array([[3, 1, 4], [9, 2, 6], [5, 8, 7]])
print("2D Array:")
print(arr_2d)
print("\nSatır bazında sıralama (axis=1):")
print(np.sort(arr_2d, axis=1))
print("\nSütun bazında sıralama (axis=0):")
print(np.sort(arr_2d, axis=0))
print()

# Searching
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Array: {arr}")

# Searchsorted - sıralı array'de eleman yerini bul
print(f"5'in yerleşeceği indeks: {np.searchsorted(arr, 5)}")
print(f"[2.5, 6.5, 8.5] değerlerinin indeksleri: {np.searchsorted(arr, [2.5, 6.5, 8.5])}")
print()

# Eleman var mı kontrol
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"3 var mı?: {3 in arr}")
print(f"10 var mı?: {10 in arr}")
print()

# Nonzero - sıfır olmayan elemanların indeksleri
arr = np.array([0, 1, 0, 2, 0, 3, 0])
print(f"Array: {arr}")
print(f"Sıfır olmayan indeksler: {np.nonzero(arr)}")
print()

print("="*60)
print("7. Set İşlemleri")
print("="*60)

# Unique - benzersiz elemanlar
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(f"Array: {arr}")
print(f"Benzersiz elemanlar: {np.unique(arr)}")

# Unique with counts
unique, counts = np.unique(arr, return_counts=True)
print(f"Elemanlar ve sayıları:")
for elem, count in zip(unique, counts):
    print(f"  {elem}: {count} kez")
print()

# İki küme işlemleri
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])

print(f"Küme A: {a}")
print(f"Küme B: {b}")
print()

print(f"Birleşim (union): {np.union1d(a, b)}")
print(f"Kesişim (intersection): {np.intersect1d(a, b)}")
print(f"Fark (A - B): {np.setdiff1d(a, b)}")
print(f"Simetrik fark: {np.setxor1d(a, b)}")
print()

# İçinde var mı kontrolü
print(f"[2, 5, 9] değerleri A'da var mı?: {np.isin([2, 5, 9], a)}")
print()

print("="*60)
print("8. Random Modülü (Gelişmiş)")
print("="*60)

# Random seed - tekrarlanabilirlik için
np.random.seed(42)
print("Random seed ayarlandı (42)")
print()

# Farklı dağılımlar
print("Normal (Gaussian) dağılım (mean=0, std=1):")
normal = np.random.randn(5)
print(normal)
print()

print("Normal dağılım (mean=100, std=15):")
normal_custom = np.random.normal(100, 15, 5)
print(normal_custom)
print()

print("Uniform dağılım (0-1 arası):")
uniform = np.random.rand(5)
print(uniform)
print()

print("Uniform dağılım (10-20 arası):")
uniform_custom = np.random.uniform(10, 20, 5)
print(uniform_custom)
print()

print("Poisson dağılımı (lambda=3):")
poisson = np.random.poisson(3, 10)
print(poisson)
print()

print("Binomial dağılım (n=10, p=0.5):")
binomial = np.random.binomial(10, 0.5, 10)
print(binomial)
print()

# Rastgele seçim
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")
print(f"5 rastgele eleman: {np.random.choice(arr, 5)}")
print(f"Tekrarsız 5 eleman: {np.random.choice(arr, 5, replace=False)}")
print()

# Shuffle - karıştırma
arr = np.array([1, 2, 3, 4, 5])
print(f"Orijinal: {arr}")
np.random.shuffle(arr)
print(f"Karıştırılmış: {arr}")
print()

# Permutation - yeni array döndür
arr = np.array([1, 2, 3, 4, 5])
print(f"Orijinal: {arr}")
permuted = np.random.permutation(arr)
print(f"Permutasyon: {permuted}")
print(f"Orijinal (değişmedi): {arr}")
print()

# Ağırlıklı seçim
choices = ['A', 'B', 'C']
probabilities = [0.1, 0.3, 0.6]  # Toplam 1 olmalı
print(f"Seçenekler: {choices}")
print(f"Olasılıklar: {probabilities}")
print("10 ağırlıklı seçim:")
selected = np.random.choice(choices, 10, p=probabilities)
print(selected)
print()

print("="*60)
print("ORTA SEVİYE TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ Broadcasting kavramı")
print("✓ Fancy indexing ve boolean masking")
print("✓ Doğrusal cebir işlemleri")
print("✓ Dosya kaydetme/yükleme")
print("✓ Gelişmiş array manipülasyonu")
print("✓ Sıralama ve arama işlemleri")
print("✓ Set işlemleri")
print("✓ Random modülü ve dağılımlar")
print()
print("Sıradaki: numpy_ileri.py")
