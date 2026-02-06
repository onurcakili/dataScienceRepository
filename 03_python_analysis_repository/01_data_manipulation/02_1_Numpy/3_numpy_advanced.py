"""
=================================================
NumPy - İleri Seviye Eğitim Notları
=================================================

Bu bölümde NumPy'ın en ileri özelliklerini ve optimizasyon tekniklerini öğreneceksiniz.

Konu Başlıkları:
1. Performans Optimizasyonu ve Vectorization
2. Memory Management ve Views vs Copies
3. Structured Arrays ve Record Arrays
4. Advanced Broadcasting Techniques
5. Polynomial Operations
6. FFT (Fast Fourier Transform)
7. Masked Arrays
8. Universal Functions (ufuncs)
9. Veri Analizi için İleri Teknikler
10. NumPy Best Practices
"""

import numpy as np
import time
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("1. Performans Optimizasyonu ve Vectorization")
print("="*60)

# Python loop vs NumPy vectorization
size = 1000000

# Python loop yaklaşımı
python_list = list(range(size))
start = time.time()
result_loop = [x**2 for x in python_list]
time_loop = time.time() - start
print(f"Python loop süresi: {time_loop:.4f} saniye")

# NumPy vectorization
numpy_array = np.arange(size)
start = time.time()
result_vectorized = numpy_array**2
time_vectorized = time.time() - start
print(f"NumPy vectorization süresi: {time_vectorized:.4f} saniye")

speedup = time_loop / time_vectorized
print(f"Hızlanma oranı: {speedup:.2f}x daha hızlı")
print()

# Memory kullanımı karşılaştırması
print("Memory Kullanımı:")
list_size = sys.getsizeof(python_list[:1000])
array_size = numpy_array[:1000].nbytes
print(f"Python list (1000 eleman): {list_size} bytes")
print(f"NumPy array (1000 eleman): {array_size} bytes")
print(f"NumPy {list_size/array_size:.2f}x daha az bellek kullanır")
print()

# Vectorization örnekleri
print("Vectorization Örnekleri:")

# Örnek 1: Koşullu işlemler
data = np.random.randn(1000)
start = time.time()
result = np.where(data > 0, data**2, data**3)
vec_time = time.time() - start
print(f"Koşullu işlem (vectorized): {vec_time:.6f} saniye")

# Örnek 2: Çoklu koşullar
arr = np.random.randint(1, 100, 1000)
conditions = [arr < 30, (arr >= 30) & (arr < 70), arr >= 70]
choices = ['düşük', 'orta', 'yüksek']
result = np.select(conditions, choices)
print(f"İlk 10 değer: {arr[:10]}")
print(f"Kategoriler: {result[:10]}")
print()

# np.vectorize kullanımı
def custom_func(x):
    if x < 0:
        return -x
    elif x == 0:
        return 0
    else:
        return x**2

vectorized_func = np.vectorize(custom_func)
data = np.array([-5, -3, 0, 2, 4])
print(f"Orijinal: {data}")
print(f"Vectorized fonksiyon: {vectorized_func(data)}")
print()

print("="*60)
print("2. Memory Management ve Views vs Copies")
print("="*60)

# View oluşturma
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
print(f"Original: {original}")
print(f"View: {view}")

view[0] = 999
print(f"\nView değiştirildi")
print(f"Original (değişti): {original}")
print(f"View: {view}")
print()

# Copy oluşturma
original = np.array([1, 2, 3, 4, 5])
copy = original.copy()
copy[0] = 999
print(f"Copy değiştirildi")
print(f"Original (değişmedi): {original}")
print(f"Copy: {copy}")
print()

# View mi Copy mi kontrolü
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
copy = arr.copy()

print("View mi kontrol:")
print(f"view.base is arr: {view.base is arr}")
print(f"copy.base is arr: {copy.base is arr}")
print()

# Memory mapping
print("Memory Mapping (Büyük dosyalar için):")
big_array = np.random.rand(1000, 1000)
np.save('/home/claude/big_array.npy', big_array)

mmap_array = np.load('/home/claude/big_array.npy', mmap_mode='r')
print(f"Memory mapped array shape: {mmap_array.shape}")
print(f"İlk eleman: {mmap_array[0, 0]}")
print()

print("="*60)
print("3. Structured Arrays ve Record Arrays")
print("="*60)

# Structured array tanımlama
dt = np.dtype([('isim', 'U10'), ('yas', 'i4'), ('maas', 'f8')])
calisanlar = np.array([
    ('Ahmet', 25, 5000.0),
    ('Ayşe', 30, 6000.0),
    ('Mehmet', 28, 5500.0),
    ('Fatma', 35, 7000.0)
], dtype=dt)

print("Structured Array:")
print(calisanlar)
print()

print(f"İsimler: {calisanlar['isim']}")
print(f"Yaşlar: {calisanlar['yas']}")
print(f"Maaşlar: {calisanlar['maas']}")
print()

# Filtreleme
yuksek_maas = calisanlar[calisanlar['maas'] > 5500]
print("Maaşı 5500'den fazla olanlar:")
print(yuksek_maas)
print()

# Record array
rec_calisanlar = calisanlar.view(np.recarray)
print(f"İlk çalışan ismi: {rec_calisanlar.isim[0]}")
print(f"İlk çalışan yaşı: {rec_calisanlar.yas[0]}")
print()

print("="*60)
print("4. Advanced Broadcasting Techniques")
print("="*60)

# Newaxis kullanımı
arr = np.array([1, 2, 3, 4, 5])
print(f"Original shape: {arr.shape}")

arr_col = arr[:, np.newaxis]
print(f"Column vector shape: {arr_col.shape}")
print(arr_col)
print()

# Dış çarpım
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30])

outer = a[:, np.newaxis] * b[np.newaxis, :]
print("Outer product (broadcasting):")
print(outer)
print()

# Mesafe matrisi
points = np.array([[1, 2], [3, 4], [5, 6]])
print(f"Noktalar:\n{points}")

diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=-1))
print("\nMesafe matrisi:")
print(distances)
print()

# Normalizasyon
data = np.random.randn(5, 3) * 10 + 50
print("Orijinal veri:")
print(data)

mean = data.mean(axis=0)
std = data.std(axis=0)
normalized = (data - mean) / std
print("\nNormalize edilmiş:")
print(normalized)
print(f"Yeni mean: {normalized.mean(axis=0)}")
print(f"Yeni std: {normalized.std(axis=0)}")
print()

print("="*60)
print("5. Polynomial Operations")
print("="*60)

# Polinom: 2x^2 + 3x + 1
coeffs = [2, 3, 1]
poly = np.poly1d(coeffs)
print(f"Polinom: {poly}")
print()

x = np.array([0, 1, 2, 3])
y = poly(x)
print(f"x: {x}")
print(f"p(x): {y}")
print()

# Polinom türevi
p_deriv = np.polyder(poly)
print(f"Türev: {p_deriv}")

# Polinom integrali  
p_int = np.polyint(poly)
print(f"İntegral: {p_int}")
print()

# Kök bulma
p = np.poly1d([1, -5, 6])
roots = np.roots(p)
print(f"Polinom: {p}")
print(f"Kökler: {roots}")
print()

# Curve fitting
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1, 3, 7, 13, 21])

coeffs = np.polyfit(x_data, y_data, 2)
poly_fit = np.poly1d(coeffs)
print(f"Fit edilen polinom: {poly_fit}")

x_new = np.array([5, 6])
y_pred = poly_fit(x_new)
print(f"Tahminler: x={x_new}, y={y_pred}")
print()

print("="*60)
print("6. FFT (Fast Fourier Transform)")
print("="*60)

# Sinyal oluşturma
t = np.linspace(0, 1, 1000)
freq1, freq2 = 5, 20
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

print("Sinyal: 5 Hz + 20 Hz")
print(f"Sinyal uzunluğu: {len(signal)}")
print()

# FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])

power = np.abs(fft_result)**2

positive_freq_idx = frequencies > 0
freq_positive = frequencies[positive_freq_idx]
power_positive = power[positive_freq_idx]

top_freq_idx = np.argsort(power_positive)[-5:]
print("En yüksek 5 frekans:")
for idx in reversed(top_freq_idx):
    print(f"  {freq_positive[idx]:.2f} Hz, Güç: {power_positive[idx]:.2f}")
print()

# Inverse FFT
signal_reconstructed = np.fft.ifft(fft_result)
print(f"Rekonstrüksiyon hatası: {np.mean(np.abs(signal - signal_reconstructed.real)):.10f}")
print()

print("="*60)
print("7. Masked Arrays")
print("="*60)

# Eksik verileri maskele
data = np.array([1, 2, -999, 4, -999, 6, 7, -999, 9])
print(f"Ham veri: {data}")

masked_data = np.ma.masked_equal(data, -999)
print(f"Masked: {masked_data}")
print(f"Mask: {masked_data.mask}")
print()

print(f"Ortalama: {masked_data.mean()}")
print(f"Toplam: {masked_data.sum()}")
print(f"Std: {masked_data.std()}")
print()

# Koşullu maskeleme
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
masked = np.ma.masked_where(data > 5, data)
print(f"5'ten büyükler maskelendi: {masked}")
print()

# 2D masked array
matrix = np.array([[1, 2, 3], [4, -999, 6], [7, 8, -999]])
masked_matrix = np.ma.masked_equal(matrix, -999)
print("2D Masked:")
print(masked_matrix)
print(f"Satır ortalamaları: {masked_matrix.mean(axis=1)}")
print()

# Filled
filled = masked_matrix.filled(0)
print("Masked değerler 0 ile dolduruldu:")
print(filled)
print()

print("="*60)
print("8. Universal Functions (ufuncs)")
print("="*60)

print(f"np.add type: {type(np.add)}")
print(f"Input sayısı: {np.add.nin}")
print(f"Output sayısı: {np.add.nout}")
print()

# Reduce
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Toplam (reduce): {np.add.reduce(arr)}")
print(f"Çarpım (reduce): {np.multiply.reduce(arr)}")
print()

# Accumulate
print(f"Kümülatif toplam: {np.add.accumulate(arr)}")
print(f"Kümülatif çarpım: {np.multiply.accumulate(arr)}")
print()

# Outer
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print("Outer product:")
print(np.multiply.outer(a, b))
print()

# Custom ufunc
def my_func(x, y):
    return x**2 + y**2

my_ufunc = np.frompyfunc(my_func, 2, 1)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
result = my_ufunc(x, y)
print(f"Custom ufunc (x^2 + y^2):")
print(f"x: {x}, y: {y}")
print(f"Result: {result}")
print()

print("="*60)
print("9. Veri Analizi için İleri Teknikler")
print("="*60)

# Rolling window
def rolling_window(arr, window):
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Veri: {data}")
windows = rolling_window(data, 3)
print(f"3'lü pencereler:\n{windows}")
print(f"Pencere ortalamaları: {windows.mean(axis=1)}")
print()

# Binning
data = np.random.randint(0, 100, 20)
print(f"Veri: {data}")

bins = [0, 25, 50, 75, 100]
digitized = np.digitize(data, bins)
print(f"Bin indeksleri: {digitized}")
print()

# Histogram
counts, bin_edges = np.histogram(data, bins=5)
print("Histogram:")
for i in range(len(counts)):
    print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {counts[i]} eleman")
print()

# Korelasyon
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

correlation = np.corrcoef(x, y)
print("Korelasyon Matrisi:")
print(correlation)
print(f"Korelasyon: {correlation[0, 1]:.4f}")
print()

# Outlier detection (IQR)
data = np.concatenate([np.random.randn(100) * 10 + 50, [150, 160, -50]])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

outliers = (data < alt_sinir) | (data > ust_sinir)
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Alt sınır: {alt_sinir:.2f}, Üst sınır: {ust_sinir:.2f}")
print(f"Outlier sayısı: {outliers.sum()}")
print(f"Outlier değerleri: {data[outliers]}")
print()

# Z-score
data = np.random.randn(100) * 15 + 100
z_scores = (data - data.mean()) / data.std()
print(f"Orijinal: mean={data.mean():.2f}, std={data.std():.2f}")
print(f"Z-score: mean={z_scores.mean():.6f}, std={z_scores.std():.6f}")
print(f"3 sigma dışı: {np.sum(np.abs(z_scores) > 3)}")
print()

print("="*60)
print("10. NumPy Best Practices")
print("="*60)

print("NumPy En İyi Uygulamaları:\n")

print("1. VECTORIZATION KULLAN")
print("   ✗ for loop kullanma")
print("   ✓ NumPy fonksiyonlarını kullan")
print()

# Performans testi
def slow_sum(arr):
    result = 0
    for x in arr:
        result += x**2
    return result

def fast_sum(arr):
    return np.sum(arr**2)

test_arr = np.random.rand(10000)
start = time.time()
_ = slow_sum(test_arr)
slow_time = time.time() - start

start = time.time()
_ = fast_sum(test_arr)
fast_time = time.time() - start

print(f"Slow: {slow_time:.6f}s, Fast: {fast_time:.6f}s")
print(f"Speedup: {slow_time/fast_time:.2f}x")
print()

print("2. IN-PLACE İŞLEMLER")
arr = np.random.rand(1000, 1000)
print(f"Orijinal memory: {arr.nbytes / 1024 / 1024:.2f} MB")
arr *= 2  # In-place
print("In-place işlem ek memory kullanmaz")
print()

print("3. DOĞRU VERİ TİPİ")
arr_64 = np.random.rand(1000, 1000)
arr_32 = arr_64.astype(np.float32)
print(f"Float64: {arr_64.nbytes / 1024 / 1024:.2f} MB")
print(f"Float32: {arr_32.nbytes / 1024 / 1024:.2f} MB")
print(f"Tasarruf: %{(1 - arr_32.nbytes/arr_64.nbytes) * 100:.0f}")
print()

print("4. VIEW vs COPY")
arr = np.arange(10)
view = arr[::2]
copy = arr[::2].copy()
print(f"View: {view.base is arr}")
print(f"Copy: {copy.base is arr}")
print()

print("5. EINSUM kullanımı")
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C = np.einsum('ij,jk->ik', A, B)  # A @ B
print(f"Einsum matrix mult shape: {C.shape}")

arr = np.random.rand(5, 5)
trace = np.einsum('ii->', arr)
print(f"Trace (einsum): {trace:.4f}")
print()

print("="*60)
print("İLERİ SEVİYE TAMAMLANDI!")
print("="*60)
print()
print("Öğrenilenler:")
print("✓ Performans optimizasyonu ve vectorization")
print("✓ Memory management (view vs copy)")
print("✓ Structured ve record arrays")
print("✓ Advanced broadcasting")
print("✓ Polynomial operations")
print("✓ FFT (Fast Fourier Transform)")
print("✓ Masked arrays")
print("✓ Universal functions")
print("✓ Veri analizi teknikleri")
print("✓ NumPy best practices")
print()
print("NumPy tamamlandı! Şimdi Pandas'a geçin.")
