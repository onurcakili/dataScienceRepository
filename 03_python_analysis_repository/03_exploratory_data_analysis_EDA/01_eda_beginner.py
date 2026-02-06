"""
=============================================================================
GİRİŞ SEVİYESİ - EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================

Bu dosya veri analizine yeni başlayanlar için temel EDA tekniklerini içerir.

KAPSAM:
-------
1. Veri Setini Tanıma ve İlk Keşif
   - Veri setinin boyutunu öğrenme
   - İlk ve son satırları inceleme
   - Veri tiplerini anlama
   - Bellek kullanımını kontrol etme

2. Temel İstatistiksel Özetler
   - Sayısal değişkenler için merkezi eğilim ölçüleri (ortalama, medyan, mod)
   - Dağılım ölçüleri (standart sapma, varyans, min, max)
   - Çeyreklik değerler (Q1, Q2, Q3)

3. Basit Görselleştirmeler
   - Histogram: Veri dağılımını görmek için
   - Boxplot: Aykırı değerleri tespit etmek için
   - Density plot: Sürekli dağılımı anlamak için
   - Bar chart: Kategorik verileri göstermek için

4. Veri Kalitesi Kontrolü
   - Eksik değerlerin tespit edilmesi
   - Tekil (unique) değer sayısının kontrolü
   - Sabit sütunların belirlenmesi

5. Eksik Veri Analizi
   - Hangi sütunlarda eksik veri var?
   - Eksik veri oranları nedir?
   - Görsel analiz ile eksik veri kalıplarını belirleme

KULLANILAN KÜTÜPHANELER:
------------------------
- pandas: Veri manipülasyonu ve analiz
- numpy: Sayısal işlemler
- matplotlib: Temel görselleştirme
- seaborn: İleri seviye görselleştirme
- scipy: İstatistiksel testler

ÖRNEK KULLANIM:
---------------
    import pandas as pd
    import seaborn as sns
    
    # Veri seti yükle
    df = sns.load_dataset('titanic')
    
    # Temel EDA başlat
    eda = BasicEDA(df)
    eda.first_look()
    eda.data_types_analysis()
    eda.basic_statistics()
    eda.missing_data_analysis()
    
    # Sayısal değişken analizi
    numerical_analysis = NumericalVariableAnalysis(df)
    numerical_analysis.distribution_analysis('age')
    numerical_analysis.plot_all_variables()
    
    # Kategorik değişken analizi
    categorical_analysis = CategoricalVariableAnalysis(df)
    categorical_analysis.category_analysis('embark_town')
    categorical_analysis.plot_all_categorical_variables()

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# =============================================================================
# 1. VERİ SETİNİ YÜKLEME VE İLK KEŞİF
# =============================================================================

class BasicEDA:
    """
    Temel EDA işlemleri için sınıf.
    
    Bu sınıf, veri setinin genel yapısını anlamak için kullanılır.
    İlk adımda veriyi tanımak, veri tiplerini öğrenmek ve temel istatistikleri
    görmek için faydalıdır.
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        
    Methods:
        first_look(): Veri setine genel bakış
        data_types_analysis(): Veri tiplerini analiz eder
        basic_statistics(): Temel istatistiksel özetler
        missing_data_analysis(): Eksik veri analizi
        unique_value_analysis(): Tekil değer analizi
        
    Örnek:
        >>> df = pd.read_csv('data.csv')
        >>> eda = BasicEDA(df)
        >>> eda.first_look()
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
            
        Not:
            Orijinal veri setini değiştirmemek için kopyası alınır.
        """
        self.df = dataframe.copy()
        
    def first_look(self):
        """
        Veri setine ilk bakış.
        
        Bu metod şunları gösterir:
        - Veri setinin boyutu (satır ve sütun sayısı)
        - Bellek kullanımı
        - İlk 5 satır
        - Son 5 satır
        - Rastgele 5 satır
        
        Neden önemli:
            İlk bakış, veriyi tanımanın en önemli adımıdır. Veri setinin
            ne kadar büyük olduğunu, hangi sütunların olduğunu ve verilerin
            nasıl göründüğünü anlamak için gereklidir.
        """
        print("=" * 80)
        print("VERİ SETİNE İLK BAKIŞ")
        print("=" * 80)
        
        print(f"\nVeri Seti Boyutu: {self.df.shape[0]} satır, {self.df.shape[1]} sütun")
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Bellek Kullanımı: {memory_usage:.2f} MB")
        
        print("\n" + "="*80)
        print("İLK 5 SATIR:")
        print("="*80)
        print(self.df.head())
        
        print("\n" + "="*80)
        print("SON 5 SATIR:")
        print("="*80)
        print(self.df.tail())
        
        print("\n" + "="*80)
        print("RASTGELE 5 SATIR:")
        print("="*80)
        print(self.df.sample(5))
        
    def data_types_analysis(self):
        """
        Veri tiplerini analiz eder.
        
        Bu metod şunları gösterir:
        - Her sütunun veri tipi (int64, float64, object vb.)
        - Veri tiplerine göre sütun sayısı
        - Sayısal değişkenlerin listesi
        - Kategorik değişkenlerin listesi
        
        Veri Tipleri Hakkında:
            - int64: Tam sayılar (yaş, sayı vb.)
            - float64: Ondalıklı sayılar (fiyat, ağırlık vb.)
            - object: Metin verileri (isim, kategori vb.)
            - category: Kategorik veriler (düşük, orta, yüksek vb.)
            - datetime64: Tarih/saat verileri
            
        Neden önemli:
            Veri tipini bilmek, hangi analiz yöntemlerinin kullanılacağını
            belirler. Örneğin, sayısal verilerde ortalama alabilirken,
            kategorik verilerde frekans sayımı yapılır.
        """
        print("\n" + "="*80)
        print("VERİ TİPLERİ ANALİZİ")
        print("="*80)
        
        # Veri tipi bilgileri
        print("\nSütun Veri Tipleri:")
        print(self.df.dtypes)
        
        # Veri tiplerine göre gruplandırma
        print("\nVeri Tipi Dağılımı:")
        type_distribution = self.df.dtypes.value_counts()
        for dtype, count in type_distribution.items():
            print(f"  - {dtype}: {count} sütun")
        
        # Sayısal ve kategorik değişkenleri ayırma
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\nSayısal Değişkenler (Toplam: {len(numerical_cols)}):")
        for col in numerical_cols:
            print(f"  - {col}")
            
        print(f"\nKategorik Değişkenler (Toplam: {len(categorical_cols)}):")
        for col in categorical_cols:
            print(f"  - {col}")
    
    def basic_statistics(self):
        """
        Temel istatistiksel özetler.
        
        Bu metod şunları hesaplar:
        
        Sayısal Değişkenler İçin:
        - count: Geçerli (null olmayan) değer sayısı
        - mean: Ortalama değer
        - std: Standart sapma (verinin yayılma derecesi)
        - min: En küçük değer
        - 25%: Birinci çeyreklik (Q1)
        - 50%: İkinci çeyreklik (Q2, Medyan)
        - 75%: Üçüncü çeyreklik (Q3)
        - max: En büyük değer
        
        Kategorik Değişkenler İçin:
        - count: Geçerli değer sayısı
        - unique: Farklı kategori sayısı
        - top: En sık görülen kategori
        - freq: En sık görülen kategorinin frekansı
        
        İstatistiksel Terimler:
            Ortalama (Mean): Tüm değerlerin toplamının, değer sayısına bölümü
            Medyan (Median): Sıralanmış verilerin ortasındaki değer
            Standart Sapma (Std): Verilerin ortalamadan ne kadar saptığı
            Çeyreklikler (Quartiles): Verileri 4 eşit parçaya bölen değerler
        """
        print("\n" + "="*80)
        print("TEMEL İSTATİSTİKSEL ÖZETLER")
        print("="*80)
        
        # Sayısal değişkenler için
        print("\nSayısal Değişkenler İçin İstatistikler:")
        print(self.df.describe().T)
        
        print("\nAçıklama:")
        print("  - count: Eksik olmayan değer sayısı")
        print("  - mean: Aritmetik ortalama")
        print("  - std: Standart sapma (veri ne kadar dağınık)")
        print("  - min: En küçük değer")
        print("  - 25%: Birinci çeyreklik (verilerin %25'i bu değerin altında)")
        print("  - 50%: Medyan (ortanca değer)")
        print("  - 75%: Üçüncü çeyreklik (verilerin %75'i bu değerin altında)")
        print("  - max: En büyük değer")
        
        # Kategorik değişkenler için
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print("\nKategorik Değişkenler İçin İstatistikler:")
            print(self.df[categorical_cols].describe().T)
            
            print("\nAçıklama:")
            print("  - count: Eksik olmayan değer sayısı")
            print("  - unique: Farklı kategori sayısı")
            print("  - top: En sık görülen kategori")
            print("  - freq: En sık görülen kategorinin tekrar sayısı")
    
    def missing_data_analysis(self):
        """
        Eksik veri analizi.
        
        Eksik veri, veri setinde bazı gözlemler için değer olmayan durumu ifade eder.
        Python'da genellikle NaN (Not a Number) veya None olarak gösterilir.
        
        Bu metod şunları yapar:
        - Her sütundaki eksik değer sayısını hesaplar
        - Eksik değer yüzdesini hesaplar
        - Eksik değer içeren sütunları listeler
        - Eksik değerleri görselleştirir
        
        Eksik Veri Neden Önemli:
            Eksik veriler analiz sonuçlarını etkileyebilir. Çoğu makine öğrenmesi
            algoritması eksik veri içeren satırları kullanamaz. Bu nedenle,
            eksik verileri tespit etmek ve uygun şekilde işlemek kritiktir.
            
        Eksik Veri Çözümleri:
            1. Silme: Eksik değeri olan satırları/sütunları silmek
            2. Doldurma: Ortalama, medyan veya mod ile doldurmak
            3. Tahmin: Machine learning ile eksik değerleri tahmin etmek
            4. Kategori olarak işaretleme: "Bilinmiyor" kategorisi oluşturmak
        """
        print("\n" + "="*80)
        print("EKSİK VERİ ANALİZİ")
        print("="*80)
        
        # Eksik değer sayısı ve yüzdesi
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100
        
        missing_table = pd.DataFrame({
            'Sütun': missing_count.index,
            'Eksik_Sayi': missing_count.values,
            'Eksik_Yuzde': missing_percentage.values
        })
        
        missing_table = missing_table[missing_table['Eksik_Sayi'] > 0].sort_values('Eksik_Sayi', ascending=False)
        
        if len(missing_table) > 0:
            print("\nEksik Değer İçeren Sütunlar:")
            print(missing_table.to_string(index=False))
            
            print("\nDegerlendirme:")
            for _, row in missing_table.iterrows():
                if row['Eksik_Yuzde'] > 50:
                    print(f"  UYARI: '{row['Sütun']}' sütununun %{row['Eksik_Yuzde']:.1f}'si eksik! Sütunu silmeyi düşünün.")
                elif row['Eksik_Yuzde'] > 20:
                    print(f"  DİKKAT: '{row['Sütun']}' sütununun %{row['Eksik_Yuzde']:.1f}'si eksik! Doldurma stratejisi gerekli.")
                else:
                    print(f"  BİLGİ: '{row['Sütun']}' sütununun %{row['Eksik_Yuzde']:.1f}'si eksik. Kabul edilebilir seviye.")
            
            # Görselleştirme
            if len(missing_table) > 0:
                plt.figure(figsize=(10, 6))
                plt.barh(missing_table['Sütun'], missing_table['Eksik_Yuzde'])
                plt.xlabel('Eksik Değer Yüzdesi (%)')
                plt.title('Eksik Değerlerin Sütunlara Göre Dağılımı')
                plt.tight_layout()
                plt.show()
        else:
            print("\nVeri setinde eksik değer bulunmamaktadır.")
            print("Bu, veri kalitesi açısından iyi bir durumdur.")
    
    def unique_value_analysis(self):
        """
        Tekil (unique) değer analizi.
        
        Tekil değer, bir sütunda kaç farklı değerin olduğunu gösterir.
        
        Bu metod şunları hesaplar:
        - Her sütundaki tekil değer sayısı
        - Tekil değer oranı (tekil değer sayısı / toplam satır sayısı)
        
        Kardinalite Seviyeleri:
            - Düşük Kardinalite: Az sayıda farklı değer (örn: cinsiyet, kan grubu)
            - Orta Kardinalite: Orta seviyede farklı değer (örn: şehir, marka)
            - Yüksek Kardinalite: Çok sayıda farklı değer (örn: müşteri ID, telefon)
        
        Özel Durumlar:
            1. Tekil değer = 1: Sabit sütun, hiçbir bilgi taşımaz, silinebilir
            2. Tekil değer = Satır sayısı: Her satırda farklı değer (genelde ID)
            3. Tekil değer < 10: Kategorik değişken olarak işlenebilir
            4. Tekil değer > Satır sayısının %50'si: Sürekli değişken olabilir
        
        Neden önemli:
            Kardinalite, bir sütunun kategorik mi yoksa sürekli mi olduğunu,
            ve o sütunun modelde nasıl kullanılacağını belirler.
        """
        print("\n" + "="*80)
        print("TEKİL DEĞER ANALİZİ")
        print("="*80)
        
        unique_values = pd.DataFrame({
            'Sütun': self.df.columns,
            'Tekil_Deger_Sayisi': [self.df[col].nunique() for col in self.df.columns],
            'Tekil_Deger_Orani': [(self.df[col].nunique() / len(self.df)) * 100 for col in self.df.columns]
        })
        
        unique_values = unique_values.sort_values('Tekil_Deger_Sayisi', ascending=False)
        print(unique_values.to_string(index=False))
        
        # Kardinalite uyarıları
        print("\nKardinalite Değerlendirmesi:")
        print("(Kardinalite: Bir sütundaki farklı değer sayısı)")
        print("")
        
        for _, row in unique_values.iterrows():
            col_name = row['Sütun']
            unique_count = row['Tekil_Deger_Sayisi']
            unique_ratio = row['Tekil_Deger_Orani']
            
            if unique_count == 1:
                print(f"  UYARI: '{col_name}' sabit değer içeriyor (tüm değerler aynı)")
                print(f"         Bu sütun analiz için faydasız olabilir, silinebilir.")
            elif unique_count == len(self.df):
                print(f"  BİLGİ: '{col_name}' her satırda farklı değer içeriyor")
                print(f"         Bu muhtemelen bir ID sütunudur.")
            elif unique_ratio < 1:
                print(f"  İYİ: '{col_name}' düşük kardinaliteye sahip (%{unique_ratio:.2f})")
                print(f"       Bu sütun kategorik değişken olarak kullanılabilir.")
            elif unique_ratio > 50:
                print(f"  BİLGİ: '{col_name}' yüksek kardinaliteye sahip (%{unique_ratio:.2f})")
                print(f"         Bu sütun sürekli değişken olabilir veya one-hot encoding için uygun değildir.")


# =============================================================================
# 2. SAYISAL DEĞİŞKENLER İÇİN GÖRSELLEŞTİRME
# =============================================================================

class NumericalVariableAnalysis:
    """
    Sayısal değişkenler için detaylı analiz sınıfı.
    
    Sayısal değişkenler, matematiksel işlemlerin yapılabileceği değişkenlerdir.
    Örneğin: yaş, gelir, fiyat, sıcaklık, ağırlık vb.
    
    Bu sınıf şunları yapar:
    - Dağılım analizi (histogram, yoğunluk grafiği)
    - Merkezi eğilim ölçüleri (ortalama, medyan, mod)
    - Dağılım ölçüleri (standart sapma, varyans, ranj)
    - Çarpıklık ve basıklık analizi
    - Aykırı değer tespiti (boxplot)
    - Normallik testi (Q-Q plot)
    
    Methods:
        distribution_analysis(): Tek bir değişken için detaylı dağılım analizi
        plot_all_variables(): Tüm sayısal değişkenler için hızlı görselleştirme
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def distribution_analysis(self, column_name):
        """
        Tek bir sayısal değişken için dağılım analizi.
        
        Dağılım Analizi Nedir:
            Dağılım, bir değişkenin değerlerinin nasıl yayıldığını gösterir.
            Örneğin, yaş değişkeni için çoğu insan 20-40 yaş arasındaysa,
            dağılım bu aralıkta yoğunlaşmıştır.
        
        Çarpıklık (Skewness):
            - Pozitif çarpıklık (sağa çarpık): Uzun sağ kuyruk
              Örnek: Gelir dağılımı (az sayıda çok zengin insan)
            - Negatif çarpıklık (sola çarpık): Uzun sol kuyruk
              Örnek: Yaş dağılımı emekli toplumda
            - Sıfıra yakın: Simetrik dağılım (normal dağılıma benzer)
        
        Basıklık (Kurtosis):
            - Pozitif: Sivri dağılım (değerler ortalamanın etrafında toplanmış)
            - Negatif: Basık dağılım (değerler geniş bir alana yayılmış)
            - Sıfıra yakın: Normal dağılıma yakın
        
        Parameters:
            column_name (str): Analiz edilecek sütun adı
            
        Görselleştirmeler:
            1. Histogram: Veri dağılımını gösterir
            2. Boxplot: Aykırı değerleri ve çeyreklikleri gösterir
            3. Kernel Density Plot: Sürekli dağılım eğrisini gösterir
            4. Q-Q Plot: Normal dağılıma ne kadar yakın olduğunu gösterir
        """
        if column_name not in self.numerical_cols:
            print(f"HATA: '{column_name}' sayısal bir değişken değil!")
            return
        
        data = self.df[column_name].dropna()
        
        print(f"\n{'='*80}")
        print(f"DAĞILIM ANALİZİ: {column_name}")
        print(f"{'='*80}")
        
        # Temel istatistikler
        print(f"\nTemel İstatistikler:")
        print(f"  - Ortalama (Mean): {data.mean():.2f}")
        print(f"    Açıklama: Tüm değerlerin ortalaması")
        print(f"  - Medyan (Median): {data.median():.2f}")
        print(f"    Açıklama: Sıralanmış değerlerin ortasındaki değer")
        
        if len(data.mode()) > 0:
            print(f"  - Mod (Mode): {data.mode().values[0]}")
            print(f"    Açıklama: En sık tekrar eden değer")
        
        print(f"  - Standart Sapma: {data.std():.2f}")
        print(f"    Açıklama: Değerlerin ortalamadan ortalama sapması")
        print(f"  - Varyans: {data.var():.2f}")
        print(f"    Açıklama: Standart sapmanın karesi, dağılımın ölçüsü")
        print(f"  - Minimum: {data.min():.2f}")
        print(f"  - Maksimum: {data.max():.2f}")
        print(f"  - Ranj (Max - Min): {data.max() - data.min():.2f}")
        
        # Çeyreklikler
        print(f"\nÇeyreklik Değerler:")
        print(f"  - Q1 (25%): {data.quantile(0.25):.2f}")
        print(f"    Açıklama: Değerlerin %25'i bu sayının altında")
        print(f"  - Q2 (50%, Medyan): {data.quantile(0.50):.2f}")
        print(f"    Açıklama: Değerlerin yarısı bu sayının altında")
        print(f"  - Q3 (75%): {data.quantile(0.75):.2f}")
        print(f"    Açıklama: Değerlerin %75'i bu sayının altında")
        print(f"  - IQR (Q3 - Q1): {data.quantile(0.75) - data.quantile(0.25):.2f}")
        print(f"    Açıklama: Çeyreklikler arası aralık, verilerin %50'sinin bulunduğu aralık")
        
        # Çarpıklık ve Basıklık
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        print(f"\nDağılım Şekli:")
        print(f"  - Çarpıklık (Skewness): {skewness:.2f}")
        if skewness > 1:
            print(f"    Yorum: Yüksek derecede sağa çarpık (uzun sağ kuyruk)")
        elif skewness > 0.5:
            print(f"    Yorum: Orta derecede sağa çarpık")
        elif skewness > 0:
            print(f"    Yorum: Hafif sağa çarpık")
        elif skewness < -1:
            print(f"    Yorum: Yüksek derecede sola çarpık (uzun sol kuyruk)")
        elif skewness < -0.5:
            print(f"    Yorum: Orta derecede sola çarpık")
        elif skewness < 0:
            print(f"    Yorum: Hafif sola çarpık")
        else:
            print(f"    Yorum: Simetrik dağılım (normal dağılıma yakın)")
        
        print(f"  - Basıklık (Kurtosis): {kurtosis:.2f}")
        if kurtosis > 3:
            print(f"    Yorum: Çok sivri dağılım (leptokurtik) - değerler merkezde yoğunlaşmış")
        elif kurtosis > 0:
            print(f"    Yorum: Sivri dağılım (leptokurtik)")
        elif kurtosis < -3:
            print(f"    Yorum: Çok basık dağılım (platykurtik) - değerler geniş alana yayılmış")
        elif kurtosis < 0:
            print(f"    Yorum: Basık dağılım (platykurtik)")
        else:
            print(f"    Yorum: Normal dağılıma yakın (mesokurtik)")
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(data.mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {data.mean():.2f}')
        axes[0, 0].axvline(data.median(), color='green', linestyle='--', 
                          label=f'Medyan: {data.median():.2f}')
        axes[0, 0].set_xlabel(column_name)
        axes[0, 0].set_ylabel('Frekans')
        axes[0, 0].set_title(f'Histogram - {column_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Boxplot
        axes[0, 1].boxplot(data, vert=True)
        axes[0, 1].set_ylabel(column_name)
        axes[0, 1].set_title(f'Boxplot - {column_name}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Kernel Density Plot
        data.plot(kind='density', ax=axes[1, 0], color='blue', linewidth=2)
        axes[1, 0].set_xlabel(column_name)
        axes[1, 0].set_ylabel('Yoğunluk')
        axes[1, 0].set_title(f'Yoğunluk Grafiği - {column_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot (Normallik testi)
        stats.probplot(data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot - {column_name}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_variables(self):
        """
        Tüm sayısal değişkenler için hızlı görselleştirme.
        
        Bu metod, tüm sayısal değişkenlerin histogramlarını yan yana gösterir.
        Hızlı bir genel bakış için kullanışlıdır.
        
        Kullanım Senaryosu:
            Elimizde 10 sayısal değişken var ve hepsinin dağılımına hızlıca
            bakmak istiyoruz. Her birini tek tek incelemek yerine, bu metod
            hepsini aynı anda gösterir.
        """
        if len(self.numerical_cols) == 0:
            print("HATA: Veri setinde sayısal değişken bulunamadı!")
            return
        
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if len(self.numerical_cols) > 1 else [axes]
        
        for idx, col in enumerate(self.numerical_cols):
            mean_value = self.df[col].mean()
            self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}\n(Ortalama: {mean_value:.2f})')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frekans')
            axes[idx].grid(True, alpha=0.3)
        
        # Boş subplotları gizle
        for idx in range(len(self.numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# 3. KATEGORİK DEĞİŞKENLER İÇİN ANALIZ
# =============================================================================

class CategoricalVariableAnalysis:
    """
    Kategorik değişkenler için detaylı analiz sınıfı.
    
    Kategorik değişkenler, sınırlı sayıda kategori içeren değişkenlerdir.
    Örneğin: cinsiyet (erkek/kadın), şehir (İstanbul/Ankara/İzmir),
    eğitim seviyesi (lise/üniversite/yüksek lisans) vb.
    
    Kategorik Değişken Türleri:
        1. Nominal: Sıralama yok (örn: renk, şehir)
        2. Ordinal: Sıralama var (örn: eğitim seviyesi, memnuniyet)
    
    Bu sınıf şunları yapar:
    - Frekans analizi (her kategorinin sayısı)
    - Yüzdelik dağılım
    - En yaygın ve en nadir kategorileri bulma
    - Bar chart ve pasta grafiği ile görselleştirme
    
    Methods:
        category_analysis(): Tek bir kategorik değişken için detaylı analiz
        plot_all_categorical_variables(): Tüm kategorik değişkenler için görselleştirme
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def category_analysis(self, column_name, top_n=10):
        """
        Tek bir kategorik değişken için analiz.
        
        Frekans Analizi:
            Frekans, her kategorinin veri setinde kaç kez geçtiğini gösterir.
            Örneğin, 100 kişilik veri setinde 60 erkek, 40 kadın varsa,
            erkek frekansı 60, kadın frekansı 40'tır.
        
        Yüzdelik Dağılım:
            Toplam içindeki oranı gösterir. Yukarıdaki örnekte,
            erkek %60, kadın %40'tır.
        
        Parameters:
            column_name (str): Analiz edilecek sütun adı
            top_n (int): Gösterilecek en yaygın kategori sayısı
            
        Kullanım Senaryosu:
            Bir e-ticaret sitesinde hangi kategorilerdeki ürünler
            daha çok satılıyor öğrenmek için frekans analizi yapılır.
        """
        if column_name not in self.categorical_cols:
            print(f"HATA: '{column_name}' kategorik bir değişken değil!")
            return
        
        print(f"\n{'='*80}")
        print(f"KATEGORİK ANALİZ: {column_name}")
        print(f"{'='*80}")
        
        # Frekans analizi
        frequency = self.df[column_name].value_counts()
        percentage = (self.df[column_name].value_counts(normalize=True) * 100).round(2)
        
        analysis_df = pd.DataFrame({
            'Kategori': frequency.index,
            'Frekans': frequency.values,
            'Yuzde': percentage.values
        })
        
        print(f"\nFrekans Tablosu (İlk {top_n} kategori):")
        print(analysis_df.head(top_n).to_string(index=False))
        
        print(f"\nÖzet İstatistikler:")
        print(f"  - Toplam Kategori Sayısı: {self.df[column_name].nunique()}")
        print(f"  - En Yaygın Kategori: {frequency.index[0]} ({frequency.values[0]} adet, %{percentage.values[0]})")
        print(f"  - En Az Yaygın Kategori: {frequency.index[-1]} ({frequency.values[-1]} adet, %{percentage.values[-1]})")
        
        # Değerlendirme
        if self.df[column_name].nunique() > 50:
            print(f"\nUYARI: Bu değişken {self.df[column_name].nunique()} farklı kategori içeriyor.")
            print("       Yüksek kardinalite nedeniyle one-hot encoding yerine")
            print("       target encoding veya frequency encoding düşünülebilir.")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        top_categories = frequency.head(top_n)
        axes[0].barh(range(len(top_categories)), top_categories.values)
        axes[0].set_yticks(range(len(top_categories)))
        axes[0].set_yticklabels(top_categories.index)
        axes[0].set_xlabel('Frekans')
        axes[0].set_title(f'En Yaygın {top_n} Kategori - {column_name}')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Pasta grafiği
        axes[1].pie(top_categories.values, labels=top_categories.index, 
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f'Dağılım (İlk {top_n}) - {column_name}')
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_categorical_variables(self):
        """
        Tüm kategorik değişkenler için hızlı görselleştirme.
        
        Bu metod, her kategorik değişken için en yaygın 10 kategoriyi
        bar chart ile gösterir. Hızlı bir genel bakış için kullanışlıdır.
        
        Not:
            Çok fazla kategorik değişken varsa (>20), performans nedeniyle
            sadece ilk 5 tanesi gösterilir.
        """
        if len(self.categorical_cols) == 0:
            print("HATA: Veri setinde kategorik değişken bulunamadı!")
            return
        
        # İlk 5 kategorik değişken (performans için)
        cols_to_plot = self.categorical_cols[:5]
        
        if len(self.categorical_cols) > 5:
            print(f"\nBİLGİ: Toplam {len(self.categorical_cols)} kategorik değişken var.")
            print(f"       Performans için sadece ilk 5 tanesi gösteriliyor.\n")
        
        for col in cols_to_plot:
            top_10 = self.df[col].value_counts().head(10)
            
            plt.figure(figsize=(10, 5))
            plt.barh(range(len(top_10)), top_10.values)
            plt.yticks(range(len(top_10)), top_10.index)
            plt.xlabel('Frekans')
            plt.title(f'En Yaygın 10 Kategori - {col}')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()


# =============================================================================
# ÖRNEK KULLANIM
# =============================================================================

if __name__ == "__main__":
    print("""
    ============================================================================
              GİRİŞ SEVİYESİ - EXPLORATORY DATA ANALYSIS
                     Temel Veri Keşfi ve Analiz
    ============================================================================
    """)
    
    # Örnek veri seti yükleme (Seaborn'dan Titanic veri seti)
    print("\nÖrnek veri seti yükleniyor (Titanic)...\n")
    df = sns.load_dataset('titanic')
    
    print("="*80)
    print("VERİ SETİ HAKKINDA")
    print("="*80)
    print("Titanic veri seti, 1912 yılında batan Titanic gemisindeki")
    print("yolcuların bilgilerini içerir. Bu veri seti, hayatta kalma")
    print("durumunu tahmin etmek için sıklıkla kullanılır.")
    print()
    print("Sütunlar:")
    print("  - survived: Hayatta kalma durumu (0: hayır, 1: evet)")
    print("  - pclass: Yolcu sınıfı (1, 2, 3)")
    print("  - sex: Cinsiyet")
    print("  - age: Yaş")
    print("  - fare: Bilet ücreti")
    print("  - embarked: Binme limanı")
    print("  - class: Yolcu sınıfı (metinsel)")
    print("  ve daha fazlası...")
    print()
    
    # Temel EDA işlemleri
    print("TEMEL EDA İŞLEMLERİ BAŞLATILIYOR...")
    print()
    eda = BasicEDA(df)
    
    # 1. İlk bakış
    print("\n[1/5] İlk Bakış Analizi Yapılıyor...")
    eda.first_look()
    
    # 2. Veri tipleri analizi
    print("\n[2/5] Veri Tipleri Analiz Ediliyor...")
    eda.data_types_analysis()
    
    # 3. Temel istatistikler
    print("\n[3/5] Temel İstatistikler Hesaplanıyor...")
    eda.basic_statistics()
    
    # 4. Eksik veri analizi
    print("\n[4/5] Eksik Veri Analizi Yapılıyor...")
    eda.missing_data_analysis()
    
    # 5. Tekil değer analizi
    print("\n[5/5] Tekil Değer Analizi Yapılıyor...")
    eda.unique_value_analysis()
    
    # Sayısal değişken analizi
    print("\n" + "="*80)
    print("SAYISAL DEĞİŞKEN ANALİZİ BAŞLATILIYOR...")
    print("="*80)
    numerical_analysis = NumericalVariableAnalysis(df)
    
    # Örnek: Age değişkeni için detaylı analiz
    print("\nÖrnek analiz: 'age' (yaş) değişkeni")
    numerical_analysis.distribution_analysis('age')
    
    # Tüm sayısal değişkenler için hızlı görselleştirme
    print("\nTüm sayısal değişkenler için hızlı görselleştirme yapılıyor...")
    numerical_analysis.plot_all_variables()
    
    # Kategorik değişken analizi
    print("\n" + "="*80)
    print("KATEGORİK DEĞİŞKEN ANALİZİ BAŞLATILIYOR...")
    print("="*80)
    categorical_analysis = CategoricalVariableAnalysis(df)
    
    # Örnek: Embarked değişkeni için detaylı analiz
    print("\nÖrnek analiz: 'embark_town' (binme limanı) değişkeni")
    categorical_analysis.category_analysis('embark_town')
    
    # Tüm kategorik değişkenler için hızlı görselleştirme
    print("\nTüm kategorik değişkenler için hızlı görselleştirme yapılıyor...")
    categorical_analysis.plot_all_categorical_variables()
    
    print("\n" + "="*80)
    print("GİRİŞ SEVİYESİ EDA TAMAMLANDI")
    print("="*80)
    print("\nÖğrenilen Konular:")
    print("  - Veri setini tanıma ve ilk keşif")
    print("  - Veri tiplerini anlama")
    print("  - Temel istatistiksel ölçüler")
    print("  - Eksik veri analizi")
    print("  - Sayısal değişken dağılım analizi")
    print("  - Kategorik değişken frekans analizi")
    print("\nSonraki Adım:")
    print("  Orta seviye EDA dosyasına geçerek korelasyon analizi,")
    print("  aykırı değer tespiti ve normallik testlerini öğrenebilirsiniz.")
