"""
=============================================================================
ORTA SEVİYE - EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================

Bu dosya orta seviye veri analizi tekniklerini içerir.

KAPSAM:
-------
1. Korelasyon ve İlişki Analizleri
   - Pearson korelasyonu (doğrusal ilişki)
   - Spearman korelasyonu (monoton ilişki)
   - Kendall korelasyonu (sıralı ilişki)
   - Korelasyon matrisi görselleştirme
   - Kategorik-sayısal ilişki analizi (ANOVA)

2. Aykırı Değer (Outlier) Tespiti ve İşlenmesi
   - IQR (Interquartile Range) yöntemi
   - Z-Score yöntemi
   - Aykırı değerleri silme
   - Aykırı değerleri ortalama/medyan ile doldurma
   - Aykırı değerleri baskılama (suppress)

3. Normallik Testleri ve Dağılım Analizleri
   - Shapiro-Wilk testi
   - Anderson-Darling testi
   - Kolmogorov-Smirnov testi
   - Q-Q plot ile görsel normallik kontrolü
   - Histogram + Normal dağılım eğrisi karşılaştırması

4. Çok Değişkenli Görselleştirmeler
   - Heatmap (ısı haritası)
   - Pairplot (ikili ilişkiler)
   - Violin plot (dağılım + yoğunluk)
   - Scatter plot (nokta grafiği)

KULLANILAN KÜTÜPHANELER:
------------------------
- pandas, numpy: Veri işleme
- matplotlib, seaborn: Görselleştirme
- scipy: İstatistiksel testler
- sklearn: Veri ön işleme

ÖRNEK KULLANIM:
---------------
    import pandas as pd
    import seaborn as sns
    
    # Veri seti yükle
    df = sns.load_dataset('diamonds')
    
    # Korelasyon analizi
    correlation = CorrelationAnalysis(df)
    correlation.correlation_matrix(method='pearson', threshold=0.7)
    correlation.categorical_numerical_relationship('cut', 'price')
    
    # Aykırı değer analizi
    outlier = OutlierAnalysis(df)
    outlier_info = outlier.iqr_method('price', k=1.5)
    df_clean = outlier.handle_outliers('price', method='iqr', action='suppress')
    
    # Normallik testi
    normality = NormalityTests(df)
    normality.shapiro_wilk_test('carat')
    normality.normality_visualization('carat')
    results = normality.test_all_variables()

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


# =============================================================================
# 1. KORELASYON VE İLİŞKİ ANALİZLERİ
# =============================================================================

class CorrelationAnalysis:
    """
    Değişkenler arası ilişkileri inceleyen sınıf.
    
    Korelasyon Nedir:
        İki değişken arasındaki doğrusal ilişkinin gücünü ve yönünü ölçer.
        Değeri -1 ile +1 arasında değişir.
        
    Korelasyon Yorumlama:
        +1.0: Mükemmel pozitif ilişki (biri artarken diğeri de artar)
        +0.7 ile +1.0: Güçlü pozitif ilişki
        +0.3 ile +0.7: Orta pozitif ilişki
        0.0: İlişki yok
        -0.3 ile -0.7: Orta negatif ilişki
        -0.7 ile -1.0: Güçlü negatif ilişki
        -1.0: Mükemmel negatif ilişki (biri artarken diğeri azalır)
    
    Korelasyon Türleri:
        1. Pearson: Doğrusal ilişkiyi ölçer (en yaygın)
           Kullanım: Normal dağılımlı sürekli değişkenler
           Örnek: Boy ve kilo arasındaki ilişki
           
        2. Spearman: Monoton ilişkiyi ölçer (sıra tabanlı)
           Kullanım: Normal dağılmayan veya ordinal veriler
           Örnek: Eğitim seviyesi ve gelir arasındaki ilişki
           
        3. Kendall: Sıra korelasyonu (Spearman'a alternatif)
           Kullanım: Küçük örneklemler, çok sayıda eşit değer
    
    Önemli Uyarılar:
        - Korelasyon, nedensellik anlamına gelmez!
          Örnek: Dondurma satışı ile boğulma olayları koreledir,
          ama biri diğerine neden olmaz (her ikisi de sıcak havadan etkilenir)
        
        - Sadece doğrusal ilişkiyi ölçer (Pearson için)
          Örnek: Y = X^2 gibi parabol ilişkiler düşük korelasyon gösterebilir
        
        - Aykırı değerlerden etkilenir
          Tek bir uç değer, korelasyonu yanıltabilir
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        numerical_cols (list): Sayısal sütun isimleri
        
    Methods:
        correlation_matrix(): Korelasyon matrisi hesaplar ve görselleştirir
        pairplot_analysis(): İkili ilişkileri görselleştirir
        categorical_numerical_relationship(): Kategorik-sayısal ilişki analizi
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def correlation_matrix(self, method='pearson', threshold=0.5):
        """
        Korelasyon matrisi hesaplar ve görselleştirir.
        
        Korelasyon Matrisi:
            Tüm sayısal değişkenlerin birbirleriyle korelasyonunu gösteren
            simetrik bir tablodur. Köşegen her zaman 1'dir (kendisiyle korelasyon).
        
        Parameters:
            method (str): 'pearson', 'spearman' veya 'kendall'
                - 'pearson': Normal dağılımlı veriler için (varsayılan)
                - 'spearman': Normal dağılmayan veriler için
                - 'kendall': Küçük örneklemler için
            threshold (float): Güçlü korelasyon eşik değeri (0-1 arası)
                Örnek: threshold=0.7 -> |korelasyon| > 0.7 olanları göster
        
        Returns:
            pd.DataFrame: Korelasyon matrisi
            
        Kullanım Örnekleri:
            >>> correlation = CorrelationAnalysis(df)
            >>> corr_matrix = correlation.correlation_matrix(method='pearson', threshold=0.7)
            >>> # Yüksek korelasyonlu değişkenleri bulur ve gösterir
        
        Yorumlama İpuçları:
            - Yüksek korelasyon (>0.9): Çoklu bağlantı (multicollinearity) riski
              Bu durumda değişkenlerden birini modelden çıkarmayı düşünün
            - Hedef değişkenle yüksek korelasyon: İyi özellik adayı
            - Düşük korelasyon (<0.1): Hedef değişkeni açıklamada zayıf
        """
        print(f"\n{'='*80}")
        print(f"KORELASYON ANALİZİ ({method.upper()} Yöntemi)")
        print(f"{'='*80}")
        
        print(f"\nKorelasyon Yöntemi: {method}")
        if method == 'pearson':
            print("Açıklama: Doğrusal ilişkiyi ölçer (en yaygın kullanılan)")
            print("Uygun: Normal dağılımlı sürekli değişkenler")
        elif method == 'spearman':
            print("Açıklama: Monoton ilişkiyi ölçer (sıra tabanlı)")
            print("Uygun: Normal dağılmayan veya ordinal veriler")
        elif method == 'kendall':
            print("Açıklama: Sıra korelasyonu (Spearman'a alternatif)")
            print("Uygun: Küçük örneklemler veya çok sayıda eşit değer")
        
        # Korelasyon matrisini hesapla
        corr_matrix = self.df[self.numerical_cols].corr(method=method)
        
        # Güçlü korelasyonları bul
        print(f"\nGüçlü Korelasyonlar (|r| > {threshold}):")
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_correlations.append({
                        'Degisken_1': corr_matrix.columns[i],
                        'Degisken_2': corr_matrix.columns[j],
                        'Korelasyon': corr_value
                    })
        
        if strong_correlations:
            strong_df = pd.DataFrame(strong_correlations).sort_values('Korelasyon', 
                                                                       key=abs, 
                                                                       ascending=False)
            print(strong_df.to_string(index=False))
            
            print("\nYorum:")
            for _, row in strong_df.iterrows():
                corr_val = row['Korelasyon']
                if abs(corr_val) > 0.9:
                    print(f"  UYARI: '{row['Degisken_1']}' ve '{row['Degisken_2']}' "
                          f"çok yüksek korelasyonlu ({corr_val:.3f})")
                    print(f"         Çoklu bağlantı (multicollinearity) riski var!")
                elif abs(corr_val) > 0.7:
                    print(f"  DİKKAT: '{row['Degisken_1']}' ve '{row['Degisken_2']}' "
                          f"güçlü korelasyonlu ({corr_val:.3f})")
        else:
            print(f"  {threshold} eşik değerinin üzerinde korelasyon bulunamadı.")
        
        # Görselleştirme 1: Heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": 0.8})
        plt.title(f'Korelasyon Matrisi ({method.capitalize()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Görselleştirme 2: Korelasyon çubuk grafiği
        if len(self.numerical_cols) > 1:
            # İlk değişkenle diğerleri arasındaki korelasyonlar
            target_col = self.numerical_cols[0]
            correlations = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            colors = ['green' if x > 0 else 'red' for x in correlations.values]
            plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
            plt.yticks(range(len(correlations)), correlations.index)
            plt.xlabel('Korelasyon Katsayısı')
            plt.title(f'"{target_col}" ile Diğer Değişkenler Arasındaki Korelasyon', 
                     fontsize=12, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.show()
        
        return corr_matrix
    
    def pairplot_analysis(self, target_col=None, sample_size=None):
        """
        Değişkenler arası ikili ilişkileri görselleştirir.
        
        Pairplot Nedir:
            Her sayısal değişken çiftinin scatter plot'unu ve
            her değişkenin histogramını aynı grafikte gösterir.
            Çok değişkenli ilişkileri hızlıca görmek için idealdir.
        
        Pairplot Kullanım Alanları:
            1. Değişkenler arası ilişki kalıplarını görmek
            2. Doğrusal olmayan ilişkileri tespit etmek
            3. Gruplar arası farklılıkları görmek (target_col ile)
            4. Çoklu değişkenli aykırı değerleri görmek
        
        Parameters:
            target_col (str, optional): Renklendirme için hedef değişken
                Örnek: target_col='species' -> Her tür farklı renkle gösterilir
            sample_size (int, optional): Örneklem boyutu
                Büyük veri setlerinde performans için kullanılır
                Örnek: sample_size=1000 -> Rastgele 1000 satır seçilir
        
        Not:
            Performans nedeniyle en fazla 5 sayısal değişken gösterilir.
            Çok fazla değişken olması durumunda grafik karmaşık olur.
        
        Örnek Kullanım:
            >>> correlation = CorrelationAnalysis(df)
            >>> # Hedef değişkene göre renklendir
            >>> correlation.pairplot_analysis(target_col='species', sample_size=500)
        """
        print(f"\n{'='*80}")
        print("PAIRPLOT ANALİZİ (Değişkenler Arası İkili İlişkiler)")
        print(f"{'='*80}")
        
        print("\nPairplot Nedir:")
        print("  Tüm sayısal değişken çiftlerinin scatter plot'unu ve")
        print("  her değişkenin dağılımını (histogram/kde) aynı anda gösterir.")
        print("\nFaydaları:")
        print("  - Değişkenler arası ilişki kalıplarını hızlıca görme")
        print("  - Doğrusal ve doğrusal olmayan ilişkileri tespit etme")
        print("  - Gruplar arası farklılıkları görselleştirme")
        print("  - Çoklu değişkenli aykırı değerleri tespit etme")
        
        # Veri boyutunu kontrol et
        df_sample = self.df.copy()
        if sample_size and len(df_sample) > sample_size:
            df_sample = df_sample.sample(sample_size, random_state=42)
            print(f"\nBİLGİ: Veri seti büyük olduğu için {sample_size} örneklem alındı.")
        
        # En fazla 5 sayısal değişken seç (performans için)
        cols_to_plot = self.numerical_cols[:5]
        
        if len(self.numerical_cols) > 5:
            print(f"BİLGİ: Performans için ilk 5 sayısal değişken gösteriliyor.")
        
        if target_col and target_col in self.df.columns:
            print(f"\nHedef değişken: {target_col}")
            print("Her grup farklı renkle gösterilecek.")
            sns.pairplot(df_sample[cols_to_plot + [target_col]], hue=target_col, 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
        else:
            sns.pairplot(df_sample[cols_to_plot], diag_kind='kde', 
                        plot_kws={'alpha': 0.6})
        
        plt.suptitle('Değişkenler Arası İkili İlişkiler', y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def categorical_numerical_relationship(self, categorical_col, numerical_col):
        """
        Kategorik ve sayısal değişken arasındaki ilişkiyi analiz eder.
        
        Bu Analiz Ne Zaman Kullanılır:
            Kategorik bir değişkenin, sayısal bir değişkeni nasıl etkilediğini
            anlamak istediğinizde kullanılır.
            
        Örnek Sorular:
            - Cinsiyet (kategorik) maaşı (sayısal) etkiliyor mu?
            - Eğitim seviyesi (kategorik) geliri (sayısal) etkiliyor mu?
            - Şehir (kategorik) ev fiyatlarını (sayısal) etkiliyor mu?
        
        ANOVA (Analysis of Variance) Testi:
            Üç veya daha fazla grubun ortalamalarının farklı olup olmadığını test eder.
            
            H0 (Null Hipotezi): Tüm grupların ortalaması eşittir
            H1 (Alternatif Hipotez): En az bir grubun ortalaması farklıdır
            
            p-değeri < 0.05: H0 reddedilir, gruplar arası fark VAR
            p-değeri >= 0.05: H0 reddedilemez, gruplar arası fark YOK
        
        Parameters:
            categorical_col (str): Kategorik değişken adı
                Örnek: 'cinsiyet', 'eğitim_seviyesi', 'şehir'
            numerical_col (str): Sayısal değişken adı
                Örnek: 'maaş', 'yaş', 'fiyat'
        
        Çıktılar:
            1. Her grup için istatistikler (ortalama, medyan, std sapma)
            2. ANOVA test sonucu (F-istatistiği, p-değeri)
            3. Görselleştirmeler:
               - Boxplot: Dağılım ve aykırı değerler
               - Violin plot: Yoğunluk ve dağılım
               - Bar plot: Grup ortalamaları
        
        Örnek Kullanım:
            >>> correlation = CorrelationAnalysis(df)
            >>> correlation.categorical_numerical_relationship('education', 'salary')
        """
        print(f"\n{'='*80}")
        print(f"KATEGORİK-SAYISAL İLİŞKİ ANALİZİ")
        print(f"Kategorik: {categorical_col} | Sayısal: {numerical_col}")
        print(f"{'='*80}")
        
        print("\nAmaç: Kategorik değişkenin grupları, sayısal değişkenin")
        print("      ortalamalarını istatistiksel olarak farklılaştırıyor mu?")
        
        # Gruplara göre istatistikler
        group_stats = self.df.groupby(categorical_col)[numerical_col].agg([
            ('Ortalama', 'mean'),
            ('Medyan', 'median'),
            ('Std_Sapma', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Sayi', 'count')
        ]).round(2)
        
        print(f"\nGruplara Göre İstatistikler:")
        print(group_stats)
        
        # ANOVA testi (gruplar arası fark testi)
        groups = [group[numerical_col].dropna() for name, group in self.df.groupby(categorical_col)]
        f_statistic, p_value = stats.f_oneway(*groups)
        
        print(f"\nANOVA Testi (Gruplar arası fark):")
        print(f"  F-istatistiği: {f_statistic:.4f}")
        print(f"  p-değeri: {p_value:.4f}")
        print(f"\nHipotez Testi:")
        print(f"  H0: Tüm grupların ortalaması eşittir")
        print(f"  H1: En az bir grubun ortalaması farklıdır")
        print(f"\nSonuç:")
        if p_value < 0.05:
            print(f"  SONUÇ: Gruplar arasında istatistiksel olarak anlamlı fark VAR (p < 0.05)")
            print(f"  Yorum: '{categorical_col}' değişkeni '{numerical_col}' değişkenini etkiliyor.")
        else:
            print(f"  SONUÇ: Gruplar arasında istatistiksel olarak anlamlı fark YOK (p >= 0.05)")
            print(f"  Yorum: '{categorical_col}' değişkeni '{numerical_col}' değişkenini etkilemiyor olabilir.")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Boxplot
        self.df.boxplot(column=numerical_col, by=categorical_col, ax=axes[0])
        axes[0].set_title(f'Boxplot: {numerical_col} by {categorical_col}')
        axes[0].set_xlabel(categorical_col)
        axes[0].set_ylabel(numerical_col)
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        # Violin plot
        sns.violinplot(data=self.df, x=categorical_col, y=numerical_col, ax=axes[1])
        axes[1].set_title(f'Violin Plot: {numerical_col} by {categorical_col}')
        axes[1].set_xlabel(categorical_col)
        axes[1].set_ylabel(numerical_col)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Bar plot (ortalamalar)
        group_mean = self.df.groupby(categorical_col)[numerical_col].mean().sort_values(ascending=False)
        axes[2].barh(range(len(group_mean)), group_mean.values, color='skyblue', edgecolor='black')
        axes[2].set_yticks(range(len(group_mean)))
        axes[2].set_yticklabels(group_mean.index)
        axes[2].set_xlabel(f'Ortalama {numerical_col}')
        axes[2].set_title(f'Gruplara Göre Ortalama {numerical_col}')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# 2. AYKIRI DEĞER (OUTLIER) TESPİTİ VE İŞLENMESİ
# =============================================================================

class OutlierAnalysis:
    """
    Aykırı değerleri tespit eden ve işleyen sınıf.
    
    Aykırı Değer (Outlier) Nedir:
        Veri setindeki diğer gözlemlerden önemli ölçüde farklı olan değerlerdir.
        Genel eğilimin dışına çıkan, olağandışı gözlemlerdir.
    
    Aykırı Değer Nedenleri:
        1. Veri Girişi Hatası: Yanlış veri girişi (örn: yaş = 200)
        2. Ölçüm Hatası: Ölçüm cihazının arızası
        3. Gerçek Aykırılık: Nadir ama gerçek durumlar (örn: Bill Gates'in geliri)
        4. Doğal Varyasyon: Populasyonun doğal değişkenliği
    
    Aykırı Değer Neden Önemli:
        - Ortalama ve standart sapmayı çarpıtır
        - Korelasyonları yanıltır
        - Model performansını düşürür
        - İstatistiksel testleri etkiler
    
    Tespit Yöntemleri:
        1. IQR (Interquartile Range): Çeyreklikler arası aralık kullanır
        2. Z-Score: Standart sapma kullanır
        3. Isolation Forest: Makine öğrenmesi tabanlı
        4. DBSCAN: Kümeleme tabanlı
    
    İşleme Stratejileri:
        1. Silme: Aykırı değeri veri setinden çıkar
        2. Doldurma: Ortalama, medyan veya mod ile değiştir
        3. Baskılama: Eşik değerlere çek (winsorization)
        4. Dönüştürme: Log, sqrt gibi dönüşümler uygula
        5. Ayrı Modelleme: Aykırı değerler için ayrı model
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        numerical_cols (list): Sayısal sütun isimleri
        
    Methods:
        iqr_method(): IQR yöntemiyle aykırı değer tespiti
        z_score_method(): Z-Score yöntemiyle aykırı değer tespiti
        handle_outliers(): Aykırı değerleri işleme
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def iqr_method(self, column_name, k=1.5):
        """
        IQR (Interquartile Range) yöntemiyle aykırı değer tespiti.
        
        IQR Yöntemi Nasıl Çalışır:
            1. Q1 (25. yüzdelik) ve Q3 (75. yüzdelik) hesaplanır
            2. IQR = Q3 - Q1 hesaplanır
            3. Alt sınır = Q1 - k × IQR
            4. Üst sınır = Q3 + k × IQR
            5. Bu sınırların dışındaki değerler aykırı kabul edilir
        
        k Parametresi:
            - k = 1.5: Standart yaklaşım (Tukey'in kuralı)
              Orta derecede aykırı değerler için
            - k = 3.0: Aşırı aykırı değerler için
              Sadece çok uç değerleri yakalar
        
        Avantajları:
            - Medyan tabanlı, ortalamadan etkilenmez
            - Dağılım varsayımı gerektirmez
            - Yorumlaması kolay (boxplot ile görselleştirilebilir)
        
        Dezavantajları:
            - Tek boyutlu (sadece bir değişkene bakar)
            - Küçük veri setlerinde yanıltıcı olabilir
        
        Parameters:
            column_name (str): Analiz edilecek sütun
            k (float): IQR çarpanı
                k=1.5: Standart (önerilen)
                k=3.0: Sadece aşırı aykırı değerler
        
        Returns:
            dict: Aykırı değer bilgileri
                - alt_sinir: Alt sınır değeri
                - ust_sinir: Üst sınır değeri
                - aykiri_alt: Alt sınırın altındaki değerler
                - aykiri_ust: Üst sınırın üstündeki değerler
                - toplam_aykiri: Toplam aykırı değer sayısı
        
        Örnek Kullanım:
            >>> outlier = OutlierAnalysis(df)
            >>> result = outlier.iqr_method('price', k=1.5)
            >>> print(f"Toplam aykırı: {result['toplam_aykiri']}")
        """
        print(f"\n{'='*80}")
        print(f"IQR YÖNTEMİ İLE AYKIRI DEĞER TESPİTİ: {column_name}")
        print(f"{'='*80}")
        
        print(f"\nYöntem: IQR (Interquartile Range - Çeyreklikler Arası Aralık)")
        print(f"k parametresi: {k}")
        if k == 1.5:
            print("  (Standart yaklaşım - Tukey'in kuralı)")
        elif k == 3.0:
            print("  (Aşırı aykırı değerler için)")
        
        data = self.df[column_name].dropna()
        
        # IQR hesaplama
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        alt_sinir = Q1 - k * IQR
        ust_sinir = Q3 + k * IQR
        
        # Aykırı değerleri bul
        outliers_lower = data[data < alt_sinir]
        outliers_upper = data[data > ust_sinir]
        total_outliers = len(outliers_lower) + len(outliers_upper)
        
        print(f"\nIQR İstatistikleri:")
        print(f"  Q1 (25. yüzdelik): {Q1:.2f}")
        print(f"  Q3 (75. yüzdelik): {Q3:.2f}")
        print(f"  IQR (Q3 - Q1): {IQR:.2f}")
        print(f"  Alt Sınır (Q1 - {k} × IQR): {alt_sinir:.2f}")
        print(f"  Üst Sınır (Q3 + {k} × IQR): {ust_sinir:.2f}")
        
        print(f"\nAykırı Değer Tespit Sonuçları:")
        print(f"  Alt sınırın altında: {len(outliers_lower)} adet ({len(outliers_lower)/len(data)*100:.2f}%)")
        print(f"  Üst sınırın üstünde: {len(outliers_upper)} adet ({len(outliers_upper)/len(data)*100:.2f}%)")
        print(f"  Toplam aykırı değer: {total_outliers} adet ({total_outliers/len(data)*100:.2f}%)")
        
        print(f"\nDeğerlendirme:")
        if total_outliers == 0:
            print("  SONUÇ: Aykırı değer tespit edilmedi.")
        elif total_outliers/len(data) < 0.01:
            print("  SONUÇ: Çok az aykırı değer var (%1'den az). Kabul edilebilir seviye.")
        elif total_outliers/len(data) < 0.05:
            print("  SONUÇ: Az sayıda aykırı değer var (%1-5 arası). İncelenmeli.")
        else:
            print("  UYARI: Yüksek oranda aykırı değer var (%5'ten fazla). Dikkat gerekli!")
            print("        Veri toplama sürecini veya k parametresini gözden geçirin.")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        axes[0].boxplot(data)
        axes[0].axhline(y=alt_sinir, color='r', linestyle='--', 
                       label=f'Alt Sınır: {alt_sinir:.2f}')
        axes[0].axhline(y=ust_sinir, color='r', linestyle='--', 
                       label=f'Üst Sınır: {ust_sinir:.2f}')
        axes[0].set_ylabel(column_name)
        axes[0].set_title(f'Boxplot - {column_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=alt_sinir, color='r', linestyle='--', 
                       label=f'Alt Sınır: {alt_sinir:.2f}')
        axes[1].axvline(x=ust_sinir, color='r', linestyle='--', 
                       label=f'Üst Sınır: {ust_sinir:.2f}')
        axes[1].set_xlabel(column_name)
        axes[1].set_ylabel('Frekans')
        axes[1].set_title(f'Histogram - {column_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'alt_sinir': alt_sinir,
            'ust_sinir': ust_sinir,
            'aykiri_alt': outliers_lower,
            'aykiri_ust': outliers_upper,
            'toplam_aykiri': total_outliers
        }
    
    def z_score_method(self, column_name, threshold=3):
        """
        Z-Score yöntemiyle aykırı değer tespiti.
        
        Z-Score Nedir:
            Bir değerin ortalamadan kaç standart sapma uzakta olduğunu gösterir.
            
            Z = (X - μ) / σ
            
            X: Gözlem değeri
            μ: Ortalama
            σ: Standart sapma
        
        Z-Score Yorumlama:
            Z = 0: Değer ortalamaya eşit
            Z = 1: Değer ortalamadan 1 standart sapma yukarıda
            Z = -1: Değer ortalamadan 1 standart sapma aşağıda
            |Z| > 3: Genellikle aykırı değer kabul edilir
        
        threshold Parametresi:
            - threshold = 2: Gevşek yaklaşım (daha çok aykırı değer)
            - threshold = 3: Standart yaklaşım (önerilen)
            - threshold = 4: Katı yaklaşım (sadece çok uç değerler)
        
        Avantajları:
            - Matematiksel olarak sağlam
            - Yorumlaması kolay
            - Standart dağılım varsayımı altında güvenilir
        
        Dezavantajları:
            - Normal dağılım varsayımı gerektirir
            - Ortalama ve std sapma aykırı değerlerden etkilenir
            - Küçük veri setlerinde güvenilir değil
        
        Ne Zaman Kullanılır:
            - Veri normal dağılımlıysa
            - Veri seti büyükse (n > 30)
            - Tek boyutlu aykırı değer tespitinde
        
        Parameters:
            column_name (str): Analiz edilecek sütun
            threshold (float): Z-score eşik değeri (genellikle 3)
        
        Returns:
            dict: Aykırı değer bilgileri
        
        Örnek Kullanım:
            >>> outlier = OutlierAnalysis(df)
            >>> result = outlier.z_score_method('age', threshold=3)
        """
        print(f"\n{'='*80}")
        print(f"Z-SCORE YÖNTEMİ İLE AYKIRI DEĞER TESPİTİ: {column_name}")
        print(f"{'='*80}")
        
        print(f"\nYöntem: Z-Score (Standardize Edilmiş Değer)")
        print(f"Eşik değer: ±{threshold}")
        print(f"Formül: Z = (X - μ) / σ")
        print(f"  X: Gözlem değeri")
        print(f"  μ: Ortalama")
        print(f"  σ: Standart sapma")
        
        data = self.df[column_name].dropna()
        
        # Z-score hesaplama
        z_scores = np.abs(stats.zscore(data))
        outlier_indices = np.where(z_scores > threshold)[0]
        outlier_values = data.iloc[outlier_indices]
        
        print(f"\nZ-Score İstatistikleri:")
        print(f"  Ortalama (μ): {data.mean():.2f}")
        print(f"  Standart Sapma (σ): {data.std():.2f}")
        print(f"  Eşik Değer: ±{threshold}")
        print(f"  Yorum: |Z| > {threshold} olan değerler aykırı kabul edilir")
        
        print(f"\nAykırı Değer Tespit Sonuçları:")
        print(f"  Toplam aykırı değer: {len(outlier_values)} adet ({len(outlier_values)/len(data)*100:.2f}%)")
        if len(outlier_values) > 0:
            print(f"  Min aykırı değer: {outlier_values.min():.2f}")
            print(f"  Max aykırı değer: {outlier_values.max():.2f}")
            print(f"  Aykırı değerlerin ortalaması: {outlier_values.mean():.2f}")
        
        print(f"\nDeğerlendirme:")
        if len(outlier_values) == 0:
            print("  SONUÇ: Z-score yöntemiyle aykırı değer tespit edilmedi.")
        elif len(outlier_values)/len(data) < 0.01:
            print("  SONUÇ: Çok az aykırı değer (%1'den az).")
        else:
            print("  DİKKAT: Aykırı değer oranı yüksek. Veriyi ve eşik değeri gözden geçirin.")
        
        return {
            'z_scores': z_scores,
            'aykiri_indeksler': outlier_indices,
            'aykiri_degerler': outlier_values
        }
    
    def handle_outliers(self, column_name, method='iqr', action='remove', k=1.5):
        """
        Aykırı değerleri işler (silme, değiştirme veya baskılama).
        
        Aykırı Değer İşleme Stratejileri:
        
        1. REMOVE (Silme):
           Ne zaman: Veri girişi hatası veya ölçüm hatası olduğunda
           Avantaj: Temiz veri seti
           Dezavantaj: Veri kaybı, örneklem boyutu azalır
           
        2. MEAN (Ortalama ile Doldurma):
           Ne zaman: Aykırı değerler az sayıdaysa
           Avantaj: Veri kaybı yok
           Dezavantaj: Dağılımı değiştirir, varyansı azaltır
           
        3. MEDIAN (Medyan ile Doldurma):
           Ne zaman: Dağılım çarpıksa
           Avantaj: Ortalamadan daha robust
           Dezavantaj: Yine dağılımı etkiler
           
        4. SUPPRESS (Baskılama - Winsorization):
           Ne zaman: Gerçek ama uç değerler olduğunda
           Avantaj: Bilgi kaybı minimum, varyans korunur
           Dezavantaj: Değerleri değiştirir
        
        Karar Ağacı:
            - Veri girişi hatası → REMOVE
            - Ölçüm hatası → REMOVE veya MEDIAN
            - Gerçek ama uç değerler → SUPPRESS
            - Az sayıda aykırı değer → MEAN/MEDIAN
            - Çok sayıda aykırı değer → Veri toplama sürecini gözden geçir
        
        Parameters:
            column_name (str): İşlenecek sütun
            method (str): 'iqr' veya 'zscore'
            action (str): İşlem türü
                'remove': Aykırı değerleri sil
                'mean': Ortalama ile doldur
                'median': Medyan ile doldur
                'suppress': Sınırlara baskıla (winsorization)
            k (float): IQR çarpanı (method='iqr' ise)
        
        Returns:
            pd.DataFrame: İşlenmiş veri seti
        
        Örnek Kullanım:
            >>> outlier = OutlierAnalysis(df)
            >>> # Aykırı değerleri baskıla
            >>> df_clean = outlier.handle_outliers('price', method='iqr', action='suppress')
        """
        df_processed = self.df.copy()
        
        print(f"\n{'='*80}")
        print(f"AYKIRI DEĞER İŞLEME")
        print(f"Sütun: {column_name} | Yöntem: {method.upper()} | İşlem: {action.upper()}")
        print(f"{'='*80}")
        
        # Aykırı değerleri tespit et
        if method == 'iqr':
            result = self.iqr_method(column_name, k=k)
            lower_bound = result['alt_sinir']
            upper_bound = result['ust_sinir']
            outlier_mask = (df_processed[column_name] < lower_bound) | (df_processed[column_name] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_processed[column_name].dropna()))
            outlier_mask = z_scores > 3
            lower_bound = df_processed[column_name].mean() - 3 * df_processed[column_name].std()
            upper_bound = df_processed[column_name].mean() + 3 * df_processed[column_name].std()
        
        previous_size = len(df_processed)
        outlier_count = outlier_mask.sum()
        
        print(f"\nİşlem Öncesi Durum:")
        print(f"  Toplam satır: {previous_size}")
        print(f"  Aykırı değer sayısı: {outlier_count} (%{outlier_count/previous_size*100:.2f})")
        
        # İşlemi uygula
        if action == 'remove':
            df_processed = df_processed[~outlier_mask]
            print(f"\nİşlem: SİLME (REMOVE)")
            print(f"  {previous_size - len(df_processed)} satır silindi")
            print(f"  Yeni satır sayısı: {len(df_processed)}")
            print(f"  Veri kaybı: %{((previous_size - len(df_processed))/previous_size*100):.2f}")
        
        elif action == 'mean':
            mean_value = df_processed[column_name].mean()
            df_processed.loc[outlier_mask, column_name] = mean_value
            print(f"\nİşlem: ORTALAMA İLE DOLDURMA (MEAN)")
            print(f"  {outlier_count} aykırı değer ortalama ile değiştirildi")
            print(f"  Kullanılan ortalama: {mean_value:.2f}")
        
        elif action == 'median':
            median_value = df_processed[column_name].median()
            df_processed.loc[outlier_mask, column_name] = median_value
            print(f"\nİşlem: MEDYAN İLE DOLDURMA (MEDIAN)")
            print(f"  {outlier_count} aykırı değer medyan ile değiştirildi")
            print(f"  Kullanılan medyan: {median_value:.2f}")
        
        elif action == 'suppress':
            df_processed.loc[df_processed[column_name] < lower_bound, column_name] = lower_bound
            df_processed.loc[df_processed[column_name] > upper_bound, column_name] = upper_bound
            print(f"\nİşlem: BASKILAMA (SUPPRESS/WINSORIZATION)")
            print(f"  {outlier_count} aykırı değer sınırlara çekildi")
            print(f"  Alt sınır: {lower_bound:.2f}")
            print(f"  Üst sınır: {upper_bound:.2f}")
            print(f"  Açıklama: Alt sınırın altındaki değerler alt sınıra,")
            print(f"            üst sınırın üstündeki değerler üst sınıra eşitlendi.")
        
        print(f"\nÖneri:")
        if action == 'remove' and outlier_count/previous_size > 0.1:
            print("  UYARI: %10'dan fazla veri silindi. Bu çok fazla olabilir.")
            print("  Alternatif: 'suppress' veya 'median' yöntemini deneyin.")
        elif action in ['mean', 'median']:
            print("  BİLGİ: Dağılım değişti. Model performansına etkisini izleyin.")
        elif action == 'suppress':
            print("  İYİ: Baskılama yöntemi veri kaybını minimuma indirir.")
        
        return df_processed


# =============================================================================
# 3. NORMALLİK TESTLERİ VE DAĞILIM ANALİZİ
# =============================================================================

class NormalityTests:
    """
    Dağılım normalliğini test eden sınıf.
    
    Normal Dağılım Nedir:
        Doğada ve sosyal bilimlerde en yaygın görülen dağılım türüdür.
        Çan şeklinde, simetrik bir dağılımdır.
        
        Özellikleri:
        - Ortalama = Medyan = Mod
        - Simetrik (çarpıklık = 0)
        - %68 veri μ ± 1σ aralığında
        - %95 veri μ ± 2σ aralığında
        - %99.7 veri μ ± 3σ aralığında
    
    Normal Dağılım Neden Önemli:
        Birçok istatistiksel test normal dağılım varsayar:
        - t-testi
        - ANOVA
        - Doğrusal regresyon
        - Pearson korelasyon
        
        Normal değilse:
        - Parametrik olmayan testler kullanılmalı
        - Veri dönüşümü yapılmalı (log, sqrt vb.)
        - Robust yöntemler tercih edilmeli
    
    Normallik Test Yöntemleri:
        1. Shapiro-Wilk: En güçlü test, küçük örneklemler için (n < 5000)
        2. Anderson-Darling: Kuyruk bölgelerine duyarlı
        3. Kolmogorov-Smirnov: Büyük örneklemler için
        4. Görsel Yöntemler: Q-Q plot, histogram
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        numerical_cols (list): Sayısal sütun isimleri
        
    Methods:
        shapiro_wilk_test(): Shapiro-Wilk normallik testi
        anderson_darling_test(): Anderson-Darling normallik testi
        normality_visualization(): Normallik için görsel analizler
        test_all_variables(): Tüm değişkenler için normallik testi
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def shapiro_wilk_test(self, column_name):
        """
        Shapiro-Wilk normallik testi.
        
        Shapiro-Wilk Testi:
            En güçlü normallik testidir. Küçük ve orta boyutlu örneklemler için önerilir.
            
            H0 (Null Hipotezi): Veri normal dağılımlıdır
            H1 (Alternatif Hipotez): Veri normal dağılımlı değildir
            
            p-değeri > 0.05: H0 reddedilmez, veri normal dağılımlı
            p-değeri <= 0.05: H0 reddedilir, veri normal dağılımlı değil
        
        Test İstatistiği (W):
            0 ile 1 arasında değer alır
            1'e yakın: Normal dağılıma yakın
            0'a yakın: Normal dağılımdan uzak
        
        Uyarılar:
            - Büyük örneklemlerde (n > 5000) her zaman anlamlı çıkabilir
            - Küçük sapmalarda bile p < 0.05 verebilir
            - Görsel kontrolle desteklenmeli (Q-Q plot)
        
        Ne Zaman Kullanılır:
            - Örneklem boyutu n < 5000
            - Normallik varsayımı kritik önemdeyse
            - Parametrik test öncesi kontrol
        
        Parameters:
            column_name (str): Test edilecek sütun
        
        Returns:
            tuple: (test_statistic, p_value)
        
        Örnek Kullanım:
            >>> normality = NormalityTests(df)
            >>> stat, p = normality.shapiro_wilk_test('age')
            >>> if p > 0.05:
            >>>     print("Normal dağılımlı")
        """
        data = self.df[column_name].dropna()
        
        if len(data) > 5000:
            print(f"\nUYARI: Veri boyutu ({len(data)}) çok büyük.")
            print("       Shapiro-Wilk testi için 5000 örneklem alınıyor...")
            data = data.sample(5000, random_state=42)
        
        stat, p_value = shapiro(data)
        
        print(f"\nShapiro-Wilk Normallik Testi - {column_name}")
        print("="*60)
        print(f"\nHipotezler:")
        print(f"  H0: Veri normal dağılımlıdır")
        print(f"  H1: Veri normal dağılımlı değildir")
        print(f"\nTest Sonuçları:")
        print(f"  W İstatistiği: {stat:.4f}")
        print(f"    (1'e yakın = normal dağılıma yakın)")
        print(f"  p-değeri: {p_value:.4f}")
        print(f"\nKarar (α = 0.05):")
        if p_value > 0.05:
            print(f"  SONUÇ: Veri normal dağılıma UYGUN (p = {p_value:.4f} > 0.05)")
            print(f"  Yorum: Parametrik testler kullanılabilir.")
        else:
            print(f"  SONUÇ: Veri normal dağılıma UYGUN DEĞİL (p = {p_value:.4f} <= 0.05)")
            print(f"  Yorum: Parametrik olmayan testler tercih edilmeli")
            print(f"         veya veri dönüşümü (log, sqrt) denenebilir.")
        
        return stat, p_value
    
    def anderson_darling_test(self, column_name):
        """
        Anderson-Darling normallik testi.
        
        Anderson-Darling Testi:
            Dağılımın kuyruk bölgelerine daha duyarlıdır.
            Shapiro-Wilk'e alternatif olarak kullanılır.
            
            Farklı anlamlılık seviyelerinde (15%, 10%, 5%, 2.5%, 1%)
            kritik değerler verir.
        
        Test İstatistiği Yorumlama:
            Test istatistiği < Kritik değer: Normal dağılımlı
            Test istatistiği >= Kritik değer: Normal dağılımlı değil
        
        Avantajları:
            - Kuyruk bölgelerine duyarlı
            - Farklı anlamlılık seviyeleri
            - Büyük örneklemler için uygun
        
        Ne Zaman Kullanılır:
            - Kuyruk bölgeleri önemliyse (örn: risk analizi)
            - Shapiro-Wilk'e alternatif olarak
            - Büyük örneklemlerde
        
        Parameters:
            column_name (str): Test edilecek sütun
        
        Returns:
            anderson_result: Test sonucu nesnesi
        
        Örnek Kullanım:
            >>> normality = NormalityTests(df)
            >>> result = normality.anderson_darling_test('price')
        """
        data = self.df[column_name].dropna()
        
        result = anderson(data)
        
        print(f"\nAnderson-Darling Normallik Testi - {column_name}")
        print("="*60)
        print(f"\nTest İstatistiği: {result.statistic:.4f}")
        print(f"\nKritik Değerler ve Anlamlılık Seviyeleri:")
        print(f"{'Anlamlılık':<15} {'Kritik Değer':<15} {'Sonuç'}")
        print("-" * 60)
        
        for i, (significance, critical_value) in enumerate(zip(result.significance_level, 
                                                               result.critical_values)):
            if result.statistic < critical_value:
                decision = "Normal (H0 kabul)"
            else:
                decision = "Normal DEĞİL (H0 red)"
            
            print(f"%{significance:<14} {critical_value:<15.3f} {decision}")
        
        print(f"\nGenel Değerlendirme:")
        # %5 seviyesine bak
        critical_5 = result.critical_values[2]  # %5 seviyesi 3. sırada
        if result.statistic < critical_5:
            print(f"  SONUÇ: %5 anlamlılık seviyesinde veri normal dağılımlı")
        else:
            print(f"  SONUÇ: %5 anlamlılık seviyesinde veri normal dağılımlı DEĞİL")
        
        return result
    
    def normality_visualization(self, column_name):
        """
        Normallik için görsel analizler.
        
        Görsel Normallik Kontrolleri:
            1. Histogram + Normal Dağılım Eğrisi:
               - Verinin dağılımını gösterir
               - Normal dağılım eğrisiyle karşılaştırır
               - Çarpıklık ve basıklık gözle değerlendirilir
            
            2. Q-Q Plot (Quantile-Quantile Plot):
               - En güvenilir görsel yöntem
               - Noktalar 45 derece çizgi üzerindeyse normal dağılımlı
               - Eğri gösterirse çarpık dağılım
               - S-şekliyse farklı varyans
            
            3. Boxplot:
               - Simetri kontrolü
               - Aykırı değerleri gösterir
               - Çeyreklikler arası mesafe
            
            4. Kernel Density Estimate (KDE):
               - Sürekli dağılım tahmini
               - Histogram'a alternatif
               - Çan şekli kontrolü
        
        Q-Q Plot Yorumlama:
            - Noktalar düz çizgi üzerinde: Normal dağılım
            - Sol üst köşe yukarı sapma: Sağa çarpık
            - Sağ alt köşe aşağı sapma: Sola çarpık
            - Her iki uç sapma: Ağır kuyruklu dağılım
            - S-şekli: Farklı varyans
        
        Parameters:
            column_name (str): Görselleştirilecek sütun
        
        Örnek Kullanım:
            >>> normality = NormalityTests(df)
            >>> normality.normality_visualization('age')
            >>> # 4 grafik birden gösterilir
        """
        data = self.df[column_name].dropna()
        
        print(f"\nNormallik Görsel Analizi: {column_name}")
        print("="*60)
        print("\nGörsel Kontrol Listesi:")
        print("  1. Histogram: Çan şekline benziyor mu?")
        print("  2. Q-Q Plot: Noktalar düz çizgi üzerinde mi?")
        print("  3. Boxplot: Simetrik mi?")
        print("  4. KDE: Tek tepeli ve simetrik mi?")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram + Normal dağılım eğrisi
        axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                       label='Normal Dağılım')
        axes[0, 0].set_xlabel(column_name)
        axes[0, 0].set_ylabel('Yoğunluk')
        axes[0, 0].set_title(f'Histogram + Normal Dağılım Eğrisi')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normallik Kontrolü)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Boxplot
        axes[1, 0].boxplot(data, vert=True)
        axes[1, 0].set_ylabel(column_name)
        axes[1, 0].set_title('Boxplot (Simetri Kontrolü)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Kernel Density Estimate
        data.plot(kind='density', ax=axes[1, 1], linewidth=2, color='blue')
        axes[1, 1].set_xlabel(column_name)
        axes[1, 1].set_ylabel('Yoğunluk')
        axes[1, 1].set_title('Kernel Density Estimate (KDE)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Normallik Analizi: {column_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def test_all_variables(self):
        """
        Tüm sayısal değişkenler için normallik testi.
        
        Bu metod, veri setindeki tüm sayısal değişkenleri test eder ve
        sonuçları özet bir tabloda gösterir.
        
        Çıktı Tablosu:
            - Değişken: Sütun adı
            - Shapiro W: Shapiro-Wilk test istatistiği
            - p-değeri: Test p-değeri
            - Normal Dağılım: Evet/Hayır
            - Çarpıklık: Dağılımın çarpıklığı
            - Basıklık: Dağılımın basıklığı
        
        Kullanım Senaryosu:
            Veri setindeki hangi değişkenlerin normal dağılımlı olduğunu
            hızlıca görmek için kullanılır. Böylece hangi değişkenlerde
            parametrik, hangi değişkenlerde parametrik olmayan testler
            kullanılacağı belirlenir.
        
        Returns:
            pd.DataFrame: Normallik test sonuçları
        
        Örnek Kullanım:
            >>> normality = NormalityTests(df)
            >>> results = normality.test_all_variables()
            >>> # Normal olmayan değişkenler
            >>> non_normal = results[results['Normal_Dagilim'] == 'Hayir']
            >>> print(non_normal['Degisken'].tolist())
        """
        print(f"\n{'='*80}")
        print("TÜM DEĞİŞKENLER İÇİN NORMALLİK TESTİ")
        print(f"{'='*80}")
        
        print(f"\nTest Edilen Değişken Sayısı: {len(self.numerical_cols)}")
        print(f"Kullanılan Test: Shapiro-Wilk")
        print(f"Anlamlılık Seviyesi: α = 0.05")
        
        results = []
        
        for col in self.numerical_cols:
            data = self.df[col].dropna()
            
            if len(data) > 5000:
                data_sample = data.sample(5000, random_state=42)
            else:
                data_sample = data
            
            # Shapiro-Wilk testi
            stat, p_value = shapiro(data_sample)
            
            results.append({
                'Degisken': col,
                'Shapiro_W': stat,
                'p_degeri': p_value,
                'Normal_Dagilim': 'Evet' if p_value > 0.05 else 'Hayir',
                'Carpiklik': data.skew(),
                'Basiklik': data.kurtosis()
            })
        
        result_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("TEST SONUÇLARI")
        print("="*80)
        print(result_df.to_string(index=False))
        
        # Özet istatistikler
        normal_count = (result_df['Normal_Dagilim'] == 'Evet').sum()
        non_normal_count = (result_df['Normal_Dagilim'] == 'Hayir').sum()
        
        print("\n" + "="*80)
        print("ÖZET İSTATİSTİKLER")
        print("="*80)
        print(f"\nNormal Dağılımlı Değişkenler: {normal_count} ({normal_count/len(result_df)*100:.1f}%)")
        print(f"Normal Olmayan Değişkenler: {non_normal_count} ({non_normal_count/len(result_df)*100:.1f}%)")
        
        if normal_count > 0:
            print(f"\nNormal Dağılımlı Değişkenler:")
            normal_vars = result_df[result_df['Normal_Dagilim'] == 'Evet']['Degisken'].tolist()
            for var in normal_vars:
                print(f"  - {var}")
        
        if non_normal_count > 0:
            print(f"\nNormal Olmayan Değişkenler:")
            non_normal_vars = result_df[result_df['Normal_Dagilim'] == 'Hayir']['Degisken'].tolist()
            for var in non_normal_vars:
                print(f"  - {var}")
            print(f"\nÖneri: Bu değişkenler için:")
            print(f"  1. Parametrik olmayan testler kullanın (Mann-Whitney, Kruskal-Wallis)")
            print(f"  2. Veri dönüşümü deneyin (log, sqrt, box-cox)")
            print(f"  3. Robust istatistiksel yöntemler tercih edin")
        
        return result_df


# =============================================================================
# ÖRNEK KULLANIM
# =============================================================================

if __name__ == "__main__":
    print("""
    ============================================================================
              ORTA SEVİYE - EXPLORATORY DATA ANALYSIS
         Korelasyon, Aykırı Değer ve Normallik Analizleri
    ============================================================================
    """)
    
    # Örnek veri seti yükleme
    print("\nÖrnek veri seti yükleniyor (Diamonds)...\n")
    df = sns.load_dataset('diamonds')
    
    print("="*80)
    print("VERİ SETİ HAKKINDA")
    print("="*80)
    print("Diamonds veri seti, elmas özelliklerini ve fiyatlarını içerir.")
    print("\nSütunlar:")
    print("  - carat: Elmasın ağırlığı")
    print("  - cut: Kesim kalitesi")
    print("  - color: Renk")
    print("  - clarity: Berraklık")
    print("  - depth: Derinlik yüzdesi")
    print("  - table: Tablo genişliği")
    print("  - price: Fiyat (USD)")
    print("  - x, y, z: Boyutlar (mm)")
    print()
    
    # 1. KORELASYON ANALİZİ
    print("="*80)
    print("[1/3] KORELASYON ANALİZİ")
    print("="*80)
    correlation = CorrelationAnalysis(df)
    
    # Pearson korelasyonu
    print("\nPearson Korelasyon Matrisi hesaplanıyor...")
    corr_matrix = correlation.correlation_matrix(method='pearson', threshold=0.7)
    
    # Kategorik-Sayısal ilişki
    print("\nKategorik-Sayısal ilişki analizi yapılıyor...")
    correlation.categorical_numerical_relationship('cut', 'price')
    
    # 2. AYKIRI DEĞER ANALİZİ
    print("\n" + "="*80)
    print("[2/3] AYKIRI DEĞER ANALİZİ")
    print("="*80)
    outlier = OutlierAnalysis(df)
    
    # IQR yöntemi
    print("\nIQR yöntemi ile aykırı değer tespiti...")
    outlier_info = outlier.iqr_method('price', k=1.5)
    
    # Z-score yöntemi
    print("\nZ-score yöntemi ile aykırı değer tespiti...")
    outlier.z_score_method('price', threshold=3)
    
    # Aykırı değerleri işleme
    print("\nAykırı değerler baskılanıyor...")
    df_clean = outlier.handle_outliers('price', method='iqr', action='suppress', k=1.5)
    
    # 3. NORMALLİK TESTLERİ
    print("\n" + "="*80)
    print("[3/3] NORMALLİK TESTLERİ")
    print("="*80)
    normality = NormalityTests(df)
    
    # Tek değişken için detaylı test
    print("\nShapiro-Wilk testi uygulanıyor...")
    normality.shapiro_wilk_test('carat')
    
    print("\nAnderson-Darling testi uygulanıyor...")
    normality.anderson_darling_test('carat')
    
    print("\nNormallik görselleştirmesi yapılıyor...")
    normality.normality_visualization('carat')
    
    # Tüm değişkenler için özet test
    print("\nTüm değişkenler için normallik testi yapılıyor...")
    normality_results = normality.test_all_variables()
    
    print("\n" + "="*80)
    print("ORTA SEVİYE EDA TAMAMLANDI")
    print("="*80)
    print("\nÖğrenilen Konular:")
    print("  - Korelasyon analizi (Pearson, Spearman, Kendall)")
    print("  - Kategorik-sayısal ilişki analizi (ANOVA)")
    print("  - Aykırı değer tespiti (IQR, Z-Score)")
    print("  - Aykırı değer işleme stratejileri")
    print("  - Normallik testleri (Shapiro-Wilk, Anderson-Darling)")
    print("  - Görsel normallik kontrolü (Q-Q plot, histogram)")
    print("\nSonraki Adım:")
    print("  İleri seviye EDA dosyasına geçerek çok değişkenli analizler,")
    print("  boyut indirgeme ve özellik mühendisliği tekniklerini öğrenebilirsiniz.")
