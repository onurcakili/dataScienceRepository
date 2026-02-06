"""
=============================================================================
İLERİ SEVİYE - EXPLORATORY DATA ANALYSIS (EDA) - PART 1
=============================================================================

Bu dosya ileri seviye veri analizi tekniklerini içerir.

KAPSAM (PART 1):
----------------
1. Çok Değişkenli İstatistiksel Analizler
   - Ki-kare bağımsızlık testi (kategorik-kategorik ilişki)
   - Bağımsız örneklem t-testi (iki grup karşılaştırma)
   - Kısmi korelasyon (üçüncü değişken kontrolü)
   - Çok yönlü ilişki analizleri

2. Boyut İndirgeme ve Görselleştirme
   - PCA (Principal Component Analysis)
   - t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - Biplot (değişken vektörleri)
   - Scree plot (açıklanan varyans)

KULLANILAN KÜTÜPHANELER:
------------------------
- pandas, numpy: Veri işleme
- matplotlib, seaborn: Görselleştirme
- scipy, statsmodels: İstatistiksel testler
- sklearn: ML algoritmaları, boyut indirgeme

ÖRNEK KULLANIM:
---------------
    import pandas as pd
    from sklearn.datasets import load_iris
    
    # Veri yükle
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Çok değişkenli analiz
    multivariate = MultivariateAnalysis(df)
    multivariate.chi_square_test('species', 'petal_size_category')
    multivariate.t_test('species', 'sepal_length')
    
    # Boyut indirgeme
    dim_reduction = DimensionReduction(df)
    pca_model, pca_result, loadings = dim_reduction.pca_analysis(n_components=4)
    tsne_model, tsne_result = dim_reduction.tsne_analysis(n_components=2)

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# Sklearn kütüphaneleri
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10


# =============================================================================
# 1. ÇOK DEĞİŞKENLİ İSTATİSTİKSEL ANALİZLER
# =============================================================================

class MultivariateAnalysis:
    """
    Çok değişkenli istatistiksel analizler için sınıf.
    
    Çok Değişkenli Analiz Nedir:
        İkiden fazla değişkenin aynı anda incelendiği analiz türüdür.
        Değişkenler arası karmaşık ilişkileri anlamak için kullanılır.
    
    Bu Sınıfın Kapsadığı Konular:
        1. Ki-Kare Testi: İki kategorik değişken bağımsız mı?
        2. t-Testi: İki grubun ortalaması farklı mı?
        3. Kısmi Korelasyon: Üçüncü değişken kontrol edildiğinde ilişki nasıl?
    
    Kullanım Alanları:
        - A/B testleri
        - Grup karşılaştırmaları
        - Karıştırıcı değişken kontrolü
        - Neden-sonuç ilişkisi araştırması
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        numerical_cols (list): Sayısal sütunlar
        categorical_cols (list): Kategorik sütunlar
        
    Methods:
        chi_square_test(): Ki-kare bağımsızlık testi
        t_test(): Bağımsız örneklem t-testi
        partial_correlation(): Kısmi korelasyon analizi
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def chi_square_test(self, col1, col2):
        """
        İki kategorik değişken arasında bağımsızlık testi (Ki-Kare).
        
        Ki-Kare Testi Nedir:
            İki kategorik değişkenin birbirinden bağımsız olup olmadığını test eder.
            
            Örnek Sorular:
            - Cinsiyet ile ürün tercihi bağımsız mı?
            - Eğitim seviyesi ile gelir kategorisi ilişkili mi?
            - Şehir ile alışveriş sıklığı arasında ilişki var mı?
        
        Çapraz Tablo (Contingency Table):
            İki kategorik değişkenin kesişim frekanslarını gösteren tablodur.
            
            Örnek:
                         Ürün A    Ürün B    Toplam
            Erkek           30        20        50
            Kadın           25        25        50
            Toplam          55        45       100
        
        Hipotezler:
            H0: İki değişken bağımsızdır (ilişki yoktur)
            H1: İki değişken bağımlıdır (ilişki vardır)
            
            p < 0.05: H0 reddedilir, ilişki VAR
            p >= 0.05: H0 reddedilemez, ilişki YOK
        
        Ki-Kare İstatistiği:
            Gözlenen frekanslar ile beklenen frekanslar arasındaki farkı ölçer.
            Büyük ki-kare = Güçlü ilişki
            Küçük ki-kare = Zayıf ilişki
        
        Serbestlik Derecesi (df):
            df = (satır sayısı - 1) × (sütun sayısı - 1)
            
        Varsayımlar:
            - Her hücrede beklenen frekans >= 5 olmalı
            - Gözlemler bağımsız olmalı
            - Kategoriler birbirini dışlamalı
        
        Parameters:
            col1 (str): Birinci kategorik değişken
            col2 (str): İkinci kategorik değişken
        
        Returns:
            tuple: (chi2_statistic, p_value)
        
        Örnek Kullanım:
            >>> multivariate = MultivariateAnalysis(df)
            >>> chi2, p = multivariate.chi_square_test('gender', 'product_choice')
            >>> if p < 0.05:
            >>>     print("Cinsiyet ile ürün tercihi ilişkili")
        """
        print(f"\n{'='*80}")
        print(f"Kİ-KARE BAĞIMSIZLIK TESTİ")
        print(f"Değişken 1: {col1} | Değişken 2: {col2}")
        print(f"{'='*80}")
        
        print(f"\nAmaç: İki kategorik değişkenin birbirinden bağımsız olup")
        print(f"      olmadığını test etmek.")
        
        # Çapraz tablo (Contingency table)
        crosstab = pd.crosstab(self.df[col1], self.df[col2])
        
        print(f"\nÇapraz Tablo (Gözlenen Frekanslar):")
        print(crosstab)
        
        # Ki-kare testi
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        
        print(f"\nBeklenen Frekanslar:")
        expected_df = pd.DataFrame(expected, 
                                   index=crosstab.index, 
                                   columns=crosstab.columns)
        print(expected_df.round(2))
        
        # Varsayım kontrolü
        print(f"\nVarsayım Kontrolü:")
        min_expected = expected.min()
        if min_expected < 5:
            print(f"  UYARI: En küçük beklenen frekans {min_expected:.2f} < 5")
            print(f"         Ki-kare testi güvenilir olmayabilir!")
            print(f"         Çözüm: Kategorileri birleştirin veya Fisher's Exact Test kullanın.")
        else:
            print(f"  İYİ: Tüm hücrelerde beklenen frekans >= 5")
        
        print(f"\nKi-Kare Test Sonuçları:")
        print(f"  Ki-Kare İstatistiği: {chi2:.4f}")
        print(f"  p-değeri: {p_value:.4f}")
        print(f"  Serbestlik Derecesi: {dof}")
        
        print(f"\nHipotez Testi:")
        print(f"  H0: '{col1}' ve '{col2}' bağımsızdır (ilişki yok)")
        print(f"  H1: '{col1}' ve '{col2}' bağımlıdır (ilişki var)")
        
        print(f"\nKarar (α = 0.05):")
        if p_value < 0.05:
            print(f"  SONUÇ: Değişkenler arasında istatistiksel olarak ANLAMLI ilişki VAR")
            print(f"         (p = {p_value:.4f} < 0.05, H0 reddedilir)")
            print(f"  Yorum: '{col1}' değişkeni '{col2}' değişkenini etkiliyor olabilir.")
        else:
            print(f"  SONUÇ: Değişkenler arasında istatistiksel olarak anlamlı ilişki YOK")
            print(f"         (p = {p_value:.4f} >= 0.05, H0 reddedilemez)")
            print(f"  Yorum: '{col1}' ve '{col2}' birbirinden bağımsız görünüyor.")
        
        # Etki büyüklüğü (Cramer's V)
        n = crosstab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
        print(f"\nEtki Büyüklüğü (Cramer's V): {cramers_v:.4f}")
        if cramers_v < 0.1:
            print(f"  Yorum: Zayıf ilişki")
        elif cramers_v < 0.3:
            print(f"  Yorum: Orta güçlükte ilişki")
        else:
            print(f"  Yorum: Güçlü ilişki")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Heatmap
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0], 
                   cbar_kws={'label': 'Frekans'})
        axes[0].set_title(f'Çapraz Tablo Heatmap\n{col1} vs {col2}')
        
        # Stacked bar chart
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        crosstab_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set3')
        axes[1].set_title(f'Yüzdesel Dağılım (Stacked)\n{col1} vs {col2}')
        axes[1].set_ylabel('Yüzde (%)')
        axes[1].legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return chi2, p_value
    
    def t_test(self, categorical_col, numerical_col):
        """
        İki grup arasında ortalama farkı testi (Bağımsız Örneklem t-testi).
        
        Bağımsız Örneklem t-Testi Nedir:
            İki bağımsız grubun ortalamalarının istatistiksel olarak
            farklı olup olmadığını test eder.
            
            Örnek Sorular:
            - Erkekler ve kadınların ortalama geliri farklı mı?
            - İlaç alan ve almayan grupların tansiyon ortalaması farklı mı?
            - İki şubenin ortalama satış performansı farklı mı?
        
        Bağımsız Örneklem Nedir:
            İki grup birbirinden tamamen ayrıdır. Bir gruptaki gözlem,
            diğer gruptaki hiçbir gözlemle ilişkili değildir.
            
            Örnek: Farklı kişilerden oluşan erkek ve kadın grupları
            
        Hipotezler:
            H0: μ1 = μ2 (İki grubun ortalaması eşittir)
            H1: μ1 ≠ μ2 (İki grubun ortalaması farklıdır)
            
            p < 0.05: H0 reddedilir, ortalamalar FARKLI
            p >= 0.05: H0 reddedilemez, ortalamalar AYNI
        
        t-İstatistiği:
            t = (x̄1 - x̄2) / SE
            
            x̄1, x̄2: Grup ortalamaları
            SE: Standart hata
            
            Büyük |t| = Büyük fark
        
        Varsayımlar:
            1. Normal dağılım: Her grup normal dağılımlı olmalı (n>30 ise esnete değişkenilir)
            2. Varyans homojenliği: İki grubun varyansı yakın olmalı
            3. Bağımsızlık: Gözlemler birbirinden bağımsız olmalı
        
        Ne Zaman Kullanılır:
            - Tam olarak 2 grup varsa
            - Sayısal bir değişkeni karşılaştırırken
            - Gruplar bağımsızsa
            - Normal dağılım varsayımı sağlanıyorsa
        
        Alternatifler:
            - 2'den fazla grup: ANOVA
            - Normal olmayan dağılım: Mann-Whitney U testi
            - Bağımlı örneklemler: Paired t-test
        
        Parameters:
            categorical_col (str): Gruplandırma değişkeni (tam 2 kategori olmalı)
            numerical_col (str): Karşılaştırılacak sayısal değişken
        
        Returns:
            tuple: (t_statistic, p_value)
        
        Örnek Kullanım:
            >>> multivariate = MultivariateAnalysis(df)
            >>> t, p = multivariate.t_test('gender', 'salary')
            >>> if p < 0.05:
            >>>     print("Cinsiyetler arası maaş farkı anlamlı")
        """
        print(f"\n{'='*80}")
        print(f"BAĞIMSIZ ÖRNEKLEM T-TESTİ")
        print(f"Grup Değişkeni: {categorical_col} | Sayısal Değişken: {numerical_col}")
        print(f"{'='*80}")
        
        print(f"\nAmaç: İki bağımsız grubun ortalamalarının farklı olup olmadığını test etmek.")
        
        # Grupları ayır
        groups = self.df[categorical_col].unique()
        
        if len(groups) != 2:
            print(f"\nUYARI: t-testi için tam olarak 2 grup gerekli.")
            print(f"       Mevcut grup sayısı: {len(groups)}")
            print(f"       İlk iki grup kullanılacak: {groups[:2]}")
            groups = groups[:2]
        
        group1_data = self.df[self.df[categorical_col] == groups[0]][numerical_col].dropna()
        group2_data = self.df[self.df[categorical_col] == groups[1]][numerical_col].dropna()
        
        # Grup istatistikleri
        print(f"\nGrup İstatistikleri:")
        print(f"\nGrup 1 - {groups[0]}:")
        print(f"  Ortalama: {group1_data.mean():.2f}")
        print(f"  Medyan: {group1_data.median():.2f}")
        print(f"  Std Sapma: {group1_data.std():.2f}")
        print(f"  Min: {group1_data.min():.2f}")
        print(f"  Max: {group1_data.max():.2f}")
        print(f"  Gözlem Sayısı: {len(group1_data)}")
        
        print(f"\nGrup 2 - {groups[1]}:")
        print(f"  Ortalama: {group2_data.mean():.2f}")
        print(f"  Medyan: {group2_data.median():.2f}")
        print(f"  Std Sapma: {group2_data.std():.2f}")
        print(f"  Min: {group2_data.min():.2f}")
        print(f"  Max: {group2_data.max():.2f}")
        print(f"  Gözlem Sayısı: {len(group2_data)}")
        
        mean_diff = group1_data.mean() - group2_data.mean()
        print(f"\nOrtalama Farkı: {mean_diff:.2f}")
        print(f"  ({groups[0]} - {groups[1]})")
        
        # t-testi
        t_statistic, p_value = ttest_ind(group1_data, group2_data)
        
        print(f"\nt-Testi Sonuçları:")
        print(f"  t-istatistiği: {t_statistic:.4f}")
        print(f"  p-değeri: {p_value:.4f}")
        
        # Serbestlik derecesi
        df_value = len(group1_data) + len(group2_data) - 2
        print(f"  Serbestlik Derecesi: {df_value}")
        
        print(f"\nHipotez Testi:")
        print(f"  H0: μ1 = μ2 (Grup ortalamaları eşittir)")
        print(f"  H1: μ1 ≠ μ2 (Grup ortalamaları farklıdır)")
        
        print(f"\nKarar (α = 0.05):")
        if p_value < 0.05:
            print(f"  SONUÇ: Gruplar arasında istatistiksel olarak ANLAMLI fark VAR")
            print(f"         (p = {p_value:.4f} < 0.05, H0 reddedilir)")
            print(f"  Yorum: '{groups[0]}' ve '{groups[1]}' gruplarının '{numerical_col}'")
            print(f"         ortalamaları farklıdır.")
        else:
            print(f"  SONUÇ: Gruplar arasında istatistiksel olarak anlamlı fark YOK")
            print(f"         (p = {p_value:.4f} >= 0.05, H0 reddedilemez)")
            print(f"  Yorum: '{groups[0]}' ve '{groups[1]}' gruplarının '{numerical_col}'")
            print(f"         ortalamaları benzerdir.")
        
        # Cohen's d (etki büyüklüğü)
        pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.std()**2 + 
                              (len(group2_data)-1)*group2_data.std()**2) / df_value)
        cohens_d = mean_diff / pooled_std
        
        print(f"\nEtki Büyüklüğü (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print(f"  Yorum: Küçük etki")
        elif abs(cohens_d) < 0.5:
            print(f"  Yorum: Orta etki")
        elif abs(cohens_d) < 0.8:
            print(f"  Yorum: Büyük etki")
        else:
            print(f"  Yorum: Çok büyük etki")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot karşılaştırma
        data_for_box = pd.DataFrame({
            numerical_col: pd.concat([group1_data, group2_data]),
            categorical_col: [groups[0]]*len(group1_data) + [groups[1]]*len(group2_data)
        })
        sns.boxplot(data=data_for_box, x=categorical_col, y=numerical_col, ax=axes[0])
        axes[0].set_title(f'Boxplot Karşılaştırma\n{numerical_col} by {categorical_col}')
        
        # Violin plot
        sns.violinplot(data=data_for_box, x=categorical_col, y=numerical_col, ax=axes[1])
        axes[1].set_title(f'Violin Plot Karşılaştırma\n{numerical_col} by {categorical_col}')
        
        plt.tight_layout()
        plt.show()
        
        return t_statistic, p_value
    
    def partial_correlation(self, x_col, y_col, control_col):
        """
        Kısmi korelasyon hesaplama (üçüncü değişken kontrol edilerek).
        
        Kısmi Korelasyon Nedir:
            Üçüncü bir değişkenin etkisi kontrol edilerek (sabit tutularak),
            iki değişken arasındaki ilişkiyi ölçer.
            
        Neden Önemli:
            Bazen iki değişken arasındaki ilişki, aslında üçüncü bir
            değişkenden kaynaklanıyor olabilir (sahte korelasyon).
            
        Sahte Korelasyon Örneği:
            Dondurma satışı ile boğulma olayları arasında pozitif korelasyon var.
            Ama bu bir neden-sonuç ilişkisi değil!
            İkisi de sıcaklıktan etkileniyor (karıştırıcı değişken).
            
            Sıcaklık kontrol edildiğinde (kısmi korelasyon), ilişki kaybolur.
        
        Kullanım Alanları:
            - Karıştırıcı değişken tespiti
            - Neden-sonuç ilişkisi araştırması
            - Regresyon analizinde değişken seçimi
            - A/B testlerinde kontrol değişkenleri
        
        Yorumlama:
            Normal Korelasyon vs Kısmi Korelasyon:
            
            1. r = 0.8, r_partial = 0.7:
               İlişki gerçek ama üçüncü değişken de katkıda bulunuyor
            
            2. r = 0.8, r_partial = 0.1:
               İlişki çoğunlukla üçüncü değişkenden kaynaklanıyor (sahte korelasyon)
            
            3. r = 0.3, r_partial = 0.8:
               Üçüncü değişken ilişkiyi gizliyor (suppressor variable)
        
        Parameters:
            x_col (str): Birinci değişken
            y_col (str): İkinci değişken
            control_col (str): Kontrol edilecek değişken
        
        Returns:
            tuple: (partial_correlation, p_value)
        
        Örnek Kullanım:
            >>> multivariate = MultivariateAnalysis(df)
            >>> # Yaş kontrol edilerek gelir-eğitim ilişkisi
            >>> r_partial, p = multivariate.partial_correlation('education', 'income', 'age')
        """
        print(f"\n{'='*80}")
        print(f"KISMI KORELASYON ANALİZİ")
        print(f"X: {x_col}, Y: {y_col} (Kontrol: {control_col})")
        print(f"{'='*80}")
        
        print(f"\nAmaç: '{control_col}' değişkeninin etkisi kontrol edilerek")
        print(f"      '{x_col}' ve '{y_col}' arasındaki ilişkiyi ölçmek.")
        
        # Veriyi hazırla
        data = self.df[[x_col, y_col, control_col]].dropna()
        
        # Normal korelasyon
        normal_corr = data[x_col].corr(data[y_col])
        
        # Kısmi korelasyon için rezidüeller
        from scipy.stats import pearsonr
        
        # X'i kontrol değişkenine regress et
        x_mean = data[x_col].mean()
        control_mean = data[control_col].mean()
        x_control_corr = data[x_col].corr(data[control_col])
        x_residuals = data[x_col] - x_mean - x_control_corr * (data[control_col] - control_mean)
        
        # Y'yi kontrol değişkenine regress et
        y_mean = data[y_col].mean()
        y_control_corr = data[y_col].corr(data[control_col])
        y_residuals = data[y_col] - y_mean - y_control_corr * (data[control_col] - control_mean)
        
        # Rezidüeller arası korelasyon = kısmi korelasyon
        partial_corr, p_value = pearsonr(x_residuals, y_residuals)
        
        print(f"\nKorelasyon Sonuçları:")
        print(f"  Normal Korelasyon ({x_col} - {y_col}): {normal_corr:.4f}")
        print(f"  Kısmi Korelasyon (Kontrol: {control_col}): {partial_corr:.4f}")
        print(f"  p-değeri: {p_value:.4f}")
        print(f"  Mutlak Fark: {abs(normal_corr - partial_corr):.4f}")
        
        print(f"\nYorum:")
        fark = abs(normal_corr - partial_corr)
        if fark > 0.3:
            print(f"  UYARI: '{control_col}' değişkeni ilişkiyi ÖNEMLI ÖLÇÜDE ETKİLİYOR!")
            print(f"         Fark: {fark:.4f}")
            if abs(partial_corr) < 0.2 and abs(normal_corr) > 0.5:
                print(f"         Bu bir SAHTE KORELASYON olabilir.")
                print(f"         '{x_col}' ve '{y_col}' arasındaki ilişki")
                print(f"         aslında '{control_col}' değişkeninden kaynaklanıyor olabilir.")
        elif fark > 0.1:
            print(f"  DİKKAT: '{control_col}' değişkeninin ORTA DÜZEYDE etkisi var.")
            print(f"          Fark: {fark:.4f}")
        else:
            print(f"  BİLGİ: '{control_col}' değişkeninin etkisi SINIRLI.")
            print(f"         Fark: {fark:.4f}")
            print(f"         '{x_col}' ve '{y_col}' arasındaki ilişki gerçek görünüyor.")
        
        # İstatistiksel anlamlılık
        if p_value < 0.05:
            print(f"\n  SONUÇ: Kısmi korelasyon istatistiksel olarak ANLAMLI (p < 0.05)")
        else:
            print(f"\n  SONUÇ: Kısmi korelasyon istatistiksel olarak anlamlı DEĞİL (p >= 0.05)")
        
        return partial_corr, p_value


# =============================================================================
# 2. BOYUT İNDİRGEME VE VİZÜALİZASYON
# =============================================================================

class DimensionReduction:
    """
    Boyut indirgeme teknikleri için sınıf.
    
    Boyut İndirgeme Nedir:
        Çok sayıda değişkeni (özellik) daha az sayıda yeni değişkene
        dönüştürme işlemidir. Bilgi kaybını minimize ederek boyutu azaltır.
    
    Neden Boyut İndirgeme:
        1. Görselleştirme: 3'ten fazla boyutu görselleştirmek zor
        2. Performans: Daha az değişken = Daha hızlı model
        3. Aşırı öğrenme: Çok değişken modeli karmaşıklaştırır
        4. Gürültü azaltma: Önemsiz değişkenler elenir
        5. Multicollinearity: Yüksek korelasyonlu değişkenler birleştirilir
    
    Temel Yöntemler:
        1. PCA (Principal Component Analysis):
           - Doğrusal boyut indirgeme
           - Varyansı maksimize eder
           - Hızlı ve verimli
           - Yorumlaması kolay
        
        2. t-SNE (t-Distributed Stochastic Neighbor Embedding):
           - Non-linear boyut indirgeme
           - Yerel yapıyı korur
           - Görselleştirme için mükemmel
           - Hesaplama maliyeti yüksek
    
    Kullanım Alanları:
        - Yüksek boyutlu veri görselleştirme
        - Özellik mühendisliği
        - Anomali tespiti
        - Kümeleme öncesi ön işleme
        - Transfer learning
    
    Attributes:
        df (pd.DataFrame): Analiz edilecek veri seti
        numerical_cols (list): Sayısal sütunlar
        
    Methods:
        pca_analysis(): PCA ile boyut indirgeme
        tsne_analysis(): t-SNE ile boyut indirgeme
    """
    
    def __init__(self, dataframe):
        """
        Sınıfın başlatıcı metodu.
        
        Parameters:
            dataframe (pd.DataFrame): Analiz edilecek veri seti
        """
        self.df = dataframe.copy()
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def pca_analysis(self, n_components=2, scale=True):
        """
        PCA (Principal Component Analysis) ile boyut indirgeme.
        
        PCA Nasıl Çalışır:
            1. Verileri standardize et (opsiyonel ama önerilen)
            2. Kovaryans matrisini hesapla
            3. Eigenvector ve eigenvalue bul
            4. En büyük eigenvalue'lara karşılık gelen eigenvector'leri seç
            5. Veriyi yeni eksenlere (PC'lere) projeksiyon et
        
        Principal Component (PC) Nedir:
            Orijinal değişkenlerin doğrusal kombinasyonudur.
            
            PC1 = a1×X1 + a2×X2 + a3×X3 + ...
            
            PC1: En fazla varyansı açıklayan bileşen
            PC2: PC1'e dik, kalan varyansın en fazlasını açıklayan
            PC3: PC1 ve PC2'ye dik, ...
        
        Açıklanan Varyans:
            Her PC'nin orijinal veri varyansının ne kadarını açıkladığı.
            
            İdeal: İlk birkaç PC toplam varyansın %80-90'ını açıklamalı
            
            Örnek:
            PC1: %65
            PC2: %20  → İlk 2 PC %85 açıklıyor (iyi)
            PC3: %10
            PC4: %5
        
        Component Loadings:
            Her PC'nin orijinal değişkenlerle ilişkisini gösterir.
            Yüksek loading = O değişken bu PC için önemli
            
            Örnek:
            PC1 = 0.9×boy + 0.3×kilo
            → PC1 çoğunlukla boy'u temsil ediyor
        
        Avantajlar:
            - Matematiksel olarak optimal (varyans maksimizasyonu)
            - Hızlı ve verimli
            - Yorumlaması nispeten kolay
            - Deterministik (her zaman aynı sonuç)
        
        Dezavantajlar:
            - Sadece doğrusal ilişkileri yakalar
            - Outlier'lara duyarlı
            - PC'ler her zaman yorumlanabilir değil
            - Standardizasyon gerektirir
        
        Ne Zaman Kullanılır:
            - Çok sayıda korelasyonlu değişken varsa
            - Doğrusal ilişkiler baskınsa
            - Hız önemliyse
            - Boyut indirgeme için
        
        Parameters:
            n_components (int): Hedef bileşen sayısı
                2-3: Görselleştirme için
                5-10: Model için
                None: Tüm bileşenler
            scale (bool): Verileri standardize et
                True: Önerilen (değişkenler farklı ölçeklerdeyse)
                False: Değişkenler aynı ölçekteyse
        
        Returns:
            tuple: (pca_model, pca_result, loadings)
                pca_model: Eğitilmiş PCA modeli
                pca_result: Dönüştürülmüş veri
                loadings: Bileşen yükleri
        
        Örnek Kullanım:
            >>> dim_reduction = DimensionReduction(df)
            >>> pca, result, loadings = dim_reduction.pca_analysis(n_components=3)
            >>> print(f"İlk 3 PC varyansın %{pca.explained_variance_ratio_.sum()*100:.1f}'ini açıklıyor")
        """
        print(f"\n{'='*80}")
        print(f"PCA (PRINCIPAL COMPONENT ANALYSIS)")
        print(f"{'='*80}")
        
        print(f"\nAmaç: Çok boyutlu veriyi daha az boyutta temsil etmek")
        print(f"      (bilgi kaybını minimize ederek)")
        
        # Veriyi hazırla
        data = self.df[self.numerical_cols].dropna()
        
        print(f"\nOrijinal Veri:")
        print(f"  Gözlem sayısı: {len(data)}")
        print(f"  Özellik sayısı: {len(self.numerical_cols)}")
        
        # Standardizasyon
        if scale:
            print(f"\nStandardizasyon: EVET")
            print(f"  (Her değişken ortalama=0, std=1'e dönüştürülüyor)")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            print(f"\nStandardizasyon: HAYIR")
            data_scaled = data.values
        
        # PCA uygula
        n_components_actual = min(n_components, len(self.numerical_cols))
        pca = PCA(n_components=n_components_actual)
        pca_result = pca.fit_transform(data_scaled)
        
        print(f"\nPCA Sonuçları:")
        print(f"  Hedef bileşen sayısı: {n_components}")
        print(f"  Gerçek bileşen sayısı: {pca.n_components_}")
        
        # Açıklanan varyans
        print(f"\nAçıklanan Varyans Oranları:")
        cumulative_variance = 0
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            cumulative_variance += var_ratio
            print(f"  PC{i+1}: %{var_ratio*100:.2f} (Kümülatif: %{cumulative_variance*100:.2f})")
        
        print(f"\nToplam Açıklanan Varyans: %{sum(pca.explained_variance_ratio_)*100:.2f}")
        
        if cumulative_variance > 0.9:
            print(f"  MÜKEMMEL: İlk {pca.n_components_} bileşen varyansın %90'ından fazlasını açıklıyor!")
        elif cumulative_variance > 0.8:
            print(f"  İYİ: İlk {pca.n_components_} bileşen varyansın %80-90'ını açıklıyor.")
        elif cumulative_variance > 0.7:
            print(f"  ORTA: İlk {pca.n_components_} bileşen varyansın %70-80'ini açıklıyor.")
            print(f"  Öneri: Daha fazla bileşen kullanmayı düşünün.")
        else:
            print(f"  DÜŞÜK: İlk {pca.n_components_} bileşen varyansın %{cumulative_variance*100:.0f}'ini açıklıyor.")
            print(f"  UYARI: Çok fazla bilgi kaybı var! Daha fazla bileşen gerekli.")
        
        # Bileşen yükleri (loadings)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.numerical_cols
        )
        
        print(f"\nBileşen Yükleri (Component Loadings):")
        print(f"  (Her PC'nin orijinal değişkenlerle ilişkisi)")
        print(loadings.iloc[:, :min(2, pca.n_components_)].round(3))
        
        print(f"\nYüklerin Yorumlanması:")
        for i in range(min(2, pca.n_components_)):
            pc_name = f'PC{i+1}'
            abs_loadings = loadings[pc_name].abs().sort_values(ascending=False)
            top_3 = abs_loadings.head(3)
            print(f"\n  {pc_name} en çok etkilendiği değişkenler:")
            for var_name, loading in top_3.items():
                direction = "pozitif" if loadings.loc[var_name, pc_name] > 0 else "negatif"
                print(f"    - {var_name}: {loading:.3f} ({direction})")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scree plot
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 
                    alpha=0.7, edgecolor='black', label='Bireysel')
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                     np.cumsum(pca.explained_variance_ratio_), 
                     'ro-', linewidth=2, label='Kümülatif')
        axes[0].axhline(y=0.8, color='g', linestyle='--', linewidth=1, label='%80 eşiği')
        axes[0].set_xlabel('Bileşen Numarası')
        axes[0].set_ylabel('Açıklanan Varyans Oranı')
        axes[0].set_title('Scree Plot (Açıklanan Varyans)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PCA scatter (ilk 2 bileşen)
        if pca.n_components_ >= 2:
            axes[1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            axes[1].set_xlabel(f'PC1 (%{pca.explained_variance_ratio_[0]*100:.1f})')
            axes[1].set_ylabel(f'PC2 (%{pca.explained_variance_ratio_[1]*100:.1f})')
            axes[1].set_title('PCA Scatter Plot (İlk 2 Bileşen)')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Biplot (değişken ok vektörleri)
        if pca.n_components_ >= 2:
            self._biplot(pca_result[:, :2], loadings.iloc[:, :2].values, 
                        loadings.index, ['PC1', 'PC2'])
        
        return pca, pca_result, loadings
    
    def _biplot(self, scores, loadings, labels, pc_labels):
        """
        PCA biplot çizimi.
        
        Biplot Nedir:
            Hem gözlemleri (noktalar) hem de değişkenleri (ok vektörleri)
            aynı grafikte gösterir.
            
        Nasıl Yorumlanır:
            1. Noktalar: Her nokta bir gözlemi temsil eder
               - Birbirine yakın noktalar: Benzer gözlemler
               - Uzak noktalar: Farklı gözlemler
            
            2. Oklar: Her ok bir orijinal değişkeni temsil eder
               - Ok uzunluğu: Değişkenin PC'deki önemi
               - Ok yönü: Değişkenin PC'deki katkısı
            
            3. Açı:
               - Dar açı (~0°): Pozitif korelasyon
               - Geniş açı (~180°): Negatif korelasyon
               - Dik açı (~90°): Korelasyon yok
            
            4. Orijine uzaklık:
               - Uzak noktalar: Uç gözlemler, potansiyel outlier'lar
               - Orijine yakın: Ortalama gözlemler
        
        Parameters:
            scores: PCA skorları (gözlemler)
            loadings: Bileşen yükleri (değişkenler)
            labels: Değişken isimleri
            pc_labels: PC isimleri
        """
        plt.figure(figsize=(10, 8))
        
        # Gözlemleri çiz (noktalar)
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=20, label='Gözlemler')
        
        # Değişken vektörlerini çiz (oklar)
        for i, label in enumerate(labels):
            arrow_scale = 3  # Ok uzunluğu çarpanı
            plt.arrow(0, 0, 
                     loadings[i, 0] * arrow_scale, 
                     loadings[i, 1] * arrow_scale, 
                     color='r', alpha=0.7, 
                     head_width=0.1, head_length=0.1, 
                     linewidth=2)
            plt.text(loadings[i, 0] * arrow_scale * 1.15, 
                    loadings[i, 1] * arrow_scale * 1.15, 
                    label, color='r', fontsize=10, fontweight='bold',
                    ha='center', va='center')
        
        plt.xlabel(pc_labels[0], fontsize=12)
        plt.ylabel(pc_labels[1], fontsize=12)
        plt.title('PCA Biplot (Gözlemler + Değişken Vektörleri)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("\nBiplot Yorumlama İpuçları:")
        print("  1. Ok uzunluğu: Değişkenin PC'deki önemi")
        print("  2. Oklar arası açı:")
        print("     - Dar açı: Pozitif korelasyon")
        print("     - Geniş açı: Negatif korelasyon")
        print("     - Dik açı: Korelasyon yok")
        print("  3. Orijinden uzak noktalar: Potansiyel outlier'lar")
    
    def tsne_analysis(self, n_components=2, perplexity=30, n_samples=1000):
        """
        t-SNE ile boyut indirgeme (görselleştirme için).
        
        t-SNE Nedir:
            t-Distributed Stochastic Neighbor Embedding
            
            Non-linear boyut indirgeme yöntemidir.
            Yüksek boyutta yakın olan noktaların düşük boyutta da
            yakın kalmasını sağlar.
        
        PCA vs t-SNE:
            PCA:
            - Doğrusal dönüşüm
            - Global yapıyı korur
            - Deterministik
            - Hızlı
            - Boyut indirgeme için
            
            t-SNE:
            - Non-linear dönüşüm
            - Yerel yapıyı korur
            - Stochastic (her seferinde farklı)
            - Yavaş
            - Sadece görselleştirme için
        
        Perplexity Nedir:
            Her noktanın kaç komşusuna dikkat edeceğini belirler.
            
            Düşük perplexity (5-15):
            - Yerel yapıya odaklanır
            - Küçük kümeleri vurgular
            - Küçük veri setleri için uygun
            
            Yüksek perplexity (30-50):
            - Global yapıya odaklanır
            - Büyük kümeleri vurgular
            - Büyük veri setleri için uygun
            
            Kural: n/3 < perplexity < n (n = örneklem sayısı)
        
        Önemli Notlar:
            1. Uzaklıklar anlamlı DEĞİL (sadece yakınlık önemli)
            2. Küme boyutları yanıltıcı olabilir
            3. Her çalıştırmada farklı sonuç verir
            4. Sadece görselleştirme için kullanılmalı
            5. Yeni veri eklenemez (transform edilemez)
        
        Avantajlar:
            - Kompleks non-linear yapıları yakalar
            - Kümeleri çok iyi ayırır
            - Görselleştirme için mükemmel
        
        Dezavantajlar:
            - Hesaplama maliyeti çok yüksek
            - Stochastic (tekrarlanamaz)
            - Parametrelere çok duyarlı
            - Global yapıyı bozabilir
            - Sadece görselleştirme için
        
        Ne Zaman Kullanılır:
            - Yüksek boyutlu veriyi görselleştirmek için
            - Kümeleri görmek için
            - Non-linear ilişkiler varsa
            - İnteraktif keşif için
        
        Parameters:
            n_components (int): Hedef boyut (genellikle 2 veya 3)
            perplexity (float): Komşu sayısı (5-50 arası)
            n_samples (int): Örneklem boyutu (büyük veri için)
        
        Returns:
            tuple: (tsne_model, tsne_result)
        
        Örnek Kullanım:
            >>> dim_reduction = DimensionReduction(df)
            >>> tsne, result = dim_reduction.tsne_analysis(perplexity=30)
            >>> # Kümeleri renklendir
            >>> plt.scatter(result[:, 0], result[:, 1], c=labels)
        """
        print(f"\n{'='*80}")
        print(f"t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        print(f"{'='*80}")
        
        print(f"\nAmaç: Yüksek boyutlu veriyi 2D/3D'de görselleştirmek")
        print(f"      (yerel yapıyı koruyarak)")
        
        # Veriyi hazırla
        data = self.df[self.numerical_cols].dropna()
        
        # Örnekleme
        if len(data) > n_samples:
            data_sampled = data.sample(n_samples, random_state=42)
            print(f"\nBİLGİ: Veri boyutu büyük ({len(data)} satır)")
            print(f"       Performans için {n_samples} örneklem alındı.")
        else:
            data_sampled = data
            print(f"\nVeri boyutu: {len(data_sampled)} gözlem")
        
        print(f"Özellik sayısı: {len(self.numerical_cols)}")
        
        # Standardizasyon (t-SNE için önemli)
        print(f"\nStandardizasyon uygulanıyor...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_sampled)
        
        # Perplexity kontrolü
        max_perplexity = len(data_sampled) / 3
        if perplexity > max_perplexity:
            perplexity = int(max_perplexity)
            print(f"\nUYARI: Perplexity çok yüksek, {perplexity} olarak ayarlandı")
        
        print(f"\nt-SNE Parametreleri:")
        print(f"  Hedef boyut: {n_components}")
        print(f"  Perplexity: {perplexity}")
        print(f"    (Her noktanın ~{perplexity} komşusuna bakacak)")
        
        # t-SNE uygula
        print(f"\nt-SNE hesaplanıyor...")
        print(f"  (Bu işlem biraz zaman alabilir...)")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=42, verbose=0)
        tsne_result = tsne.fit_transform(data_scaled)
        
        print(f"\nt-SNE tamamlandı!")
        print(f"  Yeni veri şekli: {tsne_result.shape}")
        
        print(f"\nÖnemli Notlar:")
        print(f"  1. Uzaklıklar anlamlı DEĞİL (sadece yakınlık önemli)")
        print(f"  2. Her çalıştırmada farklı sonuç verebilir (stochastic)")
        print(f"  3. Sadece görselleştirme için kullanılmalı")
        print(f"  4. Küme boyutları yanıltıcı olabilir")
        
        # Görselleştirme
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                       alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
            plt.xlabel('t-SNE Bileşen 1', fontsize=12)
            plt.ylabel('t-SNE Bileşen 2', fontsize=12)
            plt.title(f't-SNE Görselleştirme (Perplexity={perplexity})', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print("\nGörselleştirme İpuçları:")
            print("  - Yakın noktalar: Benzer gözlemler")
            print("  - Ayrı kümeler: Farklı gruplar")
            print("  - Uzaklık ölçüleri anlamlı DEĞİL")
        
        return tsne, tsne_result


# Dosya çok uzun olduğu için PART 2'de devam edecek...
# Özellik Mühendisliği, Anomali Tespiti ve Dengesiz Veri Analizi
# PART 2 dosyasında yer alacak.
