"""
DATA VISUALIZATION - ORTA SEVIYE
=================================

Bu dosya data science ve data analyst pozisyonlarına yönelik
orta seviye veri görselleştirme tekniklerini içerir.

ICERIK:
1. Correlation Heatmap ve Matrix Analizi
2. FacetGrid - Coklu Karsilastirma
3. Time Series Visualization
4. Statistical Plots (Regression, Residual, Q-Q)
5. Custom Styling ve Annotations
6. Comparative Analysis

GEREKLI KUTUPHANELER:
- matplotlib
- seaborn
- pandas
- numpy
- scipy
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)


# ==============================================================================
# 1. CORRELATION HEATMAP VE MATRIX ANALIZI
# ==============================================================================

def correlation_heatmap():
    """
    Korelasyon matrisi ve heatmap görselleştirmesi
    
    Korelasyon nedir?
    - Iki degisken arasindaki dogrusal iliski gucunu olcen
    - -1 ile +1 arasinda deger alir
    - 0'a yakin: zayif iliski
    - 1'e yakin: guclu pozitif iliski
    - -1'e yakin: guclu negatif iliski
    
    Heatmap nedir?
    - Sayisal degerleri renk yogunlugu ile gosteren grafik
    - Korelasyon matrisini gorsellesttirmek icin idealdir
    - Hizli pattern tespiti saglar
    
    Kullanim alanlari:
    - Feature selection (hangi degiskenler birbiriyle iliskili)
    - Multicollinearity tespiti
    - Veri kesfetme (exploratory data analysis)
    - Model oncesi veri analizi
    
    Yorumlama:
    - Kirmizi tonlar: Pozitif korelasyon
    - Mavi tonlar: Negatif korelasyon
    - Beyaz/acik renkler: Zayif veya yok korelasyon
    """
    print("=" * 70)
    print("1. CORRELATION HEATMAP - KORELASYON ANALIZI")
    print("=" * 70)
    
    # Örnek veri seti olusturma
    # Senaryo: Calisanlar hakkinda veriler
    np.random.seed(42)
    n_samples = 200
    
    # Veri olusturma
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'salary': np.random.randint(3000, 15000, n_samples),
        'experience': np.random.randint(0, 30, n_samples),
        'education_years': np.random.randint(12, 20, n_samples),
        'performance': np.random.randint(60, 100, n_samples),
        'bonus': np.random.randint(500, 5000, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Mantikli iliskiler ekleyelim
    # Maas, deneyim ve egitimle iliskili olmali
    df['salary'] = (df['experience'] * 300 + 
                    df['education_years'] * 200 + 
                    np.random.normal(0, 1000, n_samples))
    
    # Prim, performansla iliskili olmali
    df['bonus'] = df['performance'] * 40 + np.random.normal(0, 500, n_samples)
    
    # Korelasyon matrisini hesaplama
    correlation_matrix = df.corr()
    
    print("\nKorelasyon Matrisi:")
    print(correlation_matrix.round(2))
    
    # Gorsellesttirme
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. Temel heatmap
    # annot=True: her hucreye korelasyon degerini yazar
    # fmt='.2f': 2 ondalik basamak formatinda
    # cmap='coolwarm': renk paleti (soguktan sicaga)
    # center=0: 0 degerini beyaz yapar
    # square=True: hucreler kare sekilde
    # linewidths: hucre aralari
    # cbar_kws: colorbar ayarlari
    sns.heatmap(correlation_matrix, 
                annot=True,              # Degerleri goster
                fmt='.2f',               # Format: 2 ondalik
                cmap='coolwarm',         # Renk paleti
                center=0,                # 0 merkez (beyaz)
                square=True,             # Kare hucreler
                linewidths=1,            # Hucre araligi
                cbar_kws={"shrink": 0.8},  # Colorbar boyutu
                ax=axes[0])
    axes[0].set_title('Korelasyon Matrisi - Standart', fontsize=14, fontweight='bold', pad=15)
    
    # 2. Gelismis heatmap - Sadece alt ucgen
    # mask: ust ucgeni gizler (tekrari onler)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,               # Ust ucgeni gizle
                annot=True, 
                fmt='.2f', 
                cmap='RdYlGn',           # Kirmizi-Sari-Yesil
                center=0, 
                square=True, 
                linewidths=1,
                cbar_kws={"shrink": 0.8}, 
                ax=axes[1])
    axes[1].set_title('Korelasyon Matrisi - Alt Ucgen', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/11_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCorrelation heatmap basariyla olusturuldu")
    print("\nKorelasyon Yorumlama Rehberi:")
    print("  +1.0          : Mukemmel pozitif korelasyon")
    print("  +0.7 - +1.0   : Guclu pozitif korelasyon")
    print("  +0.3 - +0.7   : Orta pozitif korelasyon")
    print("  -0.3 - +0.3   : Zayif/yok korelasyon")
    print("  -0.7 - -0.3   : Orta negatif korelasyon")
    print("  -1.0 - -0.7   : Guclu negatif korelasyon")
    print("  -1.0          : Mukemmel negatif korelasyon")
    print("\nOnemli Bulgu:")
    print("  - Maas ile deneyim arasinda guclu pozitif korelasyon var")
    print("  - Prim ile performans arasinda guclu pozitif korelasyon var")
    print()


# ==============================================================================
# 2. FACET GRID - COKLU KARSILASTIRMA
# ==============================================================================

def facet_grid_analysis():
    """
    FacetGrid ile kategorilere gore ayrilmis grafikler
    
    FacetGrid nedir?
    - Veriyi kategorilere ayirip her kategori icin ayri grafik olusturur
    - Satirlarda ve sutunlarda farkli kategoriler olabilir
    - Tum grafiklerde ayni eksen olcegi kullanilir (karsilastirma icin)
    
    Kullanim alanlari:
    - Cinsiyet, bolge, zaman gibi kategorilere gore karsilastirma
    - Grup bazli pattern kesfetme
    - Ayni degiskenlerin farkli gruplardaki dagilimi
    
    Avantajlari:
    - Cok boyutlu veriyi 2D'de gosterebilme
    - Kategoriler arasi fark acikca gorulur
    - Tutarli olcekleme ile adil karsilastirma
    
    Parametreler:
    - col: sutun bazli ayirma
    - row: satir bazli ayirma
    - hue: renk bazli ayirma
    - height: her grafik yuksekligi
    - aspect: en/boy orani
    """
    print("=" * 70)
    print("2. FACET GRID - COKLU KARSILASTIRMA")
    print("=" * 70)
    
    # Tips veri setini yukle
    tips = sns.load_dataset('tips')
    
    print("\nVeri seti degiskenleri:")
    print(tips.columns.tolist())
    print(f"Toplam gozlem sayisi: {len(tips)}")
    
    # FacetGrid olusturma
    # col='time': sutunlarda Lunch ve Dinner
    # row='sex': satirlarda Male ve Female
    # hue='smoker': renge gore sigara icme durumu
    g = sns.FacetGrid(tips, 
                      col='time',               # Sutun: Lunch/Dinner
                      row='sex',                # Satir: Male/Female
                      hue='smoker',             # Renk: Yes/No
                      height=4,                 # Her grafik 4 inch
                      aspect=1.2,               # En/boy orani
                      palette='Set1',           # Renk paleti
                      margin_titles=True)       # Kenar basliklari
    
    # Her grafige scatter plot ekle
    # g.map: her subgrafige ayni fonksiyonu uygular
    g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.7, s=80)
    
    # Legend ekle
    g.add_legend(title='Sigara Iciyor mu?')
    
    # Ana baslik
    g.fig.suptitle('Hesap vs Bahsis (Cinsiyet, Zaman, Sigara)', 
                   y=1.02, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/12_facet_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFacetGrid basariyla olusturuldu")
    print("\nNasil okunur:")
    print("  - Her kucuk grafik bir grup kombinasyonunu gosterir")
    print("  - Satirlar: Cinsiyet (Male/Female)")
    print("  - Sutunlar: Zaman (Lunch/Dinner)")
    print("  - Renkler: Sigara icme durumu")
    print("\nKullanim: Farkli gruplardaki dagilimi karsilastirma")
    print()


# ==============================================================================
# 3. TIME SERIES VISUALIZATION
# ==============================================================================

def time_series_visualization():
    """
    Zaman serisi gorsellestirme teknikleri
    
    Zaman serisi nedir?
    - Zamana bagli olarak kaydedilmis veriler
    - Her gozlemin bir zaman damgasi vardir
    - Trend, mevsimsellik, dongusellik icerabilir
    
    Zaman serisi bilesenleri:
    - Trend: Uzun donemli yonu (yukselis/dususu)
    - Seasonality: Duzenli tekrar eden paternler (gunluk, aylik, yillik)
    - Cyclical: Duzensiz tekrar eden paternler (ekonomik donguler)
    - Noise: Rastgele dalgalanmalar
    
    Gorsellesttirme teknikleri:
    - Line plot: Temel zaman serisi
    - Moving average: Trendii gormek icin
    - Seasonal decomposition: Bilesenleri ayirma
    - Comparison plots: Yillar arasi karsilastirma
    
    Kullanim alanlari:
    - Satis tahminleme
    - Hisse senedi analizi
    - Hava durumu analizi
    - Web trafik analizi
    """
    print("=" * 70)
    print("3. ZAMAN SERISI GORSELLESTIRME")
    print("=" * 70)
    
    # Zaman serisi verisi olusturma
    # Senaryo: 4 yillik gunluk satis verisi
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Trend + Mevsimsellik + Gurultu
    # Trend: Dogrusal artis
    trend = np.linspace(100, 200, len(date_range))
    
    # Mevsimsellik: Yillik dongu
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
    
    # Gurultu: Rastgele dalgalanmalar
    noise = np.random.normal(0, 10, len(date_range))
    
    # Toplam deger
    values = trend + seasonality + noise
    
    # DataFrame olusturma
    df_ts = pd.DataFrame({'date': date_range, 'value': values})
    df_ts['month'] = df_ts['date'].dt.month
    df_ts['year'] = df_ts['date'].dt.year
    
    print(f"\nZaman serisi olusturuldu: {len(df_ts)} gun")
    print(f"Tarih araligi: {df_ts['date'].min()} - {df_ts['date'].max()}")
    
    # Gorsellesttirme
    fig = plt.figure(figsize=(18, 12))
    
    # GridSpec ile karmasik layout
    # 3 satir, 2 sutun
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Temel zaman serisi (tum sutunlar)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_ts['date'], df_ts['value'], linewidth=1, alpha=0.7, color='steelblue')
    ax1.set_title('Zaman Serisi - Gunluk Degerler', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tarih', fontsize=11)
    ax1.set_ylabel('Deger', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Hareketli ortalamalar
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Rolling window ile hareketli ortalama hesaplama
    # window: kac gunluk ortalama
    df_ts['ma_30'] = df_ts['value'].rolling(window=30).mean()
    df_ts['ma_90'] = df_ts['value'].rolling(window=90).mean()
    
    ax2.plot(df_ts['date'], df_ts['value'], 
             linewidth=0.5, alpha=0.3, color='gray', label='Gunluk')
    ax2.plot(df_ts['date'], df_ts['ma_30'], 
             linewidth=2, color='orange', label='30 Gunluk Ort.')
    ax2.plot(df_ts['date'], df_ts['ma_90'], 
             linewidth=2, color='red', label='90 Gunluk Ort.')
    ax2.set_title('Hareketli Ortalamalar', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tarih', fontsize=10)
    ax2.set_ylabel('Deger', fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Aylik box plot (dagilimlari karsilastirma)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Her ay icin veriyi toplama
    monthly_data = [df_ts[df_ts['month'] == month]['value'].values 
                    for month in range(1, 13)]
    
    # Box plot olusturma
    bp = ax3.boxplot(monthly_data, 
                     labels=['Oca', 'Sub', 'Mar', 'Nis', 'May', 'Haz',
                            'Tem', 'Agu', 'Eyl', 'Eki', 'Kas', 'Ara'],
                     patch_artist=True,     # Renklendirme icin
                     notch=True)            # Notch ekle
    
    # Kutulari renklendirme
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax3.set_title('Aylik Dagilim Karsilastirmasi', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Ay', fontsize=10)
    ax3.set_ylabel('Deger', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Yillik karsilastirma
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Her yil icin ayri cizgi
    for year in df_ts['year'].unique():
        year_data = df_ts[df_ts['year'] == year]
        # dayofyear: yilin kacinci gunu (1-365)
        ax4.plot(year_data['date'].dt.dayofyear, 
                year_data['value'], 
                label=f'{year}', 
                alpha=0.7, 
                linewidth=2)
    
    ax4.set_title('Yillik Karsilastirma', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Yilin Gunu', fontsize=10)
    ax4.set_ylabel('Deger', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Aylik ortalamalar (bar chart)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Gruplama ve ortalama hesaplama
    monthly_avg = df_ts.groupby('month')['value'].mean()
    
    ax5.bar(range(1, 13), monthly_avg.values, 
           color='teal', alpha=0.7, edgecolor='black')
    ax5.set_title('Ortalama Aylik Degerler', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Ay', fontsize=10)
    ax5.set_ylabel('Ortalama Deger', fontsize=10)
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(['Oca', 'Sub', 'Mar', 'Nis', 'May', 'Haz',
                        'Tem', 'Agu', 'Eyl', 'Eki', 'Kas', 'Ara'])
    ax5.grid(axis='y', alpha=0.3)
    
    plt.savefig('/mnt/user-data/outputs/13_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nZaman serisi gorsellestirmeleri basariyla olusturuldu")
    print("\nHareketli Ortalama:")
    print("  - 30 gunluk: Kisa vadeli trendii gosterir")
    print("  - 90 gunluk: Uzun vadeli trendii gosterir")
    print("\nMevsimsellik:")
    print("  - Aylik box plot mevsimsel farkliliklari gosterir")
    print("  - Yillar arasi karsilastirma tekrar eden patternleri gosterir")
    print()


# ==============================================================================
# 4. STATISTICAL PLOTS
# ==============================================================================

def statistical_plots():
    """
    Istatistiksel gorsellestirmeler
    
    Regression Plot:
    - Iki degisken arasi dogrusal iliskiyi gosterir
    - En iyi uyum dogrusu (best fit line) cizilir
    - Guven araligi (confidence interval) eklenir
    
    Residual Plot:
    - Model tahminlerinin hata dagitimini gosterir
    - Artik degerler (actual - predicted) gosterilir
    - Homoscedasticity (varyans homojen mi) kontrolu
    
    Q-Q Plot (Quantile-Quantile):
    - Verinin dagilimini teorik dagilimla karsilastirir
    - Normallik varsayimi kontrolu
    - Noktal eger dogruya yakinsa dagilim normale yakindir
    
    Kullanim alanlari:
    - Model varsayimlari kontrolu
    - Veri dagilimi analizi
    - Outlier tespiti
    - Regresyon kalitesi degerlendirme
    """
    print("=" * 70)
    print("4. ISTATISTIKSEL GORSELLESTIRMELER")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Gorsellesttirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Regression Plot
    # Senaryo: X ve Y arasi dogrusal iliski
    x = np.random.uniform(0, 100, 100)
    y = 2.5 * x + np.random.normal(0, 20, 100)
    df_reg = pd.DataFrame({'X': x, 'Y': y})
    
    # regplot: regression line ile scatter plot
    # scatter_kws: scatter plot ayarlari
    # line_kws: dogru cizgisi ayarlari
    sns.regplot(data=df_reg, x='X', y='Y', ax=axes[0, 0], 
                scatter_kws={'alpha': 0.5, 's': 50},
                line_kws={'color': 'red', 'linewidth': 2})
    axes[0, 0].set_title('Regression Plot (Dogrusal Iliski)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    # Artik degerler: Gercek - Tahmin edilen
    # residplot: artiklari otomatik hesaplar ve cizer
    sns.residplot(data=df_reg, x='X', y='Y', ax=axes[0, 1],
                  scatter_kws={'alpha': 0.5})
    axes[0, 1].set_title('Residual Plot (Artik Degerler)', fontsize=12, fontweight='bold')
    # y=0 cizgisi ekle (referans)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot (Normallik Kontrolu)
    # Normal dagilimdan veri
    data_normal = np.random.normal(0, 1, 1000)
    
    # probplot: quantile-quantile plot olusturur
    # dist="norm": normal dagilim ile karsilastir
    stats.probplot(data_normal, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normallik Kontrolu)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ECDF (Empirical Cumulative Distribution Function)
    # Kumülatif olasilik dagilimi
    sorted_data = np.sort(data_normal)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # step: basamak grafigi
    # where='post': sonraki degere kadar yatay git
    axes[1, 1].step(sorted_data, ecdf, where='post', linewidth=2, color='green')
    axes[1, 1].set_title('ECDF (Kumulatif Dagilim)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Deger')
    axes[1, 1].set_ylabel('Kumulatif Olasilik')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/14_statistical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nIstatistiksel gorsellestirmeler basariyla olusturuldu")
    print("\nRegression Plot Yorumlama:")
    print("  - Guven araligi (golge alan): Tahmin belirsizligi")
    print("  - Noktalarin dogruya yakinligi: Iyi fit")
    print("\nResidual Plot Yorumlama:")
    print("  - Rastgele dagilim: Iyi model")
    print("  - Pattern var: Model yetersiz")
    print("  - Huni sekli: Heteroscedasticity var")
    print("\nQ-Q Plot Yorumlama:")
    print("  - Noktalar dogruya yakin: Normal dagilim")
    print("  - S-seklinde: Skewed dagilim")
    print("  - Uclar ayrilir: Heavy-tailed dagilim")
    print()


# ==============================================================================
# 5. CUSTOM STYLING VE ANNOTATIONS
# ==============================================================================

def custom_styling_annotations():
    """
    Ozel stil ve aciklamalar
    
    Annotation nedir?
    - Grafige ek metin, ok, sekil ekleme
    - Onemli noktalari vurgulama
    - Aciklayici bilgi ekleme
    
    Annotation tipleri:
    - annotate(): Ok ile metin
    - text(): Sadece metin
    - arrow(): Sadece ok
    - shapes: Daire, dikdortgen, vb.
    
    Kullanim alanlari:
    - Outlier aciklama
    - Onemli olaylari isaretleme
    - Grafik uzerinde aciklama
    - Vurgu ve dikkat cekme
    
    Styling ozellikleri:
    - Renkler: Color palettes
    - Fontlar: Font family, size, weight
    - Cizgiler: Linewidth, style, color
    - Isiklar: Alpha, transparency
    """
    print("=" * 70)
    print("5. OZEL STIL VE ACIKLAMALAR")
    print("=" * 70)
    
    # Tips veri seti
    tips = sns.load_dataset('tips')
    
    # Gorsellesttirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Annotated Bar Plot
    # Gunlere gore ortalama hesap
    day_avg = tips.groupby('day')['total_bill'].mean().sort_values(ascending=False)
    
    # Bar plot olusturma
    bars = axes[0, 0].bar(day_avg.index, day_avg.values, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                          edgecolor='black', linewidth=1.5)
    
    # Her bar'in ustune deger yazma
    for bar in bars:
        height = bar.get_height()
        # text(): metin ekleme
        # bar.get_x(): bar'in x konumu
        # bar.get_width(): bar'in genisligi
        # ha='center': yatayda ortala
        # va='bottom': dikeyden altta
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.2f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[0, 0].set_title('Gunlere Gore Ortalama Hesap (Annotated)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Ortalama Hesap ($)', fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Vurgulu Scatter Plot
    scatter = axes[0, 1].scatter(tips['total_bill'], tips['tip'], 
                                c=tips['size'], s=100, alpha=0.6,
                                cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # En yuksek bahsisi vurgula
    max_tip_idx = tips['tip'].idxmax()
    
    # Daire ile vurgulama
    axes[0, 1].scatter(tips.loc[max_tip_idx, 'total_bill'], 
                      tips.loc[max_tip_idx, 'tip'],
                      s=500,                    # Buyuk boyut
                      facecolors='none',        # Ic bos
                      edgecolors='red',         # Kirmizi kenar
                      linewidth=3)              # Kalin kenar
    
    # Ok ve metin ile aciklama
    # annotate(): ok ve metin birlestir
    # xy: isaretlenecek nokta
    # xytext: metin konumu
    # arrowprops: ok ayarlari
    axes[0, 1].annotate('En Yuksek Bahsis!', 
                       xy=(tips.loc[max_tip_idx, 'total_bill'], 
                           tips.loc[max_tip_idx, 'tip']),
                       xytext=(30, 8),          # Metin konumu
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    axes[0, 1].set_title('Vurgulu Scatter Plot', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Toplam Hesap ($)')
    axes[0, 1].set_ylabel('Bahsis ($)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Grup Buyuklugu')
    
    # 3. Fill Between (Alan doldurma)
    # Senaryo: Guven araligi gosterimi
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y_upper = np.sin(x) + 0.5      # Ust limit
    y_lower = np.sin(x) - 0.5      # Alt limit
    
    axes[1, 0].plot(x, y, 'b-', linewidth=2, label='Ortalama')
    axes[1, 0].plot(x, y_upper, 'r--', linewidth=1, alpha=0.7, label='Ust Limit')
    axes[1, 0].plot(x, y_lower, 'r--', linewidth=1, alpha=0.7, label='Alt Limit')
    
    # fill_between: iki cizgi arasini doldur
    axes[1, 0].fill_between(x, y_upper, y_lower, alpha=0.2, color='red')
    
    axes[1, 0].set_title('Guven Araligi Gosterimi', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Ozel Renk Paleti
    custom_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    sns.violinplot(data=tips, x='day', y='total_bill', 
                   palette=custom_palette, ax=axes[1, 1])
    
    axes[1, 1].set_title('Ozel Renk Paleti', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Gun')
    axes[1, 1].set_ylabel('Toplam Hesap ($)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/15_custom_styling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nOzel stil ve aciklamalar eklendi")
    print("\nAnnotation Kullanim Ipuclari:")
    print("  - Onemli noktalari vurgula")
    print("  - Acik ve kisa aciklamalar kullan")
    print("  - Renkleri dikkat cekmek icin kullan")
    print("  - Ok kullanimi net olmali")
    print()


# ==============================================================================
# 6. COMPARATIVE ANALYSIS
# ==============================================================================

def comparative_analysis():
    """
    Karsilastirmali analiz gorsellestirmeleri
    
    Grouped Bar Chart:
    - Ayni kategorilerde farkli gruplari yanyana gosterir
    - Karsilastirma yapmak icin idealdir
    
    Stacked Bar Chart:
    - Toplam ve alt parcalari ayni anda gosterir
    - Parça-butun iliskisini korur
    
    Radar Chart (Spider Chart):
    - Cok boyutlu karsilastirma
    - Her eksen farkli bir metrik
    - Circuler gosterim
    
    Kullanim alanlari:
    - Urun karsilastirmasi
    - Performans analizi
    - Departman karsilastirmasi
    - Bolgesel analiz
    """
    print("=" * 70)
    print("6. KARSILASTIRMALI ANALIZ")
    print("=" * 70)
    
    # Ornek veri: Kampanya performansi
    np.random.seed(42)
    campaigns = ['Kampanya A', 'Kampanya B', 'Kampanya C', 'Kampanya D']
    metrics = ['Tiklama', 'Goruntulenme', 'Donusum', 'Harcama']
    
    data = {
        'campaign': campaigns * len(metrics),
        'metric': [m for m in metrics for _ in campaigns],
        'value': np.random.randint(50, 500, len(campaigns) * len(metrics))
    }
    df_campaign = pd.DataFrame(data)
    
    # Gorsellesttirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Grouped Bar Chart
    # pivot: Wide format (kampanyalar x metrikler)
    df_pivot = df_campaign.pivot(index='campaign', columns='metric', values='value')
    
    # Grouped bar plot
    df_pivot.plot(kind='bar', ax=axes[0, 0], width=0.8, edgecolor='black')
    axes[0, 0].set_title('Kampanya Performans Karsilastirmasi', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Kampanya')
    axes[0, 0].set_ylabel('Deger')
    axes[0, 0].legend(title='Metrik', bbox_to_anchor=(1.05, 1))
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
    
    # 2. Stacked Bar Chart
    df_pivot.plot(kind='bar', stacked=True, ax=axes[0, 1], 
                  colormap='viridis', edgecolor='black')
    axes[0, 1].set_title('Yigilmis Bar Chart', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Kampanya')
    axes[0, 1].set_ylabel('Toplam Deger')
    axes[0, 1].legend(title='Metrik', bbox_to_anchor=(1.05, 1))
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
    
    # 3. Radar Chart
    # Radar chart icin polar koordinat sistemi kullanilir
    categories = df_pivot.columns.tolist()
    N = len(categories)
    
    # Acilar: Her kategori icin aci
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Daireyi kapatmak icin ilk degeri ekle
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.set_theta_offset(np.pi / 2)      # Yukaridan basla
    ax3.set_theta_direction(-1)          # Saat yonunun tersi
    
    # Her kampanya icin cizgi ciz
    for idx, campaign in enumerate(campaigns):
        values = df_pivot.loc[campaign].tolist()
        values += values[:1]  # Daireyi kapat
        
        ax3.plot(angles, values, 'o-', linewidth=2, label=campaign)
        ax3.fill(angles, values, alpha=0.15)
    
    # Etiketler
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_title('Radar Chart (Cok Boyutlu Karsilastirma)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True)
    
    # 4. Normalized Comparison (Yuzde bazli)
    # Her kampanyanin toplam icindeki yuzdesi
    df_normalized = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
    
    df_normalized.plot(kind='barh', stacked=True, ax=axes[1, 1], 
                      colormap='tab10', edgecolor='black')
    axes[1, 1].set_title('Normalize Edilmis Karsilastirma (%)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Yuzde (%)')
    axes[1, 1].set_ylabel('Kampanya')
    axes[1, 1].legend(title='Metrik', bbox_to_anchor=(1.05, 1))
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/16_comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKarsilastirmali analiz grafikleri basariyla olusturuldu")
    print("\nGrafik Tipleri:")
    print("  - Grouped Bar: Yan yana karsilastirma")
    print("  - Stacked Bar: Toplam ve parcalar")
    print("  - Radar Chart: Cok boyutlu profil")
    print("  - Normalized Bar: Yuzdelik dagitim")
    print()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Ana fonksiyon - tum ornekleri calistir
    
    Bu fonksiyon tum orta seviye gorsellestirme orneklerini sirayla calistirir.
    """
    print("\n" + "="*70)
    print("DATA VISUALIZATION - ORTA SEVIYE EGITIMI")
    print("="*70 + "\n")
    
    # Tum fonksiyonlari calistir
    correlation_heatmap()
    facet_grid_analysis()
    time_series_visualization()
    statistical_plots()
    custom_styling_annotations()
    comparative_analysis()
    
    print("\n" + "="*70)
    print("TUM ORTA SEVIYE GORSELLESTIRMELER TAMAMLANDI")
    print("="*70)
    print(f"\nGrafikler su klasore kaydedildi: /mnt/user-data/outputs/")
    print("\nOGRENILEN KONULAR:")
    print("  - Correlation Heatmap ve Matrix")
    print("  - FacetGrid ile coklu karsilastirma")
    print("  - Zaman Serisi gorsellestirme")
    print("  - Istatistiksel Plotlar (Regression, Q-Q, ECDF)")
    print("  - Custom Styling ve Annotations")
    print("  - Karsilastirmali Analiz (Radar Chart)")
    print("\nSONRAKI ADIM: Ileri seviye dosyaya gecin")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
