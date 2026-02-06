"""
DATA VISUALIZATION - GIRIS SEVIYESI
====================================

Bu dosya data science ve data analyst pozisyonlarına yönelik
temel veri görselleştirme tekniklerini içerir.

ICERIK:
1. Temel Grafik Türleri (Line, Bar, Scatter, Pie)
2. Matplotlib Temelleri
3. Seaborn ile Görselleştirme
4. Veri Dağılım Grafikleri
5. Kategorisel Veri Görselleştirme

GEREKLI KUTUPHANELER:
- matplotlib
- seaborn
- pandas
- numpy
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# 1. TEMEL GRAFIK TURLERI
# ==============================================================================

def basic_line_plot():
    """
    Temel çizgi grafiği oluşturma
    
    Çizgi grafikleri nedir?
    - Zaman içindeki değişimi göstermek için kullanılır
    - Trend analizi için idealdir
    - Sürekli veriler için uygundur
    
    Kullanım alanları:
    - Hisse senedi fiyat değişimleri
    - Aylık satış trendleri
    - Sıcaklık değişimleri
    - Kullanıcı büyüme grafikleri
    """
    print("=" * 60)
    print("1. TEMEL CIZGI GRAFIGI (LINE PLOT)")
    print("=" * 60)
    
    # Örnek veri oluşturma
    # Senaryo: 6 aylık satış verisi
    months = ['Ocak', 'Subat', 'Mart', 'Nisan', 'Mayis', 'Haziran']
    sales = [150, 180, 165, 220, 240, 280]
    
    # Figure ve axes oluşturma
    # figsize: grafiğin boyutunu belirler (genişlik, yükseklik) inch cinsinden
    plt.figure(figsize=(10, 6))
    
    # Çizgi grafiği çizme
    # marker: her veri noktasında işaret gösterir
    # linestyle: çizgi stili ('-' düz çizgi, '--' kesikli, ':' noktalı)
    # linewidth: çizgi kalınlığı
    # markersize: işaret boyutu
    # color: renk belirleme
    # label: legend için etiket
    plt.plot(months, sales, 
             marker='o',           # Nokta işareti
             linestyle='-',        # Düz çizgi
             linewidth=2,          # Çizgi kalınlığı
             markersize=8,         # Nokta boyutu
             color='blue',         # Mavi renk
             label='Satislar')     # Etiket
    
    # Grafik başlığı ekleme
    plt.title('Aylik Satis Grafigi', fontsize=16, fontweight='bold')
    
    # Eksen etiketleri
    plt.xlabel('Aylar', fontsize=12)
    plt.ylabel('Satis Miktari (Bin TL)', fontsize=12)
    
    # Legend (açıklama kutusu) gösterme
    plt.legend()
    
    # Grid (ızgara) ekleme
    # alpha: saydamlık (0-1 arası, 1 tamamen opak)
    plt.grid(True, alpha=0.3)
    
    # Layout düzenleme
    plt.tight_layout()
    
    # Grafiği kaydetme
    plt.savefig('/mnt/user-data/outputs/01_line_plot.png', dpi=300, bbox_inches='tight')
    
    # Grafiği gösterme
    plt.show()
    
    print("Cizgi grafigi basariyla olusturuldu")
    print("Kullanim Alani: Zaman serisi verileri, trend analizi")
    print()


def basic_bar_plot():
    """
    Temel bar grafiği oluşturma
    
    Bar grafikleri nedir?
    - Kategorik verileri karşılaştırmak için kullanılır
    - Her kategori bir bar ile temsil edilir
    - Dikey veya yatay olabilir
    
    Kullanım alanları:
    - Ürün satış karşılaştırması
    - Şehirlere göre nüfus
    - Departman bazlı performans
    - Ankete verilen cevap dağılımı
    
    Dikey vs Yatay Bar:
    - Dikey: Zaman bazlı veriler için tercih edilir
    - Yatay: Kategori isimleri uzun olduğunda daha okunabilir
    """
    print("=" * 60)
    print("2. BAR GRAFIGI (BAR PLOT)")
    print("=" * 60)
    
    # Örnek veri
    # Senaryo: 5 farklı ürünün satış adedi
    categories = ['Urun A', 'Urun B', 'Urun C', 'Urun D', 'Urun E']
    values = [45, 60, 38, 72, 55]
    
    # 1 satır, 2 sütunlu subplot oluşturma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dikey bar grafiği
    # color: her bar için farklı renk listesi
    ax1.bar(categories, values, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    ax1.set_title('Dikey Bar Grafigi', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Urunler', fontsize=12)
    ax1.set_ylabel('Satis Adedi', fontsize=12)
    # axis='y': sadece y ekseninde grid göster
    ax1.grid(axis='y', alpha=0.3)
    
    # Yatay bar grafiği
    # barh: horizontal (yatay) bar için kullanılır
    ax2.barh(categories, values, 
             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    ax2.set_title('Yatay Bar Grafigi', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Satis Adedi', fontsize=12)
    ax2.set_ylabel('Urunler', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/02_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Bar grafikleri basariyla olusturuldu")
    print("Kullanim Alani: Kategorik veriler arasi karsilastirma")
    print()


def basic_scatter_plot():
    """
    Scatter (dağılım) grafiği oluşturma
    
    Scatter plot nedir?
    - İki değişken arasındaki ilişkiyi gösterir
    - Her veri noktası bir nokta olarak gösterilir
    - Korelasyon analizi için kullanılır
    
    Kullanım alanları:
    - Reklam harcaması vs satış ilişkisi
    - Yaş vs maaş ilişkisi
    - Çalışma saati vs verimlilik
    - Sıcaklık vs dondurma satışı
    
    Korelasyon türleri:
    - Pozitif: Bir artarken diğeri de artar
    - Negatif: Bir artarken diğeri azalır
    - Yok: Belirgin bir ilişki yoktur
    """
    print("=" * 60)
    print("3. SCATTER PLOT (DAGILIM GRAFIGI)")
    print("=" * 60)
    
    # Örnek veri oluşturma
    # Senaryo: Reklam harcaması ile satış arasındaki ilişki
    np.random.seed(42)  # Tekrarlanabilir rastgele sayılar için
    
    # 50 adet reklam harcaması verisi (10-100 bin TL arası)
    ad_spending = np.random.uniform(10, 100, 50)
    
    # Satışlar reklam harcamasıyla ilişkili + rastgele gürültü
    # Formül: satış = 2.5 * reklam_harcaması + gürültü
    sales = 2.5 * ad_spending + np.random.normal(0, 20, 50)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot oluşturma
    # alpha: nokta saydamlığı (üst üste binme durumunda faydalı)
    # s: nokta boyutu
    # c: renk (tek renk veya değer listesi)
    # cmap: renk haritası (değer listesi kullanıldığında)
    # edgecolors: nokta kenar rengi
    # linewidth: kenar çizgi kalınlığı
    plt.scatter(ad_spending, sales, 
                alpha=0.6,              # %60 saydamlık
                s=100,                  # Nokta boyutu
                c=sales,                # Satış değerine göre renklendirme
                cmap='viridis',         # Renk haritası
                edgecolors='black',     # Siyah kenar
                linewidth=1)            # Kenar kalınlığı
    
    # Trend çizgisi ekleme
    # np.polyfit: polinom fit (1. derece = doğrusal)
    z = np.polyfit(ad_spending, sales, 1)
    p = np.poly1d(z)
    plt.plot(ad_spending, p(ad_spending), 
             "r--",                 # Kırmızı kesikli çizgi
             linewidth=2, 
             label='Trend Cizgisi')
    
    plt.title('Reklam Harcamasi vs Satis Iliskisi', fontsize=16, fontweight='bold')
    plt.xlabel('Reklam Harcamasi (Bin TL)', fontsize=12)
    plt.ylabel('Satis (Bin TL)', fontsize=12)
    
    # Colorbar ekleme (renk değer eşleştirmesi)
    plt.colorbar(label='Satis Degeri')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/03_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Scatter plot basariyla olusturuldu")
    print("Kullanim Alani: Iki degisken arasi iliski analizi")
    print("Yorumlama: Pozitif korelasyon gorulmektedir - reklam harcamasi arttikca satis artar")
    print()


def basic_pie_chart():
    """
    Pasta grafiği oluşturma
    
    Pie chart nedir?
    - Bütünün parçalara ayrılmasını gösterir
    - Her dilim bir kategoriyi temsil eder
    - Yüzdelik dağılımları gösterir
    
    Kullanım alanları:
    - Pazar payı dağılımı
    - Bütçe tahsisi
    - Kategori bazlı satış oranları
    - Anket sonucu dağılımı
    
    Dikkat edilmesi gerekenler:
    - Çok fazla kategori olmamalı (ideal 5-7)
    - Küçük dilimler "Diğer" altında toplanabilir
    - Renk seçimi önemlidir (ayırt edilebilir olmalı)
    """
    print("=" * 60)
    print("4. PIE CHART (PASTA GRAFIGI)")
    print("=" * 60)
    
    # Örnek veri
    # Senaryo: Satışların kategori bazlı dağılımı
    categories = ['Teknoloji', 'Gida', 'Giyim', 'Kozmetik', 'Diger']
    shares = [30, 25, 20, 15, 10]  # Yüzdelik paylar
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # explode: belirli dilimleri merkezden uzaklaştırma
    # İlk dilimi (Teknoloji) 0.1 birim ayır
    explode = (0.1, 0, 0, 0, 0)
    
    plt.figure(figsize=(10, 8))
    
    # Pie chart oluşturma
    # labels: dilim etiketleri
    # autopct: yüzde gösterimi (format: %1.1f%% -> 1 ondalık basamak)
    # startangle: başlangıç açısı (90 = saat 12'den başla)
    # colors: renk listesi
    # explode: dilim ayrımı
    # shadow: gölge efekti
    # textprops: metin özellikleri
    plt.pie(shares, 
            labels=categories, 
            autopct='%1.1f%%',          # Yüzde formatı
            startangle=90,              # 12 saat yönünden başla
            colors=colors, 
            explode=explode,
            shadow=True,                # Gölge ekle
            textprops={'fontsize': 12})
    
    plt.title('Kategori Bazinda Satis Dagilimi', fontsize=16, fontweight='bold', pad=20)
    
    # axis('equal'): daireyi tam yuvarlak yapar
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/04_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Pie chart basariyla olusturuldu")
    print("Kullanim Alani: Parca-butun iliskilerini gosterme")
    print()


# ==============================================================================
# 2. SEABORN ILE GELISMIS GORSELLESTIRME
# ==============================================================================

def seaborn_histogram():
    """
    Histogram ve dağılım grafikleri
    
    Histogram nedir?
    - Sürekli değişkenlerin frekans dağılımını gösterir
    - Verileri aralıklara (bins) böler
    - Her aralıktaki veri sayısını gösterir
    
    KDE (Kernel Density Estimation) nedir?
    - Histogramın düzgünleştirilmiş hali
    - Olasılık yoğunluk fonksiyonunu tahmin eder
    - Dağılımın şeklini daha net gösterir
    
    Kullanım alanları:
    - Yaş dağılımı
    - Maaş dağılımı
    - Test skorları dağılımı
    - Normallik kontrolü
    
    Histogram yorumlama:
    - Simetrik: Normal dağılım ihtimali
    - Sağa çarpık: Çoğu değer solda, birkaç büyük değer
    - Sola çarpık: Çoğu değer sağda, birkaç küçük değer
    - İki tepeli: İki farklı grup olabilir
    """
    print("=" * 60)
    print("5. HISTOGRAM VE DAGILIM GRAFIKLERI")
    print("=" * 60)
    
    # Örnek veri oluşturma
    # Senaryo: 1000 kişinin yaş dağılımı
    np.random.seed(42)
    # Normal dağılım: ortalama=35, standart sapma=10
    ages = np.random.normal(35, 10, 1000)
    
    # 2x2 subplot düzeni oluşturma
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Temel histogram (Matplotlib)
    # bins: aralık sayısı (fazla bins -> detaylı, az bins -> genel görünüm)
    # edgecolor: bar kenar rengi
    axes[0, 0].hist(ages, 
                    bins=30,                # 30 aralık
                    color='skyblue', 
                    edgecolor='black', 
                    alpha=0.7)
    axes[0, 0].set_title('Histogram', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Yas', fontsize=12)
    axes[0, 0].set_ylabel('Frekans', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Seaborn Histogram + KDE
    # kde=True: Kernel Density Estimation ekler
    sns.histplot(ages, 
                 bins=30, 
                 kde=True,              # KDE çizgisi ekle
                 color='coral', 
                 ax=axes[0, 1])
    axes[0, 1].set_title('Histogram + KDE (Kernel Density)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Yas', fontsize=12)
    axes[0, 1].set_ylabel('Frekans', fontsize=12)
    
    # 3. Sadece KDE Plot
    # shade=True: eğrinin altını doldur (deprecated, fill=True kullanılabilir)
    sns.kdeplot(ages, 
                shade=True,             # Altını doldur
                color='green', 
                ax=axes[1, 0])
    axes[1, 0].set_title('KDE (Yogunluk) Grafigi', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Yas', fontsize=12)
    axes[1, 0].set_ylabel('Yogunluk', fontsize=12)
    
    # 4. Distribution Plot
    # stat='density': y eksenini yoğunluk olarak gösterir
    sns.histplot(ages, 
                 kde=True, 
                 stat='density',        # Yoğunluk bazlı
                 color='purple', 
                 ax=axes[1, 1])
    axes[1, 1].set_title('Distribution Plot', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Yas', fontsize=12)
    axes[1, 1].set_ylabel('Yogunluk', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/05_histogram_kde.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Histogram ve dagilim grafikleri basariyla olusturuldu")
    print("Kullanim Alani: Veri dagilimini inceleme, normallik kontrolu")
    print("Yorum: Veri yaklasik normal dagilim gostermektedir (cani sekilli)")
    print()


def seaborn_boxplot_violinplot():
    """
    Box plot ve violin plot
    
    Box Plot (Kutu Grafiği) nedir?
    - 5 sayı özeti gösterir: min, Q1, median, Q3, max
    - Aykırı değerleri nokta olarak gösterir
    - Dağılımın merkezini ve yayılımını gösterir
    
    Box plot elemanları:
    - Alt çizgi: Minimum (aykırı değerler hariç)
    - Kutu alt sınırı: 1. Çeyrek (Q1) - %25
    - Kutu içi çizgi: Medyan (Q2) - %50
    - Kutu üst sınırı: 3. Çeyrek (Q3) - %75
    - Üst çizgi: Maximum (aykırı değerler hariç)
    - Noktalar: Aykırı değerler (outliers)
    
    IQR (Interquartile Range) = Q3 - Q1
    Aykırı değer eşiği:
    - Alt sınır: Q1 - 1.5 * IQR
    - Üst sınır: Q3 + 1.5 * IQR
    
    Violin Plot nedir?
    - Box plot + KDE birleşimi
    - Dağılımın yoğunluğunu gösterir
    - Her noktadaki veri yoğunluğunu genişlik ile gösterir
    
    Kullanım alanları:
    - Aykırı değer tespiti
    - Grup karşılaştırması
    - Dağılım şekli analizi
    - Varyans karşılaştırması
    """
    print("=" * 60)
    print("6. BOX PLOT VE VIOLIN PLOT")
    print("=" * 60)
    
    # Örnek veri oluşturma
    # Senaryo: 4 farklı kategorinin değer dağılımı
    np.random.seed(42)
    data = {
        'Kategori': ['A']*100 + ['B']*100 + ['C']*100 + ['D']*100,
        'Deger': np.concatenate([
            np.random.normal(50, 10, 100),   # Kategori A: ort=50, std=10
            np.random.normal(60, 15, 100),   # Kategori B: ort=60, std=15
            np.random.normal(55, 8, 100),    # Kategori C: ort=55, std=8
            np.random.normal(70, 12, 100)    # Kategori D: ort=70, std=12
        ])
    }
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box Plot
    # palette: renk paleti
    sns.boxplot(data=df, 
                x='Kategori', 
                y='Deger', 
                ax=ax1, 
                palette='Set2')
    ax1.set_title('Box Plot - Kategori Bazli Dagilim', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Kategori', fontsize=12)
    ax1.set_ylabel('Deger', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Violin Plot
    sns.violinplot(data=df, 
                   x='Kategori', 
                   y='Deger', 
                   ax=ax2, 
                   palette='muted')
    ax2.set_title('Violin Plot - Yogunluk Dagilimi', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Kategori', fontsize=12)
    ax2.set_ylabel('Deger', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/06_boxplot_violinplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Box plot ve violin plot basariyla olusturuldu")
    print("Kullanim Alani: Aykiri deger tespiti, grup karsilastirmasi")
    print("Yorum:")
    print("  - Kategori D en yuksek ortalamaya sahip")
    print("  - Kategori C en az varyansa sahip (en dar dagilim)")
    print("  - Kategori B en fazla varyansa sahip (en genis dagilim)")
    print()


# ==============================================================================
# 3. GERCEK VERI SETI ILE UYGULAMALAR
# ==============================================================================

def real_data_analysis():
    """
    Gerçek veri seti ile görselleştirme örnekleri
    
    Tips Dataset:
    - Seaborn ile gelen örnek veri seti
    - Restoran bahşiş verileri
    - 244 gözlem, 7 değişken
    
    Değişkenler:
    - total_bill: Toplam hesap miktarı
    - tip: Bahşiş miktarı
    - sex: Cinsiyet (Male/Female)
    - smoker: Sigara içiyor mu (Yes/No)
    - day: Gün (Thur/Fri/Sat/Sun)
    - time: Öğün zamanı (Lunch/Dinner)
    - size: Grup büyüklüğü
    
    Bu bölümde farklı plot türlerini gerçek veri ile uygulayacağız
    """
    print("=" * 60)
    print("7. GERCEK VERI SETI ANALIZI (TIPS DATASET)")
    print("=" * 60)
    
    # Tips veri setini yükleme
    tips = sns.load_dataset('tips')
    
    # Veri seti hakkında bilgi
    print("\nVeri Seti Ilk 5 Satir:")
    print(tips.head())
    print(f"\nVeri Seti Boyutu: {tips.shape}")
    print(f"Sutunlar: {tips.columns.tolist()}")
    
    # 2x2 subplot düzeni
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Count Plot - Kategorik değişken frekansı
    # Kullanım: Bir kategorideki öğe sayısını gösterir
    sns.countplot(data=tips, 
                  x='day',              # X ekseninde gün
                  palette='pastel',     # Pastel renk paleti
                  ax=axes[0, 0])
    axes[0, 0].set_title('Gunlere Gore Musteri Sayisi', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Gun', fontsize=12)
    axes[0, 0].set_ylabel('Musteri Sayisi', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Bar Plot - Ortalama değerler
    # estimator: değerleri özetleme fonksiyonu (ortalama, toplam, vb.)
    sns.barplot(data=tips, 
                x='day', 
                y='total_bill', 
                estimator=np.mean,      # Ortalama hesapla
                palette='coolwarm', 
                ax=axes[0, 1])
    axes[0, 1].set_title('Gunlere Gore Ortalama Hesap', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Gun', fontsize=12)
    axes[0, 1].set_ylabel('Ortalama Hesap ($)', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Box Plot - Dağılım karşılaştırması
    sns.boxplot(data=tips, 
                x='day', 
                y='tip', 
                palette='Set3', 
                ax=axes[1, 0])
    axes[1, 0].set_title('Gunlere Gore Bahsis Dagilimi', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Gun', fontsize=12)
    axes[1, 0].set_ylabel('Bahsis ($)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Scatter Plot - İlişki analizi
    # hue: üçüncü bir değişkene göre renklendirme
    # style: işaret şekline göre ayırma
    sns.scatterplot(data=tips, 
                    x='total_bill', 
                    y='tip', 
                    hue='time',         # Zamana göre renk
                    style='sex',        # Cinsiyete göre şekil
                    s=100,              # Nokta boyutu
                    ax=axes[1, 1])
    axes[1, 1].set_title('Hesap vs Bahsis Iliskisi', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Toplam Hesap ($)', fontsize=12)
    axes[1, 1].set_ylabel('Bahsis ($)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/07_tips_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTips veri seti analizi tamamlandi")
    print("\nOnemli bulgular:")
    print("  - En fazla musteri Cumartesi ve Pazar gunleri geliyor")
    print("  - Pazar gunu ortalama hesap daha yuksek")
    print("  - Hesap miktari arttikca bahsis de genellikle artiyor")
    print()


def multivariate_analysis():
    """
    Çoklu değişken görselleştirme
    
    Pairplot nedir?
    - Tüm sayısal değişkenlerin birbirleriyle ilişkisini gösterir
    - Matris formatında düzenlenir
    - Diagonal: Her değişkenin dağılımı
    - Off-diagonal: İki değişken arası ilişki
    
    Kullanım alanları:
    - Değişkenler arası korelasyon keşfi
    - Feature engineering için değişken seçimi
    - Çok değişkenli normallik kontrolü
    - Grup ayrımı analizi
    
    Pairplot yorumlama:
    - Doğrusal ilişkiler: Pozitif/negatif korelasyon
    - Kümelenme: Farklı grupların varlığı
    - Dağılım şekli: Normal/skewed dağılım
    - Aykırı değerler: Noktalar ana kümeden uzak
    """
    print("=" * 60)
    print("8. COKLU DEGISKEN ANALIZI")
    print("=" * 60)
    
    tips = sns.load_dataset('tips')
    
    # Pairplot oluşturma
    # hue: kategorik değişkene göre renklendirme
    # diag_kind: diagonal grafik türü ('kde', 'hist')
    # plot_kws: scatter plot ayarları
    print("Pairplot olusturuluyor...")
    pairplot_fig = sns.pairplot(tips, 
                                 hue='sex',             # Cinsiyete göre renk
                                 palette='husl',        # Renk paleti
                                 diag_kind='kde',       # Diagonal: KDE
                                 plot_kws={'alpha': 0.6})  # Scatter saydamlık
    
    pairplot_fig.fig.suptitle('Tum Degiskenler Arasi Iliski Matrisi', 
                               y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/08_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Pairplot basariyla olusturuldu")
    print("Kullanim Alani: Tum degiskenler arasi iliskileri tek seferde gorme")
    print("\nPairplot nasil okunur:")
    print("  - Diagonal (kosegen): Her degiskenin yogunluk dagilimi")
    print("  - Alt ucgen: Degisken ciftlerinin scatter plot'u")
    print("  - Ust ucgen: Alt ucgenin aynisi (simetrik)")
    print("  - Renkler: Farkli kategorileri temsil eder (burada cinsiyet)")
    print()


# ==============================================================================
# 4. SUBPLOT VE FIGURE CUSTOMIZATION
# ==============================================================================

def subplot_examples():
    """
    Alt grafikler (subplots) oluşturma
    
    Subplot nedir?
    - Tek bir figure içinde birden fazla grafik
    - Karşılaştırma ve panel gösterimler için kullanılır
    - Tutarlı formatlamaya olanak sağlar
    
    Subplot oluşturma yöntemleri:
    1. plt.subplots(rows, cols): Grid düzeni
    2. plt.subplot(xyz): Tek tek ekleme
    3. GridSpec: Karmaşık düzenler için
    
    Kullanım alanları:
    - Dashboard oluşturma
    - Karşılaştırmalı analiz
    - Raporlama
    - Akademik yayınlar
    """
    print("=" * 60)
    print("9. SUBPLOTS - COK GRAFIKLI GOSTERIM")
    print("=" * 60)
    
    # Örnek veri: Matematiksel fonksiyonlar
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    
    # 2 satır, 3 sütun subplot oluşturma
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Figure başlığı
    fig.suptitle('Farkli Matematiksel Fonksiyonlar', fontsize=18, fontweight='bold')
    
    # axes[satır, sütun] formatında erişim
    
    # Subplot 1: Sin fonksiyonu
    axes[0, 0].plot(x, np.sin(x), 'r-', linewidth=2)
    axes[0, 0].set_title('Sin(x)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Cos fonksiyonu
    axes[0, 1].plot(x, np.cos(x), 'b-', linewidth=2)
    axes[0, 1].set_title('Cos(x)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Tan fonksiyonu
    axes[0, 2].plot(x, np.tan(x), 'g-', linewidth=2)
    axes[0, 2].set_title('Tan(x)', fontsize=12)
    axes[0, 2].set_ylim(-10, 10)  # Y ekseni sınırı
    axes[0, 2].grid(True, alpha=0.3)
    
    # Subplot 4: Exp fonksiyonu
    axes[1, 0].plot(x, np.exp(x/5), 'm-', linewidth=2)
    axes[1, 0].set_title('Exp(x/5)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 5: Log fonksiyonu
    axes[1, 1].plot(x[1:], np.log(x[1:]), 'c-', linewidth=2)
    axes[1, 1].set_title('Log(x)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Subplot 6: Kare fonksiyonu
    axes[1, 2].plot(x, x**2, 'y-', linewidth=2)
    axes[1, 2].set_title('x²', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/09_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Subplots basariyla olusturuldu")
    print("\nSubplot kullanim ipuclari:")
    print("  - axes dizisini kullanarak her subplot'a erisim")
    print("  - axes[satir, sutun] formatinda indeksleme")
    print("  - Her subplot bagimsiz olarak ozellestirilir")
    print("  - tight_layout() otomatik bosluk ayari yapar")
    print()


# ==============================================================================
# 5. BEST PRACTICES VE IPUCLARI
# ==============================================================================

def best_practices_demo():
    """
    Görselleştirme en iyi pratikleri
    
    Iyi bir görselleştirme özellikleri:
    1. Anlaşılır: Mesaj net olmalı
    2. Doğru: Verileri doğru temsil etmeli
    3. Estetik: Görsel olarak çekici olmalı
    4. Tutarlı: Renkler ve stiller tutarlı olmalı
    5. Erişilebilir: Renk körü dostu olmalı
    
    Kaçınılması gerekenler:
    - Çok fazla renk kullanımı
    - Etiket eksikliği
    - Okunaksız fontlar
    - Yanıltıcı eksen ölçekleri
    - Gereksiz dekorasyon (chartjunk)
    
    Profesyonel görselleştirme için:
    - Her zaman eksen etiketleri ekleyin
    - Anlamlı başlıklar kullanın
    - Uygun renk paleti seçin
    - Grid ekleyerek okunabilirliği artırın
    - Gerektiğinde legend ekleyin
    - Yüksek çözünürlükte kaydedin
    """
    print("=" * 60)
    print("10. GORSELLESTIRME EN IYI PRATIKLERI")
    print("=" * 60)
    
    tips = sns.load_dataset('tips')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # KOTU ORNEK: Eksik etiketler, grid yok, renk bilgisi yok
    ax1.scatter(tips['total_bill'], tips['tip'], c='red')
    ax1.set_title('Kotu Gorsellestirme Ornegi')
    # Not: Eksen etiketleri yok, grid yok, renk anlamsiz
    
    # IYI ORNEK: Tam etiketler, grid, anlamli renkler, trend cizgisi
    scatter = ax2.scatter(tips['total_bill'], tips['tip'], 
                         c=tips['size'],        # Grup buyuklugune gore renk
                         cmap='viridis',        # Profesyonel renk paleti
                         s=100,                 # Okunabilir boyut
                         alpha=0.6,             # Uygun saydamlik
                         edgecolors='black',    # Kenar ile ayirt edilebilirlik
                         linewidth=0.5)
    
    ax2.set_title('Iyi Gorsellestirme Ornegi', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Toplam Hesap ($)', fontsize=12)
    ax2.set_ylabel('Bahsis ($)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Colorbar ekleme
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Grup Buyuklugu', fontsize=10)
    
    # Trend çizgisi ekle
    z = np.polyfit(tips['total_bill'], tips['tip'], 1)
    p = np.poly1d(z)
    ax2.plot(tips['total_bill'], p(tips['total_bill']), 
            "r--", alpha=0.8, linewidth=2, label='Trend')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/10_best_practices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBest practices gosterildi")
    print("\nGORSELLESTIRME IPUCLARI:")
    print("=" * 60)
    print("1. Her zaman eksen etiketleri ekleyin")
    print("   - X ve Y ekseninin ne oldugu acik olmali")
    print("   - Birim bilgisi ekleyin (TL, $, adet, vb.)")
    print()
    print("2. Anlamli basliklar kullanin")
    print("   - Grafik neyi gosteriyor acikca belirtin")
    print("   - Icerik hakkinda bilgi verin")
    print()
    print("3. Uygun renk paleti secin")
    print("   - Renk koru dostu paletler kullanin")
    print("   - Anlamli renklendirme yapin (renk bir degiskeni temsil etmeli)")
    print()
    print("4. Grid ekleyerek okunabilirligi artirin")
    print("   - Degerleri okumak kolaylasir")
    print("   - Alpha degerini dusuk tutun (0.3 ideal)")
    print()
    print("5. Gerektiginde legend/colorbar kullanin")
    print("   - Renkler veya semboller bir sey ifade ediyorsa aciklama ekleyin")
    print()
    print("6. Yuksek cozunurlukde kaydedin")
    print("   - dpi=300 profesyonel yayinlar icin yeterli")
    print("   - bbox_inches='tight' gereksiz boslugu keser")
    print()
    print("7. Grafigi baglam icin optimize edin")
    print("   - Rapor: Detayli, aciklamali")
    print("   - Sunum: Buyuk font, basit")
    print("   - Dashboard: Kompakt, cok panel")
    print()
    print("8. Veri-murekkep oranini optimize edin")
    print("   - Gereksiz dekorasyondan kacinin")
    print("   - Her element bir amaca hizmet etmeli")
    print()
    print("9. Tutarli stil kullanin")
    print("   - Ayni raporda ayni renk paleti")
    print("   - Ayni font boyutlari")
    print()
    print("10. Erisilebilirligi goz onunde bulundurun")
    print("   - Sadece renge guvenmeyin")
    print("   - Sembol ve desen de kullanin")
    print()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Ana fonksiyon - tum ornekleri calistir
    
    Bu fonksiyon tum gorsellestirme orneklerini sirayla calistirir.
    Her fonksiyon kendi grafigini olusturur ve kaydeder.
    
    Calistirma siralamasi:
    1. Temel grafikler (Line, Bar, Scatter, Pie)
    2. Seaborn grafikleri (Histogram, Box, Violin)
    3. Gercek veri analizi
    4. Coklu degisken analizi
    5. Subplots
    6. Best practices
    """
    print("\n" + "="*60)
    print("DATA VISUALIZATION - GIRIS SEVIYESI EGITIMI")
    print("="*60 + "\n")
    
    # Tum fonksiyonlari calistir
    basic_line_plot()
    basic_bar_plot()
    basic_scatter_plot()
    basic_pie_chart()
    seaborn_histogram()
    seaborn_boxplot_violinplot()
    real_data_analysis()
    multivariate_analysis()
    subplot_examples()
    best_practices_demo()
    
    print("\n" + "="*60)
    print("TUM GORSELLESTIRMELER TAMAMLANDI")
    print("="*60)
    print(f"\nGrafikler su klasore kaydedildi: /mnt/user-data/outputs/")
    print("\nOGRENILEN KONULAR:")
    print("  - Line, Bar, Scatter, Pie grafikleri")
    print("  - Histogram ve KDE")
    print("  - Box plot ve Violin plot")
    print("  - Seaborn ile ileri gorsellestirme")
    print("  - Gercek veri analizi")
    print("  - Subplots ve figure customization")
    print("  - Best practices")
    print("\nSONRAKI ADIM: Orta seviye dosyaya gecin")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
