"""
DATA VISUALIZATION - ILERI SEVIYE
==================================

Bu dosya data science ve data analyst pozisyonlarına yönelik
ileri seviye veri görselleştirme tekniklerini içerir.

ICERIK:
1. 3D Visualizations
2. Network Graphs
3. Advanced Statistical Plots
4. Complex Dashboard Layouts
5. Publication-Ready Figures

GEREKLI KUTUPHANELER:
- matplotlib
- seaborn
- pandas
- numpy
- scipy
- networkx
- mpl_toolkits (3D plotting)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)


# ==============================================================================
# 1. 3D VISUALIZATIONS
# ==============================================================================

def three_d_visualizations():
    """
    3 boyutlu gorsellestirmeler
    
    3D Scatter Plot:
    - Uc degiskeni ayni anda gosterir
    - Kumeleri 3 boyutta gosterme
    - Uzaysal iliskileri anlama
    
    3D Surface Plot:
    - Fonksiyon yuzeylerini gosterir
    - Z = f(X, Y) seklindeki fonksiyonlar icin
    - Topografik haritalar, maliyet fonksiyonlari
    
    Kullanim alanlari:
    - Kumeleme sonuclari (3 principal component)
    - Optimizasyon problemleri (cost surface)
    - Fiziksel simulasyonlar
    - Jeolojik veriler
    
    Dikkat edilmesi gerekenler:
    - 3D grafikler yorumlamasi zor olabilir
    - Eksen acilari onemlidir
    - 2D alternatifleri dusunun (contour plot)
    - Interactive plotlar tercih edilebilir
    """
    print("=" * 80)
    print("1. 3D VISUALIZATIONS - UC BOYUTLU GORSELLESTIRMELER")
    print("=" * 80)
    
    np.random.seed(42)
    n_points = 300
    
    # 3 farkli kume olustur
    cluster1 = np.random.randn(n_points, 3) * 0.5 + [0, 0, 0]
    cluster2 = np.random.randn(n_points, 3) * 0.5 + [3, 3, 3]
    cluster3 = np.random.randn(n_points, 3) * 0.5 + [-2, 3, -2]
    
    # Tum kumeleri birlestir
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*n_points + [1]*n_points + [2]*n_points)
    
    # Gorsellesttirme
    fig = plt.figure(figsize=(18, 6))
    
    # 1. 3D Scatter Plot
    # projection='3d': 3 boyutlu eksen olusturur
    ax1 = fig.add_subplot(131, projection='3d')
    
    # scatter: 3D nokta grafigi
    scatter = ax1.scatter(data[:, 0], data[:, 1], data[:, 2], 
                         c=labels, cmap='viridis', s=50, alpha=0.6, 
                         edgecolors='black', linewidth=0.5)
    
    # Eksen etiketleri
    # labelpad: etiket ile eksen arasi mesafe
    ax1.set_xlabel('X Degiskeni', fontsize=10, labelpad=10)
    ax1.set_ylabel('Y Degiskeni', fontsize=10, labelpad=10)
    ax1.set_zlabel('Z Degiskeni', fontsize=10, labelpad=10)
    ax1.set_title('3D Scatter Plot', fontsize=12, fontweight='bold', pad=20)
    plt.colorbar(scatter, ax=ax1, label='Kume', shrink=0.5)
    
    # 2. 3D Surface Plot
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Meshgrid: 2D grid olusturur
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Z degerleri: fonksiyon hesaplama
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # plot_surface: yuzey grafigi
    # cmap: renk haritasi
    # alpha: saydamlik
    # linewidth=0: cizgileri gizle
    # antialiased: yumusak kenarlari
    surf = ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax2.set_xlabel('X', fontsize=10, labelpad=10)
    ax2.set_ylabel('Y', fontsize=10, labelpad=10)
    ax2.set_zlabel('Z', fontsize=10, labelpad=10)
    ax2.set_title('3D Surface Plot', fontsize=12, fontweight='bold', pad=20)
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    # 3. 3D Wireframe + Contour
    ax3 = fig.add_subplot(133, projection='3d')
    
    # plot_wireframe: tel kafes grafigi
    ax3.plot_wireframe(X, Y, Z, color='green', alpha=0.5, linewidth=0.8)
    
    # contour: kontur cizgileri
    # zdir='z': Z eksenine projekte et
    # offset: projeksiyon yuksekligi
    ax3.contour(X, Y, Z, zdir='z', offset=-2, cmap='viridis', alpha=0.6)
    
    ax3.set_xlabel('X', fontsize=10, labelpad=10)
    ax3.set_ylabel('Y', fontsize=10, labelpad=10)
    ax3.set_zlabel('Z', fontsize=10, labelpad=10)
    ax3.set_title('3D Wireframe + Contour', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/17_3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n3D gorsellestirmeler basariyla olusturuldu")
    print("\n3D Grafik Kullanim Ipuclari:")
    print("  - Eksen acilarini ayarlayin (view_init)")
    print("  - Renk kullanimi onemli (dorduncu boyut)")
    print("  - Cok nokta varsa alpha kullanin")
    print("  - Alternatif olarak 2D projeksiyonlar dusunun")
    print()


# ==============================================================================
# 2. NETWORK GRAPHS
# ==============================================================================

def network_graphs():
    """
    Network graph (Ag grafigi) gorsellestirmeleri
    
    Network Graph nedir?
    - Dugumleri (nodes) ve baglantilari (edges) gosterir
    - Iliskisel verileri gorsellesttirir
    - Graf teorisi uygulamalari
    
    Network turleri:
    - Undirected: Yonsuz baglantilar
    - Directed: Yonlu baglantilar (oklar)
    - Weighted: Agirlikli baglantilar
    - Unweighted: Agiriklik yok
    
    Kullanim alanlari:
    - Sosyal ag analizi
    - Organizasyon semasi
    - Bilgi agi (knowledge graph)
    - Yol ve ulasim aglari
    - Protein etkilesim aglari
    
    Onemli kavramlar:
    - Node (Dugum): Bir varlik
    - Edge (Kenar): Baglanti/iliski
    - Degree: Bir dugumdeki baglanti sayisi
    - Path: Dugumler arasi yol
    - Community: Alt gruplar
    """
    print("=" * 80)
    print("2. NETWORK GRAPHS - AG GRAFIKLERI")
    print("=" * 80)
    
    # Gorsellesttirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Simple Network (Karate Club)
    # Karate Club: Unlu sosyal ag ornegi
    G1 = nx.karate_club_graph()
    
    # spring_layout: Dugumleri estetik sekilde yerlestirir
    # seed: Tekrarlanabilirlik icin
    pos1 = nx.spring_layout(G1, seed=42)
    
    # Dugumleri ciz
    nx.draw_networkx_nodes(G1, pos1, 
                          node_color='lightblue', 
                          node_size=500, 
                          alpha=0.9, 
                          ax=axes[0, 0])
    
    # Kenarlari ciz
    nx.draw_networkx_edges(G1, pos1, alpha=0.3, ax=axes[0, 0])
    
    # Etiketleri ciz
    nx.draw_networkx_labels(G1, pos1, font_size=8, ax=axes[0, 0])
    
    axes[0, 0].set_title('Karate Club Network', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')  # Eksenleri gizle
    
    # 2. Weighted Network
    # Agirlikli ag: Her baglantinin bir agili var
    G2 = nx.Graph()
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    G2.add_nodes_from(nodes)
    
    # Agirlikli kenarlar ekle (node1, node2, weight)
    edges = [('A', 'B', 5), ('A', 'C', 3), ('B', 'C', 8), 
             ('B', 'D', 2), ('C', 'E', 4), ('D', 'E', 6), 
             ('D', 'F', 1), ('E', 'F', 7)]
    G2.add_weighted_edges_from(edges)
    
    pos2 = nx.spring_layout(G2, seed=42)
    
    # Agirliklari al
    weights = [G2[u][v]['weight'] for u, v in G2.edges()]
    
    # Dugumleri ciz
    nx.draw_networkx_nodes(G2, pos2, 
                          node_color='coral', 
                          node_size=1000, 
                          alpha=0.9, 
                          ax=axes[0, 1])
    
    # Kenarlari agirliga gore kalinlikla ciz
    # width: kenar kalinligi
    # edge_color: kenarlari agirliklarina gore renklendir
    # edge_cmap: kenar renk haritasi
    nx.draw_networkx_edges(G2, pos2, 
                          width=[w*0.5 for w in weights], 
                          alpha=0.6, 
                          edge_color=weights, 
                          edge_cmap=plt.cm.Blues, 
                          ax=axes[0, 1])
    
    # Etiketler
    nx.draw_networkx_labels(G2, pos2, font_size=14, font_weight='bold', ax=axes[0, 1])
    
    # Kenar etiketleri (agirliklar)
    edge_labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels, font_size=10, ax=axes[0, 1])
    
    axes[0, 1].set_title('Weighted Network Graph', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Directed Graph (Yonlu Graf)
    G3 = nx.DiGraph()
    G3.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'),
                      ('A', 'C'), ('B', 'D'), ('C', 'E'), ('E', 'A')])
    
    # circular_layout: Dairesel yerlesim
    pos3 = nx.circular_layout(G3)
    
    nx.draw_networkx_nodes(G3, pos3, 
                          node_color='lightgreen', 
                          node_size=1200, 
                          alpha=0.9, 
                          ax=axes[1, 0])
    
    # Oklu kenarlar
    # arrows=True: Oklari goster
    # arrowsize: Ok boyutu
    # arrowstyle: Ok stili
    # connectionstyle: Baglanti egrisi
    nx.draw_networkx_edges(G3, pos3, 
                          arrows=True, 
                          arrowsize=20, 
                          arrowstyle='->', 
                          edge_color='gray', 
                          width=2, 
                          connectionstyle='arc3,rad=0.1', 
                          ax=axes[1, 0])
    
    nx.draw_networkx_labels(G3, pos3, font_size=14, font_weight='bold', ax=axes[1, 0])
    
    axes[1, 0].set_title('Directed Graph', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Community Detection (Topluluk Tespiti)
    G4 = nx.karate_club_graph()
    
    # Greedy modularity: Topluluk tespiti algoritmasi
    communities = nx.community.greedy_modularity_communities(G4)
    
    # Her dugum icin topluluk numarasi bul
    colors = []
    for node in G4.nodes():
        for i, community in enumerate(communities):
            if node in community:
                colors.append(i)
                break
    
    pos4 = nx.spring_layout(G4, seed=42)
    
    # Topluluklara gore renklendir
    nx.draw_networkx_nodes(G4, pos4, 
                          node_color=colors, 
                          node_size=400, 
                          cmap='tab10', 
                          alpha=0.9, 
                          ax=axes[1, 1])
    
    nx.draw_networkx_edges(G4, pos4, alpha=0.2, ax=axes[1, 1])
    
    axes[1, 1].set_title('Community Detection', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/18_network_graphs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nNetwork grafikleri basariyla olusturuldu")
    print("\nNetwork Analizi Metrikleri:")
    print("  - Degree centrality: En cok baglantili dugum")
    print("  - Betweenness centrality: Yollar uzerinde onemli dugum")
    print("  - Closeness centrality: Digerlere yakinlik")
    print("  - Community structure: Alt grup yapisi")
    print("\nKullanim: Sosyal ag analizi, organizasyon semasi, ilişki haritalama")
    print()


# ==============================================================================
# 3. ADVANCED STATISTICAL PLOTS
# ==============================================================================

def advanced_statistical_plots():
    """
    Ileri seviye istatistiksel gorsellestirmeler
    
    Hexbin Density Plot:
    - 2D histogram alternatifi
    - Altigen hucreler kullanir
    - Yogunluk analizi icin
    
    Bootstrap Confidence Intervals:
    - Orneklemden yeniden ornekleme
    - Parametre guven araligi
    - Belirsizlik olcumu
    
    Hierarchical Clustering Dendrogram:
    - Kumeleme sonuclarini gosterir
    - Agac yapisi
    - Benzerlik iliskilerini gosterir
    
    Kullanim alanlari:
    - Yogunluk analizi
    - Belirsizlik tahminleme
    - Kumeleme gorsellesttirme
    - Taxonomic analiz
    """
    print("=" * 80)
    print("3. ILERI SEVIYE ISTATISTIKSEL GORSELLESTIRMELER")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Gorsellesttirme
    fig = plt.figure(figsize=(18, 12))
    
    # GridSpec ile karmasik layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hexbin Density Plot
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 2D normal dagilim
    x = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
    
    # hexbin: Altigen hucreli yogunluk grafigi
    # gridsize: Hucre sayisi
    # mincnt: Minimum gozlem sayisi (bos hucreler gosterilmez)
    ax1.hexbin(x[:, 0], x[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
    ax1.set_title('Hexbin Density Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 2. KDE Contour Plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Kernel Density Estimation
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x.T)
    
    # Mesh grid olustur
    xx, yy = np.mgrid[-3:3:100j, -3:3:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = np.reshape(kde(positions).T, xx.shape)
    
    # contourf: Doldurulmus kontur grafigi
    contour = ax2.contourf(xx, yy, density, levels=20, cmap='viridis')
    ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=10, color='white', edgecolors='black')
    ax2.set_title('KDE Contour Plot', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(contour, ax=ax2)
    
    # 3. Bootstrap Confidence Intervals
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Ornek veri
    data = np.random.normal(100, 15, 500)
    
    # Bootstrap: Yeniden ornekleme
    bootstrap_means = []
    for _ in range(1000):
        # replace=True: Ayni deger birden fazla secile bilir
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Histogram
    ax3.hist(bootstrap_means, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Guven araliklari
    # percentile: Yuzdelik dilimleri
    ci_lower = np.percentile(bootstrap_means, 2.5)   # %2.5
    ci_upper = np.percentile(bootstrap_means, 97.5)  # %97.5
    
    # Dikey cizgiler ekle
    # axvline: Dikey cizgi
    ax3.axvline(ci_lower, color='red', linestyle='--', linewidth=2, 
               label=f'95% CI Lower: {ci_lower:.2f}')
    ax3.axvline(ci_upper, color='red', linestyle='--', linewidth=2, 
               label=f'95% CI Upper: {ci_upper:.2f}')
    ax3.axvline(np.mean(data), color='green', linestyle='-', linewidth=2, 
               label=f'Original Mean: {np.mean(data):.2f}')
    
    ax3.set_title('Bootstrap Confidence Interval', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Mean Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Hierarchical Clustering Dendrogram
    ax4 = fig.add_subplot(gs[1, :])
    
    # Iris veri seti
    iris = sns.load_dataset('iris')
    X = iris.drop('species', axis=1).values
    
    # Hierarchical clustering
    # method='ward': Ward's minimum variance method
    linkage_matrix = hierarchy.linkage(X, method='ward')
    
    # dendrogram: Agac diyagrami
    # color_threshold: Renk degistirme esigi
    # above_threshold_color: Esik ustu renk
    dendro = hierarchy.dendrogram(linkage_matrix, ax=ax4, 
                                  color_threshold=20, 
                                  above_threshold_color='gray')
    ax4.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Distance')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.savefig('/mnt/user-data/outputs/19_advanced_statistical.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nIleri seviye istatistiksel gorsellestirmeler basariyla olusturuldu")
    print("\nBootstrap Yorumlama:")
    print("  - Kirmizi cizgiler: %95 guven araligi")
    print("  - Yesil cizgi: Orjinal ortalama")
    print("  - Dar aralik: Yuksek kesinlik")
    print("\nDendrogram Yorumlama:")
    print("  - Yukseklik: Benzemezlik derecesi")
    print("  - Dusuk birlesme: Benzer ogeleri")
    print("  - Kesme noktasi: Kume sayisi belirleme")
    print()


# ==============================================================================
# 4. DASHBOARD LAYOUT
# ==============================================================================

def dashboard_layout():
    """
    Dashboard layout ornegi
    
    Dashboard nedir?
    - Birden fazla grafigi bir arada gosteren panel
    - KPI'lari (Key Performance Indicators) gosterir
    - Ozet bilgi ve detayli grafikler iceren
    
    Dashboard elemanlari:
    - KPI kartlari: Onemli metrikleri gosteren kartlar
    - Trend grafikleri: Zaman serisi
    - Dagilim grafikleri: Kategori bazli
    - Karsilastirma grafikleri: Gruplar arasi
    
    Tasarim ilkeleri:
    - Hiyerarsi: En onemli bilgi en ustte
    - Tutarlilik: Ayni renk ve stil
    - Basitlik: Gerekenli bilgi
    - Erisilebilirlik: Kolay okuma
    
    Kullanim alanlari:
    - Is zekasi (Business Intelligence)
    - Performans izleme
    - Satış raporları
    - Operasyonel metrikler
    """
    print("=" * 80)
    print("4. DASHBOARD LAYOUT - PROFESYONEL GORUNTUMLEME")
    print("=" * 80)
    
    # Veri hazirlama
    np.random.seed(42)
    
    # Tarih araligi
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    # Kumulatif gelir
    revenue = np.cumsum(np.random.randn(len(dates)) * 1000 + 5000)
    
    # Kategori satislari
    categories = ['Teknoloji', 'Gida', 'Giyim', 'Kozmetik', 'Diger']
    sales = [35000, 28000, 22000, 18000, 12000]
    
    # Bolge satislari
    regions = ['Marmara', 'Ege', 'Akdeniz', 'Ic Anadolu', 'Karadeniz']
    region_sales = [45, 25, 15, 10, 5]
    
    # Dashboard olusturma
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Ana baslik
    fig.suptitle('2023 YILI SATIS ANALIZ DASHBOARD', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # KPI Cards (Manuel olarak cizilecek)
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')
    
    # KPI verileri
    kpis = [
        ('Toplam Satis', f'${sum(sales):,}', '#4ECDC4'),
        ('Musteri Sayisi', '12,547', '#FF6B6B'),
        ('Ortalama Siparis', '$89.50', '#45B7D1'),
        ('Buyume Orani', '+23.5%', '#98D8C8')
    ]
    
    # Her KPI icin kutu ciz
    from matplotlib.patches import FancyBboxPatch
    for i, (title, value, color) in enumerate(kpis):
        x_pos = 0.15 + i * 0.22
        
        # Kutu (fancy box)
        rect = FancyBboxPatch((x_pos, 0.3), 0.18, 0.5,
                             boxstyle="round,pad=0.02",
                             facecolor=color, 
                             edgecolor='black', 
                             linewidth=2, 
                             transform=ax_kpi.transAxes)
        ax_kpi.add_patch(rect)
        
        # Deger metni
        ax_kpi.text(x_pos + 0.09, 0.65, value, 
                   ha='center', va='center',
                   fontsize=18, fontweight='bold', color='white',
                   transform=ax_kpi.transAxes)
        
        # Baslik metni
        ax_kpi.text(x_pos + 0.09, 0.42, title, 
                   ha='center', va='center',
                   fontsize=11, color='white', 
                   transform=ax_kpi.transAxes)
    
    # Gelir Trendi
    ax1 = fig.add_subplot(gs[1, :3])
    ax1.plot(dates, revenue, linewidth=2, color='#4ECDC4')
    ax1.fill_between(dates, revenue, alpha=0.3, color='#4ECDC4')
    ax1.set_title('Yillik Gelir Trendi', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Tarih', fontsize=10)
    ax1.set_ylabel('Gelir ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Y eksenini format etme
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Kategori Dagilimi (Pie Chart)
    ax2 = fig.add_subplot(gs[1, 3])
    colors_cat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    wedges, texts, autotexts = ax2.pie(sales, labels=categories, autopct='%1.1f%%',
                                       colors=colors_cat, startangle=90,
                                       textprops={'fontsize': 9})
    # Yuzde metinlerini beyaz yap
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Kategori Dagilimi', fontsize=13, fontweight='bold', pad=10)
    
    # Bolgesel Performans
    ax3 = fig.add_subplot(gs[2, :2])
    bars = ax3.barh(regions, region_sales, color=colors_cat)
    ax3.set_title('Bolgesel Performans (%)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Satis Payi (%)', fontsize=10)
    ax3.grid(axis='x', alpha=0.3)
    
    # Bar uzerine deger yazma
    for bar, value in zip(bars, region_sales):
        ax3.text(value + 1, bar.get_y() + bar.get_height()/2, 
                f'{value}%', va='center', fontweight='bold')
    
    # Aylik Karsilastirma
    ax4 = fig.add_subplot(gs[2, 2:])
    months = ['Oca', 'Sub', 'Mar', 'Nis', 'May', 'Haz', 
             'Tem', 'Agu', 'Eyl', 'Eki', 'Kas', 'Ara']
    sales_2022 = np.random.randint(8000, 12000, 12)
    sales_2023 = np.random.randint(9000, 14000, 12)
    
    x = np.arange(len(months))
    width = 0.35
    
    ax4.bar(x - width/2, sales_2022, width, label='2022', color='#FFB6C1', alpha=0.8)
    ax4.bar(x + width/2, sales_2023, width, label='2023', color='#4ECDC4', alpha=0.8)
    
    ax4.set_title('Yillar Arasi Karsilastirma', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Ay', fontsize=10)
    ax4.set_ylabel('Satis ($)', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(months)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.savefig('/mnt/user-data/outputs/20_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDashboard layout basariyla olusturuldu")
    print("\nDashboard Tasarim Ilkeleri:")
    print("  - KPI'lari en ustte goster")
    print("  - Renk tutarliligi sagla")
    print("  - Okunabilir font boyutlari kullan")
    print("  - Grid ile duzeni olustur")
    print("\nKullanim: Rapor sunumlari, executive ozet, KPI gosterimleri")
    print()


# ==============================================================================
# 5. PUBLICATION-READY FIGURES
# ==============================================================================

def publication_ready_figures():
    """
    Yayin kalitesinde figur olusturma
    
    Akademik yayin standartlari:
    - Yuksek cozunurluk (300+ DPI)
    - Profesyonel fontlar (Times, Arial)
    - Uygun boyutlandirma
    - Subfigure etiketleme (a, b, c, d)
    - Tutarli stil
    
    IEEE/Nature/Science standartlari:
    - Siyah-beyaz baskilarda okunabilir
    - Renk koru dostu
    - Minimal dekorasyon
    - Acik etiketler
    - Yuksek kontrast
    
    Kullanim alanlari:
    - Bilimsel makaleler
    - Konferans sunumlari
    - Tez calismalari
    - Teknik raporlar
    """
    print("=" * 80)
    print("5. YAYIN KALITESINDE FIGUR OLUSTURMA")
    print("=" * 80)
    
    # Matplotlib parametrelerini guncelle
    # rcParams: Runtime configuration parameters
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })
    
    np.random.seed(42)
    
    # Gorsellesttirme
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    
    # Ana baslik
    fig.suptitle('Performance Comparison of Machine Learning Algorithms',
                fontsize=11, fontweight='bold')
    
    # Veri
    algorithms = ['KNN', 'SVM', 'RF', 'XGBoost']
    accuracy = [0.85, 0.88, 0.91, 0.93]
    precision = [0.83, 0.87, 0.90, 0.92]
    recall = [0.84, 0.86, 0.89, 0.91]
    f1_score = [0.835, 0.865, 0.895, 0.915]
    
    # (a) Accuracy Comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, accuracy, color='#2E86AB', alpha=0.8, 
                 edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.set_ylim([0.8, 0.95])
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    # loc='left': Basligi sola hizala
    ax.set_title('(a) Accuracy', loc='left')
    
    # Bar uzerine deger yaz
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.005, 
               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # (b) Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(precision, recall, s=100, alpha=0.7, 
              edgecolors='black', linewidth=0.8)
    
    # Etiketler ekle
    for i, alg in enumerate(algorithms):
        ax.annotate(alg, (precision[i], recall[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title('(b) Precision vs Recall', loc='left')
    ax.set_xlim([0.82, 0.93])
    ax.set_ylim([0.83, 0.92])
    
    # (c) All Metrics Comparison
    ax = axes[1, 0]
    x = np.arange(len(algorithms))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
          color='#2E86AB', alpha=0.8)
    ax.bar(x - 0.5*width, precision, width, label='Precision', 
          color='#A23B72', alpha=0.8)
    ax.bar(x + 0.5*width, recall, width, label='Recall', 
          color='#F18F01', alpha=0.8)
    ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', 
          color='#C73E1D', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_title('(c) Comprehensive Metric Comparison', loc='left')
    ax.set_ylim([0.8, 0.95])
    
    # (d) Error Analysis
    ax = axes[1, 1]
    training_error = [0.15, 0.12, 0.09, 0.07]
    validation_error = [0.18, 0.14, 0.11, 0.09]
    
    ax.plot(algorithms, training_error, 'o-', label='Training Error', 
           linewidth=1.5, markersize=6)
    ax.plot(algorithms, validation_error, 's--', label='Validation Error', 
           linewidth=1.5, markersize=6)
    
    ax.set_ylabel('Error Rate')
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title('(d) Error Analysis', loc='left')
    ax.set_ylim([0.05, 0.20])
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/21_publication_ready.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # rcParams'i varsayilana don
    plt.rcParams.update(plt.rcParamsDefault)
    
    print("\nYayin kalitesinde figur basariyla olusturuldu")
    print("\nOzellikler:")
    print("  - 300 DPI cozunurluk")
    print("  - Times New Roman font")
    print("  - Subfigure labeling (a, b, c, d)")
    print("  - Profesyonel grid ve linewidth degerleri")
    print("  - IEEE/Academic standartlarinda")
    print("\nKullanim: Bilimsel makaleler, konferans sunumlari, tez calismalari")
    print()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Ana fonksiyon - tum ornekleri calistir
    """
    print("\n" + "="*80)
    print("DATA VISUALIZATION - ILERI SEVIYE EGITIMI")
    print("="*80 + "\n")
    
    # Tum fonksiyonlari calistir
    three_d_visualizations()
    network_graphs()
    advanced_statistical_plots()
    dashboard_layout()
    publication_ready_figures()
    
    print("\n" + "="*80)
    print("TUM ILERI SEVIYE GORSELLESTIRMELER TAMAMLANDI")
    print("="*80)
    print(f"\nGrafikler su klasore kaydedildi: /mnt/user-data/outputs/")
    print("\nOGRENILEN KONULAR:")
    print("  - 3D Gorsellestirmeler (Scatter, Surface, Wireframe)")
    print("  - Network Graphs (Simple, Weighted, Directed, Community)")
    print("  - Ileri Istatistiksel Plotlar (Hexbin, Bootstrap, Dendrogram)")
    print("  - Dashboard Layout (KPI Cards, Multi-panel)")
    print("  - Yayin Kalitesinde Figurler (IEEE/Academic standartlari)")
    print("\nPORTFOLIO ICIN HAZIRSINIZ")
    print("\nEK ONERILER:")
    print("  - Plotly ile interaktif grafikler kesfedin")
    print("  - Matplotlib Animation modulunu inceleyin")
    print("  - Gercek projelerinizde bu teknikleri uygulayın")
    print("  - GitHub README'nize bu grafikleri ekleyin")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
