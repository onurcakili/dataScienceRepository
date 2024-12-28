----------------------------------------------------------------------------------------------------------------------------
-- Data Science için SQL Proje Yönetim Süreci (RoadMap)
-- [MSSQL ile Uygulamalı Olarak Hazırlanmıştır.]
-- Bu repo uzmanlığımı pekiştirmek ve topluluğa kaynak oluşturmak amacıyla veri bilimi projelerinin SQL yönetim sürecini adım adım ele alır.
-- MSSQL kullanılarak bir proje içerisindeki tüm adımlar, yöntemler, temel rehberler ve örnek uygulamalar sunulmaktadır.

/* 

Bu süreç:
 
	1. Proje Tanımı ve Planlama
Bir projenin gerçek hayat senaryosu nedir?
Hangi iş problemini çözmek için bu projeyi yapıyoruz?
Verilerin genel tanımı ve hedefleri nelerdir?
Kullanılacak araçlar ve SQL'in proje içindeki rolü nedir?

	2. Veri Kaynakları ve Veri Toplama
Veri tabanı yapısını (tablolar, ilişkiler, veri türleri) ve örnek veri modellerini açıklar.
SQL ile veri tabanı bağlantısı nasıl kurulur?
Veri toplamak için gerçek hayatta kullanılan yöntemleri açıklar.

	3. Veri Manipülasyonu ve Dönüşümü
SQL ile verileri filtreleme, gruplama, birleştirme, dönüştürme ve temizleme işlemlerini detaylıca gösterir.
Örnek veri senaryoları üzerinde uygulamalı örnekler sunar. (örn. bir satış raporu oluşturma, eksik veri doldurma).

	4. Veri Analizi ve Görselleştirme
MSSQL ile KPI hesaplamaları nasıl yapılır?
SQL'den Power BI veya başka bir görselleştirme aracına veri aktarma sürecini açıklar.
Veri analizi için SQL sorgularıyla örnek uygulama.

	5. Makine Öğrenimi için Veri Hazırlığı
SQL ile veri seçimi ve veri seti oluşturma süreçlerini göster.
Modelleme için SQL'de veri normalizasyonu, öznitelik mühendisliği gibi konuları açıklar.

	6. Raporlama ve Sonuçların Sunumu
SQL'de prosedürler ve raporlama araçlarıyla sonuçların sunulması.
Gerçek iş hayatında kullanabileceğim SQL tabanlı bir rapor örneği oluştur.

	7. En İyi Uygulamalar ve Optimizasyon
SQL sorgularında performans optimizasyonu (ör. indeksleme, sorgu optimizasyonu).
Veri güvenliği ve erişim kontrolleri.


*/


-- Repo, veri bilimci adaylarına ve profesyonellere SQL projelerini sistematik bir şekilde yönetmeleri için yol gösterecek bir kaynak sunmayı hedeflemektedir.

-- İyi Çalışmalar,
-- Onur

----------------------------------------------------------------------------------------------------------------------------
/* 

1. Proje Tanımı ve Planlama
Projenin başında çözmeye çalıştığın iş problemini ve veri setini netleştirmek çok önemli.
Örneğin, bir perakende mağazasının satış verilerini analiz etmeyi amaçladığımızı düşünürsek

Örnek Senaryo:
Bir perakende mağazası, satışları üzerinden en çok tercih edilen ürünlerini belirlemek ve müşteri segmentasyonu 
yapmak istiyor. Bunun için geçmiş verileri (satış, ürünler, müşteriler) dayanarak bir analiz yapılması gerekiyor.

Hedefler:
 * En çok satılan ürünleri belirlemek,
 * Müşteri segmentasyonları oluşturmak (örn. yaş, gelir gibi parametrelere göre)
 * Satışları etkileyen diğer faktörler belirlenmek isteniyor.
 
** Kullanılacak Araçlar ve SQL'in Projedeki Rolü ** 
Veri analizi için SQL kullanarak veri manipülasyonu, analiz ve raporlama işlemleri yapılacak.
SQL özellikle veri sorgulama, filtreleme, ve gruplandırma işlemlerinde kullanılacaktır.
Veri Görselleştirme için ise Power BI veya Excel gibi araçlar devreye girebilir.
*/

----------------------------------------------------------------------------------------------------------------------------
/* 

2. Veri Kaynakları ve Veri Toplama

Veri Tabanı Yapısı : Veri tabanındaki tabloları, ilişkileri, veri türlerini tanımlamak veri toplama sürecine başlamak için gereklidir.
Bu, verilerin düzenli ve anlamlı hale gelmesini sağlar.

Örnek Veri Tabanı Yapısı : 

* Customers : Müşteri bilgileri (ID, Ad, Soyad, Yaş, Cinsiyet, Gelir, vb.)
* Products : Ürün bilgileri (ID, Ürün Adı, Fiyat, Kategori, vb.)
* Sales : Satış bilgileri (Satış ID, Müşteri ID, Ürün ID, Satış Tutarı, Satış Tarihi)

Veri Toplama ve SQL Bağlantısı  : Veri toplama için önce veri tabanına bağlanılır.
Örneğin, MSSQL kullanıyorsak, SQL Server Management Studio (SSMS) veya bir uygulama örn. Python ile de yapabiliriz.

Biz ROADMAP için kendimiz bir ECommerce DB create edeceğiz ve sorguları buradan değerlendireceğiz.
*/


-- Yeni bir veritabanı oluşturuyoruz
-- CREATE DATABASE ECommerceDB

-- 2.1 Customers tablosu Oluşturmak
CREATE TABLE Customers 
(
    CustomerID INT PRIMARY KEY IDENTITY(1,1),    -- Müşteri ID, otomatik artan
    FirstName VARCHAR(50),                       -- Müşterinin adı
    LastName VARCHAR(50),                        -- Müşterinin soyadı
    Age INT,                                     -- Müşterinin yaşı
    Gender VARCHAR(10),                          -- Müşterinin cinsiyeti
    Income DECIMAL(10, 2),                       -- Müşterinin yıllık geliri
    Email VARCHAR(100),                          -- Müşterinin e-posta adresi
    Phone VARCHAR(20),                           -- Müşterinin telefon numarası
    Address VARCHAR(200)                         -- Müşterinin adresi
)

-- 2.1.1 Products Tablosu Oluşturmak
CREATE TABLE Products 
(
    ProductID INT PRIMARY KEY IDENTITY(1,1),      -- Ürün ID, otomatik artan
    ProductName VARCHAR(100),                     -- Ürünün adı
    Price DECIMAL(10, 2),                         -- Ürünün fiyatı
    Category VARCHAR(50),                         -- Ürün kategorisi (Elektronik, Giyim vb.)
    StockQuantity INT,                            -- Ürünün mevcut stok durumu
    Description TEXT    --varchar(max)            -- Ürünün açıklaması
)

-- 2.1.2 Sales Tablosu Oluşturmak
-- Sales tablosu Products ve Customers tablosuyla ilişkilendirilecektir. 
CREATE TABLE Sales 
(
    SaleID INT PRIMARY KEY IDENTITY(1,1),         -- Satış ID, otomatik artan
    CustomerID INT,                               -- Satışı yapan müşteri ID'si
    ProductID INT,                                -- Satılan ürün ID'si
    Quantity INT,                                 -- Satılan ürün adedi
    SaleAmount DECIMAL(10, 2),                    -- Satış tutarı
    SaleDate DATETIME,                            -- Satış tarihi ve saati ile birlikte
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),  -- Customers tablosuna ilişki
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)       -- Products tablosuna ilişki
)

CREATE TABLE Orders (
    OrderID INT IDENTITY(1,1) PRIMARY KEY,    -- Unique OrderID
    CustomerID INT NOT NULL,                   -- Müşteri kimliği
    OrderDate DATETIME NOT NULL,               -- Siparişin verildiği tarih
    TotalAmount DECIMAL(10, 2) NOT NULL,       -- Toplam tutar
    ShippingAddress NVARCHAR(255) NOT NULL,    -- Teslimat adresi
    OrderStatus NVARCHAR(50) NOT NULL,         -- Sipariş durumu
    PaymentMethod NVARCHAR(50) NOT NULL,       -- Ödeme yöntemi
    LastUpdated DATETIME DEFAULT GETDATE(),    -- Güncelleme tarihi (varsayılan olarak şu an)
    ShippingDate DATETIME NULL,                -- Gönderim tarihi
    ProductID INT NOT NULL,                    -- Siparişte yer alan ürün kimliği
    Quantity INT NOT NULL,                     -- Ürün adedi
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),  -- Müşteri tablosuna ilişki
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)      -- Ürün tablosuna ilişki
)


-- 2.2 Customers tablosuna veri ekleme işlemi
INSERT INTO Customers (FirstName, LastName, Age, Gender, Income, Email, Phone, Address)
VALUES 
('Ahmet', 'Yılmaz', 30, 'Erkek', 50000.00, 'ahmet@example.com', '5551234567', 'İstanbul, Türkiye'),
('Ayşe', 'Kaya', 28, 'Kadın', 45000.00, 'ayse@example.com', '5552345678', 'Ankara, Türkiye'),
('Mustafa', 'Çelik', 50, 'Erkek', 75000.00, 'mustafa@example.com', '5557845678', 'Hatay, Türkiye'),
('Mehmet', 'Demir', 35, 'Erkek', 60000.00, 'mehmet@example.com', '5553456789', 'İzmir, Türkiye')



-- 2.2.1 Products tablosuna veri ekleme işlemi
INSERT INTO Products (ProductName, Price, Category, StockQuantity, Description)
VALUES 
('Smartphone', 1500.00, 'Elektronik', 50, 'Yüksek çözünürlüklü ekran, 64GB depolama alanı'),
('T-shirt', 100.00, 'Giyim', 200, 'Pamuklu, rahat kullanım'),
('Laptop', 3500.00, 'Elektronik', 30, 'Intel i7 işlemci, 16GB RAM, 512GB SSD')



-- 2.2.2 Satış tablosuna veri ekleme işlemi
INSERT INTO Sales (CustomerID, ProductID, Quantity, SaleAmount, SaleDate)
VALUES 
(1, 1, 1, 1500.00, '2024-12-25 10:30:00'),
(2, 2, 2, 200.00, '2024-12-26 14:15:00'),
(3, 3, 1, 3500.00, '2024-12-27 16:00:00')



-- 2.2.3 Orders tablosuna veri ekleme işlemi
INSERT INTO Orders (CustomerID, OrderDate, TotalAmount, ShippingAddress, OrderStatus, PaymentMethod, ShippingDate, ProductID, Quantity)
VALUES
(1, '2024-12-26 10:00:00', 5000.00, 'İstanbul, Türkiye', 'Beklemede', 'Kredi Kartı', '2024-12-26 09:00:00', 1, 1),
(2, '2024-12-25 14:30:00', 1500.00, 'Ankara, Türkiye', 'Tamamlandı', 'Havale', '2024-12-26 09:00:00', 2, 1), 
(1, '2024-12-26 12:00:00', 100.00, 'İstanbul, Türkiye', 'Tamamlandı', 'Kredi Kartı', '2024-12-27 10:00:00', 3, 2)

-- 2.3 Customers Tablosundaki Verileri Görüntüleme
SELECT * FROM Customers;


-- 2.3.1 Products Tablosundaki Verileri Görüntüleme
SELECT * FROM Products;


-- 2.3.2 Sales Tablosundaki Verileri Görüntüleme
SELECT * FROM Sales;

-- 2.3.3 Orders Tablosundaki Verileri Görüntüleme
SELECT * FROM Orders;


----------------------------------------------------------------------------------------------------------------------------
/* 

3. Veri Manipülasyonu ve Dönüşümü

Veri Manipülasyonu ve dönüşümü, ham veriyi analiz ve raporlama için uygun bir formata dönüştürme sürecidir.
SQL, bu süreçte güçlü araçlar sunar. Veri Manipülasyonu genellikle veri temizleme, format dönüştürme, birleştirme,
özetleme ve türetilmiş sütunlar oluşturma gibi adımları içerir.
*/

-- 3.1 Veri Temizleme
-- Ham veride eksik, tutarsız veya hatalı bilgiler olabilir. SQL ile veri temizleme işlemi yapılabilir.

-- Eksik Değerleri Kontrol Etmek
Select * FROM Customers
WHERE Age IS NULL

-- Eksik Değerleri Güncellemek
UPDATE Customers
SET AGE = 30
WHERE AGE IS NULL


-- Eksik Değerleri Ortalama ile Güncellemek
UPDATE Customers
Set AGE = (SELECT AVG(AGE) FROM Customers WHERE AGE IS NOT NULL) -- NULL olmayan değerlerden ort. hesaplamak
WHERE AGE IS NULL

-- Tutarsız Verilerin Düzeltilmesi
UPDATE Customers
SET GENDER = 'Male'
WHERE GENDER = 'Erkek'


-- 3.2 Veri Formatlama Sorguları

 -- firstName değerlerini büyük harf yapar
Select UPPER(firstname) AS Upper_FirstName
FROM Customers

 -- firstName değerlerini sadece baş harfini büyük harf yapar
SELECT UPPER(LEFT(firstname, 1)) + LOWER(SUBSTRING(firstname, 2, LEN(firstname))) AS Capitalized_FirstName
FROM Customers

-- firstName değerlerini küçük harf yapar
Select LOWER(firstname) AS Upper_FirstName 
FROM Customers

-- iki column'u birleştirerek getirir
SELECT CONCAT(firstname, ' ', lastname) as fullName
FROM Customers

-- date degiskenini varchar'a dönüştürür
SELECT 
    CONVERT(VARCHAR, SaleDate, 101) AS US_Format,   -- 101 US Format (mm/dd/yyyy)
    CONVERT(VARCHAR, SaleDate, 104) AS German_Format -- 104 German Format (DD.MM.YYY)
FROM Sales;

-- CAST fonks. da dönüşüm içindir fakat özelleştirme desteği yoktur.
SELECT CAST(SaleDate AS VARCHAR) AS DateAsText
FROM Sales;

-- Formatlama, özellikle tarih, para birimi veya sayısal değerler için kullanılır.
SELECT 
    FORMAT(saleamount, 'C', 'tr-TR') AS FormattedAmount
FROM Sales;

-- 3.3 Veri Birleştirme 
-- Veri Birleştirme, SQL'de farklı tablolarda saklanan ilişkili verileri, anahtar sütunlar kullanarak bir araya getirme işlemidir. 
-- Bu işlem genellikle JOIN ifadeleri ile gerçekleştirilir ve veri analizi, raporlama, veri temizliği gibi birçok senaryoda kullanılır.

-- INNER JOIN, sadece eşleşen kayıtları döndürür. 
-- Örneğin hangi müşterilerin sipariş verdiğini listelemek
SELECT
	Customers.FirstName,
	Orders.OrderID,
	Orders.OrderDate
FROM Customers
INNER JOIN Orders on Customers.CustomerID = Orders.CustomerID  -- JOIN'in default değeri INNER JOIN

-- firstName ve lastName birleştirerek inner join işlemi
SELECT
	CONCAT(Customers.FirstName, ' ', Customers.LastName) AS fullName,
	Orders.OrderID,
	Orders.OrderDate
FROM Customers
INNER JOIN Orders on Customers.CustomerID = Orders.CustomerID 


-- LEFT JOIN, soldaki tablodaki tüm kayıtları, sağdaki tablodaki eşleşen kayıtları döndürür
-- Eşleşme yoksa sağdakı sütunlar NULL döner
-- Örn. sipariş veren - sipariş vermeyen tüm müşterileri listelemek

Select 
	Customers.FirstName,
    Orders.OrderID,
    Orders.OrderDate
From Customers
LEFT JOIN Orders on Customers.CustomerID = Orders.CustomerID


-- RIGHT JOIN, sağdaki tablodaki tüm kayıtları, soldaki tablodaki eşleşen kayıtları döndürür.
-- Eşleşme yoksa soldaki sütunlar NULL döner
-- Örn. tüm siparişleri ve sipariş veren/vermeyen müşterileri listelemek
SELECT
	Customers.FirstName,
    Orders.OrderID,
    Orders.OrderDate
FROM Customers
RIGHT JOIN Orders on Customers.CustomerID = Orders.CustomerID


-- FULL JOIN, her iki tablodaki tüm kayıtları döndürür. Eşleşme olmayan kayıtları NULL getirir 
SELECT
	Customers.FirstName,
    Orders.OrderID,
    Orders.orderdate
FROM Customers
FULL JOIN Orders on Customers.CustomerID = Orders.CustomerID

-- NULL Değerler: LEFT, RIGHT ve FULL JOIN kullanıldığında
-- Eşleşmeyen kayıtlar NULL döner. 
-- Tutulan table'daki tüm kayıtlar gelirken, ikinci table'daki null kayıtlar döner
-- Bu durumun raporlamada göz önünde bulundurulması gerekir.


-- NULL değerlerden kurtulmanın en basit çözümü WHERE Komutu ile onları elemektir.
SELECT 
    Customers.FirstName, 
    Orders.OrderID, 
    Orders.TotalAmount
FROM Customers
LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID
WHERE Orders.OrderID IS NOT NULL	-- NULL DEĞİLSE (2. table için)



-- CROSS JOIN 
-- Her iki tablodaki tüm kayıtları döndürür (kartesyen çarpımı)
-- Örn. Tüm ürünlerin satış sayıları ile eşleştirilmesi
SELECT 
    Products.ProductName, 
    Sales.Quantity
FROM Products
CROSS JOIN Sales;


-- 3.4 Veri Özetleme
-- Verilerin genel özelliklerini daha kolay anlayabilmek için belirli istatistiksel ve analizsel yöntemler kullanılır.
-- SQL'de veri özetleme genellikle gruplama, filtreleme merkezi dağılım ölçüleriyle gerçekleştirilir.

-- Toplam Satış Miktarını Hesaplamak
SELECT SUM(saleamount) as Total_Sales
FROM Sales

-- Ürün bazında satış miktarını hesaplamak
SELECT productid, SUM(saleamount) AS Total_Amount
FROM Sales
GROUP BY productid	-- aggregate fonks.(SUM(saleamount)) kullanıldığı için GROUP BY gereklidir.


SELECT S.productid, P.productname, SUM(S.saleamount) AS Total_Amount
FROM Sales S
JOIN Products P ON S.productid = P.productid
GROUP BY S.ProductID, P.productname

-- Ürün bazında 1000 TL ve üzerinde olan ürünlerin satış miktarını hesaplamak
SELECT productid, SUM(saleamount) AS Total_Sales
FROM Sales
GROUP BY productid
HAVING SUM(saleamount) > 1000;

-- Ürün Bazında Ortalama Satış Tutarını Hesaplamak
SELECT productid, AVG(saleamount) AS Total_Amount
FROM Sales
GROUP BY productid;

-- Her Bir Tarih İçin Satış Sayısını Hesaplamak
SELECT saledate, COUNT(quantity) AS Sale_Quantity
FROM Sales
GROUP BY saledate



-- 3.5 Türetilmiş Sütunlar Oluşturmak
-- Mevcut verilerden yeni bilgiler türetmek için amacıyla oluşturulurlar.
-- Bu tür sütunlar genellikle hesaplama veya birleştirme işleminde kullanılır.

-- Müşteri Gelir Durumunu Kategorize Etme

Select
	customerid,
    firstname,
    lastname,
    income,
    
    CASE
    	WHEN income <= 45000.00 THEN 'Düşük'
        WHEN income BETWEEN 50000.00 AND 60000.00 THEN 'Orta'
        ELSE 'Yüksek'
    END as IncomeCategory
    
from Customers


-- Satış tutarından indirimli fiyat hesaplamak
-- Satış tablosunda her ürün için %10 indirim uygulanmış bir satış tutarı üretmek istiyoruz

select saleid, 
	   productid,
       saleamount,
       saleamount * 0.9 as DiscountedAmount
From Sales


-- Ürün Satış Performansı, her ürün için toplam satış miktarı ve toplam kazanç bilgisi türetmek istersek
SELECT 
    p.ProductID,
    p.ProductName,
    SUM(s.Quantity) AS TotalQuantitySold,
    SUM(s.SaleAmount) AS TotalSales
FROM  Products p
LEFT JOIN Sales s ON p.ProductID = s.ProductID
GROUP BY p.ProductID, p.ProductName
ORDER By SUM(s.SaleAmount) DESC
    
   
 
-- Siparişlerin teslim süresi
-- Her siparişin teslim süresini (gün cinsinden) hesaplayarak türetilmiş bir sütun oluşturacağız.
SELECT 
    OrderID as UrunID,
    OrderDate AS SiparisTarihi,
    ShippingDate AS GonderimTarihi,
    DATEDIFF(DAY, OrderDate, ShippingDate) AS TeslimatSuresi
FROM Orders


-- En çok harcama yapan müşteri
SELECT 
    c.CustomerID,
    CONCAT(c.FirstName, ' ', c.LastName) AS FullName,
    SUM(s.SaleAmount) AS TotalSpent
FROM Customers c
JOIN Sales s ON c.CustomerID = s.CustomerID
GROUP BY c.CustomerID, c.FirstName, c.LastName
ORDER BY TotalSpent DESC;


----------------------------------------------------------------------------------------------------------------------------

-- 4. SQL ile Veri Analizi 
-- Veri Analizi verileri gruplandırmak, filtrelemek ve özetlemek için kullanılan işlemlerden oluşur.

-- Müşteri bazında satış tutarlarını hesaplamak
Select
	c.customerID,
    c.FirstName,
    c.LastName,
    SUM(s.SaleAmount) as Total_Spent
From Sales s
INNER JOIN Customers c on s.CustomerID = c.CustomerID
GROUP BY c.CustomerID, c.FirstName, c.LastName

-- Ürün kategorisine göre satış tutarı
Select
	p.Category,
    SUM(s.SaleAmount) as TotalSales
From Sales s
INNER JOIN Products p on s.ProductID = p.ProductID
GROUP BY p.Category


-- Satış tarihlerine göre satış miktarlarını hesaplamak
Select
	CONVERT(VARCHAR, SaleDate, 23) as SaleDate,  -- DATETIME YYYY-MM-DD formatına dönüştürüldü
    SUM(SaleAmount) as TotalSales
FROM Sales
GROUP BY CONVERT(VARCHAR, SaleDate, 23) -- Aggregation kullanıldığı için group by
ORDER BY SaleDate


-- Müşteri Yaş Gruplarına Göre Satış Dağılımı
SELECT 
    CASE 
        WHEN Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN Age BETWEEN 36 AND 45 THEN '36-45'
        ELSE '46+' 
    END AS AgeGroup,
    SUM(s.SaleAmount) AS TotalSales
FROM Sales s
INNER JOIN Customers c ON s.CustomerID = c.CustomerID
GROUP BY 
    CASE 
        WHEN Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN Age BETWEEN 36 AND 45 THEN '36-45'
        ELSE '46+' 
    END

 ----------------------------------------------------------------------------------------------------------------------------       
-- 5. Makine Öğrenmesi için Veri Hazırlığı

/*

Veri hazırlığı, makine öğrenmesi projelerinin en kritik adımlarındandır.
İyi bir veri hazırlığı süreci modelin başarısını doğrudan etkiler. 
Veri Temizliği, future engineering, dönüştürme ve verilerin uygun formata getirilmesi çok önemlidir.

 */
        

-- 5.1 Veri Temizleme (Data Cleaning)

-- SQL ile Eksik Verilerin Tespiti
SELECT * From Customers
WHERE Email IS NULL


-- SQL ile Eksik Verileri Silmek
DELETE FROM Customers
Where Email IS NULL

-- 5.2 Eksik Verilerle İşlemler 
-- SQL ile Eksik Verileri Doldurmak
Update Customers
Set income = (SELECT AVG(income) from Customers)
Where income IS NULL

-- 5.3 Aykırı Değerlerle İşlemler 
-- SQL ile Aykırı Değer Tespiti

-- Ortak Sorgular için Geçici Tablo (CTE (Common Table Expression))
WITH Quartiles AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY SaleAmount) OVER () AS Q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY SaleAmount) OVER () AS Q3
    FROM Sales
),
Bounds AS (
    SELECT 
        MAX(Q1) AS Q1, 
        MAX(Q3) AS Q3, 
        MAX(Q3) - MAX(Q1) AS IQR,
        GREATEST(MAX(Q1) - 1.5 * (MAX(Q3) - MAX(Q1)), 0) AS LowerBound,   -- Alt sınır (0'dan küçük olamaz)
        MAX(Q3) + 1.5 * (MAX(Q3) - MAX(Q1)) AS UpperBound                 -- Üst sınır
    FROM Quartiles
)
SELECT 
    S.SaleAmount,
    B.LowerBound,
    B.UpperBound,
    CASE
        WHEN S.SaleAmount < B.LowerBound THEN 'Alt Aykırı'
        WHEN S.SaleAmount > B.UpperBound THEN 'Üst Aykırı'
        ELSE 'Normal'
    END AS Durum
FROM Sales S
CROSS JOIN Bounds B;



-- SQL ile Çift Kayıtların (duplicate records) silinmesi
SELECT firstname, lastname, COUNT(*)	-- çift kayıtlar DELETE yapılarak silinebilir.
FROM Customers
GROUP BY firstname, lastname
HAVING COUNT(*) > 1  

-- 5.4 Veri Dönüşümleri 
-- Kategorik verilerin sayısal veriye dönüştürülmesi
SELECT

	CASE
    	WHEN Gender = 'Male' THEN 0
        WHEN Gender = 'Female' THEN 1
        ELSE 'Belirtilmedi'
	END as Gender,
    firstname, lastname
FROM Customers

-- SQL ile min-max normalizasyon işlemi
-- (MAAŞ - MİN(MAAS)) / (MAX(MAAS) - Min(MAAS))  >>>> Maaş değr. [0,1] aralığına çekecek

SELECT
	(income - (SELECT min(income) from Customers)) 
    /
    ((SELECT max(income) from Customers) - (SELECT min(income) from Customers)) 
    as NormalizedIncome
FROM Customers


-- Müşterinin son satın alma tarihinden itibaren geçen süre
SELECT
    firstname,
    lastname,
    datediff(day, MAX(saledate), GETDATE()) as DaysSinceLastPurchase
FROM Sales s
INNER JOIN Customers c on c.CustomerID = s.CustomerID 
GROUP BY firstname, lastname

----------------------------------------------------------------------------------------------------------------------------
















 
























