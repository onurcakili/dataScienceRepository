----    DİĞER SQL KOMUTLARI -----

-- DISTINCT KOMUTU
-- SELECT DISTINCE SUTUN1, SUTUN2.. FROM TABLE ŞEKLİNDE KULLANILIR
-- SORGU SONUCLARINI TEKİLLEŞTİREN SQL KOMUTUDUR

-- ŞEHİRLERİ TEKİL OLARAK GETİREN SORGU 
SELECT DISTINCT CITY FROM EMPLOYEES 

-- ISTANBUL İLÇELERİNİ TEKİL OLARAK GETİREN SORGU
SELECT DISTINCT CITY, TOWN FROM EMPLOYEES
WHERE CITY = 'ISTANBUL'

-- DEPARTMAN SAYISINI TEKİLLEŞTİREREK GETİREN SORGU
SELECT DISTINCT DEPARTMENT FROM EMPLOYEES


------ ORDER BY KOMUTU -----
-- SQL SORGUSUNU  BÜYÜKTEN KÜÇÜĞE VEYA KÜÇÜKTEN BÜYÜĞE SIRALAYAN KOMUTTUR
-- HEM SAYISAL VE HEM DE KATEGORİK DEGİSKENLERDE AYNI İŞLEVİ GÖRMEKTEDİR

-- ÇALIŞANLARI ALFABETİK OLARAK (A'DAN Z'YE) SIRALAYAN SQL KOMUTU
SELECT * FROM EMPLOYEES
ORDER BY FULLNAME

-- ÇALIŞANLARI Z'DEN A'YA SIRALAYAN SQL KOMUTU
SELECT * FROM EMPLOYEES
ORDER BY FULLNAME DESC

-- BİRDEN FAZLA SIRALAMA İŞLEMİ YAPAN SQL KOMUTU
-- İSİMLERE GÖRE ASCENDING(ARTAN), ŞEHİRLERE GÖRE DESCENDİNG(AZALAN)
SELECT * FROM EMPLOYEES
ORDER BY FULLNAME ASC, CITY DESC

-- ŞARTLI ORDER BY KOMUTU
-- İSTANBULDA YAŞAYANLARI A'DAN Z'YE İSİMLERİNE GÖRE SIRALAYAN SQL KOMUTU
SELECT * FROM EMPLOYEES
WHERE CITY = 'ISTANBUL'
ORDER BY FULLNAME


-- ORDER BY KOMUTU YAZARKEN SÜTUN İSİMLERİ YERİNE
-- SÜTUN NUMARALARI GİREREK AYNI İŞLEMİ YAPABİLİRİZ
-- 2 NUMARALI KOLON = FULLNAME KOLONU

SELECT * FROM EMPLOYEES
ORDER BY 2


				---TOP KOMUTU--- 
/*

SQL SORGUSU SONUCUNDA BELİRLİ BİR SAYIDA SONUÇ GETİRMESİNİ İSTEDİĞİMİZDE KULLANDIĞIMIZ SORGUDUR.
ÖRN. EN ÇOK SATANLAR veya EN ÇOK SATAN İLK 10

GENELLİKLE ORDER BY KOMUTU İLE BERABER KULLANILIR

*/

-- EMPLOYEES TABLOSUNDAKİ İLK 10 ÇALIŞAN
SELECT TOP 10 * FROM EMPLOYEES

SELECT TOP 10 * FROM EMPLOYEES
ORDER BY FULLNAME

SELECT TOP 50 * FROM EMPLOYEES
ORDER BY FULLNAME DESC

SELECT TOP 50 * FROM EMPLOYEES
WHERE CITY = 'ISTANBUL'
ORDER BY FULLNAME DESC


-- PERCENT İLE KULLANIMI

-- TOP 50 PERCENT, VERİ SETİNİN YARISINI GETİREN SORGUDUR
SELECT TOP 50 PERCENT * FROM EMPLOYEES

-- TOP 10 PERCENT, VERİ SETİNİN %10'UNU GETİREN SORGUDUR
SELECT TOP 10 PERCENT * FROM EMPLOYEES



-- ALIAS (TAKMA AD) KULLANIMI

SELECT 
E.ID AS CALISANNO,
E.FULLNAME AS ADSOYAD,
E.CITY AS IL,
E.TOWN AS ILCE
FROM EMPLOYEES AS E


-- ALIAS VE ORDER BY KULLANIMI

SELECT 
E.ID AS CALISANNO,
E.FULLNAME AS ADSOYAD,
E.CITY AS IL,
E.TOWN AS ILCE
FROM EMPLOYEES AS E
ORDER BY ADSOYAD




