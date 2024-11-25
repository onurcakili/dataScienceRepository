--Cursor, bir tablonun içindeki verilere iþlem yapmak için kullanýlýr
-- Örn. Select,update,insert,delete.
-- Peki neden diðer komutlar dururken Cursor kullanýyoruz?
--Çünkü Cursor, tablo içindeki Her satýrý tek tek inceleyebilme fýrsatý sunuyor

-- 1. ADIM ÜZERÝNDE ÝÞLEM YAPACAGIMIZ SÜTUNLARI GÝRÝYORUZ (BÝZDEN ÝSTENENLERÝ)

Declare @vadi nvarchar(10), @vsoyadi nvarchar(20), @vunvan nvarchar(30)

-- 2. Adým Cursoru tanýmlýyoruz (oluþturuyoruz)
--Declare (Cursore vermek istedgmz isim) CURSOR

Declare PersonelCursor CURSOR
FOR -- FOR (için) demektir. Bu cursor ne için? 1. Adýmda tanýmladýgýmýz degiskenler için

-- 3. adým yani FOR'un cevabýný veriyoruz burada

Select Adi,SoyAdi,Unvan from Personeller

--Cursorun tüm ihtiyaclarýný karþýladýk. ona ne yapmasý gerektiðini öðrettik
--Þimdi Cursoru çalýþtýralým
Open PersonelCursor

--Cursor için ne demiþtik? Satýr satýr veriyi inceleme fýrsatý verirdi bize
--Bu yüzden satýrlar arasýnda geçiþ yapmak için next(sonraki) deyimini tanýmlýyoruz
--Bunun aksi yani geriye olaný da mevcut ama onu simdilik kullanmayacagýz

FETCH NEXT FROM PersonelCursor into @vadi,@vsoyadi,@vunvan
-- Ýþlem baþarýlý ise @@FETCH_STATUS deðiþken deðeri '0' ise iþlem baþarýlýdýr ve bir sonraki kayýt var demektir.
--Bu yüzden bunu tanýmlamamýz gerek 

while @@FETCH_STATUS = 0
-- Artýk tüm adýmlar bitti ve Cursor'un ne yapmasý gerektigini belirtliyoruz. (Select, insort, update tümünü yapabiliriz)
--Biz personel adý ve soy adýný falan yazdýracagýz
begin

print @vadi+ ' ' + @vsoyadi+ ' ' + @vunvan 
	FETCH NEXT FROM PersonelCursor into @vadi,@vsoyadi,@vunvan -- ÜST SATIRDA YAZDIGIMIZIN AYNISINI YAZIYORUZ
	-- Sýradaki veriye geçilir ve tekrardan cursor üzerinden kolon deðerleri ilgili deðiþkenlere atanýr.
	--Bunu yapma amacýmýz While döngüsünün devamlý olabilmesi için

END -- iþimiz bitti
CLOSE PersonelCursor
Deallocate PersonelCursor 
--Deallocate (serbest býrak demektir)
--bunlarý Ram'den silebilirsin artýk demek 