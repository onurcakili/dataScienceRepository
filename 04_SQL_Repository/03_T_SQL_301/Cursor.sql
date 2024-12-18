--Cursor, bir tablonun i�indeki verilere i�lem yapmak i�in kullan�l�r
-- �rn. Select,update,insert,delete.
-- Peki neden di�er komutlar dururken Cursor kullan�yoruz?
--��nk� Cursor, tablo i�indeki Her sat�r� tek tek inceleyebilme f�rsat� sunuyor

-- 1. ADIM �ZER�NDE ��LEM YAPACAGIMIZ S�TUNLARI G�R�YORUZ (B�ZDEN �STENENLER�)

Declare @vadi nvarchar(10), @vsoyadi nvarchar(20), @vunvan nvarchar(30)

-- 2. Ad�m Cursoru tan�ml�yoruz (olu�turuyoruz)
--Declare (Cursore vermek istedgmz isim) CURSOR

Declare PersonelCursor CURSOR
FOR -- FOR (i�in) demektir. Bu cursor ne i�in? 1. Ad�mda tan�mlad�g�m�z degiskenler i�in

-- 3. ad�m yani FOR'un cevab�n� veriyoruz burada

Select Adi,SoyAdi,Unvan from Personeller

--Cursorun t�m ihtiyaclar�n� kar��lad�k. ona ne yapmas� gerekti�ini ��rettik
--�imdi Cursoru �al��t�ral�m
Open PersonelCursor

--Cursor i�in ne demi�tik? Sat�r sat�r veriyi inceleme f�rsat� verirdi bize
--Bu y�zden sat�rlar aras�nda ge�i� yapmak i�in next(sonraki) deyimini tan�ml�yoruz
--Bunun aksi yani geriye olan� da mevcut ama onu simdilik kullanmayacag�z

FETCH NEXT FROM PersonelCursor into @vadi,@vsoyadi,@vunvan
-- ��lem ba�ar�l� ise @@FETCH_STATUS de�i�ken de�eri '0' ise i�lem ba�ar�l�d�r ve bir sonraki kay�t var demektir.
--Bu y�zden bunu tan�mlamam�z gerek 

while @@FETCH_STATUS = 0
-- Art�k t�m ad�mlar bitti ve Cursor'un ne yapmas� gerektigini belirtliyoruz. (Select, insort, update t�m�n� yapabiliriz)
--Biz personel ad� ve soy ad�n� falan yazd�racag�z
begin

print @vadi+ ' ' + @vsoyadi+ ' ' + @vunvan 
	FETCH NEXT FROM PersonelCursor into @vadi,@vsoyadi,@vunvan -- �ST SATIRDA YAZDIGIMIZIN AYNISINI YAZIYORUZ
	-- S�radaki veriye ge�ilir ve tekrardan cursor �zerinden kolon de�erleri ilgili de�i�kenlere atan�r.
	--Bunu yapma amac�m�z While d�ng�s�n�n devaml� olabilmesi i�in

END -- i�imiz bitti
CLOSE PersonelCursor
Deallocate PersonelCursor 
--Deallocate (serbest b�rak demektir)
--bunlar� Ram'den silebilirsin art�k demek 