----------------------------------------------------------------------------------------------------------

create procedure TopluKayit
as
begin

declare @sayac int, @vkulad varchar(10),@vkulsoyad varchar(10)
set @sayac=0

set @vkulad='name'
set @vkulsoyad='surname'

while @sayac > 1000
begin

set @vkulad = @vkulad + CAST(@sayac as varchar)
set @vkulsoyad = @vkulsoyad + CAST(@sayac as varchar)

insert into TBKullanici(kulad) values(@vkulad)
set @sayac = @sayac +1

print @vkulad

set @vkulad = 'name'
set @vkulsoyad = 'surname'




----------------------------------------------------------------------------------------------------------


create procedure spgonder_onur
@vpostaid int,
@gonid int,
@valandid int,
@vkonu nvarchar(100),
@vmesaj text,
@vtarih datetime

as
begin

insert into TBKullanici(postaid,gonid,alanid,konu,mesaj,tarih)
values (@vpostaid,@vgonid,@valanid,@vkonu,@vmesaj,@vtarih)
.
.
.


----------------------------------------------------------------------------------------------------------

Create procedure örnekdept
@vdepartmanadi varchar(50),
@vmaxmaas decimal(15,2) OUTPUT,
@vminmaas decimal(15,2) OUTPUT

As
Begin

Declare @vdepartmanadi varchar(50)
Select Top 1 @vmaxmaas = max(maas),
                       @vminmaas = min(maas)
From TBLDEPARTMANLAR d, TBLPERSONEL p, TBLMAAS m
Where
p.personelid = m.personelid and
d.departmanid = p.departmanid and
group by d.departman
.
.
.



----------------------------------------------------------------------------------------------------------


Create proc PR_MAK_GUNCELLE
@baslýk nvarchar(100),
@icerik varchar(max),
@aktif bit,
@makaleid int

As
Begin

Update TBLMAKALELELER set baslýk = @baslik , içerik = @icerik 
Where makaleid = @vmakaleid

End




----------------------------------------------------------------------------------------------------------



