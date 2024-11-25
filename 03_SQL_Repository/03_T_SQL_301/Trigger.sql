alter trigger trginsert17
on [Satis Detaylari]
after insert
as
begin
set nocount on;
declare @vurunid int, @vbirimfiyat  float,@vsatisid int

select @vurunid=UrunID,@vsatisid=SatisID from inserted
select * from Urunler where UrunID=@vurunid

select @vbirimfiyat=BirimFiyati from Urunler where UrunID=@vurunid
update [Satis Detaylari] set BirimFiyati=@vbirimfiyat where SatisID=@vsatisid

end
go
