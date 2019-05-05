create table boris_classify
(
	filename varchar(1000)
	,[class] varchar(1000)
	,createdon datetime
	,createdby varchar(1000)
)
go

ALTER TABLE boris_classify ADD  DEFAULT (getdate()) FOR [CreatedOn]
go
