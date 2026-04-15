--1.
SELECT Customers.FirstName, Customers.LastName
FROM Customers
WHERE Customers.Country = "Brazil";

--2 

SELECT DISTINCT Employees.Title
FROM Employees;

SELECT Employees.FirstName, Employees.LastName
FROM Employees
WHERE Employees.Title = 'Sales Support Agent';

--3
SELECT Tracks.Name
FROM Tracks
    INNER JOIN  Albums ON Artists.ArtistId = Albums.ArtistId
    INNER JOIN  Artists ON Tracks.AlbumId = Albums.ArtistId
WHERE Artists.Name ='AC/DC';

--4
SELECT Customers.FirstName, Customers.LastName, Customers.CustomerId, Customers.Country
FROM Customers
WHERE Customers.Country != 'USA';

--5
SELECT Employees.FirstName, Employees.LastName, Employees.City, Employees.State, Employees.Country, Employees.Country
FROM Employees
WHERE Employees.Title = 'Sales Support Agent';

--6
SELECT DISTINCT Invoices.BillingCountry
FROM Invoices;

--7 empiezo a perderme y estamos en el 7. He tenido que mirarlo
SELECT Customers.State, COUNT(Customers.CustomerId) AS NumeroClientes
FROM Customers
WHERE Customers.Country = 'USA'
GROUP BY Customers.State;

--8
SELECT COUNT(Invoice_items.InvoiceLineId) AS NumeroArticulos
FROM Invoice_items
WHERE Invoice_items.InvoiceId = 37;

--9 CONTAR LAS DEL EJERCICIO 3
SELECT COUNT(Tracks.Name)
FROM Tracks
    INNER JOIN  Albums ON Artists.ArtistId = Albums.ArtistId
    INNER JOIN  Artists ON Tracks.AlbumId = Albums.ArtistId
WHERE Artists.Name ='AC/DC';

--10
SELECT Invoices.InvoiceId, COUNT(Invoice_items.InvoiceLineId) AS NumeroArticulos
FROM Invoice_items
    INNER JOIN Invoices ON Invoices.InvoiceId = Invoice_items.InvoiceId
GROUP BY Invoices.InvoiceId;

--11  
SELECT Invoices.BillingCountry, COUNT(Invoices.InvoiceId) AS NumeroFacturas
FROM Invoices
GROUP BY Invoices.BillingCountry;

--12
-- extraer el año
SELECT COUNT (Invoices.InvoiceId) AS NumeroFacturas
FROM Invoices
WHERE strftime('%Y', InvoiceDate) IN ('2009','2011');

--13
SELECT COUNT (Invoices.InvoiceId) AS NumeroFacturas
FROM Invoices
WHERE Invoices.InvoiceDate BETWEEN '2009-01-01' AND '2011-12-31';

--14
SELECT COUNT (Customers.CustomerId) AS NumeroClientesEspañaItalia
FROM Customers
WHERE Customers.Country IN ('Brazil','Spain');

--15
SELECT Tracks.Name 
FROM Tracks
WHERE Tracks.Name LIKE 'You%';

--SEGUNDA PARTE

--1
SELECT Customers.FirstName, Customers.LastName, Invoices.InvoiceId, Invoices.InvoiceDate, Invoices.BillingCountry
FROM Invoices
    INNER JOIN Customers ON Customers.CustomerId = Invoices.CustomerId
WHERE Customers.Country = 'Brazil';

--2
SELECT Invoices.InvoiceId, Employees.FirstName, Employees.LastName
FROM Customers
    INNER JOIN Invoices ON Invoices.CustomerId = Customers.CustomerId
    INNER JOIN Employees ON Employees.EmployeeId = Customers.SupportRepId; --estoy suponiendo que esta es la relacion

--3
-- Es lo que he interpretado pero sospecho que está mal 
--Obténelnombredelcliente,elpaís,elnombredelagenteyeltotal
SELECT  Customers.FirstName AS NombreCliente, Customers.LastName AS ApellidoCliente, Customers.Country, Employees.FirstName AS NombreAgente, Employees.LastName AS ApellidoAgente, SUM (Invoices.Total) AS Total
FROM Customers
    INNER JOIN Invoices ON Invoices.CustomerId = Customers.CustomerId
    INNER JOIN Employees ON Employees.EmployeeId = Customers.SupportRepId --estoy suponiendo que esta es la relacion
GROUP BY Customers.CustomerId;

--4 Obténcadaartículodelafacturaconelnombredelacanción ¿qué factura? No entiendo

--5 uestratodaslascancionesconsunombre,formato,álbumygénero

SELECT Tracks.Name, Media_Types.Name AS Tipo, Albums.Title AS Album, Tracks.GenreId
FROM Tracks
    INNER JOIN Albums ON Albums.AlbumId = Tracks.AlbumId
    INNER JOIN Media_Types ON Media_Types.MediaTypeId = Tracks.MediaTypeId;

--6
SELECT COUNT (Playlist_track.TrackId) AS NumeroCanciones, Playlists.Name
FROM Playlists
    INNER JOIN Playlist_track ON Playlist_track.PlaylistId=Playlists.PlaylistId
GROUP BY (Playlists.Name);

--7
SELECT Employees.EmployeeId, SUM(Invoices.Total) AS Ventas
FROM Invoices
    INNER JOIN Customers ON Customers.CustomerId=Invoices.CustomerId
    INNER JOIN Employees ON Employees.EmployeeId = Customers.SupportRepId
GROUP BY (Employees.EmployeeId);

SELECT Employees.EmployeeId, SUM(Invoices.Total) AS Ventas
FROM Invoices
    INNER JOIN Customers ON Customers.CustomerId=Invoices.CustomerId
    INNER JOIN Employees ON Employees.EmployeeId = Customers.SupportRepId
WHERE strftime('%Y', InvoiceDate) = '2009';

SELECT Artists.Name, SUM(Invoice_items.Quantity*Invoice_items.UnitPrice) AS Total
FROM Invoice_items
    INNER JOIN Tracks ON Invoice_items.TrackId = Tracks.TrackId
    INNER JOIN Albums ON Tracks.AlbumId = Albums.AlbumId
    INNER JOIN Artists ON Artists.ArtistId = Albums.ArtistId
GROUP BY (Artists.Name)

