-- Find the name of the company that placed order 10290.
SELECT CompanyName
FROM Customers
WHERE CustomerID = (SELECT CustomerID
			FROM Orders
			WHERE OrderID = 10290);

-- Find the Companies that placed orders in 2016
SELECT CompanyName
FROM Customers
WHERE CustomerID IN (SELECT CustomerID
			FROM Orders
			WHERE OrderDate BETWEEN '2016-01-01' AND '2016-12-31');