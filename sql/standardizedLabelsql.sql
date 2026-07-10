UPDATE cic_typed
SET label = 'Web Attack - Brute Force'
WHERE label LIKE 'Web Attack%Brute Force%';

UPDATE cic_typed
SET label = 'Web Attack - XSS'
WHERE label LIKE 'Web Attack%XSS%';

UPDATE cic_typed
SET label = 'Web Attack - SQL Injection'
WHERE label LIKE 'Web Attack%Sql Injection%';