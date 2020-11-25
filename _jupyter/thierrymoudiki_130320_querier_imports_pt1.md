# Import data into the querier (Part 1)

We illustrate how to import data into [the querier](https://github.com/Techtonique/querier), a query language for Data Frames
. 

First, in [example 1](#example-1), we import data __from csv__. Then in [example 2](#example-2) we import data __from a [relational database](https://en.wikipedia.org/wiki/Relational_database)__ (sqlite3). 

We start by installing the package from GitHub (command line):


```
!pip install git+https://github.com/Techtonique/querier.git
```

## Example 1

Import data __from csv__, and chain the querier's operations [`select`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_select.ipynb), [`filtr`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_filtr.ipynb), [`summarize`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_summarize.ipynb).


```
import pandas as pd
import querier as qr
import sqlite3 
import sys


# data -----

url = ('https://raw.github.com/pandas-dev'
   '/pandas/master/pandas/tests/data/tips.csv')


# Example 1 - Import from csv -----

qrobj1 = qr.Querier(source=url)

df1 = qrobj1\
.select(req="tip, sex, smoker, time")\
.filtr(req="smoker == 'No'")\
.summarize(req="sum(tip), sex, time", group_by="sex, time")\
.df

print(df1)
```

       sum_tip     sex    time
    0    88.28  Female  Dinner
    1    61.49  Female   Lunch
    2   243.17    Male  Dinner
    3    58.83    Male   Lunch


## Example 2

Import data __from an sqlite3 database__, and chain the querier's operations [`select`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_select.ipynb), [`filter`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_filtr.ipynb), [`summarize`](https://github.com/Techtonique/querier/blob/master/querier/demo/thierrymoudiki_231019_summarize.ipynb).


```
# Example 2 - Import from sqlite3 -----

# an sqlite3 database  connexion
con = sqlite3.connect('people.db')
 
with con:
    cur = con.cursor()    
    cur.execute("CREATE TABLE Population(id INTEGER PRIMARY KEY, name TEXT, age INT, sex TEXT)")
    cur.execute("INSERT INTO Population VALUES(NULL,'Michael',19, 'M')")
    cur.execute("INSERT INTO Population VALUES(NULL,'Sandy', 41, 'F')")
    cur.execute("INSERT INTO Population VALUES(NULL,'Betty', 34, 'F')")
    cur.execute("INSERT INTO Population VALUES(NULL,'Chuck', 12, 'M')")
    cur.execute("INSERT INTO Population VALUES(NULL,'Rich', 24, 'M')")
    
# create querier object from the sqlite3 database 
qrobj2 = qr.Querier(source='people.db', table="Population")    

# filter on people with age >= 20
df2 = qrobj2.select(req="name, age, sex").filtr(req="age >= 20").df

print("df2: ")
print(df2)
print("\n")

# avg. age for people with age >= 20, groupped by sex
qrobj3 = qr.Querier(source='people.db', table="Population")  
df3 = qrobj3.select(req="name, age, sex").filtr(req="age >= 20")\
.summarize("avg(age), sex", group_by="sex").df

print("df3: ")
print(df3)
print("\n")
```

    df2: 
        name  age sex
    1  Sandy   41   F
    2  Betty   34   F
    4   Rich   24   M
    
    
    df3: 
       avg_age sex
    0     37.5   F
    1     24.0   M
    
    

