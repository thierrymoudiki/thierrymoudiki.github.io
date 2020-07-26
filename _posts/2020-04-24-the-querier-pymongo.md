---
layout: post
title: "Import data from a NoSQL database using the querier"
description: Import data from a NoSQL database using the querier
date: 2020-04-15
categories: [DataBases, Python]
---

NoSQL are alternatives to SQL databases (are they?). Personally, I like both; each has their strengths and weaknesses. 

For example, __I'll go for a SQL database__, if I want some kind of _semantics_ in the way my tables are organized. And conversely, __I'd go for a NoSQL database__ for the relative ease of use, and if I wanted a rapid prototyping/schema of my tables. 


The [querier](https://thierrymoudiki.github.io/blog/#DataBases) implements a query language for Data Frames. A __documentation for the tool__ is available on [readthedocs](https://querier.readthedocs.io/en/latest/). This post is about importing data from a NoSQL database, here MongoDB, and using the querier.

![image-title-here]({{base}}/images/2020-04-03/2020-04-03-image1.png){:class="img-responsive"}

 I use fake customer data from this [excellent tutorial](https://www.w3schools.com/python/python_mongodb_getstarted.asp), and create a MongoDB database using a Python driver called `pymongo`.

```python
import pymongo

mongodb_uri = "mongodb://localhost:27017/"
myclient = pymongo.MongoClient(mongodb_uri)
mydb = myclient["mydatabase"]
mycol = mydb["customers"]

mylist = [
  { "name": "Amy", "address": "Apple st 652", "age":"32"},
  { "name": "Hannah", "address": "Mountain 21", "age":"35"},
  { "name": "Michael", "address": "Valley 345", "age":"24"},
  { "name": "Sandy", "address": "Ocean blvd 2", "age":"36"},
  { "name": "Betty", "address": "Green Grass 1", "age":"20"},
  { "name": "Richard", "address": "Sky st 331", "age":"22"},
  { "name": "Susan", "address": "One way 98", "age":"16"},
  { "name": "Vicky", "address": "Yellow Garden 2", "age":"42"},
  { "name": "Ben", "address": "Park Lane 38", "age":"52"},
  { "name": "William", "address": "Central st 954", "age":"42"},
  { "name": "Chuck", "address": "Main Road 989", "age":"32"},
  { "name": "Viola", "address": "Sideway 1633", "age":"62"}
]

x = mycol.insert_many(mylist)

# print list of the _id values of the inserted documents:

print(x.inserted_ids) 
```


The `querier` is proud to (try to) do only [a few things](https://thierrymoudiki.github.io/blog/#DataBases) well. So, if you want to carry out a very specific operation on a MongoDB database, you should still use the  `pymongo` driver, or MongoDB itself. 

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!



    



