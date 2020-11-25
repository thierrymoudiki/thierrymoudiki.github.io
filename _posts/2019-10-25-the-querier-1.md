---
layout: post
title: "Introducing the querier"
description: the querier, a query language for pandas Data Frames
date: 2019-10-25
categories: DataBases
---


Data Frames are a way to represent tabular data, that is widely used and useful for Statistical Learning. Basically, a Data Frame = Tabular data + Named columns, and there are different implementations of this data structure, notably in R, Python and Apache Spark. The [querier](https://github.com/Techtonique/querier) exposes a query language to retrieve data from Python `pandas` Data Frames, inspired from [SQL](https://en.wikipedia.org/wiki/SQL)'s relational databases querying. Currently, the `querier` can be installed from Github as:

```bash
pip install git+https://github.com/Techtonique/querier.git
```

There are __9 types of operations__ available in the `querier`, with no plan to extend that list much further (to maintain a relatively simple mental model). These verbs will look familiar to `dplyr` users, but the implementation (`numpy`, `pandas` and `SQLite3` are used) and functions' signatures are different: 


- `concat`: __concatenates 2 Data Frames__, either horizontally or vertically

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image1.png){:class="img-responsive"}


- `delete`: __deletes rows__ from a Data Frame based on given criteria

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image2.png){:class="img-responsive"}


- `drop`: __drops columns__ from a Data Frame

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image3.png){:class="img-responsive"}


- `filtr`: __filters rows__ of the Data Frame based on given criteria


![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image4.png){:class="img-responsive"}



- `join`: __joins 2 Data Frames__ based on given criteria (available for _completeness_ of the interface, this operation is already straightforward in pandas)


![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image5.png){:class="img-responsive"}


- `select`: __selects columns__ from the Data Frame

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image6.png){:class="img-responsive"}


- `summarize`: obtains __summaries of data__ based on grouping columns

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image7.png){:class="img-responsive"}


- `update`: __updates a column/creates a new column__, using an operation given by the user

![image-title-here]({{base}}/images/2019-10-25/2019-10-25-image8.png){:class="img-responsive"}


- `request`: for operations more complex than the previous 8 ones, makes it possible to directly use a __SQL query on the Data Frame__



The following notebooks present multiple __examples of use__ of the `querier`: 

- [`concat` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_251019_concat.ipynb)
- [`delete` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_241019_delete.ipynb)
- [`drop` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_241019_drop.ipynb)
- [`filtr` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_231019_filtr.ipynb)
- [`join` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_231019_join.ipynb)
- [`select` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_231019_select.ipynb)
- [`summarize` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_231019_summarize.ipynb)
- [`update` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_251019_update.ipynb)
- [`request` example](https://github.com/Techtonique/querier/tree/master/querier/demo/thierrymoudiki_231019_request.ipynb)


Contributions/remarks are welcome as usual, you can submit a pull request [on Github](https://github.com/Techtonique/querier).


__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!


<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Licence Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />Under License <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>.


