---
layout: post
title: "Documentation+Pypi for the `teller`, a model-agnostic tool for Machine Learning explainability"
description: Documentation+Pypi for the `teller`, a model-agnostic tool for Machine Learning explainability 
date: 2020-05-01
categories: [Python, ExplainableML]
---

The `teller` is a model-agnostic tool for Machine Learning (ML) explainability. It uses [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) and finite differences to explain ML models predictions: 

__a little increase__ in model's explanatory variables __+__ a __little decrease__ = approximate __sensitivities__ of its predictions to changes in these explanatory variables

![image-title-here]({{base}}/images/2020-05-01/2020-05-01-image1.png){:class="img-responsive"}  


The `teller` is now available on Pypi (yeaaah!), and can be installed from the command line as:

```bash
pip install the-teller
```

The code is also documented on readthedocs (it's a work in progress):

[https://the-teller.readthedocs.io/en/latest/?badge=latest](https://the-teller.readthedocs.io/en/latest/?badge=latest)


For those who haven't had a taste of the teller yet, these notebooks will constitute a good (and fun) introduction:

- [Heterogeneity of marginal effects](https://github.com/Techtonique/teller/blob/master/teller/demo/thierrymoudiki_011119_boston_housing.ipynb)
- [Significance of marginal effects](https://github.com/Techtonique/teller/blob/master/teller/demo/thierrymoudiki_081119_boston_housing.ipynb)
- [Model comparison](https://github.com/Techtonique/teller/blob/master/teller/demo/thierrymoudiki_151119_boston_housing.ipynb)
- [Classification](https://github.com/Techtonique/teller/blob/master/teller/demo/thierrymoudiki_041219_breast_cancer_classif.ipynb)
- [Interactions](https://github.com/Techtonique/teller/blob/master/teller/demo/thierrymoudiki_041219_boston_housing_interactions.ipynb)


__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!

