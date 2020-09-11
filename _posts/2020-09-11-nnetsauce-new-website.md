---
layout: post
title: "A new version of nnetsauce, and a new Techtonique website"
description: Announcements.
date: 2020-09-11
categories: [Misc, Python, QuasiRandomizedNN]
---


As a reminder `nnetsauce`, `mlsauce`, the `querier` and the `teller` are now stored under [Techtonique](https://github.com/Techtonique). A new [Techtonique website](https://techtonique.github.io/) is also out know; it contains  **documentation + examples** for `nnetsauce`, `mlsauce`, the `querier` and the `teller`, and is a __work in progress__. 

![new-techtonique-website]({{base}}/images/2020-09-11/2020-09-11-image1.png){:class="img-responsive"}
_Figure: [New Techtonique Website](https://techtonique.github.io/)_


In addition, a new version of `nnetsauce`  including nonlinear [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLM) has been released, both on [Pypi](https://pypi.org/project/nnetsauce/) and GitHub.  

- __Installing__ by using `pip`:

```bash
pip install nnetsauce
```


- __Installing__ from Github: 

```bash
pip install git+https://github.com/thierrymoudiki/nnetsauce.git
```


I've been experiencing some issues when installing from Pypi lately. Please, feel free to report any issue to me, if you're experiencing some, __or use the GitHub version instead__. You can still execute the following jupyter notebook -- at the bottom for __nonlinear GLMs__ in particular`-- to see what's changed:

[https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_040920_examples.ipynb](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_040920_examples.ipynb)



__How do these nonlinear GLMs work?__ Let's say (a highly simplified example) that we want to predict the  future, final height of a basketball player, using his current height and his age. We'd use historical information of heights, ages (covariates), and final heights (response) of multiple players: 

1 - Input data, the covariates, are transformed into new covariates, as we've already seen it before, for nnetsauce. You can visit this page for a refresher: [References](https://techtonique.github.io/nnetsauce/REFERENCES/).

![nnetsauce-input-transformation]({{base}}/images/2020-09-11/2019-10-18-image1.png){:class="img-responsive"}

2 -  A loss function, a function measuring the quality of adjustment of the model to the observed data, is optimized as represented in the __figure below__. Currently, two optimization methods are used: __Stochastic [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)__ and __Stochastic [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)__. With, notably and amongst other hyperparameters, [early stopping](https://en.wikipedia.org/wiki/Early_stopping) criteria and regularization to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting).

![glm-loss-function]({{base}}/images/2020-09-04/2020-09-04-image1.png){:class="img-responsive"}
_Figure: GLM loss function (from notebook)_

I'll write down a more formal description of these algorithms in the future. 
