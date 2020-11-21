---
layout: post
title: "Boosting nonlinear penalized least squares"
description: Boosting nonlinear penalized least squares
date: 2020-11-21
categories: [ExplainableML, R, Python, LSBoost]
---

For some reasons I couldn't foresee, there's been no blog post here on november 13 
and november 20. So, here is the post about LSBoost announced here [a few weeks ago](https://thierrymoudiki.github.io/blog/2020/10/30/misc/news). 

First things first, what is LSBoost? **Gradient boosted nonlinear penalized least squares**. More precisely in LSBoost, the ensembles' base learners are penalized, randomized _neural_ networks. 


These previous posts, with **several Python and R examples**, constitute a good introduction to LSBoost: 

- [https://thierrymoudiki.github.io/blog/2020/07/24/python/r/lsboost/explainableml/mlsauce/xai-boosting](https://thierrymoudiki.github.io/blog/2020/07/24/python/r/lsboost/explainableml/mlsauce/xai-boosting)

- [https://thierrymoudiki.github.io/blog/2020/07/31/python/r/lsboost/explainableml/mlsauce/xai-boosting-2](https://thierrymoudiki.github.io/blog/2020/07/31/python/r/lsboost/explainableml/mlsauce/xai-boosting-2)

More recently, I've also written a more formal, short introduction to LSBoost: 

- [LSBoost, gradient boosted penalized nonlinear least squares (pdf)](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares). 

The paper's code -- and **more** insights on LSBoost -- can be found in the following Jupyter notebook: 

- [https://github.com/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki_211120_lsboost_sensi_to_hyperparams.ipynb](https://github.com/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki_211120_lsboost_sensi_to_hyperparams.ipynb)

Comments, suggestions are welcome as usual.

![pres-image]({{base}}/images/2020-11-21/2020-11-21-image6.png){:class="img-responsive"}

