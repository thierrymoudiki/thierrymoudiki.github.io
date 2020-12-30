---
layout: post
title: "2020 recap, Gradient Boosting, Generalized Linear Models, AdaOpt with nnetsauce and mlsauce"
description: 2020 recap, Gradient Boosting, Generalized Linear Models, AdaOpt with nnetsauce and mlsauce
date: 2020-12-29
categories: [Python, R, Misc]
---

A few **highlights from 2020** in this blog include:

- The introduction of [mlsauce](https://techtonique.github.io/mlsauce/)'s AdaOpt and LSBoost
- The introduction of Generalized Linear Models ([GLMs](https://en.wikipedia.org/wiki/Generalized_linear_model)) in [nnetsauce](https://techtonique.github.io/nnetsauce/)

**What are AdaOpt, LSBoost and nnetsauce's GLMs?**


- **mlsauce's [AdaOpt](https://thierrymoudiki.github.io/blog/#AdaOpt)** is a _probabilistic_ classifier based on a mix of **multivariable optimization** and a **nearest neighbors** algorithm. [This document](https://www.researchgate.net/publication/341409169_AdaOpt_Multivariable_optimization_for_classification) explains AdaOpt with more details, in English and without formulas. Hopefully that makes it accessible to more people. Other resources on AdaOpt can be 
found [through this link](https://thierrymoudiki.github.io/blog/index.html#AdaOpt). 

- **mlsauce's LSBoost** implements [**Gradient Boosting**](https://en.wikipedia.org/wiki/Gradient_boosting) of augmented base learners (base learners = basic components in [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning)). In LSBoost, the base learners are penalized regression models augmented through randomized hidden nodes and activation functions. Examples in both R and Python are presented in [these posts](https://thierrymoudiki.github.io/blog/#LSBoost). And if anyone reading this is a **Windows + R specialist**, I'd love to hear from him/her, because, I get sometimes notified that mlsauce doesn't work well at this intersection (Windows + R). 

- Regarding **GLMs in nnetsauce**, this [post from november 28th](https://thierrymoudiki.github.io/blog/2020/11/28/explainableml/python/glms) will offer you a brief introduction to what they are. nnetsauce's GLMs are actually nonlinear models since the features (covariates) are transformed by using randomized/quasi-randomized hidden nodes and activation functions. The current optimizers for GLMs loss functions in nnetsauce are based on various gradient descent algorithms. There are probably some more efficient  ways that can be explored. This is a work in progress.  

In general, and not only for GLMs, the best way to read nnetsauce things is: [https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN). In #QuasiRandomizedNN, you'll find nnetsauce's posts you might have missed. For example, [this one](https://thierrymoudiki.github.io/blog/2020/12/11/r/quasirandomizednn/classify-penguins-nnetsauce), in which nnetsauce's MultitaskClassifier **perfectly classifies penguins** (in R).

I can see that **nnetsauce and mlsauce are downloaded thousands of times each month**. But that's not the most important thing to me! 
If you're using mlsauce, nnetsauce or any other tool presented in this blog, feel free and do not hesitate to [**contribute**](https://thierrymoudiki.github.io/blog/2020/02/14/misc/git-github), or **star** the repository. That way, we could create and keep alive a _cool_ community around these tools. That's ultimately the most important thing to me.


**Best wishes for 2021!** 
