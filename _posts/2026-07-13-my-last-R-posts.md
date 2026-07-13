---
layout: post
title: "My last R posts: How conformalization helps weak models, fast conformal prediction with jackknife+ (and no refitting), and sklearn in R"
description: "My last R posts: How conformalization helps weak models, fast conformal prediction with jackknife+ (and no refitting), and sklearn in R."
date: 2026-07-13
categories: R
comments: true
---

This post is mainly (but not only) a test, because I had a broken xml feed for my R posts, and I wanted to see if it was fixed. 

It's about my last R posts from june and july, which are: 

- Using scikit-learn models in R easily with the `tisthemachinelearner` R package
- How conformalization helps weak models
- Fast Conformal Prediction for Some Machine Learning Models (jackknife+ and no refitting)
  

# Using scikit-learn models in R easily with the `tisthemachinelearner` R package

This post is about the [`tisthemachinelearner` R package](https://github.com/Techtonique/tisthemachinelearner_r/tree/main), that allows to use scikit-learn models in R. It is a wrapper around the [tisthemachinelearner Python package](https://github.com/Techtonique/tisthemachinelearner/tree/main). Prediction intervals can be computed using either split conformal prediction, surrogate methods or the bootstrap. 

Read: [https://thierrymoudiki.github.io/blog/2026/06/21/r/tisthemllearner](https://thierrymoudiki.github.io/blog/2026/06/21/r/tisthemllearner)


# How conformalization helps weak models

In this post, we compare [split conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction)
across several predictive models, using [R package mlS3](https://cran.r-project.org/web/packages/mlS3/index.html).

Read: [https://thierrymoudiki.github.io/blog/2026/06/07/r/conformalization-helps-weak-models](https://thierrymoudiki.github.io/blog/2026/06/07/r/conformalization-helps-weak-models)

# Fast Conformal Prediction for Some Machine Learning Models (jackknife+ and no refitting)

It's surprisingly fast to obtain conformal [jackknife+](https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2F20-AOS1965) prediction intervals for Machine Learning models of the form $$\hat{y} = Sy$$ (including Ordinary
Least Squares, Ridge Regression, Random Vector Functional Link Networks,
Kernel Ridge Regression, smoothing splines, and local polynomial regression). **No refitting involved, just Linear Algebra**. Read [https://www.researchgate.net/publication/408161842_Fast_Conformal_Prediction_for_Some_Machine_Learning_Models_via_Closed-Form_Jackknife](https://www.researchgate.net/publication/408161842_Fast_Conformal_Prediction_for_Some_Machine_Learning_Models_via_Closed-Form_Jackknife).  
    
![image-title-here]({{base}}/images/2026-06-27/2026-06-27-jackknife-plus-smoothers_6_1.png){:class="img-responsive"}
    
<a target="_blank" href="https://colab.research.google.com/github/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki-2026-06-27_jackknife_plus_smoothers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; height: auto; width: 120px;"/>
</a>

