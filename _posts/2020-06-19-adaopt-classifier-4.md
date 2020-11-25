---
layout: post
title: "Parallel AdaOpt classification"
description: Parallel AdaOpt classification
date: 2020-06-19
categories: [Python, AdaOpt, mlsauce]
---

In these previous posts:

- [AdaOpt classification on MNIST handwritten digits (without preprocessing)]({% post_url 2020-05-29-adaopt-classifier-3 %}) (on 05/29/2020)
- [AdaOpt (a probabilistic classifier based on a mix of multivariable optimization and nearest neighbors) for R]({% post_url 2020-05-22-adaopt-classifier-2 %}) (on 05/22/2020)
- [AdaOpt]({% post_url 2020-05-15-adaopt-classifier-1 %}) (on 05/15/2020)

I introduced `AdaOpt`, a novel _probabilistic_ classifier based on a **mix of multivariable optimization and a _nearest neighbors_** algorithm. More details about the algorithm can also be found in [this (short) paper](https://www.researchgate.net/publication/341409169_AdaOpt_Multivariable_optimization_for_classification). [`mlsauce`](https://github.com/Techtonique/mlsauce)'s development version now contains a __parallel implementation__ of `AdaOpt`. In order to install this development version from the command line, you'll need to type: 

```bash
pip install git+https://github.com/Techtonique/mlsauce.git --upgrade
```

And in order to use parallel processing, create the `AdaOpt` object (see previous post) with:

```bash
n_jobs = 2 # or 4, or -1 or 13
```

