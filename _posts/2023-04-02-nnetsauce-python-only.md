---
layout: post
title: "A new version of nnetsauce (randomized and quasi-randomized 'neural' networks)"
description: "Randomized and quasi-randomized 'neural' networks"
date: 2023-04-02
categories: [Python, R, QuasiRandomizedNN]
comments: true
---

**Content:**

<ul>
  <li> nnetsauce's new version </li>
  <li> Installing nnetsauce for Python </li>
  <li> About nnetsauce for R </li>
</ul>

## nnetsauce's new version

A new version of [nnetsauce](https://github.com/Techtonique/nnetsauce), v0.12.0, is available on PyPI and for conda. It's been mostly tested on Linux and macOS platforms so far. For Windows users: you can use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about) in case it doesn't work on your computer.

**As a reminder**, nnetsauce does Statistical/Machine Learning (regression, classification, and time series forecasting for now) using randomized and quasi-randomized _neural_ networks layers. More precisely, every model in nnetsauce is based on components g(XW + b), where:

<ul>
    <li> X is a matrix containing explanatory variables and optional clustering information. Clustering the inputs helps in taking into account data’s heterogeneity before model fitting. </li>
    <li> W creates new, additional explanatory variables from X. W can be drawn from various random and quasi-random sequences. </li>
    <li> b is an optional bias parameter. </li>
    <li> g is an activation function such as ReLU or the hyperbolic tangent, that makes the combination of explanatory variables – through W – nonlinear. </li>
</ul>

**Examples** of use of nnetsauce are available on GitHub,  [here](https://github.com/Techtonique/nnetsauce/tree/master/nnetsauce/demo) (including R Markdown  examples) and [here](https://github.com/Techtonique/nnetsauce/tree/master/examples). 

![RVFL]({{base}}/images/2023-04-02/2023-04-02-image1.png){:class="img-responsive"}


**v0.12.0 is an important release**, because it's totally written in Python (using numpy, scipy, jax, and  scikit-learn), and doesn't use _C++_ nor _Cython_ anymore. Because of this, nnetsauce is faster to install, and easier to maintain. In addition, when faster calculations are needed, I'd like to try out tools like [numba](https://numba.pydata.org/), or [jax](https://github.com/google/jax)'s `jit` in the near future.


## Installing nnetsauce for Python

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```

- __2nd method__: using `conda` (Linux and macOS only for now)

```bash
conda install -c conda-forge nnetsauce 
```

- __3rd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

or in a virtual environment: 

```bash
git clone https://github.com/Techtonique/nnetsauce.git
cd nnetsauce
make install
```

## About nnetsauce for R

The R version is discontinued. Well, 'discontinued' until I finally wrap 
my head around it... If you're interested in using nnetsauce for R, 
everything happens in [this R script](https://github.com/Techtonique/nnetsauce/blob/master/R-package/R/zzz.R). 
