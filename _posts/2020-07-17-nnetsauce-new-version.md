---
layout: post
title: "nnetsauce version 0.5.0, randomized neural networks on GPU"
description: nnetsauce version 0.5.0, randomized neural networks on GPU.
date: 2020-07-17
categories: [Python, R, QuasiRandomizedNN]
---



[nnetsauce](https://thierrymoudiki.github.io/software/nnetsauce/) is a general purpose tool for Statistical/Machine Learning, in which __pattern recognition__ is achieved by using [quasi-randomized networks](https://www.researchgate.net/project/Quasi-randomized-neural-networks). A new version, `0.5.0`, is out on [Pypi](https://pypi.org/project/nnetsauce/0.5.0/) and for R:

- Install by using `pip` (stable version):

```bash
pip install nnetsauce --upgrade
```

- Install from Github (development version):

```bash
pip install git+https://github.com/thierrymoudiki/nnetsauce.git --upgrade
```

- Install from Github, in R console:

```r
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
```



This could be the occasion for you to __re-read__ all the previous [posts about nnetsauce](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN), or to play with various examples in [Python](https://github.com/thierrymoudiki/nnetsauce/tree/master/examples) or [R](https://github.com/thierrymoudiki/nnetsauce/blob/master/nnetsauce/demo). Here are a few __other ways to interact__ with the nnetsauce: 

__1) Forms__

- If you're not comfortable with version control yet: a [__feedback form__](https://forms.gle/HQVbrUsvZE7o8xco8).

__2) Submit Pull Requests on GitHub__

- As detailed [in this post]({% post_url 2020-02-14-git-github %}). [Raising issues](https://guides.github.com/features/issues/) is another constructive way to interact. You can also contribute examples to this [demo repo](https://github.com/thierrymoudiki/nnetsauce/blob/master/nnetsauce/demo), using the following naming convention: 

`yourgithubname_ddmmyy_shortdescriptionofdemo.[ipynb|Rmd]`

 If it's a jupyter notebook written in __R__, then just add `_R` to the suffix. 

__3) Reaching out directly _via_ email__

- Use the address: thierry __dot__ moudiki __at__ pm __dot__ me

To those who are contacting me through LinkedIn: no, I'm not declining, __please, add a short message to your request__, so that I'd know a bit more about who you are, and/or how we can envisage to work together. 

![image-title-here]({{base}}/images/2020-07-17/2020-07-17-image1.png){:class="img-responsive"}

This **new version**, `0.5.0`:
- contains a refactorized code for the [`Base`](https://github.com/thierrymoudiki/nnetsauce/nnetsauce/base/base.py) class, and for many other utilities.
- makes use of [randtoolbox](https://cran.r-project.org/web/packages/randtoolbox/index.html) for a faster, more scalable generation of quasi-random numbers.
- contains __a (work in progress) implementation of most algorithms on GPUs__, using [JAX](https://github.com/google/jax). Most of the nnetsauce's changes related to GPUs are currently made on potentially time consuming operations such as matrices multiplications and matrices inversions. Though, to see a _GPU effect_, __you need to have loads of data__ at hand, and a relatively high `n_hidden_features` parameter. __How do you try it out?__ By instantiating a class with the option:

```python
backend = "gpu"
```

or

```python
backend = "tpu"
```

An __example__ can be found in [this notebook, on GitHub](https://github.com/thierrymoudiki/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_170720_nnetsauce_gpu.ipynb). 

[nnetsauce](https://github.com/thierrymoudiki/nnetsauce)'s future release is planned to be much faster on CPU, due the use of [Cython](https://cython.org/), as with [mlsauce](https://github.com/thierrymoudiki/mlsauce). There are indeed a lot of nnetsauce's parts which can be *cythonized*. If you've ever considered joining the project, now is the right time. For example, among other things, I'm looking for a volunteer to do some testing in R+Python on Microsoft Windows. __Envisage a smooth onboarding, even if you don't have a lot of experience__.