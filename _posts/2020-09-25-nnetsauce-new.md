---
layout: post
title: "New nnetsauce"
description: New nnetsauce.
date: 2020-09-25
categories: [Misc, QuasiRandomizedNN]
---


A new version of nnetsauce has been released. The best way to use it right now, is to install from GitHub:

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

The __main change__ is the [cythonization](https://cython.org/) of [RandomBagClassifier](https://techtonique.github.io/nnetsauce/documentation/classifiers/). I didn't see a noticeable change, and I think it's mainly because in my code, there are a lot of [interactions between Python and Cython](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/randombag/_randombagc.html). 