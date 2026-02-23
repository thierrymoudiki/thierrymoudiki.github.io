---
layout: post
title: "nnetsauce with and without jax for GPU acceleration"
description: "How to install nnetsauce with and without jax for GPU acceleration"
date: 2026-02-23
categories: [R, Python]
comments: true
---

In the new version (`0.51.2`) of nnetsauce (for Python, but also [for R](https://thierrymoudiki.github.io/blog/2025/12/17/r/python/new-nnetsauce-R-uv)), available on PyPI and for conda, I removed jax and jaxlib (for GPU) from the default version, because jaxlib is heavy. 

It means that if you want to use GPUs with nnetsauce (as in [https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas](https://www.researchgate.net/publication/382589729_Probabilistic_Forecasting_with_nnetsauce_using_Density_Estimation_Bayesian_inference_Conformal_prediction_and_Vine_copulas)), you'd want to explicitly install jax: 

```
pip install nnetsauce[jax]
```

or

```
uv pip install nnetsauce[jax]
```

or 

```
conda install -c conda-forge nnetsauce jax jaxlib
```