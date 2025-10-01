---
layout: post
title: "Real-time pricing with a pretrained probabilistic stock return model"
description: "Real-time pricing with a pretrained probabilistic stock return model, using Python FastAPI and R Plumber"
date: 2025-10-01
categories: [R, Python]
comments: true
---

[https://alphathon2025-344c7b0f914a.herokuapp.com/](https://alphathon2025-344c7b0f914a.herokuapp.com/) is a [Python FastAPI](https://fastapi.tiangolo.com/) Web app submitted at [https://alphathon.org/](https://alphathon.org/). 

This web app [interacts](https://pretrainedridge2f-8aee3d9572cc.herokuapp.com/__docs__/) with a [pretrained model](https://thierrymoudiki.github.io/blog/2025/09/09/r/python/pretraining-ridge2f-part2) for probabilistic stock return forecasting. 

The methodology for _pretraining_ is described concisely in: 

- [https://thierrymoudiki.github.io/blog/2025/09/08/r/python/pretraining-ridge2f](https://thierrymoudiki.github.io/blog/2025/09/08/r/python/pretraining-ridge2f)
- [https://thierrymoudiki.github.io/blog/2025/09/09/r/python/pretraining-ridge2f-part2](https://thierrymoudiki.github.io/blog/2025/09/09/r/python/pretraining-ridge2f-part2)
- [https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f](https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f)

The synthetic data generated are available at: 

[https://github.com/Techtonique/datasets/blob/main/time_series/multivariate/synthetic_stock_returns.csv](https://github.com/Techtonique/datasets/blob/main/time_series/multivariate/synthetic_stock_returns.csv)

The [R Plumber](https://www.rplumber.io/) API ([its code](/plumberAPI)): 

[https://pretrainedridge2f-8aee3d9572cc.herokuapp.com/__docs__/](https://pretrainedridge2f-8aee3d9572cc.herokuapp.com/__docs__/)

![image-title-here]({{base}}/images/2025-10-01/2025-10-01-image1.png){:class="img-responsive"}



