---
layout: post
title: "Using caretS3 with ahead::ctxthetaf for Time Series Forecasting"
description: "A demonstration of using caretS3 models for time series forecasting with ahead::ctxthetaf."
date: 2026-11-07
categories: R
comments: true
---

[`ahead::ctxthetaf`](https://techtonique.r-universe.dev/ahead/doc/manual.html#ctxthetaf) has no more contextual information about the time series, but it still works, and can still be used notably with `caretS3` (a `caret` derivative introduced in [https://thierrymoudiki.github.io/blog/2026/10/25/r/caretS3-ahead](https://thierrymoudiki.github.io/blog/2026/10/25/r/caretS3-ahead)) to forecast time series data. The following code demonstrates how to use `caretS3` models for time series forecasting with `ahead::ctxthetaf`.

```R
list_models <- caret::modelLookup()[caret::modelLookup()[,"forReg"]==TRUE, 1]

n_models <- length(list_models)

pb <- utils::txtProgressBar(min = 0, max = n_models, style = 3)

for (i in seq_len(n_models))
{
  ts_model <- try(ahead::ctxthetaf(y=fdeaths, 
                                   theta = 1,
                                   h=25L, 
                                   fit_func = function (x, y) caretS3::caretS3(x, y, 
                                                                               method=list_models[i])), 
                  silent = TRUE)
  if (class(ts_model) != "try-error")
  {
    plot(ts_model) 
  } 
  utils::setTxtProgressBar(pb, i)
}

close(pb)
```

![image-title-here]({{base}}/images/2026-10-25/2026-10-25-caretS3-ahead_6_0.png){:class="img-responsive"}
