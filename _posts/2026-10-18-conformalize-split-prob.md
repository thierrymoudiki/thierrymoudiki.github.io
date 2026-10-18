---
layout: post
title: "Choosing the size of the calibration set in conformal prediction with ahead::conformalize()"
description: "This post explores the impact of the split_prob parameter in the ahead::conformalize() function on the resulting prediction intervals."
date: 2026-10-18
categories: R
comments: true
---

In version 0.39.0 of the `ahead` package, the `conformalize()` function has been updated to include a new parameter called `split_prob`. This parameter allows users to specify the proportion of the data that should be used for calibration when constructing conformal prediction intervals.

# 0 - Install packages

```R
install.packages("pak")
```


```R
pak::pak("Techtonique/ahead")
```


```R
library(ahead)
```

# 1 - `USAccDeaths`


```R
par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN = forecast::ets, y = USAccDeaths,
                           h = 20L, split_prob = 0.4)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = USAccDeaths,
                           h = 20L, split_prob = 0.3)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = USAccDeaths,
                           h = 20L, split_prob = 0.6)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = USAccDeaths,
                           h = 20L, split_prob = 0.7)
plot(obj)
```


    
![image-title-here]({{base}}/images/2026-10-18/2026-10-18-conformalize-split-prob_5_0.png){:class="img-responsive"}
    


# 2 - `AirPassengers`


```R
par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN = forecast::ets, y = AirPassengers,
                           h = 20L, split_prob = 0.4)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = AirPassengers,
                           h = 20L, split_prob = 0.3)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = AirPassengers,
                           h = 20L, split_prob = 0.6)
plot(obj)
obj <- ahead::conformalize(FUN = forecast::ets, y = AirPassengers,
                           h = 20L, split_prob = 0.7)
plot(obj)
```


    
![image-title-here]({{base}}/images/2026-10-18/2026-10-18-conformalize-split-prob_7_0.png){:class="img-responsive"}
    


# 3 - `fdeaths`


```R
par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = fdeaths,
                           h = 20L, split_prob = 0.4)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = fdeaths,
                           h = 20L, split_prob = 0.3)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = fdeaths,
                           h = 20L, split_prob = 0.6)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = fdeaths,
                           h = 20L, split_prob = 0.7)
plot(obj)
```


    
![image-title-here]({{base}}/images/2026-10-18/2026-10-18-conformalize-split-prob_9_0.png){:class="img-responsive"}
    


# 4 - `UKDriverDeaths`


```R
par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = UKDriverDeaths,
                           h = 20L, split_prob = 0.4)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = UKDriverDeaths,
                           h = 20L, split_prob = 0.3)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = UKDriverDeaths,
                           h = 20L, split_prob = 0.6)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = UKDriverDeaths,
                           h = 20L, split_prob = 0.7)
plot(obj)
```


    
![image-title-here]({{base}}/images/2026-10-18/2026-10-18-conformalize-split-prob_11_0.png){:class="img-responsive"}
    


# 5 - `WWWusage`


```R
par(mfrow=c(2, 2))
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = WWWusage,
                           h = 20L, split_prob = 0.4)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = WWWusage,
                           h = 20L, split_prob = 0.3)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = WWWusage,
                           h = 20L, split_prob = 0.6)
plot(obj)
obj <- ahead::conformalize(FUN = ahead::dynrmf, y = WWWusage,
                           h = 20L, split_prob = 0.5)
plot(obj)
```


    
![image-title-here]({{base}}/images/2026-10-18/2026-10-18-conformalize-split-prob_13_0.png){:class="img-responsive"}
    

