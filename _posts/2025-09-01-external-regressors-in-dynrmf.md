---
layout: post
title: "external regressors in dynrmf"
date: 2025-09-01
categories: [R, Python]
comments: true
---


```R
options(repos = c(
                    techtonique = "https://r-packages.techtonique.net",
                    CRAN = "https://cloud.r-project.org"
                ))

install.packages(c("ahead", "misc", "fpp2", "glmnet"))
```


```R
sets <- list(USAccDeaths, AirPassengers, fpp2::a10, fdeaths)

# Default: ridge
par(mfrow=c(2, 2))
for (x in sets)
{
  xreg <- ahead::createtrendseason(x)
  train_test_x <- misc::splitts(x, split_prob = 0.8)
  xreg_training <- window(xreg, start=start(train_test_x$training),
                          end=end(train_test_x$training))
  xreg_testing <- window(xreg, start=start(train_test_x$testing),
                         end=end(train_test_x$testing))
  h <- length(train_test_x$testing)
  plot(ahead::dynrmf(y=train_test_x$training,
                     xreg_fit=xreg_training,
                     xreg_predict=xreg_testing,
                     level=99,
                     h=h))
}

# Default: glmnet::cv.glmnet
par(mfrow=c(2, 2))
for (x in sets)
{
  xreg <- ahead::createtrendseason(x)
  train_test_x <- misc::splitts(x, split_prob = 0.8)
  xreg_training <- window(xreg, start=start(train_test_x$training),
                          end=end(train_test_x$training))
  xreg_testing <- window(xreg, start=start(train_test_x$testing),
                         end=end(train_test_x$testing))
  h <- length(train_test_x$testing)
  plot(ahead::dynrmf(y=train_test_x$training,
                     xreg_fit=xreg_training,
                     xreg_predict=xreg_testing,
                     fit_func = glmnet::cv.glmnet,
                     level=99,
                     h=h))
}

```


    
![image-title-here]({{base}}/images/2025-09-01/2025-09-01-external-regressors-in-dynrmf_1_0.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2025-09-01/2025-09-01-external-regressors-in-dynrmf_1_1.png){:class="img-responsive"}
    

