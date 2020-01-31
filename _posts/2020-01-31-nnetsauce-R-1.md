---
layout: post
title: "nnetsauce for R"
description: nnetsauce for R
date: 2020-01-31
---


[`nnetsauce`](https://github.com/thierrymoudiki/nnetsauce/R-package) is now available to R users, in the form of a development version. Not all the functions available in Python are available in R so far, but the R implementation is catching up fast. The general rule for invoking methods in R as we'll see in the [example](## Example), is to __mirror the Python way, but replacing `.`'s by `$`'s__. Contributions/remarks are welcome as usual, and you can submit a pull request [on Github](https://github.com/thierrymoudiki/nnetsauce/R-package).


## Installation 

Here is how to install `nnetsauce` from Github, using R console: 

```r
# use library devtools
library(devtools)
# install nnetsauce from Github 
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
# load nnetsauce
library(nnetsauce)
```

Having installed and loaded `nnetsauce`, we can now showcase a simple classification based on `Ridge2Classifier` model, and `iris` dataset. 

## Example 

```r
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
```

```r
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
```



__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!



