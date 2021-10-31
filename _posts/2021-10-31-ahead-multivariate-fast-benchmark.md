---
layout: post
title: "Fast and scalable forecasting with `ahead::ridge2f`"
description: Fast and scalable multivariate time series forecasting with `ahead::ridge2f`.
date: 2021-10-31
categories: [R, Misc]
---


[Two weeks ago](https://thierrymoudiki.github.io/blog/2021/10/15/r/misc/ahead-intro) I presented `ahead`, an R package for univariate and multivariate time series forecasting. And [last  week](https://thierrymoudiki.github.io/blog/2021/10/22/r/misc/ahead-ridge), I've shown how 
`ahead::dynrmf` could be used for automatic univariate forecasting. 

This week, I compare the speeds of execution of [`ahead::ridge2f`](https://www.mdpi.com/2227-9091/6/1/22) (quasi-randomized autoregressive network) and `ahead::varf` (Vector AutoRegressive model), with their default parameters (notably 1 lag, and 5-steps-ahead forecasting). For more examples of multivariate time series forecasting with `ahead`, you can type `?ahead::ridge2f`, `?ahead::varf(x)`, or `?ahead::plot.mtsforecast` in R console, once the package is installed and loaded. 


Here's how to **install** `ahead`:

- __1st method__: from [R-universe](https://ropensci.org/r-universe/)

    In R console:
    
    ```R
    options(repos = c(
        techtonique = 'https://techtonique.r-universe.dev',
        CRAN = 'https://cloud.r-project.org'))
        
    install.packages("ahead")
    ```

- __2nd method__: from Github

    In R console:
    
    ```R
    devtools::install_github("Techtonique/ahead")
    ```
    
    Or
    
    ```R
    remotes::install_github("Techtonique/ahead")
    ```

**Loading packages** for the demo: 

```R
library(ahead)
library(microbenchmark)
```

**Benchmarks**: 

With 10 time series and 10000 observations: 

```R
x <- ts(matrix(rnorm(100000), ncol = 10))
(res <- microbenchmark::microbenchmark(ahead::ridge2f(x), ahead::varf(x), times = 10L))
Unit: milliseconds
             expr       min        lq      mean    median        uq      max neval
ahead::ridge2f(x)  28.74215  32.22568  48.10457  35.70773  39.54476 104.5946    10
ahead::varf(x)    126.64908 138.91396 171.74475 157.16234 207.46105 237.5916    10
```
```R
ggplot2::autoplot(res)
```

![image-title-here]({{base}}/images/2021-10-31/2021-10-31-image1.png){:class="img-responsive"}


With 100 time series and 1000 observations: 

```R
x <- ts(matrix(rnorm(100000), ncol = 100))
(res <- microbenchmark::microbenchmark(ahead::ridge2f(x), ahead::varf(x), times = 10L))
Unit: milliseconds
             expr       min         lq      mean     median         uq       max neval
ahead::ridge2f(x)   46.8317   48.44567   81.1854   53.52305   61.06889  220.5755    10
ahead::varf(x)    2276.9425 2293.05932 2360.1591 2316.90078 2362.63500 2696.0487    10
```
```R
ggplot2::autoplot(res)
```

![image-title-here]({{base}}/images/2021-10-31/2021-10-31-image2.png){:class="img-responsive"}

With 1000 time series and 100 observations: 

```R
x <- ts(matrix(rnorm(100000), ncol = 1000))
(res <- microbenchmark::microbenchmark(ahead::ridge2f(x), ahead::varf(x), times = 10L))
Unit: seconds
             expr        min        lq       mean     median         uq        max neval
ahead::ridge2f(x)   1.891717   2.18807   2.315703   2.253376   2.286887   3.048088    10
ahead::varf(x)    226.133743 234.94063 240.931083 239.235557 247.780707 259.656456    10
```
```R
ggplot2::autoplot(res)
```

![image-title-here]({{base}}/images/2021-10-31/2021-10-31-image3.png){:class="img-responsive"}

`ahead::ridge2f` is fast and scalable, mostly because I implemented some of its parts in C++, _via_  [`Rcpp`](https://cran.r-project.org/web/packages/Rcpp/index.html). In addition, the algorithm is mainly made of matrices products and inversions -- with the inversions being the most _expensive_ parts, according to [`profvis::profvis`](https://rstudio.github.io/profvis/). 

If you are interested in making the algorithm even faster in a GitHub fork, notice that: the most training time is spent at lines 267--275 of [this file](https://github.com/Techtonique/ahead/blob/main/R/ridge2.R), which corresponds to section 2.3 of [the paper](https://www.mdpi.com/2227-9091/6/1/22).

```R
sessionInfo()
```
```
R version 4.0.4 (2021-02-15)
Platform: x86_64-apple-darwin17.0 (64-bit)
Running under: macOS Big Sur 10.16

Matrix products: default
LAPACK: /Library/Frameworks/R.framework/Versions/4.0/Resources/lib/libRlapack.dylib

locale:
[1] fr_FR.UTF-8/fr_FR.UTF-8/fr_FR.UTF-8/C/fr_FR.UTF-8/fr_FR.UTF-8

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] bigtime_0.2.1

loaded via a namespace (and not attached):
 [1] Rcpp_1.0.7           urca_1.3-0           compiler_4.0.4       pillar_1.4.6         iterators_1.0.13    
 [6] vars_1.5-3           tools_4.0.4          digest_0.6.25        corrplot_0.84        ahead_0.2.0         
[11] lifecycle_0.2.0      tibble_3.0.3         gtable_0.3.0         nlme_3.1-152         lattice_0.20-41     
[16] pkgconfig_2.0.3      rlang_0.4.10         foreach_1.5.1        rstudioapi_0.11      microbenchmark_1.4-7
[21] dplyr_1.0.2          generics_0.0.2       vctrs_0.3.4          tidyselect_1.1.0     lmtest_0.9-38       
[26] grid_4.0.4           glue_1.4.2           R6_2.5.0             randtoolbox_1.30.1   fpp_0.5             
[31] farver_2.0.3         purrr_0.3.4          ggplot2_3.3.3        magrittr_1.5         scales_1.1.1        
[36] codetools_0.2-18     ellipsis_0.3.1       MASS_7.3-53          strucchange_1.5-2    colorspace_1.4-1    
[41] sandwich_3.0-0       rngWELL_0.10-6       munsell_0.5.0        crayon_1.3.4         zoo_1.8-8
```


