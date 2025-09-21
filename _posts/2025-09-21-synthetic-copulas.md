---
layout: post
title: "Generating Synthetic Data with R-vine Copulas using esgtoolkit in R"
description: "Generating synthetic data using R-vine copulas with the esgtoolkit package in R"
date: 2025-09-21
categories: R
comments: true
---

**R-vine copulas** are powerful tools for modeling complex dependencies among multiple variables. The `esgtoolkit` package in R provides a user-friendly interface to fit R-vine copula models and generate synthetic data that preserves the statistical properties of the original dataset.

See also:

- [https://docs.techtonique.net/ESGtoolkit/index.html](https://docs.techtonique.net/ESGtoolkit/index.html)
- [https://docs.techtonique.net/ESGtoolkit/articles/syntheticopula.html](https://docs.techtonique.net/ESGtoolkit/articles/syntheticopula.html)


```R
devtools::install_github("Techtonique/esgtoolkit")
```

```R

library(esgtoolkit)

y <- esgtoolkit::calculatereturns(ts(EuStockMarkets[1:250, ], start=start(EuStockMarkets),
                                     frequency=frequency(EuStockMarkets)), type = "log")

# Run simulation
result <- simulate_rvine(y, n = 500, verbose = TRUE, n_trials = 5)

# Print summary
print(result)

# Create different types of plots
plot(result, type = "distribution")  # Default

plot(result, type = "correlation")

#plot(result, type = "both")

# Access detailed diagnostics
str(result$diagnostics)

# Access simulated data
sim_data <- result$simulated_data

head(sim_data)
```

```R

    Transforming data to uniform margins with improved boundary handling...
    
    Fitting R-vine copula model...
    
    V1 + V3 --> V1,V3 ; V2
    
    V2 + V4 --> V2,V4 ; V3
    
    V1 + V4 --> V1,V4 ; V3,V2
    
    R-vine copula model fitted successfully
    


    tree     edge | family   cop   par  par2 |  tau   utd   ltd 
    ----------------------------------------------------------- 
       1      2,1 |     19  SBB7  2.03  0.69 | 0.47  0.36  0.59
              3,2 |     19  SBB7  1.74  0.79 | 0.44  0.41  0.51
              4,3 |      1     N  0.63  0.00 | 0.43     -     -
       2    3,1;2 |      1     N  0.33  0.00 | 0.21     -     -
            4,2;3 |      1     N  0.30  0.00 | 0.19     -     -
       3  4,1;3,2 |     14    SG  1.07  0.00 | 0.06     -  0.09
    ---
    type: D-vine    logLik: 249.86    AIC: -483.71    BIC: -455.57    
    ---
    1 <-> V1,   2 <-> V2,   3 <-> V3,   4 <-> V4  tree    edge family  cop       par      par2        tau       utd        ltd
    1    1     4,3      1    N 0.6297866 0.0000000 0.43371534 0.0000000 0.00000000
    2    1     3,2     19 SBB7 1.7362736 0.7864981 0.43642959 0.4142407 0.50934532
    3    1     2,1     19 SBB7 2.0325295 0.6856104 0.46867955 0.3638576 0.59360896
    4    2   4,2;3      1    N 0.2984383 0.0000000 0.19293140 0.0000000 0.00000000
    5    2   3,1;2      1    N 0.3282026 0.0000000 0.21288575 0.0000000 0.00000000
    6    3 4,1;3,2     14   SG 1.0683573 0.0000000 0.06398351 0.0000000 0.08676182


    Running 5 simulation trials...
    
    Best simulation achieved quality score: 0.0984
    
    Score weights used: [0.4, 0.2, 0.2, 0.1, 0.1]
    
    Mean absolute correlation error (Kendall): 0.0077
    
    Mean absolute correlation error (Pearson): 0.0428
    


    R-vine Copula Simulation Results
    ================================
    
    Original observations: 249
    Variables: 4
    Simulated observations: 500
    Quality score: 0.0984
    Successful trials: 5/5
    Mean absolute correlation error (Kendall): 0.0077
    Mean absolute correlation error (Pearson): 0.0428
    
    Use plot() to visualize results and $diagnostics for detailed metrics.
```
    
![image-title-here]({{base}}/images/2025-09-21/2025-09-21-synthetic-copulas_2_4.png){:class="img-responsive"}
    

```R
    List of 24
     $ original_correlation_tau     : num [1:4, 1:4] 1 0.465 0.413 0.353 0.465 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ simulated_correlation_tau    : num [1:4, 1:4] 1 0.45 0.421 0.355 0.45 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ correlation_error_tau        : num [1:4, 1:4] 0 -0.01499 0.00825 0.00173 -0.01499 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ original_correlation_pearson : num [1:4, 1:4] 1 0.815 0.728 0.507 0.815 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ simulated_correlation_pearson: num [1:4, 1:4] 1 0.766 0.682 0.404 0.766 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ correlation_error_pearson    : num [1:4, 1:4] 0 -0.0483 -0.0454 -0.1027 -0.0483 ...
      ..- attr(*, "dimnames")=List of 2
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
      .. ..$ : chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ mean_absolute_error_tau      : num 0.0077
     $ max_absolute_error_tau       : num 0.0232
     $ mean_absolute_error_pearson  : num 0.0428
     $ max_absolute_error_pearson   : num 0.103
     $ quality_score                : num 0.0984
     $ score_weights_used           : num [1:5] 0.4 0.2 0.2 0.1 0.1
     $ trial_scores                 : num [1:5] 0.219 0.1222 0.0984 0.1231 0.1884
     $ successful_trials            : int 5
     $ RVM_model                    :List of 20
      ..$ Matrix     : num [1:4, 1:4] 1 4 3 2 0 2 4 3 0 0 ...
      ..$ family     : num [1:4, 1:4] 0 14 1 19 0 0 1 19 0 0 ...
      ..$ par        : num [1:4, 1:4] 0 1.068 0.328 2.033 0 ...
      ..$ par2       : num [1:4, 1:4] 0 0 0 0.686 0 ...
      ..$ names      : chr [1:4] "V1" "V2" "V3" "V4"
      ..$ MaxMat     : num [1:4, 1:4] 1 2 2 2 0 2 3 3 0 0 ...
      ..$ CondDistr  :List of 2
      .. ..$ direct  : logi [1:4, 1:4] FALSE TRUE TRUE TRUE FALSE FALSE ...
      .. ..$ indirect: logi [1:4, 1:4] FALSE FALSE FALSE FALSE FALSE TRUE ...
      ..$ type       : chr "D-vine"
      ..$ tau        : num [1:4, 1:4] 0 0.064 0.213 0.469 0 ...
      ..$ taildep    :List of 2
      .. ..$ upper: num [1:4, 1:4] 0 0 0 0.364 0 ...
      .. ..$ lower: num [1:4, 1:4] 0 0.0868 0 0.5936 0 ...
      ..$ beta       : num [1:4, 1:4] 0 0.062 0.213 0.443 0 ...
      ..$ call       : language VineCopula::RVineStructureSelect(data = U, familyset = valid_families,      type = 0, selectioncrit = "BIC", trun| __truncated__ ...
      ..$ nobs       : int 249
      ..$ logLik     : num 250
      ..$ pair.logLik: num [1:4, 1:4] 0 2.59 14.33 84.82 0 ...
      ..$ AIC        : num -484
      ..$ pair.AIC   : num [1:4, 1:4] 0 -3.18 -26.66 -165.65 0 ...
      ..$ BIC        : num -456
      ..$ pair.BIC   : num [1:4, 1:4] 0 0.341 -23.138 -158.615 0 ...
      ..$ emptau     : num [1:4, 1:4] 0 0.0639 0.2156 0.4653 0 ...
      ..- attr(*, "class")= chr "RVineMatrix"
     $ n_observations               : int 249
     $ n_variables                  : int 4
     $ n_simulations                : num 500
     $ original_means               : Named num [1:4] 0.000372 0.000456 0.000338 0.000255
      ..- attr(*, "names")= chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ simulated_means              : Named num [1:4] -3.17e-05 2.18e-04 -4.52e-05 1.83e-04
      ..- attr(*, "names")= chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ original_sds                 : Named num [1:4] 0.00931 0.00877 0.01049 0.00815
      ..- attr(*, "names")= chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ simulated_sds                : Named num [1:4] 0.00954 0.00886 0.01119 0.00849
      ..- attr(*, "names")= chr [1:4] "DAX" "SMI" "CAC" "FTSE"
     $ ks_test_statistics           : Named num [1:4] 0.0259 0.0354 0.0507 0.04
      ..- attr(*, "names")= chr [1:4] "D" "D" "D" "D"
     $ ks_test_pvalues              : num [1:4] 1 0.985 0.786 0.953
```


<table class="dataframe">
<caption>A matrix: 6 × 4 of type dbl</caption>
<thead>
	<tr><th scope=col>DAX</th><th scope=col>SMI</th><th scope=col>CAC</th><th scope=col>FTSE</th></tr>
</thead>
<tbody>
	<tr><td>-0.001398544</td><td>-0.0015309795</td><td> 0.003170410</td><td> 0.0008758254</td></tr>
	<tr><td> 0.004458917</td><td> 0.0026640098</td><td> 0.011666435</td><td> 0.0057322484</td></tr>
	<tr><td>-0.001597764</td><td>-0.0001979084</td><td>-0.004832143</td><td> 0.0011097778</td></tr>
	<tr><td>-0.001501251</td><td>-0.0034774275</td><td>-0.003218613</td><td> 0.0020315141</td></tr>
	<tr><td> 0.000000000</td><td> 0.0045969506</td><td> 0.001912011</td><td>-0.0044810433</td></tr>
	<tr><td>-0.002419004</td><td>-0.0004654500</td><td>-0.004832588</td><td>-0.0055430360</td></tr>
</tbody>
</table>




    
![image-title-here]({{base}}/images/2025-09-21/2025-09-21-synthetic-copulas_2_7.png){:class="img-responsive"}
    

