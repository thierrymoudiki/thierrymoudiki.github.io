---
layout: post
title: "Probabilistic Time Series Cross-Validation with R package crossvalidation"
description: "Examples of use of R package crossvalidation for Probabilistic Time Series Cross-Validation (measuring coverage and Winkler score)"
date: 2026-05-16
categories: R
comments: true
---

A previous post introduced the `crossvalidation` package for R. This time, the focus is on probabilistic forecasting — evaluating not just how accurate point forecasts are, but how well-calibrated prediction intervals are, using empirical coverage rates and Winkler scores -- and `crossvalidation`.


```R
install.packages("remotes")
```


```R
install.packages("forecast")
```


```R
remotes::install_github("Techtonique/crossvalidation")
```


```R
library(crossvalidation)
```

# Example 1


```R
require(forecast)
data("AirPassengers")



eval_metric <- function(predicted, observed)
{
  error <- observed - predicted$mean

  me <- mean(error)
  rmse <- sqrt(mean(error^2))
  mae <- mean(abs(error))

  # ----- 80% interval -----

  lower80 <- predicted$lower[, 1]
  upper80 <- predicted$upper[, 1]

  coverage80 <- mean(
    observed >= lower80 & observed <= upper80
  )

  alpha80 <- 0.20

  winkler80 <- ifelse(
    observed < lower80,
    (upper80 - lower80) + (2 / alpha80) * (lower80 - observed),
    ifelse(
      observed > upper80,
      (upper80 - lower80) + (2 / alpha80) * (observed - upper80),
      (upper80 - lower80)
    )
  )

  # ----- 95% interval -----

  lower95 <- predicted$lower[, 2]
  upper95 <- predicted$upper[, 2]

  coverage95 <- mean(
    observed >= lower95 & observed <= upper95
  )

  alpha95 <- 0.05

  winkler95 <- ifelse(
    observed < lower95,
    (upper95 - lower95) + (2 / alpha95) * (lower95 - observed),
    ifelse(
      observed > upper95,
      (upper95 - lower95) + (2 / alpha95) * (observed - upper95),
      (upper95 - lower95)
    )
  )

  c(
    ME = me,
    RMSE = rmse,
    MAE = mae,
    Coverage80 = coverage80,
    Winkler80 = mean(winkler80),
    Coverage95 = coverage95,
    Winkler95 = mean(winkler95)
  )
}

(res <- crossval_ts(y=AirPassengers, initial_window = 10,
horizon = 3, fcast_func = forecast::thetaf, eval_metric = eval_metric))
print(colMeans(res))

```

    Loading required package: forecast
    


      |======================================================================| 100%



<table class="dataframe">
<caption>A matrix: 132 × 7 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>ME</th><th scope=col>RMSE</th><th scope=col>MAE</th><th scope=col>Coverage80</th><th scope=col>Winkler80</th><th scope=col>Coverage95</th><th scope=col>Winkler95</th></tr>
</thead>
<tbody>
	<tr><th scope=row>result.1</th><td>-28.794660</td><td>29.300287</td><td>28.794660</td><td>0.0000000</td><td>153.10992</td><td>0.3333333</td><td>207.58384</td></tr>
	<tr><th scope=row>result.2</th><td> 16.198526</td><td>16.894302</td><td>16.198526</td><td>1.0000000</td><td> 45.01795</td><td>1.0000000</td><td> 68.84902</td></tr>
	<tr><th scope=row>result.3</th><td> 11.201494</td><td>15.993359</td><td>12.578276</td><td>1.0000000</td><td> 45.05996</td><td>1.0000000</td><td> 68.91326</td></tr>
	<tr><th scope=row>result.4</th><td> 21.430125</td><td>22.483895</td><td>21.430125</td><td>0.6666667</td><td> 63.01207</td><td>1.0000000</td><td> 68.84778</td></tr>
	<tr><th scope=row>result.5</th><td> 10.055765</td><td>11.527746</td><td>10.055765</td><td>1.0000000</td><td> 45.99967</td><td>1.0000000</td><td> 70.35043</td></tr>
	<tr><th scope=row>result.6</th><td> -2.640822</td><td>10.676714</td><td> 9.999466</td><td>1.0000000</td><td> 46.56907</td><td>1.0000000</td><td> 71.22125</td></tr>
	<tr><th scope=row>result.7</th><td> 14.296434</td><td>23.709132</td><td>20.531135</td><td>0.6666667</td><td> 75.04186</td><td>1.0000000</td><td> 67.58381</td></tr>
	<tr><th scope=row>result.8</th><td> 38.247497</td><td>39.529998</td><td>38.247497</td><td>0.0000000</td><td>198.74990</td><td>0.3333333</td><td>212.44029</td></tr>
	<tr><th scope=row>result.9</th><td> 23.043159</td><td>23.947630</td><td>23.043159</td><td>0.3333333</td><td> 93.83463</td><td>1.0000000</td><td> 64.19366</td></tr>
	<tr><th scope=row>result.10</th><td>-21.689067</td><td>27.907560</td><td>21.689067</td><td>0.6666667</td><td> 90.23377</td><td>1.0000000</td><td> 84.12361</td></tr>
	<tr><th scope=row>result.11</th><td>-41.782157</td><td>46.664199</td><td>41.782157</td><td>0.3333333</td><td>222.06310</td><td>0.3333333</td><td>345.16553</td></tr>
	<tr><th scope=row>result.12</th><td>-34.934831</td><td>36.512081</td><td>34.934831</td><td>0.3333333</td><td>162.38092</td><td>0.6666667</td><td>212.58117</td></tr>
	<tr><th scope=row>result.13</th><td> -4.002700</td><td>12.728771</td><td> 9.999100</td><td>1.0000000</td><td> 59.64475</td><td>1.0000000</td><td> 91.21878</td></tr>
	<tr><th scope=row>result.14</th><td> 30.349582</td><td>30.588761</td><td>30.349582</td><td>0.6666667</td><td> 72.14355</td><td>1.0000000</td><td> 99.76932</td></tr>
	<tr><th scope=row>result.15</th><td> 21.192349</td><td>25.806712</td><td>21.192349</td><td>0.6666667</td><td> 71.39094</td><td>1.0000000</td><td>101.02401</td></tr>
	<tr><th scope=row>result.16</th><td> 23.193143</td><td>25.914875</td><td>23.193143</td><td>0.6666667</td><td> 91.70660</td><td>1.0000000</td><td> 76.57925</td></tr>
	<tr><th scope=row>result.17</th><td> 30.081542</td><td>30.679960</td><td>30.081542</td><td>0.3333333</td><td>111.58689</td><td>1.0000000</td><td> 75.78459</td></tr>
	<tr><th scope=row>result.18</th><td> -6.530509</td><td> 9.111376</td><td> 6.999059</td><td>1.0000000</td><td> 69.51704</td><td>1.0000000</td><td>106.31714</td></tr>
	<tr><th scope=row>result.19</th><td> 19.907586</td><td>23.010762</td><td>19.907586</td><td>1.0000000</td><td> 67.03506</td><td>1.0000000</td><td>102.52128</td></tr>
	<tr><th scope=row>result.20</th><td> 17.631089</td><td>19.829355</td><td>17.631089</td><td>1.0000000</td><td> 67.97573</td><td>1.0000000</td><td>103.95991</td></tr>
	<tr><th scope=row>result.21</th><td> 11.738022</td><td>14.718185</td><td>12.229846</td><td>1.0000000</td><td> 61.61617</td><td>1.0000000</td><td> 94.23380</td></tr>
	<tr><th scope=row>result.22</th><td>-21.787490</td><td>28.489509</td><td>21.787490</td><td>0.6666667</td><td> 93.30920</td><td>1.0000000</td><td> 88.70090</td></tr>
	<tr><th scope=row>result.23</th><td>-43.557571</td><td>47.527244</td><td>43.557571</td><td>0.3333333</td><td>206.77368</td><td>0.6666667</td><td>216.50078</td></tr>
	<tr><th scope=row>result.24</th><td>-34.473558</td><td>35.514155</td><td>34.473558</td><td>0.3333333</td><td>146.63288</td><td>0.6666667</td><td>173.17046</td></tr>
	<tr><th scope=row>result.25</th><td> -4.699360</td><td>10.498595</td><td> 7.201224</td><td>1.0000000</td><td> 60.07550</td><td>1.0000000</td><td> 91.87755</td></tr>
	<tr><th scope=row>result.26</th><td> 25.974138</td><td>26.581272</td><td>25.974138</td><td>1.0000000</td><td> 63.01942</td><td>1.0000000</td><td> 96.37989</td></tr>
	<tr><th scope=row>result.27</th><td> 16.905109</td><td>19.474600</td><td>16.905109</td><td>1.0000000</td><td> 58.04472</td><td>1.0000000</td><td> 88.77173</td></tr>
	<tr><th scope=row>result.28</th><td> 15.218760</td><td>16.352917</td><td>15.218760</td><td>1.0000000</td><td> 55.27721</td><td>1.0000000</td><td> 84.53920</td></tr>
	<tr><th scope=row>result.29</th><td>  7.625241</td><td> 8.933828</td><td> 7.625241</td><td>1.0000000</td><td> 55.27718</td><td>1.0000000</td><td> 84.53916</td></tr>
	<tr><th scope=row>result.30</th><td>  2.261970</td><td>17.595326</td><td>15.666212</td><td>1.0000000</td><td> 57.13292</td><td>1.0000000</td><td> 87.37725</td></tr>
	<tr><th scope=row>⋮</th><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><th scope=row>result.103</th><td>  95.047754</td><td>111.26440</td><td> 95.04775</td><td>0.3333333</td><td> 485.7096</td><td>0.6666667</td><td> 594.3549</td></tr>
	<tr><th scope=row>result.104</th><td> 121.335201</td><td>125.76554</td><td>121.33520</td><td>0.0000000</td><td> 646.5750</td><td>0.3333333</td><td> 772.4818</td></tr>
	<tr><th scope=row>result.105</th><td>  27.661546</td><td> 53.66952</td><td> 52.33567</td><td>0.6666667</td><td> 149.4669</td><td>1.0000000</td><td> 226.7499</td></tr>
	<tr><th scope=row>result.106</th><td> -82.928463</td><td>106.53675</td><td> 87.39838</td><td>0.3333333</td><td> 439.0476</td><td>0.6666667</td><td> 391.0034</td></tr>
	<tr><th scope=row>result.107</th><td>-168.429957</td><td>174.86402</td><td>168.42996</td><td>0.0000000</td><td>1125.8534</td><td>0.0000000</td><td>2680.3671</td></tr>
	<tr><th scope=row>result.108</th><td> -86.047368</td><td> 89.34969</td><td> 86.04737</td><td>0.6666667</td><td> 241.5086</td><td>1.0000000</td><td> 281.3325</td></tr>
	<tr><th scope=row>result.109</th><td> -35.392983</td><td> 38.64620</td><td> 35.39298</td><td>1.0000000</td><td> 192.3314</td><td>1.0000000</td><td> 294.1455</td></tr>
	<tr><th scope=row>result.110</th><td>  32.273683</td><td> 33.69167</td><td> 32.27368</td><td>1.0000000</td><td> 199.9978</td><td>1.0000000</td><td> 305.8702</td></tr>
	<tr><th scope=row>result.111</th><td>  35.911969</td><td> 45.52857</td><td> 35.91197</td><td>1.0000000</td><td> 195.2069</td><td>1.0000000</td><td> 298.5432</td></tr>
	<tr><th scope=row>result.112</th><td>  28.584481</td><td> 41.79144</td><td> 38.16654</td><td>1.0000000</td><td> 196.5409</td><td>1.0000000</td><td> 300.5833</td></tr>
	<tr><th scope=row>result.113</th><td>  78.144295</td><td> 79.31310</td><td> 78.14430</td><td>1.0000000</td><td> 196.9343</td><td>1.0000000</td><td> 301.1850</td></tr>
	<tr><th scope=row>result.114</th><td>  37.152546</td><td> 52.61404</td><td> 39.21044</td><td>1.0000000</td><td> 192.5487</td><td>1.0000000</td><td> 294.4778</td></tr>
	<tr><th scope=row>result.115</th><td>  95.078342</td><td>110.88602</td><td> 95.07834</td><td>0.6666667</td><td> 366.3676</td><td>1.0000000</td><td> 274.9151</td></tr>
	<tr><th scope=row>result.116</th><td> 109.166178</td><td>116.17612</td><td>109.16618</td><td>0.3333333</td><td> 406.7397</td><td>1.0000000</td><td> 277.4405</td></tr>
	<tr><th scope=row>result.117</th><td>  41.289554</td><td> 62.02085</td><td> 57.33490</td><td>0.3333333</td><td> 215.1577</td><td>1.0000000</td><td> 222.4127</td></tr>
	<tr><th scope=row>result.118</th><td> -92.399494</td><td>116.61777</td><td> 92.82407</td><td>0.3333333</td><td> 466.7285</td><td>0.6666667</td><td> 445.3571</td></tr>
	<tr><th scope=row>result.119</th><td>-175.618445</td><td>183.27955</td><td>175.61845</td><td>0.0000000</td><td>1143.5479</td><td>0.0000000</td><td>2574.2409</td></tr>
	<tr><th scope=row>result.120</th><td> -94.580461</td><td> 97.36039</td><td> 94.58046</td><td>0.6666667</td><td> 277.7847</td><td>1.0000000</td><td> 293.2590</td></tr>
	<tr><th scope=row>result.121</th><td> -27.751828</td><td> 32.93559</td><td> 27.75183</td><td>1.0000000</td><td> 202.1374</td><td>1.0000000</td><td> 309.1425</td></tr>
	<tr><th scope=row>result.122</th><td>  36.177008</td><td> 38.16646</td><td> 36.17701</td><td>1.0000000</td><td> 208.6352</td><td>1.0000000</td><td> 319.0800</td></tr>
	<tr><th scope=row>result.123</th><td>   5.992278</td><td> 14.16185</td><td> 13.99743</td><td>1.0000000</td><td> 200.0098</td><td>1.0000000</td><td> 305.8885</td></tr>
	<tr><th scope=row>result.124</th><td>  12.637863</td><td> 33.65269</td><td> 27.98030</td><td>1.0000000</td><td> 200.1828</td><td>1.0000000</td><td> 306.1532</td></tr>
	<tr><th scope=row>result.125</th><td>  71.834372</td><td> 76.95073</td><td> 71.83437</td><td>1.0000000</td><td> 200.5753</td><td>1.0000000</td><td> 306.7534</td></tr>
	<tr><th scope=row>result.126</th><td>  85.518711</td><td> 93.75094</td><td> 85.51871</td><td>0.6666667</td><td> 252.5638</td><td>1.0000000</td><td> 295.0496</td></tr>
	<tr><th scope=row>result.127</th><td>  94.429064</td><td>115.52397</td><td> 94.42906</td><td>0.6666667</td><td> 407.3636</td><td>0.6666667</td><td> 417.2566</td></tr>
	<tr><th scope=row>result.128</th><td> 173.325805</td><td>177.66652</td><td>173.32580</td><td>0.0000000</td><td>1129.6141</td><td>0.0000000</td><td>2547.8618</td></tr>
	<tr><th scope=row>result.129</th><td>  33.890665</td><td> 63.84191</td><td> 61.66861</td><td>0.6666667</td><td> 242.6901</td><td>1.0000000</td><td> 230.3885</td></tr>
	<tr><th scope=row>result.130</th><td>-119.059067</td><td>137.73685</td><td>119.05907</td><td>0.3333333</td><td> 619.4166</td><td>0.3333333</td><td> 668.9786</td></tr>
	<tr><th scope=row>result.131</th><td>-180.821172</td><td>190.45241</td><td>180.82117</td><td>0.0000000</td><td>1152.4949</td><td>0.0000000</td><td>2469.3936</td></tr>
	<tr><th scope=row>result.132</th><td>-103.156396</td><td>108.61881</td><td>103.15640</td><td>0.6666667</td><td> 330.0400</td><td>1.0000000</td><td> 302.1675</td></tr>
</tbody>
</table>



             ME        RMSE         MAE  Coverage80   Winkler80  Coverage95 
      2.6570822  51.4271704  46.5118747   0.6590909 218.4527816   0.8459596 
      Winkler95 
    312.1383104 


# Example 2


```R
eval_metric <- function(predicted, observed)
{
  error <- observed - predicted$mean

  me <- mean(error)
  rmse <- sqrt(mean(error^2))
  mae <- mean(abs(error))

  # Only one interval returned
  lower <- predicted$lower
  upper <- predicted$upper

  coverage <- mean(
    observed >= lower & observed <= upper
  )

  alpha <- 0.05

  winkler <- ifelse(
    observed < lower,
    (upper - lower) + (2 / alpha) * (lower - observed),
    ifelse(
      observed > upper,
      (upper - lower) + (2 / alpha) * (observed - upper),
      (upper - lower)
    )
  )

  c(
    ME = me,
    RMSE = rmse,
    MAE = mae,
    Coverage95 = coverage,
    Winkler95 = mean(winkler)
  )
}

fcast_func <- function(y, h, ...)
{
  forecast::thetaf(
    y,
    h = h,
    level = 95
  )
}

res <- crossval_ts(
  y = AirPassengers,
  initial_window = 10,
  horizon = 3,
  fcast_func = fcast_func,
  eval_metric = eval_metric
)

print(colMeans(res))
```

      |======================================================================| 100%
             ME        RMSE         MAE  Coverage95   Winkler95 
      2.6570822  51.4271704  46.5118747   0.8459596 312.1383104 



```R
boxplot(res[, "Coverage95"])
```


    
![image-title-here]({{base}}/images/2026-05-16/2026-05-16-crossvalidation_10_0.png){:class="img-responsive"}
    

