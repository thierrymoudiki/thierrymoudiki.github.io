---
layout: post
title: "Stock price forecasting with Deep Learning: throwing power at the problem (and why it won't make you rich)"
description: "Stock price forecasting with Deep Learning: throwing power at the problem (and why it won't make you rich)"
date: 2024-12-29
categories: R
comments: true
---

You may (surely) have seen many posts on **Stock price forecasting with Deep Learning** on the web. LSTMs, GANs, Transformers, etc. You name it. Sometimes with beautiful graphics showing how well the forecast tracks the actual data. I'll show you how to create these exact same graphics without cheating, overfitting, leakage, or GPUs. I'll use **Random Walk forecasts** (predicting the next day's value as the last day's value), and try to tell you why these **Stock price forecasting with Deep Learning** posts are deceptive and misleading.  

# Packages 

We start by installing the packages we'll need.

```
utils::install.packages("pak")

pak::pkg_install(c("tseries", "forecast", "caret", 
                   "fpp2", "quantmod"))
```


# Function for studying the data 

We'll use the `stock_forecasting_summary` function to study the data. It takes a stock's identifier as input and returns the observed and forecast values and proportion of correct guesses about the direction of stock's price.

```
stock_forecasting_summary <- function(ticker = c("DAX", "CAC", 
                                                 "SMI", "FTSE", 
                                                 "GOOG", "AAPL", "MSFT"))
{
  data(EuStockMarkets)
  # stock's identifier
  ticker <- match.arg(ticker)
  if (ticker %in% c("DAX", "CAC", "SMI", "FTSE"))
  {
   y <- EuStockMarkets[, ticker] 
  }
  if (ticker == "GOOG"){
    y <- fpp2::goog
  }
  if (ticker %in% c("AAPL", "MSFT"))
  {
    quantmod::getSymbols(ticker, src="yahoo")
    y <- switch(ticker,
                AAPL=as.numeric(AAPL[,4]),
                MSFT=as.numeric(MSFT[,4]))
  }

# distribution of stock's price
print(summary(y))
n <- length(y)
half_n <- n%/%2

# Create time slices for rolling window 1-day ahead forecasting  
time_slices <- caret::createTimeSlices(y, initialWindow = half_n,
fixedWindow=FALSE)

n_slices <- length(time_slices$train)

observed <- ts(rep(0, n_slices), end=end(y),
frequency = frequency(y))
forecasts <- ts(rep(0, n_slices), end=end(y),
frequency = frequency(y))

# correct guesses = number of times we get the right direction of change for stock price
correct_guesses <- rep(0, n_slices-1)

# Loop over time slices to get the observed and forecast values and correct guesses
for (i in seq_len(n_slices))
{
  observed[i] <- y[time_slices$test[[i]]]
  # 1-day ahead Random Walk forecast 
  forecasts[i] <- forecast::rwf(y[time_slices$train[[i]]], h=1)$mean
  if (i >= 2)
    correct_guesses[i] <- (base::sign((observed[i]-observed[i-1])*(forecasts[i] - forecasts[i-1])) == 1)
}

# Plot the observed and forecast values and correct guesses
par(mfrow = c(2, 2))
plot(observed, col="blue", lwd=2, type = 'l', 
     main="observed and forecast")
lines(forecasts, col="red", type = 'l')
plot(observed, forecasts, type='p', 
     main="observed vs forecast")
abline(a = 0, b = 1, lty=2, col="green")
plot(observed-forecasts, type='l',
     main="forecasting residuals")
hist(correct_guesses, border=FALSE)

resids <- observed-forecasts
print(summary(lm(resids ~ seq_len(length(resids)))))

cat("proportion of ")
print(table(correct_guesses)/length(correct_guesses))
}
```

# Examples 

Here are examples of use of `stock_forecasting_summary` on multiple stock prices time series examples (DAX, CAC, SMI, FTSE, GOOG, AAPL, MSFT).

```
stock_forecasting_summary("DAX")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       1402    1744    2141    2531    2722    6186 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_4_1.png){:class="img-responsive"}  
    
    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -231.25  -15.54   -0.44   16.78  155.41 
    
    Coefficients:
                            Estimate Std. Error t value Pr(>|t|)
    (Intercept)             0.285300   2.778666   0.103    0.918
    seq_len(length(resids)) 0.007294   0.005171   1.411    0.159
    
    Residual standard error: 42.33 on 928 degrees of freedom
    Multiple R-squared:  0.002139,  Adjusted R-squared:  0.001064 
    F-statistic:  1.99 on 1 and 928 DF,  p-value: 0.1587
    
    proportion of correct_guesses
            0         1 
    0.5569892 0.4430108 


The first 2 graphics look just perfect -- the same observation holds for the stocks studied in the next examples. No overfitting, no leakage as you noticed. Try to beat them with Deep Learning, Transformer, LSTM, etc. And **Good luck**! ;) (more details also in [the pdf file https://cbergmeir.com/talks/neurips2024/](https://cbergmeir.com/talks/neurips2024/)). And **it's not about simplicity**, but about the inherent nature of the problem/data. 

The residuals (difference between observed and forecast values) reveal a more interesting reality. These little _pesky_   (increasing, decreasing, around 0, as high and low as -200 and 200, with a volatility that changes with time) residuals are the reason why you won't become rich by applying advanced neural networks (or anything) to stock price forecasting. **Why?**

(One of?) The point(s) of stock price forecasting is to correctly "guess" how a stock's price will move tomorrow. Even though it may either leave you in trouble or make you rich (to actually know the future), see [https://eu.usatoday.com/story/news/factcheck/2024/12/12/unitedhealthcare-ceo-nancy-pelosi-insider-trading-fact-check/76919053007/](https://eu.usatoday.com/story/news/factcheck/2024/12/12/unitedhealthcare-ceo-nancy-pelosi-insider-trading-fact-check/76919053007/) and [https://www.lemonde.fr/argent/article/2017/02/21/assurance-vie-la-mauvaise-foi-d-aviva_5082920_1657007.html](https://www.lemonde.fr/argent/article/2017/02/21/assurance-vie-la-mauvaise-foi-d-aviva_5082920_1657007.html). However, in addition to the fact that their distribution is symmetric around 0 and they have a volatility that changes with time, we observe that there's no trend in the residuals (slope is not significantly different from 0). 

IMHO, anyone claiming to forecast stock prices with more GPU power must at least be able to beat Random Walk forecasts, because that's the hard part.

The other examples reveal the same patterns. Can you beat these Random Walk forecasts with Deep Learning, Transformer, LSTM, etc?

```
stock_forecasting_summary("SMI")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       1587    2166    2796    3376    3812    8412 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_5_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -282.771  -18.270   -0.341   22.373  230.626 
    
    Coefficients:
                            Estimate Std. Error t value Pr(>|t|)
    (Intercept)             1.566404   3.458020   0.453    0.651
    seq_len(length(resids)) 0.008420   0.006435   1.308    0.191
    
    Residual standard error: 52.69 on 928 degrees of freedom
    Multiple R-squared:  0.001841,  Adjusted R-squared:  0.0007657 
    F-statistic: 1.712 on 1 and 928 DF,  p-value: 0.1911
    
    proportion of correct_guesses
            0         1 
    0.5182796 0.4817204 


```
stock_forecasting_summary("CAC")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       1611    1875    1992    2228    2274    4388 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_6_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -122.064  -14.298   -0.715   15.507  162.931 
    
    Coefficients:
                             Estimate Std. Error t value Pr(>|t|)
    (Intercept)             -0.227564   2.011967  -0.113     0.91
    seq_len(length(resids))  0.005528   0.003744   1.477     0.14
    
    Residual standard error: 30.65 on 928 degrees of freedom
    Multiple R-squared:  0.002344,  Adjusted R-squared:  0.001269 
    F-statistic:  2.18 on 1 and 928 DF,  p-value: 0.1401
    
    proportion of correct_guesses
            0         1 
    0.5419355 0.4580645 


```
stock_forecasting_summary("FTSE")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       2281    2843    3247    3566    3994    6179 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_7_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -159.722  -18.179   -0.108   18.415  158.361 
    
    Coefficients:
                              Estimate Std. Error t value Pr(>|t|)
    (Intercept)              3.0524685  2.4188420   1.262    0.207
    seq_len(length(resids)) -0.0008771  0.0045013  -0.195    0.846
    
    Residual standard error: 36.85 on 928 degrees of freedom
    Multiple R-squared:  4.091e-05, Adjusted R-squared:  -0.001037 
    F-statistic: 0.03797 on 1 and 928 DF,  p-value: 0.8456
    
    proportion of correct_guesses
           0        1 
    0.511828 0.488172 


```
stock_forecasting_summary("GOOG")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      380.5   524.9   568.9   599.4   716.9   835.7 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_8_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -40.892  -4.723  -0.234   4.676  92.424 
    
    Coefficients:
                             Estimate Std. Error t value Pr(>|t|)
    (Intercept)              0.727695   0.916782   0.794    0.428
    seq_len(length(resids)) -0.000694   0.003171  -0.219    0.827
    
    Residual standard error: 10.23 on 498 degrees of freedom
    Multiple R-squared:  9.617e-05, Adjusted R-squared:  -0.001912 
    F-statistic: 0.0479 on 1 and 498 DF,  p-value: 0.8269
    
    proportion of correct_guesses
       0    1 
    0.51 0.49 


```
stock_forecasting_summary("AAPL")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      2.793  12.726  28.247  58.862  91.071 259.020 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_9_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -10.7823  -0.6651   0.0107   0.7106  13.8410 
    
    Coefficients:
                             Estimate Std. Error t value Pr(>|t|)
    (Intercept)             4.345e-04  8.647e-02   0.005    0.996
    seq_len(length(resids)) 8.870e-05  6.613e-05   1.341    0.180
    
    Residual standard error: 2.057 on 2262 degrees of freedom
    Multiple R-squared:  0.0007947, Adjusted R-squared:  0.0003529 
    F-statistic: 1.799 on 1 and 2262 DF,  p-value: 0.18
    
    proportion of correct_guesses
            0         1 
    0.5061837 0.4938163 


```
stock_forecasting_summary("MSFT")
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      15.15   29.36   51.20  117.34  200.57  467.56 


    
![xxx]({{base}}/images/2024-12-29/2024-12-29-mlf_10_1.png){:class="img-responsive"}  
    


    
    Call:
    lm(formula = resids ~ seq_len(length(resids)))
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -26.4489  -1.2892  -0.0007   1.4016  19.7173 
    
    Coefficients:
                             Estimate Std. Error t value Pr(>|t|)
    (Intercept)             5.767e-02  1.588e-01   0.363    0.717
    seq_len(length(resids)) 9.494e-05  1.214e-04   0.782    0.434
    
    Residual standard error: 3.777 on 2262 degrees of freedom
    Multiple R-squared:  0.0002701, Adjusted R-squared:  -0.0001719 
    F-statistic: 0.611 on 1 and 2262 DF,  p-value: 0.4345
    
    proportion of correct_guesses
            0         1 
    0.5273852 0.4726148 


