---
layout: post
title: "Counterfactual Scenario Analysis with ahead::ridge2f"
description: "Counterfactual Scenario Analysis with ahead::ridge2f"
date: 2025-12-11
categories: [R, Python]
comments: true
---

In this post, we will explore how to perform counterfactual scenario analysis using the `ridge2f` function from the `ahead` package in R. 

Counterfactual scenario analysis is a powerful tool for understanding the impact of different scenarios on time series data. It allows us to evaluate the performance of different models under different scenarios, and to compare the performance of different models under different scenarios.

We will use the `insurance` dataset from the `ahead` package, which contains monthly data on insurance quotes and TV advertising.

The data set is split into three parts:

- `train`: historical data to learn from
- `scenario`: period where we apply "what-if" scenarios
- `test`: true future where we evaluate forecasts

The `train` data is used to fit the model, the `scenario` data is used to generate counterfactual scenarios, and the `test` data is used to evaluate the performance of the model under the counterfactual scenarios.

```R
install.packages("remotes")
install.packages("gridExtra")

remotes::install_github("Techtonique/ahead")
```

```R
url <- "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/multivariate/insurance_quotes_advert.csv"
insurance <- read.csv(url)
```


```R
insurance <- ts(insurance, start=c(2002, 1), frequency = 12L)
```


```R
library(ahead)
library(ggplot2)
library(gridExtra)

y <- insurance[, "Quotes"]
TV <- insurance[, "TV.advert"]
t  <- as.numeric(time(insurance))
T <- length(y)

# ========== 3-PART SPLIT ==========
# TRAIN: Historical data to learn from
# SCENARIO: Period where we apply "what-if" scenarios
# TEST: True future where we evaluate forecasts

train_pct <- 0.50
scenario_pct <- 0.30
# test_pct <- 0.20 (remainder)

train_end <- floor(train_pct * T)
scenario_end <- floor((train_pct + scenario_pct) * T)

# Split data
y_train <- y[1:train_end]
y_scenario <- y[(train_end + 1):scenario_end]
y_test <- y[(scenario_end + 1):T]

TV_train <- TV[1:train_end]
TV_scenario <- TV[(train_end + 1):scenario_end]
TV_test <- TV[(scenario_end + 1):T]

t_train <- t[1:train_end]
t_scenario <- t[(train_end + 1):scenario_end]
t_test <- t[(scenario_end + 1):T]

h_test <- length(y_test)

cat("\n=== 3-PART DATA SPLIT ===\n")
cat("TRAIN:    periods", 1, "to", train_end, "(n =", train_end, ")\n")
cat("SCENARIO: periods", train_end + 1, "to", scenario_end, "(n =", length(y_scenario), ")\n")
cat("TEST:     periods", scenario_end + 1, "to", T, "(n =", h_test, ")\n\n")

# ========== THE KEY INSIGHT ==========
cat("=== THE APPROACH ===\n")
cat("1. Train on: TRAIN data only\n")
cat("2. Apply scenarios to: SCENARIO period (with actual y values)\n")
cat("3. Forecast into: TEST period (what we're evaluating)\n")
cat("4. Compare: How do different SCENARIO assumptions affect TEST forecasts?\n\n")

# ========== DEFINE SCENARIOS ==========
# Baseline: TV advertising unchanged
# Scenario A: TV advertising was +1 higher during scenario period
# Scenario B: TV advertising was -1 lower during scenario period

TV_scenario_A <- TV_scenario + 1
TV_scenario_B <- TV_scenario - 1

cat("=== SCENARIOS ===\n")
cat("Scenario A: TV during scenario period = actual +1\n")
cat("Scenario B: TV during scenario period = actual -1\n")
cat("(We're asking: 'What if TV had been different in the recent past?')\n\n")

# ========== BUILD TRAINING DATA WITH SCENARIOS ==========

# For Baseline scenario
y_train_Baseline <- c(y_train, y_scenario)
xreg_train_Baseline <- rbind(
  cbind(TV = TV_train, trend = t_train),
  cbind(TV = TV_scenario, trend = t_scenario)
)

# For Scenario A: combine TRAIN + SCENARIO (with modified TV)
y_train_A <- y_train_Baseline
xreg_train_A <- rbind(
  cbind(TV = TV_train, trend = t_train),
  cbind(TV = TV_scenario_A, trend = t_scenario)
)

# For Scenario B: combine TRAIN + SCENARIO (with different TV)
y_train_B <- y_train_Baseline
xreg_train_B <- rbind(
  cbind(TV = TV_train, trend = t_train),
  cbind(TV = TV_scenario_B, trend = t_scenario)
)

# ========== FIT MODELS ==========


cat("Fitting Scenario A model...\n")
set.seed(123)
res_Baseline <- ridge2f(
  y_train_Baseline,
  h = h_test,
  xreg = xreg_train_Baseline,
  lags = 5,
  type_pi = "blockbootstrap",
  B = 200
)

cat("Fitting Scenario A model...\n")
set.seed(123)
res_A <- ridge2f(
  y_train_A,
  h = h_test,
  xreg = xreg_train_A,
  lags = 5,
  type_pi = "blockbootstrap",
  B = 200
)

cat("Fitting Scenario B model...\n")
set.seed(123)
res_B <- ridge2f(
  y_train_B,
  h = h_test,
  xreg = xreg_train_B,
  lags = 5,
  type_pi = "blockbootstrap",
  B = 200
)

# ========== COMPARISON TABLE ==========
comparison <- data.frame(
  Period = time(y_test),
  Actual = as.numeric(y_test),
  Forecast_Baseline = as.numeric(res_Baseline$mean),
  Forecast_A = as.numeric(res_A$mean),
  Forecast_B = as.numeric(res_B$mean),
  Diff_A_B = as.numeric(res_A$mean) - as.numeric(res_B$mean),
  Diff_A_Baseline = as.numeric(res_A$mean) - as.numeric(res_Baseline$mean),
  Diff_B_Baseline = as.numeric(res_B$mean) - as.numeric(res_Baseline$mean),
  Impact_A = as.numeric(res_A$mean) - as.numeric(y_test),
  Impact_B = as.numeric(res_B$mean) - as.numeric(y_test),
  Lower_A = as.numeric(res_A$lower),
  Upper_A = as.numeric(res_A$upper),
  Lower_B = as.numeric(res_B$lower),
  Upper_B = as.numeric(res_B$upper)
)

cat("\n=== TEST PERIOD FORECASTS ===\n")
print(round(comparison, 2))


# ========== SCENARIO IMPACT ==========

colnames_comparison <- colnames(comparison)
print(summary(comparison[, 6:10]))
for (i in 6:10)
{
  print(colnames_comparison[i])
  print(t.test(comparison[, i]))
}

# ========== COVERAGE ANALYSIS ==========
in_A <- sum(comparison$Actual >= comparison$Lower_A &
              comparison$Actual <= comparison$Upper_A)
in_B <- sum(comparison$Actual >= comparison$Lower_B &
              comparison$Actual <= comparison$Upper_B)
coverage_A <- in_A / h_test * 100
coverage_B <- in_B / h_test * 100

cat("\n=== PREDICTION INTERVAL COVERAGE ===\n")
cat(sprintf("Scenario A: %.1f%% (%d/%d)\n", coverage_A, in_A, h_test))
cat(sprintf("Scenario B: %.1f%% (%d/%d)\n", coverage_B, in_B, h_test))

# ========== PLOTS ==========


```

    
    === 3-PART DATA SPLIT ===
    TRAIN:    periods 1 to 20 (n = 20 )
    SCENARIO: periods 21 to 32 (n = 12 )
    TEST:     periods 33 to 40 (n = 8 )
    
    === THE APPROACH ===
    1. Train on: TRAIN data only
    2. Apply scenarios to: SCENARIO period (with actual y values)
    3. Forecast into: TEST period (what we're evaluating)
    4. Compare: How do different SCENARIO assumptions affect TEST forecasts?
    
    === SCENARIOS ===
    Scenario A: TV during scenario period = actual +1
    Scenario B: TV during scenario period = actual -1
    (We're asking: 'What if TV had been different in the recent past?')
    
    Fitting Scenario A model...
      |======================================================================| 100%
    Fitting Scenario A model...
      |======================================================================| 100%
    Fitting Scenario B model...
      |======================================================================| 100%
    
    === TEST PERIOD FORECASTS ===
      Period Actual Forecast_Baseline Forecast_A Forecast_B Diff_A_B
    1      1  12.86             12.44      12.08      12.49    -0.42
    2      2  12.09             12.16      11.43      12.75    -1.33
    3      3  12.93             11.49      10.29      11.53    -1.24
    4      4  11.72             10.70       9.23       9.41    -0.19
    5      5  15.47             10.93      11.11       8.83     2.28
    6      6  18.44             11.47      14.21      11.22     2.99
    7      7  17.49             12.03      14.38      14.36     0.02
    8      8  14.49             12.59      13.28      16.04    -2.76
      Diff_A_Baseline Diff_B_Baseline Impact_A Impact_B Lower_A Upper_A Lower_B
    1           -0.36            0.05    -0.78    -0.37   10.97   12.80   10.50
    2           -0.74            0.59    -0.66     0.66   10.51   12.12   10.58
    3           -1.20            0.04    -2.64    -1.40    7.65   12.97    9.90
    4           -1.47           -1.28    -2.50    -2.31    6.77   11.75    6.25
    5            0.18           -2.10    -4.36    -6.64    8.64   13.69    5.16
    6            2.74           -0.24    -4.23    -7.21   10.70   18.00    6.87
    7            2.35            2.33    -3.11    -3.13   11.35   19.12    9.14
    8            0.69            3.45    -1.21     1.54   10.56   17.72   12.62
      Upper_B
    1   13.81
    2   15.34
    3   13.73
    4   12.89
    5   12.38
    6   15.08
    7   19.60
    8   20.18
        Diff_A_B        Diff_A_Baseline    Diff_B_Baseline       Impact_A      
     Min.   :-2.75662   Min.   :-1.46811   Min.   :-2.10057   Min.   :-4.3592  
     1st Qu.:-1.26217   1st Qu.:-0.85445   1st Qu.:-0.50317   1st Qu.:-3.3915  
     Median :-0.30014   Median :-0.09231   Median : 0.04573   Median :-2.5692  
     Mean   :-0.08017   Mean   : 0.27400   Mean   : 0.35417   Mean   :-2.4371  
     3rd Qu.: 0.58402   3rd Qu.: 1.10542   3rd Qu.: 1.02461   3rd Qu.:-1.1051  
     Max.   : 2.98501   Max.   : 2.74174   Max.   : 3.44581   Max.   :-0.6627  
        Impact_B      
     Min.   :-7.2143  
     1st Qu.:-4.0080  
     Median :-1.8562  
     Mean   :-2.3569  
     3rd Qu.:-0.1095  
     Max.   : 1.5440  
    [1] "Diff_A_B"
    
    	One Sample t-test
    
    data:  comparison[, i]
    t = -0.11961, df = 7, p-value = 0.9082
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -1.665035  1.504705
    sample estimates:
      mean of x 
    -0.08016502 
    
    [1] "Diff_A_Baseline"
    
    	One Sample t-test
    
    data:  comparison[, i]
    t = 0.49381, df = 7, p-value = 0.6366
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -1.038059  1.586063
    sample estimates:
    mean of x 
    0.2740019 
    
    [1] "Diff_B_Baseline"
    
    	One Sample t-test
    
    data:  comparison[, i]
    t = 0.55518, df = 7, p-value = 0.5961
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -1.154295  1.862629
    sample estimates:
    mean of x 
    0.3541669 
    
    [1] "Impact_A"
    
    	One Sample t-test
    
    data:  comparison[, i]
    t = -4.7416, df = 7, p-value = 0.002104
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -3.652457 -1.221723
    sample estimates:
    mean of x 
     -2.43709 
    
    [1] "Impact_B"
    
    	One Sample t-test
    
    data:  comparison[, i]
    t = -2.0825, df = 7, p-value = 0.07581
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -5.0332109  0.3193609
    sample estimates:
    mean of x 
    -2.356925 
    
    
    === PREDICTION INTERVAL COVERAGE ===
    Scenario A: 62.5% (5/8)
    Scenario B: 75.0% (6/8)



```R
library(ggplot2)
library(tidyr)

df_long <- comparison[, 6:10] %>%
  pivot_longer(cols = everything(),
               names_to = "Variable",
               values_to = "Value")

# Create violin plot
ggplot(df_long, aes(x = Variable, y = Value, fill = Variable)) +
  geom_violin(trim = FALSE) +
  geom_jitter(width = 0.1, size = 1, alpha = 0.7) + # optional: show individual points
  theme_minimal() +
  labs(title = "Violin Plot of Comparison Columns",
       y = "Value",
       x = "Variable")

```


    
![image-title-here]({{base}}/images/2025-12-11/2025-12-11-Counterfactual-Scenario-Analysis-with-ridge2f_4_0.png){:class="img-responsive"}
    

