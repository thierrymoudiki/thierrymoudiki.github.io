---
layout: post
title: "Deterministic Shift Adjustment in Arbitrage-Free Pricing (historical to risk-neutral short rates)"
description: "This post demonstrates the implementation of deterministic shift adjustments
  in an arbitrage-free pricing framework using R and the NMOF package. I
  generate synthetic yield curve data using Nelson-Siegel and
  Nelson-Siegel-Svensson models, estimate short rates using three different
  methods, simulate interest rate paths, compute unadjusted zero-coupon bond
  prices, apply deterministic shift adjustments, and evaluate the results."  
date: 2025-10-28
categories: R
comments: true
---


```R
install.packages(c('randomForest', 'NMOF'))
```


```R
install.packages('gridExtra')
```


```R
library(tidyverse)
library(randomForest)
library(NMOF)  # For NS() and NSS() functions

set.seed(123)

# ============================================================================
# 1. Generate Synthetic Yield Curve Data (Vectorized with NMOF)
# ============================================================================
generate_yield_curve_ns <- function(n_dates = 100,
                                     maturities = c(0.25, 0.5, 1, 2, 3, 5, 7, 10)) {
  dates <- 1:n_dates

  # Time-varying Nelson-Siegel factors
  beta1 <- 0.06 + 0.01 * sin(2 * pi * dates / 50) + rnorm(n_dates, 0, 0.002)
  beta2 <- -0.02 + cumsum(rnorm(n_dates, 0, 0.005)) / sqrt(dates)
  beta3 <- 0.01 + rnorm(n_dates, 0, 0.003)
  lambda <- 0.0609

  # Vectorized NS yield generation using NMOF::NS
  # NS(param, tm) where param = c(beta1, beta2, beta3, lambda)
  yields <- matrix(0, n_dates, length(maturities))
  for (i in 1:n_dates) {
    yields[i, ] <- NS(param = c(beta1[i], beta2[i], beta3[i], lambda),
                      tm = maturities)
  }

  list(yields = yields, maturities = maturities,
       beta1 = beta1, beta2 = beta2, beta3 = beta3, lambda = lambda,
       type = "NS")
}

generate_yield_curve_nss <- function(n_dates = 100,
                                      maturities = c(0.25, 0.5, 1, 2, 3, 5, 7, 10)) {
  dates <- 1:n_dates

  # Time-varying Nelson-Siegel-Svensson factors
  beta1 <- 0.06 + 0.01 * sin(2 * pi * dates / 50) + rnorm(n_dates, 0, 0.002)
  beta2 <- -0.02 + cumsum(rnorm(n_dates, 0, 0.005)) / sqrt(dates)
  beta3 <- 0.01 + rnorm(n_dates, 0, 0.003)
  beta4 <- 0.005 + rnorm(n_dates, 0, 0.002)  # Additional NSS factor
  lambda1 <- 0.0609
  lambda2 <- 0.50    # Second decay parameter for NSS

  # Vectorized NSS yield generation using NMOF::NSS
  # NSS(param, tm) where param = c(beta1, beta2, beta3, beta4, lambda1, lambda2)
  yields <- matrix(0, n_dates, length(maturities))
  for (i in 1:n_dates) {
    yields[i, ] <- NSS(param = c(beta1[i], beta2[i], beta3[i], beta4[i],
                                  lambda1, lambda2),
                       tm = maturities)
  }

  list(yields = yields, maturities = maturities,
       beta1 = beta1, beta2 = beta2, beta3 = beta3, beta4 = beta4,
       lambda1 = lambda1, lambda2 = lambda2,
       type = "NSS")
}

# ============================================================================
# 2. Method 1: NS/NSS Extrapolation to Zero Maturity (Analytical Limit)
# ============================================================================
method1_ns_extrapolation <- function(data) {
  if (data$type == "NS") {
    # r(t) = lim_{tau->0} NS(tau) = beta1 + beta2
    # Since lim_{tau->0} (1-exp(-tau/lambda))/(tau/lambda) = 1
    # and lim_{tau->0} exp(-tau/lambda) = 1
    short_rates <- data$beta1 + data$beta2
  } else if (data$type == "NSS") {
    # r(t) = lim_{tau->0} NSS(tau) = beta1 + beta2 + beta4
    # Both slope terms converge to 1 as tau->0
    short_rates <- data$beta1 + data$beta2 + data$beta4
  }
  short_rates
}

# ============================================================================
# 3. Method 2: NS/NSS Features with Random Forest (Using Small Tau)
# ============================================================================
method2_ns_ml <- function(data) {
  n_dates <- nrow(data$yields)
  short_rates <- numeric(n_dates)

  # Use very small tau to approximate zero (e.g., 1 day = 1/252 years)
  tau_tiny <- 1/252

  if (data$type == "NS") {
    for (i in 1:n_dates) {
      # Create NS features using NMOF::NSf (factor loadings)
      # NSf(lambda, tm) returns matrix: rows = maturities, cols = 3 factors
      loadings_matrix <- NSf(lambda = data$lambda, tm = data$maturities)

      # Convert to data frame
      features <- as.data.frame(loadings_matrix)
      colnames(features) <- c("L1", "L2", "L3")
      target <- data$yields[i, ]

      # Train RF
      rf <- randomForest(x = features, y = target, ntree = 50, nodesize = 2)

      # Predict at tau ≈ 0: Use analytical limit values (1, 1, 0)
      # OR use very small tau
      # Analytical approach (more accurate):
      features_zero <- data.frame(L1 = 1, L2 = 1, L3 = 0)

      short_rates[i] <- suppressWarnings(predict(rf, features_zero))
    }
  } else if (data$type == "NSS") {
    for (i in 1:n_dates) {
      # Create NSS features using NMOF::NSSf
      # NSSf(lambda1, lambda2, tm) returns matrix: rows = maturities, cols = 4 factors
      loadings_matrix <- NSSf(lambda1 = data$lambda1,
                              lambda2 = data$lambda2,
                              tm = data$maturities)

      # Convert to data frame
      features <- as.data.frame(loadings_matrix)
      colnames(features) <- c("L1", "L2", "L3", "L4")
      target <- data$yields[i, ]

      # Train RF
      rf <- randomForest(x = features, y = target, ntree = 50, nodesize = 2)

      # Predict at tau ≈ 0: Use analytical limit values (1, 1, 0, 0)
      # Note: L4 = (1-exp(-aux2))/aux2 - exp(-aux2) -> 1 - 1 = 0 as tau->0
      features_zero <- data.frame(L1 = 1, L2 = 1, L3 = 0, L4 = 0)

      short_rates[i] <- suppressWarnings(predict(rf, features_zero))
    }
  }

  short_rates
}

# ============================================================================
# 4. Method 3: Direct Linear Regression to Zero Maturity
# ============================================================================
method3_direct_regression <- function(data) {
  n_dates <- nrow(data$yields)
  short_rates <- numeric(n_dates)

  for (i in 1:n_dates) {
    # Fit linear model: R(tau) = a + b*tau
    model <- lm(data$yields[i, ] ~ data$maturities)
    # Extrapolate to tau=0
    short_rates[i] <- coef(model)[1]
  }

  short_rates
}

# ============================================================================
# 5. Simulate Short Rate Paths
# ============================================================================
simulate_paths <- function(short_rates, n_sims = 100,
                           horizon = 10, dt = 0.25) {
  n_steps <- horizon / dt
  paths <- matrix(0, n_sims, n_steps + 1)
  # AR(1) model
  r0 <- tail(short_rates, 1)
  mu <- mean(short_rates)
  phi <- 0.95
  sigma <- sd(diff(short_rates)) * sqrt(dt)

  paths[, 1] <- r0

  for (t in 2:(n_steps + 1)) {
    paths[, t] <- mu + phi * (paths[, t-1] - mu) + rnorm(n_sims, 0, sigma)
    paths[, t] <- pmax(paths[, t], 0.001)
  }
  paths
}

# ============================================================================
# 6. Compute Unadjusted ZCB Prices (Vectorized)
# ============================================================================
compute_unadjusted_zcb <- function(paths, maturities, dt = 0.25) {
   n_sims <- nrow(paths)
   zcb_prices <- numeric(length(maturities))

   for (i in 1:length(maturities)) {
     T_mat <- maturities[i]
     n_steps <- round(T_mat / dt)
     # Vectorized integration
     integrals <- rowSums(paths[, 1:n_steps, drop = FALSE]) * dt
     zcb_prices[i] <- mean(exp(-integrals))
   }
   zcb_prices
}

# ============================================================================
# 7. Market ZCB Prices
# ============================================================================
get_market_zcb <- function(data, date_idx, target_maturities) {
  yields <- data$yields[date_idx, ]
  maturities <- data$maturities
  # Interpolate yields
  yields_interp <- approx(maturities, yields, target_maturities)$y
  return(exp(-yields_interp * target_maturities))
}

# ============================================================================
# 8. Deterministic Shift Adjustment (Vectorized)
# ============================================================================
apply_deterministic_shift <- function(paths, zcb_unadjusted, zcb_market,
                                       maturities, dt = 0.25) {
  n_sims <- nrow(paths)
  # Compute simulated forward rates (vectorized)
  f_sim <- numeric(length(maturities))
  for (i in 1:length(maturities)) {
    T_mat <- maturities[i]
    n_steps <- round(T_mat / dt)
    r_T <- paths[, n_steps]
    integrals <- rowSums(paths[, 1:n_steps, drop = FALSE]) * dt
    discount <- exp(-integrals)
    f_sim[i] <- sum(r_T * discount) / sum(discount)
  }
  # Market forward rates
  f_market <- -diff(log(zcb_market)) / diff(maturities)
  f_market <- c(f_market[1], f_market)
  # Shift function
  phi <- f_market - f_sim
  # Adjusted ZCB prices (vectorized)
  zcb_adjusted <- numeric(length(maturities))
  for (i in 1:length(maturities)) {
    if (i == 1) {
      int_phi <- phi[1] * maturities[1]
    } else {
      int_phi <- sum(phi[1:i] * diff(c(0, maturities[1:i])))
    }
    zcb_adjusted[i] <- exp(-int_phi) * zcb_unadjusted[i]
  }
  return(list(zcb_adjusted = zcb_adjusted,
              phi = phi))
}

# ============================================================================
# 9. Calibrated Confidence Intervals for Adjusted Prices
# ============================================================================
compute_adjusted_confidence_intervals <- function(paths, zcb_market, maturities,
                                                   n_boot = 500, dt = 0.25,
                                                   alpha = 0.01) {
  n_sims <- nrow(paths)
  boot_adjusted_prices <- matrix(0, n_boot, length(maturities))

  for (b in 1:n_boot) {
    idx <- sample(1:n_sims, n_sims, replace = TRUE)
    boot_paths <- paths[idx, ]

    boot_unadj <- compute_unadjusted_zcb(boot_paths, maturities, dt)
    boot_adjustment <- apply_deterministic_shift(boot_paths, boot_unadj,
                                                  zcb_market, maturities, dt)
    boot_adjusted_prices[b, ] <- boot_adjustment$zcb_adjusted
  }

  ci_lower <- apply(boot_adjusted_prices, 2, quantile, probs = alpha/2)
  ci_upper <- apply(boot_adjusted_prices, 2, quantile, probs = 1 - alpha/2)
  return(list(lower = ci_lower, upper = ci_upper))
}

# ============================================================================
# 10. Run Analysis Function
# ============================================================================
run_analysis <- function(data, model_name) {
  cat(sprintf("\n========== %s ANALYSIS ==========\n\n", model_name))

  test_maturities <- c(1, 3, 5, 7, 10)
  zcb_market_test <- get_market_zcb(data, nrow(data$yields), test_maturities)

  cat("Market ZCB Prices:\n")
  print(data.frame(Maturity = test_maturities, Price = round(zcb_market_test, 6)))
  cat("\n")

  results_list <- list()

  # Method 1: NS/NSS Extrapolation
  cat("--- METHOD 1: Short rates \n with NS/NSS Extrapolation ---\n")
  r1 <- method1_ns_extrapolation(data)
  cat(sprintf("Short rate (last obs): %.4f%%\n", r1[length(r1)] * 100))

  paths1 <- simulate_paths(r1, n_sims = 100L)
  zcb_unadj1 <- compute_unadjusted_zcb(paths1, test_maturities)
  adjustment1 <- apply_deterministic_shift(paths1, zcb_unadj1, zcb_market_test,
                                           test_maturities)
  cat("Computing calibrated confidence intervals...\n")
  ci1 <- compute_adjusted_confidence_intervals(paths1, zcb_market_test,
                                                test_maturities, n_boot = 500)

  results1 <- data.frame(
    Maturity = test_maturities,
    Market = round(zcb_market_test, 6),
    Unadjusted = round(zcb_unadj1, 6),
    Adjusted = round(adjustment1$zcb_adjusted, 6),
    Error_pct = round(abs(adjustment1$zcb_adjusted - zcb_market_test) /
                        zcb_market_test * 100, 3),
    CI_Lower = round(ci1$lower, 6),
    CI_Upper = round(ci1$upper, 6),
    CI_Width_bps = round((ci1$upper - ci1$lower) * 10000, 1),
    Market_In_CI = zcb_market_test >= ci1$lower & zcb_market_test <= ci1$upper
  )

  cat("\nResults:\n")
  print(results1)
  cat("\n")
  results_list[[1]] <- results1 %>% mutate(Method = "Method 1: NS/NSS Extrapolation")

  # Method 2: NS/NSS + Random Forest
  cat("--- METHOD 2: Short rates \n with NS/NSS + Random Forest ---\n")
  r2 <- method2_ns_ml(data)
  cat(sprintf("Short rate (last obs): %.4f%%\n", r2[length(r2)] * 100))

  paths2 <- simulate_paths(r2, n_sims = 100L)
  zcb_unadj2 <- compute_unadjusted_zcb(paths2, test_maturities)
  adjustment2 <- apply_deterministic_shift(paths2, zcb_unadj2, zcb_market_test,
                                           test_maturities)
  cat("Computing calibrated confidence intervals...\n")
  ci2 <- compute_adjusted_confidence_intervals(paths2, zcb_market_test,
                                                test_maturities, n_boot = 500)

  results2 <- data.frame(
    Maturity = test_maturities,
    Market = round(zcb_market_test, 6),
    Unadjusted = round(zcb_unadj2, 6),
    Adjusted = round(adjustment2$zcb_adjusted, 6),
    Error_pct = round(abs(adjustment2$zcb_adjusted - zcb_market_test) /
                        zcb_market_test * 100, 3),
    CI_Lower = round(ci2$lower, 6),
    CI_Upper = round(ci2$upper, 6),
    CI_Width_bps = round((ci2$upper - ci2$lower) * 10000, 1),
    Market_In_CI = zcb_market_test >= ci2$lower & zcb_market_test <= ci2$upper
  )

  cat("\nResults:\n")
  print(results2)
  cat("\n")
  results_list[[2]] <- results2 %>% mutate(Method = "Method 2: NS/NSS + RF")

  # Method 3: Direct Regression
  cat("--- METHOD 3: Short rates \n with Direct Regression ---\n")
  r3 <- method3_direct_regression(data)
  cat(sprintf("Short rate (last obs): %.4f%%\n", r3[length(r3)] * 100))

  paths3 <- simulate_paths(r3, n_sims = 100L)
  zcb_unadj3 <- compute_unadjusted_zcb(paths3, test_maturities)
  adjustment3 <- apply_deterministic_shift(paths3, zcb_unadj3, zcb_market_test,
                                           test_maturities)
  cat("Computing calibrated confidence intervals...\n")
  ci3 <- compute_adjusted_confidence_intervals(paths3, zcb_market_test,
                                                test_maturities, n_boot = 500)

  results3 <- data.frame(
    Maturity = test_maturities,
    Market = round(zcb_market_test, 6),
    Unadjusted = round(zcb_unadj3, 6),
    Adjusted = round(adjustment3$zcb_adjusted, 6),
    Error_pct = round(abs(adjustment3$zcb_adjusted - zcb_market_test) /
                        zcb_market_test * 100, 3),
    CI_Lower = round(ci3$lower, 6),
    CI_Upper = round(ci3$upper, 6),
    CI_Width_bps = round((ci3$upper - ci3$lower) * 10000, 1),
    Market_In_CI = zcb_market_test >= ci3$lower & zcb_market_test <= ci3$upper
  )

  cat("\nResults:\n")
  print(results3)
  cat("\n")
  results_list[[3]] <- results3 %>% mutate(Method = "Method 3: Direct Regression")

  # Summary
  cat("=== SUMMARY ===\n")
  cat(sprintf("Method 1 - Mean absolute error: %.3f%%\n", mean(results1$Error_pct)))
  cat(sprintf("Method 2 - Mean absolute error: %.3f%%\n", mean(results2$Error_pct)))
  cat(sprintf("Method 3 - Mean absolute error: %.3f%%\n", mean(results3$Error_pct)))

  coverage1 <- mean(results1$Market_In_CI) * 100
  coverage2 <- mean(results2$Market_In_CI) * 100
  coverage3 <- mean(results3$Market_In_CI) * 100

  cat("\n=== CI Coverage ===\n")
  cat(sprintf("Method 1 - Market in 99%% CI: %.0f%%\n", coverage1))
  cat(sprintf("Method 2 - Market in 99%% CI: %.0f%%\n", coverage2))
  cat(sprintf("Method 3 - Market in 99%% CI: %.0f%%\n", coverage3))
  cat(sprintf("\nAvg CI Width: %.1f bps (M1), %.1f bps (M2), %.1f bps (M3)\n",
              mean(results1$CI_Width_bps), mean(results2$CI_Width_bps),
              mean(results3$CI_Width_bps)))

  return(bind_rows(results_list))
}

# ============================================================================
# 11. Main Execution
# ============================================================================
cat("=== Arbitrage-Free Framework: NMOF Implementation ===\n")
cat("Using vectorized NS() and NSS() functions\n\n")

# Generate data for both NS and NSS
cat("Generating synthetic yield curves...\n")
data_ns <- generate_yield_curve_ns(n_dates = 100)
data_nss <- generate_yield_curve_nss(n_dates = 100)

# Run analysis for NS
results_ns <- run_analysis(data_ns, "NELSON-SIEGEL (NS)")

# Run analysis for NSS
results_nss <- run_analysis(data_nss, "NELSON-SIEGEL-SVENSSON (NSS)")

# ============================================================================
# 12. Comparative Visualization
# ============================================================================
cat("\n=== Generating Comparative Visualization ===\n")

library(gridExtra)

# Add model type to results
results_ns <- results_ns %>% mutate(Model = "NS")
results_nss <- results_nss %>% mutate(Model = "NSS")
all_results <- bind_rows(results_ns, results_nss)

# Plot 3: Adjusted vs Market Prices
p3 <- ggplot(all_results, aes(x = Maturity)) +
  geom_point(aes(y = Market), size = 3, shape = 4, stroke = 2) +
  geom_point(aes(y = Adjusted, color = Model), size = 2, alpha = 0.7) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper, color = Model),
                width = 0.2, alpha = 0.5) +
  facet_wrap(~Method, ncol = 3) +
  scale_color_manual(values = c("NS" = "#E41A1C", "NSS" = "#377EB8")) +
  labs(title = "Adjusted Prices with Confidence Intervals",
       subtitle = "Market prices (×) vs Adjusted prices (•) with 95% CI",
       x = "Maturity (Years)", y = "Zero-Coupon Bond Prices") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

print(p3)

```

    === Arbitrage-Free Framework: NMOF Implementation ===
    Using vectorized NS() and NSS() functions
    
    Generating synthetic yield curves...
    
    ========== NELSON-SIEGEL (NS) ANALYSIS ==========
    
    Market ZCB Prices:
      Maturity    Price
    1        1 0.944368
    2        3 0.841025
    3        5 0.748991
    4        7 0.667029
    5       10 0.560591
    
    --- METHOD 1: Short rates 
     with NS/NSS Extrapolation ---
    Short rate (last obs): 3.2570%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.944368   0.967691 0.943862     0.054 0.943584 0.944152
    2        3 0.841025   0.905163 0.841066     0.005 0.840352 0.841831
    3        5 0.748991   0.846230 0.749097     0.014 0.748257 0.750021
    4        7 0.667029   0.791175 0.667194     0.025 0.666362 0.668134
    5       10 0.560591   0.715174 0.560692     0.018 0.559592 0.561671
      CI_Width_bps Market_In_CI
    1          5.7        FALSE
    2         14.8         TRUE
    3         17.6         TRUE
    4         17.7         TRUE
    5         20.8         TRUE
    
    --- METHOD 2: Short rates 
     with NS/NSS + Random Forest ---
    Short rate (last obs): 5.6769%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.944368   0.944570 0.943958     0.043 0.943678 0.944300
    2        3 0.841025   0.841244 0.841482     0.054 0.840561 0.842174
    3        5 0.748991   0.748978 0.749251     0.035 0.748509 0.749930
    4        7 0.667029   0.666754 0.667387     0.054 0.666451 0.668276
    5       10 0.560591   0.558617 0.561682     0.195 0.560760 0.562701
      CI_Width_bps Market_In_CI
    1          6.2        FALSE
    2         16.1         TRUE
    3         14.2         TRUE
    4         18.3         TRUE
    5         19.4        FALSE
    
    --- METHOD 3: Short rates 
     with Direct Regression ---
    Short rate (last obs): 5.6515%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.944368   0.944990 0.943743     0.066 0.943456 0.944000
    2        3 0.841025   0.843021 0.841146     0.014 0.840457 0.841841
    3        5 0.748991   0.751161 0.749289     0.040 0.748411 0.750119
    4        7 0.667029   0.668924 0.667445     0.062 0.666569 0.668325
    5       10 0.560591   0.561686 0.560619     0.005 0.559444 0.561621
      CI_Width_bps Market_In_CI
    1          5.4        FALSE
    2         13.8         TRUE
    3         17.1         TRUE
    4         17.6         TRUE
    5         21.8         TRUE
    
    === SUMMARY ===
    Method 1 - Mean absolute error: 0.023%
    Method 2 - Mean absolute error: 0.076%
    Method 3 - Mean absolute error: 0.037%
    
    === CI Coverage ===
    Method 1 - Market in 99% CI: 80%
    Method 2 - Market in 99% CI: 60%
    Method 3 - Market in 99% CI: 80%
    
    Avg CI Width: 15.3 bps (M1), 14.8 bps (M2), 15.1 bps (M3)
    
    ========== NELSON-SIEGEL-SVENSSON (NSS) ANALYSIS ==========
    
    Market ZCB Prices:
      Maturity    Price
    1        1 0.941887
    2        3 0.836276
    3        5 0.742752
    4        7 0.659696
    5       10 0.552198
    
    --- METHOD 1: Short rates 
     with NS/NSS Extrapolation ---
    Short rate (last obs): 4.6355%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.941887   0.954688 0.942270     0.041 0.941853 0.942716
    2        3 0.836276   0.871008 0.835715     0.067 0.834544 0.836739
    3        5 0.742752   0.794980 0.741836     0.123 0.740349 0.743138
    4        7 0.659696   0.726242 0.658950     0.113 0.657747 0.660143
    5       10 0.552198   0.633806 0.551628     0.103 0.549925 0.553257
      CI_Width_bps Market_In_CI
    1          8.6         TRUE
    2         21.9         TRUE
    3         27.9         TRUE
    4         24.0         TRUE
    5         33.3         TRUE
    
    --- METHOD 2: Short rates 
     with NS/NSS + Random Forest ---
    Short rate (last obs): 5.9567%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.941887   0.942313 0.942164     0.029 0.941784 0.942523
    2        3 0.836276   0.836567 0.836628     0.042 0.835661 0.837590
    3        5 0.742752   0.742821 0.743231     0.064 0.741899 0.744219
    4        7 0.659696   0.658894 0.660377     0.103 0.659165 0.661587
    5       10 0.552198   0.549714 0.552640     0.080 0.551440 0.554016
      CI_Width_bps Market_In_CI
    1          7.4         TRUE
    2         19.3         TRUE
    3         23.2         TRUE
    4         24.2         TRUE
    5         25.8         TRUE
    
    --- METHOD 3: Short rates 
     with Direct Regression ---
    Short rate (last obs): 5.9770%
    Computing calibrated confidence intervals...
    
    Results:
      Maturity   Market Unadjusted Adjusted Error_pct CI_Lower CI_Upper
    1        1 0.941887   0.942102 0.942063     0.019 0.941738 0.942405
    2        3 0.836276   0.836988 0.836231     0.005 0.835452 0.837112
    3        5 0.742752   0.742843 0.743059     0.041 0.742083 0.744069
    4        7 0.659696   0.658711 0.659669     0.004 0.658707 0.660866
    5       10 0.552198   0.550538 0.552046     0.028 0.550710 0.553238
      CI_Width_bps Market_In_CI
    1          6.7         TRUE
    2         16.6         TRUE
    3         19.9         TRUE
    4         21.6         TRUE
    5         25.3         TRUE
    
    === SUMMARY ===
    Method 1 - Mean absolute error: 0.089%
    Method 2 - Mean absolute error: 0.064%
    Method 3 - Mean absolute error: 0.019%
    
    === CI Coverage ===
    Method 1 - Market in 99% CI: 100%
    Method 2 - Market in 99% CI: 100%
    Method 3 - Market in 99% CI: 100%
    
    Avg CI Width: 23.1 bps (M1), 20.0 bps (M2), 18.0 bps (M3)
    
    === Generating Comparative Visualization ===



    
![image-title-here]({{base}}/images/2025-10-28/2025-10-28-deterministic-shift_2_1.png){:class="img-responsive"}
    

