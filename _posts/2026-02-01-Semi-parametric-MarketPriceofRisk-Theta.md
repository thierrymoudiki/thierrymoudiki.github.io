---
layout: post
title: "Option pricing using time series models as market price of risk"
description: "Option pricing using time series models (here, Theta) as market price of risk"
date: 2026-02-01
categories: R
comments: true
---

Following [https://thierrymoudiki.github.io/blog/2025/12/07/r/forecasting/ARIMA-Pricing], I present how to use time series models as market price of risk in option pricing. `auto.arima` worked well because it enforces stationarity of residuals. This post shows that, if the chosen model can obtain stationarity of residuals, then it can be used as market price of risk.



```R
devtools::install_github("Techtonique/esgtoolkit")
devtools::install_github("Techtonique/ahead")
```


```R
library(esgtoolkit)
library(forecast)
library(ahead)

# =============================================================================
# 1 - SVJD SIMULATION (Physical Measure)
# =============================================================================

set.seed(123)
n <- 250L
h <- 5
freq <- "daily"
r <- 0.05
maturity <- 5
S0 <- 100
mu <- 0.08
sigma <- 0.04

# Simulate under physical measure with stochastic volatility and jumps
sim_GBM <- esgtoolkit::simdiff(n=n, horizon = h, frequency = freq, x0=S0, theta1 = mu, theta2 = sigma)
sim_SVJD <- esgtoolkit::rsvjd(n=n, r0=mu)
sim_Heston <- esgtoolkit::rsvjd(n=n, r0=mu,
                                lambda = 0,
                                mu_J = 0,
                                sigma_J = 0)

cat("Simulation dimensions:\n")
cat("Start:", start(sim_SVJD), "\n")
cat("End:", end(sim_SVJD), "\n")
cat("Paths:", ncol(sim_SVJD), "\n")
cat("Time steps:", nrow(sim_SVJD), "\n\n")

par(mfrow=c(1, 3))
# Plot historical (physical measure) paths
esgtoolkit::esgplotbands(sim_GBM, main="GBM Paths under the Physical Measure",
                         xlab="Time",
                         ylab="Stock prices")
esgtoolkit::esgplotbands(sim_SVJD, main="SVJD Paths under the Physical Measure",
                         xlab="Time",
                         ylab="Stock prices")
esgtoolkit::esgplotbands(sim_Heston, main="Heston Paths under the Physical Measure",
                         xlab="Time",
                         ylab="Stock prices")

# Summary statistics
cat("Physical measure statistics (GBM):\n")
terminal_prices_physical_GBM <- sim_GBM[nrow(sim_GBM), ]
cat("Mean terminal price:", mean(terminal_prices_physical_GBM), "\n")
cat("Std terminal price:", sd(terminal_prices_physical_GBM), "\n")
cat("Expected under P:", S0 * exp(mu * maturity), "\n\n")

cat("Physical measure statistics (SVJD):\n")
terminal_prices_physical_SVJD <- sim_SVJD[nrow(sim_SVJD), ]
cat("Mean terminal price:", mean(terminal_prices_physical_SVJD), "\n")
cat("Std terminal price:", sd(terminal_prices_physical_SVJD), "\n")
cat("Expected under P:", S0 * exp(mu * maturity), "\n\n")

cat("Physical measure statistics (Heston):\n")
terminal_prices_physical_Heston <- sim_Heston[nrow(sim_Heston), ]
cat("Mean terminal price:", mean(terminal_prices_physical_Heston), "\n")
cat("Std terminal price:", sd(terminal_prices_physical_Heston), "\n")
cat("Expected under P:", S0 * exp(mu * maturity), "\n\n")


# =============================================================================
# 2 - COMPUTE DISCOUNTED PRICES (Transform to Martingale Domain)
# =============================================================================

discounted_prices_GBM <- esgtoolkit::esgdiscountfactor(r=r, X=sim_GBM)
discounted_prices_SVJD <- esgtoolkit::esgdiscountfactor(r=r, X=sim_SVJD)
discounted_prices_Heston <- esgtoolkit::esgdiscountfactor(r=r, X=sim_Heston)
martingale_diff_GBM <- discounted_prices_GBM - S0
martingale_diff_SVJD <- discounted_prices_SVJD - S0
martingale_diff_Heston <- discounted_prices_Heston - S0

cat("Martingale differences dimensions (GBM):", dim(martingale_diff_GBM), "\n")
cat("Mean martingale diff (should be ≠ 0 under P):\n")
print(t.test(rowMeans(martingale_diff_GBM)))

cat("\nMartingale differences dimensions (SVJD):", dim(martingale_diff_SVJD), "\n")
cat("Mean martingale diff (should be ≠ 0 under P):\n")
print(t.test(rowMeans(martingale_diff_SVJD)))

cat("\nMartingale differences dimensions (Heston):", dim(martingale_diff_Heston), "\n")
cat("Mean martingale diff (should be ≠ 0 under P):\n")
print(t.test(rowMeans(martingale_diff_Heston)))

# =============================================================================
# 3 - VISUALIZE RISK PREMIUM
# =============================================================================

par(mfrow=c(2,2))

matplot(discounted_prices_GBM, type='l', col=rgb(0,0,1,0.3),
        main="Discounted Stock Prices (Physical Measure - GBM)",
        ylab="exp(-rt) * S_t", xlab="Time step")
abline(h=S0, col='red', lwd=2, lty=2)

matplot(discounted_prices_SVJD, type='l', col=rgb(0,0,1,0.3),
        main="Discounted Stock Prices (Physical Measure - SVJD)",
        ylab="exp(-rt) * S_t", xlab="Time step")
abline(h=S0, col='red', lwd=2, lty=2)

matplot(discounted_prices_Heston, type='l', col=rgb(0,0,1,0.3),
        main="Discounted Stock Prices (Physical Measure - Heston)",
        ylab="exp(-rt) * S_t", xlab="Time step")
abline(h=S0, col='red', lwd=2, lty=2)

par(mfrow=c(1,1))

mean_disc_path_GBM <- rowMeans(discounted_prices_GBM)
times_plot <- as.numeric(time(discounted_prices_GBM))
plot(times_plot, mean_disc_path_GBM, type='l', lwd=2, col='blue',
     main="Risk Premium in Discounted Prices (GBM)",
     xlab="Time (years)", ylab="E[exp(-rt)*S_t]")
abline(h=S0, col='red', lwd=2, lty=2)
lines(times_plot, S0 * exp((mu-r)*times_plot), col='green', lwd=2, lty=3)
legend("topleft",
       legend=c("Empirical mean", "S0", "Theoretical (μ-r drift)"),
       col=c('blue','red','green'), lty=c(1,2,3), lwd=2)

mean_disc_path_SVJD <- rowMeans(discounted_prices_SVJD)
times_plot <- as.numeric(time(discounted_prices_SVJD))
plot(times_plot, mean_disc_path_SVJD, type='l', lwd=2, col='blue',
     main="Risk Premium in Discounted Prices (SVJD)",
     xlab="Time (years)", ylab="E[exp(-rt)*S_t]")
abline(h=S0, col='red', lwd=2, lty=2)
lines(times_plot, S0 * exp((mu-r)*times_plot), col='green', lwd=2, lty=3)
legend("topleft",
       legend=c("Empirical mean", "S0", "Theoretical (μ-r drift)"),
       col=c('blue','red','green'), lty=c(1,2,3), lwd=2)

mean_disc_path_Heston <- rowMeans(discounted_prices_Heston)
times_plot <- as.numeric(time(discounted_prices_Heston))
plot(times_plot, mean_disc_path_Heston, type='l', lwd=2, col='blue',
     main="Risk Premium in Discounted Prices (Heston)",
     xlab="Time (years)", ylab="E[exp(-rt)*S_t]")
abline(h=S0, col='red', lwd=2, lty=2)
lines(times_plot, S0 * exp((mu-r)*times_plot), col='green', lwd=2, lty=3)
legend("topleft",
       legend=c("Empirical mean", "S0", "Theoretical (μ-r drift)"),
       col=c('blue','red','green'), lty=c(1,2,3), lwd=2)


# =============================================================================
# 4 - FIT Theta MODELS TO EXTRACT RISK PREMIUM
# =============================================================================

n_periods <- nrow(martingale_diff_GBM)
n_paths <- ncol(martingale_diff_GBM)

martingale_increments_GBM <- diff(martingale_diff_GBM)
martingale_increments_SVJD <- diff(martingale_diff_SVJD)
martingale_increments_Heston <- diff(martingale_diff_Heston)

# Initialize storage arrays
theta_residuals_GBM <- array(NA, dim = c(nrow(martingale_increments_GBM), n_paths))
centered_theta_residuals_GBM <- array(NA, dim = c(nrow(martingale_increments_GBM), n_paths))
means_theta_residuals_GBM <- rep(NA, n_paths)
theta_models_GBM <- list()

theta_residuals_SVJD <- array(NA, dim = c(nrow(martingale_increments_SVJD), n_paths))
centered_theta_residuals_SVJD <- array(NA, dim = c(nrow(martingale_increments_SVJD), n_paths))
means_theta_residuals_SVJD <- rep(NA, n_paths)
theta_models_SVJD <- list()

theta_residuals_Heston <- array(NA, dim = c(nrow(martingale_increments_Heston), n_paths))
centered_theta_residuals_Heston <- array(NA, dim = c(nrow(martingale_increments_Heston), n_paths))
means_theta_residuals_Heston <- rep(NA, n_paths)
theta_models_Heston <- list()

# Fit theta models to GBM
cat("Fitting theta models to", n_paths, "GBM paths...\n")
for (i in 1:n_paths) {
  y <- as.numeric(martingale_increments_GBM[, i])
  fit <- forecast::thetaf(y)
  theta_models_GBM[[i]] <- fit

  res <- as.numeric(residuals(fit))
  theta_residuals_GBM[, i] <- res

  centre_theta_residuals <- scale(res, center = TRUE, scale = FALSE)
  means_theta_residuals_GBM[i] <- attr(centre_theta_residuals, "scaled:center")
  centered_theta_residuals_GBM[, i] <- centre_theta_residuals[,1]
}

# Fit theta models to SVJD
cat("Fitting theta models to", n_paths, "SVJD paths...\n")
for (i in 1:n_paths) {
  y <- as.numeric(martingale_increments_SVJD[, i])
  fit <- forecast::thetaf(y)
  theta_models_SVJD[[i]] <- fit

  res <- as.numeric(residuals(fit))
  theta_residuals_SVJD[, i] <- res

  centre_theta_residuals <- scale(res, center = TRUE, scale = FALSE)
  means_theta_residuals_SVJD[i] <- attr(centre_theta_residuals, "scaled:center")
  centered_theta_residuals_SVJD[, i] <- centre_theta_residuals[,1]
}

# Fit theta models to Heston
cat("Fitting theta models to", n_paths, "Heston paths...\n")
for (i in 1:n_paths) {
  y <- as.numeric(martingale_increments_Heston[, i])
  fit <- forecast::thetaf(y)
  theta_models_Heston[[i]] <- fit

  res <- as.numeric(residuals(fit))
  theta_residuals_Heston[, i] <- res

  centre_theta_residuals <- scale(res, center = TRUE, scale = FALSE)
  means_theta_residuals_Heston[i] <- attr(centre_theta_residuals, "scaled:center")
  centered_theta_residuals_Heston[, i] <- centre_theta_residuals[,1]
}

# Box-Ljung tests
pvalues_GBM <- sapply(1:ncol(centered_theta_residuals_GBM),
                      function(i) Box.test(centered_theta_residuals_GBM[,i])$p.value)
cat("\nBox-Ljung test p-values (GBM):\n")
cat("Mean p-value:", mean(pvalues_GBM), "\n")
cat("Proportion > 0.05:", mean(pvalues_GBM > 0.05), "\n")

pvalues_SVJD <- sapply(1:ncol(centered_theta_residuals_SVJD),
                       function(i) Box.test(centered_theta_residuals_SVJD[,i])$p.value)
cat("\nBox-Ljung test p-values (SVJD):\n")
cat("Mean p-value:", mean(pvalues_SVJD), "\n")
cat("Proportion > 0.05:", mean(pvalues_SVJD > 0.05), "\n")

pvalues_Heston <- sapply(1:ncol(centered_theta_residuals_Heston),
                         function(i) Box.test(centered_theta_residuals_Heston[,i])$p.value)
cat("\nBox-Ljung test p-values (Heston):\n")
cat("Mean p-value:", mean(pvalues_Heston), "\n")
cat("Proportion > 0.05:", mean(pvalues_Heston > 0.05), "\n")

par(mfrow=c(1,3))
hist(pvalues_GBM, breaks=20, col='lightgreen',
     main="Box-Ljung P-values (GBM)",
     xlab="P-value")
abline(v=0.05, col='red', lwd=2, lty=2)

hist(pvalues_SVJD, breaks=20, col='lightblue',
     main="Box-Ljung P-values (SVJD)",
     xlab="P-value")
abline(v=0.05, col='red', lwd=2, lty=2)

hist(pvalues_Heston, breaks=20, col='lightcoral',
     main="Box-Ljung P-values (Heston)",
     xlab="P-value")
abline(v=0.05, col='red', lwd=2, lty=2)
par(mfrow=c(1,1))


# =============================================================================
# 5 - GENERATE RISK-NEUTRAL PATHS
# =============================================================================

cat("\n\nGenerating risk-neutral paths from ALL historical paths...\n")
n_sim_per_path <- 20  # Generate 10 paths per historical path
times <- seq(0, maturity, length.out = n_periods)
discount_factor <- exp(r * times)

# Storage for all risk-neutral paths
all_S_tilde_GBM <- list()
all_S_tilde_SVJD <- list()
all_S_tilde_Heston <- list()

# Generate GBM risk-neutral paths
for (i in 1:n_paths) {
  resampled_residuals <- ahead::rgaussiandens(centered_theta_residuals_GBM[, i],
                                              p = n_sim_per_path)
  fit <- theta_models_GBM[[i]]
  fitted_increments <- as.numeric(fitted(fit))

  discounted_path <- matrix(0, nrow = n_periods, ncol = n_sim_per_path)
  discounted_path[1, ] <- S0

  increments <- matrix(scale(fitted_increments, center = TRUE,
                             scale = FALSE)[,1], nrow = n_periods - 1, ncol = n_sim_per_path) +
    resampled_residuals[1:(n_periods - 1), ]

  discounted_path[-1, ] <- S0 + apply(increments, 2, cumsum)
  S_tilde_price <- discounted_path * discount_factor
  all_S_tilde_GBM[[i]] <- S_tilde_price
}

# Generate SVJD risk-neutral paths
for (i in 1:n_paths) {
  resampled_residuals <- ahead::rgaussiandens(centered_theta_residuals_SVJD[, i],
                                              p = n_sim_per_path)
  fit <- theta_models_SVJD[[i]]
  fitted_increments <- as.numeric(fitted(fit))

  discounted_path <- matrix(0, nrow = n_periods, ncol = n_sim_per_path)
  discounted_path[1, ] <- S0

  increments <- matrix(scale(fitted_increments, center = TRUE,
                             scale = FALSE)[,1], nrow = n_periods - 1, ncol = n_sim_per_path) +
    resampled_residuals[1:(n_periods - 1), ]

  discounted_path[-1, ] <- S0 + apply(increments, 2, cumsum)
  S_tilde_price <- discounted_path * discount_factor
  all_S_tilde_SVJD[[i]] <- S_tilde_price
}

# Generate Heston risk-neutral paths
for (i in 1:n_paths) {
  resampled_residuals <- ahead::rgaussiandens(centered_theta_residuals_Heston[, i],
                                              p = n_sim_per_path)
  fit <- theta_models_Heston[[i]]
  fitted_increments <- as.numeric(fitted(fit))

  discounted_path <- matrix(0, nrow = n_periods, ncol = n_sim_per_path)
  discounted_path[1, ] <- S0

  increments <- matrix(scale(fitted_increments, center = TRUE,
                             scale = FALSE)[,1], nrow = n_periods - 1, ncol = n_sim_per_path) +
    resampled_residuals[1:(n_periods - 1), ]

  discounted_path[-1, ] <- S0 + apply(increments, 2, cumsum)
  S_tilde_price <- discounted_path * discount_factor
  all_S_tilde_Heston[[i]] <- S_tilde_price
}

# Combine all paths
S_tilde_combined_GBM <- do.call(cbind, all_S_tilde_GBM)
cat("Total GBM risk-neutral paths generated:", ncol(S_tilde_combined_GBM), "\n")

S_tilde_combined_SVJD <- do.call(cbind, all_S_tilde_SVJD)
cat("Total SVJD risk-neutral paths generated:", ncol(S_tilde_combined_SVJD), "\n")

S_tilde_combined_Heston <- do.call(cbind, all_S_tilde_Heston)
cat("Total Heston risk-neutral paths generated:", ncol(S_tilde_combined_Heston), "\n\n")

# Convert to time series
S_tilde_ts_GBM <- ts(S_tilde_combined_GBM, start = start(sim_GBM),
                     frequency = frequency(sim_GBM))
S_tilde_ts_SVJD <- ts(S_tilde_combined_SVJD, start = start(sim_SVJD),
                      frequency = frequency(sim_SVJD))
S_tilde_ts_Heston <- ts(S_tilde_combined_Heston, start = start(sim_Heston),
                        frequency = frequency(sim_Heston))

# Visualize risk-neutral paths
par(mfrow=c(1,3))
esgtoolkit::esgplotbands(S_tilde_ts_GBM,
                         main="Risk-Neutral Paths - GBM")
esgtoolkit::esgplotbands(S_tilde_ts_SVJD,
                         main="Risk-Neutral Paths - SVJD")
esgtoolkit::esgplotbands(S_tilde_ts_Heston,
                         main="Risk-Neutral Paths - Heston")

# Sample plots
par(mfrow=c(1,3))
matplot(S_tilde_combined_GBM[, sample(ncol(S_tilde_combined_GBM), 200)],
        type='l', col=rgb(0,0,1,0.1),
        main="Risk-Neutral Paths (GBM)",
        xlab="Time step", ylab="Stock Price")
lines(rowMeans(S_tilde_combined_GBM), col='red', lwd=3)
abline(h=S0, col='green', lwd=2, lty=2)

matplot(S_tilde_combined_SVJD[, sample(ncol(S_tilde_combined_SVJD), 200)],
        type='l', col=rgb(0,0,1,0.1),
        main="Risk-Neutral Paths (SVJD)",
        xlab="Time step", ylab="Stock Price")
lines(rowMeans(S_tilde_combined_SVJD), col='red', lwd=3)
abline(h=S0, col='green', lwd=2, lty=2)

matplot(S_tilde_combined_Heston[, sample(ncol(S_tilde_combined_Heston), 200)],
        type='l', col=rgb(0,0,1,0.1),
        main="Risk-Neutral Paths (Heston)",
        xlab="Time step", ylab="Stock Price")
lines(rowMeans(S_tilde_combined_Heston), col='red', lwd=3)
abline(h=S0, col='green', lwd=2, lty=2)
par(mfrow=c(1,1))


# =============================================================================
# 6 - VERIFY RISK-NEUTRAL PROPERTY
# =============================================================================

cat("\n=== RISK-NEUTRAL VERIFICATION ===\n\n")

terminal_prices_rn_GBM <- S_tilde_combined_GBM[n_periods, ]
terminal_prices_rn_SVJD <- S_tilde_combined_SVJD[n_periods, ]
terminal_prices_rn_Heston <- S_tilde_combined_Heston[n_periods, ]
capitalized_stock_price <- S0 * exp(r * maturity)

cat("GBM Risk-Neutral Verification:\n")
cat("Expected terminal price (Q):", capitalized_stock_price, "\n")
cat("Empirical mean:", mean(terminal_prices_rn_GBM), "\n")
cat("Difference:", mean(terminal_prices_rn_GBM) - capitalized_stock_price, "\n")
print(t.test(terminal_prices_rn_GBM - capitalized_stock_price))

cat("\nSVJD Risk-Neutral Verification:\n")
cat("Expected terminal price (Q):", capitalized_stock_price, "\n")
cat("Empirical mean:", mean(terminal_prices_rn_SVJD), "\n")
cat("Difference:", mean(terminal_prices_rn_SVJD) - capitalized_stock_price, "\n")
print(t.test(terminal_prices_rn_SVJD - capitalized_stock_price))

cat("\nHeston Risk-Neutral Verification:\n")
cat("Expected terminal price (Q):", capitalized_stock_price, "\n")
cat("Empirical mean:", mean(terminal_prices_rn_Heston), "\n")
cat("Difference:", mean(terminal_prices_rn_Heston) - capitalized_stock_price, "\n")
print(t.test(terminal_prices_rn_Heston - capitalized_stock_price))

# Visualization comparison
par(mfrow=c(3, 2))
hist(terminal_prices_physical_GBM, breaks=30, col=rgb(1,0,0,0.5),
     main="Terminal Prices: Physical (GBM)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_physical_GBM), col='red', lwd=2)
abline(v=S0*exp(mu*maturity), col='blue', lwd=2, lty=2)

hist(terminal_prices_rn_GBM, breaks=30, col=rgb(0,0,1,0.5),
     main="Terminal Prices: Risk-Neutral (GBM)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_rn_GBM), col='blue', lwd=2)
abline(v=S0*exp(r*maturity), col='red', lwd=2, lty=2)

hist(terminal_prices_physical_SVJD, breaks=30, col=rgb(1,0,0,0.5),
     main="Terminal Prices: Physical (SVJD)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_physical_SVJD), col='red', lwd=2)
abline(v=S0*exp(mu*maturity), col='blue', lwd=2, lty=2)

hist(terminal_prices_rn_SVJD, breaks=30, col=rgb(0,0,1,0.5),
     main="Terminal Prices: Risk-Neutral (SVJD)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_rn_SVJD), col='blue', lwd=2)
abline(v=S0*exp(r*maturity), col='red', lwd=2, lty=2)

hist(terminal_prices_physical_Heston, breaks=30, col=rgb(1,0,0,0.5),
     main="Terminal Prices: Physical (Heston)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_physical_Heston), col='red', lwd=2)
abline(v=S0*exp(mu*maturity), col='blue', lwd=2, lty=2)

hist(terminal_prices_rn_Heston, breaks=30, col=rgb(0,0,1,0.5),
     main="Terminal Prices: Risk-Neutral (Heston)",
     xlab="Price", xlim=c(50, 300))
abline(v=mean(terminal_prices_rn_Heston), col='blue', lwd=2)
abline(v=S0*exp(r*maturity), col='red', lwd=2, lty=2)
par(mfrow=c(1,1))

# =============================================================================
# 7 - OPTION PRICING
# =============================================================================

cat("\n=== OPTION PRICING ===\n\n")

bs_price <- function(S, K, r, sigma, T, q = 0) {

  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)

  call <- S * exp(-q * T) * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
  put  <- K * exp(-r * T) * pnorm(-d2) - S * exp(-q * T) * pnorm(-d1)

  list(call = call, put = put)
}


strikes <- seq(80, 160, by=10)
d_f <- exp(-r * maturity)

# Function to price options
price_options <- function(terminal_prices, strikes, discount_factor) {
  n_strikes <- length(strikes)
  call_prices <- numeric(n_strikes)
  bs_call_prices <- numeric(n_strikes)
  put_prices <- numeric(n_strikes)
  bs_put_prices <- numeric(n_strikes)
  call_se <- numeric(n_strikes)
  put_se <- numeric(n_strikes)

  for (k in 1:n_strikes) {
    K <- strikes[k]
    call_payoffs <- pmax(terminal_prices - K, 0)
    call_prices[k] <- mean(call_payoffs) * discount_factor
    call_se[k] <- sd(call_payoffs) / sqrt(length(call_payoffs)) * discount_factor

    put_payoffs <- pmax(K - terminal_prices, 0)
    put_prices[k] <- mean(put_payoffs) * discount_factor
    put_se[k] <- sd(put_payoffs) / sqrt(length(put_payoffs)) * discount_factor

    bs_prices <- bs_price(S0, K, r, sigma=sigma, T=5, q = 0)
    bs_call_prices[k] <- bs_prices$call
    bs_put_prices[k] <- bs_prices$put
  }

  list(call_prices = call_prices, put_prices = put_prices,
       bs_call_prices = bs_call_prices, bs_put_prices = bs_put_prices,
       call_se = call_se, put_se = put_se)
}

# Price options for all models
options_GBM <- price_options(terminal_prices_rn_GBM, strikes, d_f)
options_SVJD <- price_options(terminal_prices_rn_SVJD, strikes, d_f)
options_Heston <- price_options(terminal_prices_rn_Heston, strikes, d_f)
```

    Simulation dimensions:
    Start: 0 1 
    End: 5 1 
    Paths: 250 
    Time steps: 1261 
    
    Physical measure statistics (GBM):
    Mean terminal price: 149.8948 
    Std terminal price: 13.29507 
    Expected under P: 149.1825 
    
    Physical measure statistics (SVJD):
    Mean terminal price: 149.8261 
    Std terminal price: 20.02665 
    Expected under P: 149.1825 
    
    Physical measure statistics (Heston):
    Mean terminal price: 149.7758 
    Std terminal price: 15.76233 
    Expected under P: 149.1825 
    
    Martingale differences dimensions (GBM): 1261 250 
    Mean martingale diff (should be ≠ 0 under P):
    
    	One Sample t-test
    
    data:  rowMeans(martingale_diff_GBM)
    t = 61.145, df = 1260, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     8.133524 8.672756
    sample estimates:
    mean of x 
      8.40314 
    
    
    Martingale differences dimensions (SVJD): 1261 250 
    Mean martingale diff (should be ≠ 0 under P):
    
    	One Sample t-test
    
    data:  rowMeans(martingale_diff_SVJD)
    t = 57.811, df = 1260, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     7.823370 8.372999
    sample estimates:
    mean of x 
     8.098184 
    
    
    Martingale differences dimensions (Heston): 1261 250 
    Mean martingale diff (should be ≠ 0 under P):
    
    	One Sample t-test
    
    data:  rowMeans(martingale_diff_Heston)
    t = 58.195, df = 1260, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     7.751804 8.292687
    sample estimates:
    mean of x 
     8.022246 
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_3.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_4.png){:class="img-responsive"}
    


    Fitting theta models to 250 GBM paths...
    Fitting theta models to 250 SVJD paths...
    Fitting theta models to 250 Heston paths...
    
    Box-Ljung test p-values (GBM):
    Mean p-value: 0.5185777 
    Proportion > 0.05: 0.964 
    
    Box-Ljung test p-values (SVJD):
    Mean p-value: 0.4223444 
    Proportion > 0.05: 0.844 
    
    Box-Ljung test p-values (Heston):
    Mean p-value: 0.3734787 
    Proportion > 0.05: 0.788 



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_6.png){:class="img-responsive"}
    


    
    
    Generating risk-neutral paths from ALL historical paths...
    Total GBM risk-neutral paths generated: 5000 
    Total SVJD risk-neutral paths generated: 5000 
    Total Heston risk-neutral paths generated: 5000 
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_8.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_9.png){:class="img-responsive"}
    


    
    === RISK-NEUTRAL VERIFICATION ===
    
    GBM Risk-Neutral Verification:
    Expected terminal price (Q): 128.4025 
    Empirical mean: 128.0431 
    Difference: -0.359446 
    
    	One Sample t-test
    
    data:  terminal_prices_rn_GBM - capitalized_stock_price
    t = -1.9526, df = 4999, p-value = 0.05092
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -0.72033322  0.00144115
    sample estimates:
    mean of x 
    -0.359446 
    
    
    SVJD Risk-Neutral Verification:
    Expected terminal price (Q): 128.4025 
    Empirical mean: 128.2914 
    Difference: -0.1111722 
    
    	One Sample t-test
    
    data:  terminal_prices_rn_SVJD - capitalized_stock_price
    t = -0.43071, df = 4999, p-value = 0.6667
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -0.6171844  0.3948400
    sample estimates:
     mean of x 
    -0.1111722 
    
    
    Heston Risk-Neutral Verification:
    Expected terminal price (Q): 128.4025 
    Empirical mean: 128.1919 
    Difference: -0.210678 
    
    	One Sample t-test
    
    data:  terminal_prices_rn_Heston - capitalized_stock_price
    t = -0.97728, df = 4999, p-value = 0.3285
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     -0.6333009  0.2119449
    sample estimates:
    mean of x 
    -0.210678 
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_11.png){:class="img-responsive"}
    


    
    === OPTION PRICING ===
    



    
![image-title-here]({{base}}/images/2026-02-01/2026-02-01-Semi-parametric-MarketPriceofRisk-Theta_1_13.png){:class="img-responsive"}
    



```R
## ------------------------------------------------------------------------------
print(as.data.frame(options_GBM))
print(as.data.frame(options_Heston))
print(as.data.frame(options_SVJD))
```

      call_prices   put_prices bs_call_prices bs_put_prices     call_se      put_se
    1 37.41600050  0.000000000    37.69593743  7.659025e-08 0.143365478 0.000000000
    2 29.63055025  0.002557579    29.90798974  6.021910e-05 0.143253001 0.001385215
    3 21.89202510  0.052040266    22.12602385  6.102158e-03 0.141542828 0.007968062
    4 14.41872513  0.366748124    14.47265887  1.407450e-01 0.133717540 0.023614412
    5  7.92138098  1.657411806     7.66440485  1.120499e+00 0.111762045 0.053011875
    6  3.30948845  4.833527100     3.00141348  4.245515e+00 0.077117988 0.090590260
    7  1.00426018 10.316306659     0.82807301  9.860183e+00 0.041860043 0.121063890
    8  0.21365063 17.313704945     0.16088172  1.698100e+01 0.017299977 0.137019858
    9  0.02392382 24.911985962     0.02263566  2.463076e+01 0.005074836 0.142441085
      call_prices  put_prices bs_call_prices bs_put_prices    call_se      put_se
    1 37.54125545  0.00939426    37.69593743  7.659025e-08 0.16741035 0.004460958
    2 29.78664975  0.04279639    29.90798974  6.021910e-05 0.16609502 0.009468927
    3 22.13586786  0.18002234    22.12602385  6.102158e-03 0.16190123 0.019517031
    4 14.86963745  0.70179976    14.47265887  1.407450e-01 0.15004865 0.038698660
    5  8.55724430  2.17741443     7.66440485  1.120499e+00 0.12633142 0.069087158
    6  4.02887344  5.43705141     3.00141348  4.245515e+00 0.09105286 0.105511930
    7  1.46254973 10.65873553     0.82807301  9.860183e+00 0.05510963 0.137525742
    8  0.42110415 17.40529778     0.16088172  1.698100e+01 0.02796964 0.156437070
    9  0.08973658 24.86193804     0.02263566  2.463076e+01 0.01153056 0.164808005
      call_prices put_prices bs_call_prices bs_put_prices    call_se     put_se
    1   37.701067  0.0917107    37.69593743  7.659025e-08 0.19623450 0.02273294
    2   30.004951  0.1836023    29.90798974  6.021910e-05 0.19309085 0.03033105
    3   22.467550  0.4342098    22.12602385  6.102158e-03 0.18634069 0.04221605
    4   15.372049  1.1267162    14.47265887  1.407450e-01 0.17219740 0.06186092
    5    9.236629  2.7793037     7.66440485  1.120499e+00 0.14785800 0.09096966
    6    4.759295  6.0899780     3.00141348  4.245515e+00 0.11440939 0.12539000
    7    2.088066 11.2067571     0.82807301  9.860183e+00 0.08072279 0.15662010
    8    0.827147 17.7338455     0.16088172  1.698100e+01 0.05468438 0.17762074
    9    0.324822 25.0195283     0.02263566  2.463076e+01 0.03801850 0.18897425

