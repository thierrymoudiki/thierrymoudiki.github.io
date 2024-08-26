---
layout: post
title: "A new method for deriving a nonparametric confidence interval for the mean"
description: "Deriving a nonparametric confidence interval for the mean using stratified 
sampling, the bootstrap, surrogates and density estimation"
date: 2024-08-26
categories: R
comments: true
---

Last week, I was looking for a way to construct nonparametric confidence intervals for average 
effects in [`learningmachine`](https://thierrymoudiki.github.io/blog/2024/07/08/r/learningmachine-docs) (using Student-T tests to construct confidence intervals for now). So, I thought of one and implemented it. The methodology combines stratified sampling with the generation of pseudo-observations derived from standardized residuals.

Not sure if it's completely new or a _breakthrough_ for reviewer no.2, but it seems to be working pretty well in many different use cases. As illustrated below (in particular in function `compute_ci_mean`). 

### The Methodology: Breaking It Down

The approach can be broken down into four key steps:

#### 1. **Stratified Sampling**
The first step involves splitting your data into three subsets: training, calibration, and test sets. This is done using stratified sampling, ensuring that each subset is representative of the entire dataset. This method reduces sampling bias, leading to more reliable and accurate estimates.

#### 2. **Estimation and Standardization**
Next, the mean and standard deviation of the training set are calculated. These values are used to standardize the residuals from the calibration set. Standardization stabilizes the variance of the residuals, making them homoscedastic (i.e., having constant variance). This is crucial for the next steps as it simplifies the distribution of residuals, which will be used to generate pseudo-observations.

#### 3. **Generation of Pseudo-Observations**
Pseudo-observations are created by adding back the standardized residuals to the test set values. These residuals are sampled from the standardized calibration residuals. The idea is to mimic potential future observations, allowing for a more accurate estimation of the sampling distribution of the mean.

#### 4. **Construction of the Confidence Interval**
Finally, the means of the pseudo-observations are computed. The empirical quantiles of these means are used to form the lower and upper bounds of the confidence interval. This nonparametric CI captures the true mean with a specified level of confidence without relying on parametric assumptions about the underlying distribution.

### Why Does This Work?

The robustness of this method lies in a few key mathematical principles:

- **Law of Large Numbers (LLN)**: Ensures that as the sample size increases, the sample mean converges to the true population mean.
- **Central Limit Theorem (CLT)**: Even if the original data isn’t normally distributed, the distribution of the sample means (from the pseudo-observations) will approximate a normal distribution as the sample size grows.
- **Stratified Sampling**: Helps in reducing variance and ensuring that the subsamples are representative of the overall data, which improves the accuracy of the estimated confidence interval.
- **Independence of Residuals**: Ensures that the sampling of residuals is valid and that the pseudo-observations properly reflect the true data distribution.

### Code

```python
%load_ext rpy2.ipython
```


```r
%%R

# Install required packages
utils::install.packages("evd")
utils::install.packages("microbenchmark")
utils::install.packages("kde1d")
utils::install.packages("ggplot2")
utils::install.packages("tseries")
utils::install.packages("caret")

# Load packages
library("evd")
library("microbenchmark")
library("kde1d")
library("ggplot2")
library("tseries")
library("caret")
```


```python

```


```r
%%R

# 1. Direct Sampling
direct_sampling <- function(data = NULL, n = 1000,
                            method = c("kde",
                                       "surrogate",
                                       "bootstrap"),
                            kde = NULL,
                            seed = NULL,
                            ...) {
  method <- match.arg(method)
  if (!is.null(seed))
  {
    set.seed(seed)
  }
  if (identical(method, "kde"))
  {
    if (is.null(kde)) {
      stopifnot(!is.null(data))
      kde <- density(data, bw = "SJ", ...)
    } else if (is.null(data))
    {
      stopifnot(!is.null(kde))
    }
    prob <- kde$y / sum(kde$y)
    return(sample(kde$x, size = n, replace = TRUE, prob = prob))
  }

  if (identical(method, "surrogate"))
  {
    return(sample(tseries::surrogate(data, ns = 1, ...),
                  size = n,
                  replace = TRUE))
  }

  if (identical(method, "bootstrap"))
  {
    return(sample(tseries::tsbootstrap(data, nb = 1, type = "block", b = 1, ...),
                  size = n,
                  replace = TRUE))
  }
}

# 2. Approximate Inverse Transform Sampling
# Function for approximate inverse transform sampling with duplicate handling
inverse_transform_kde <- function(data, n = 1000) {
  #kde <- density(data, bw = "SJ", kernel = "epanechnikov")
  kde <- density(data, bw = "SJ")
  prob <- kde$y / sum(kde$y)
  cdf <- cumsum(prob)

  # Ensure x-values and CDF values are unique
  unique_indices <- !duplicated(cdf)
  cdf_unique <- cdf[unique_indices]
  x_unique <- kde$x[unique_indices]

  # Generate uniform random numbers
  u <- runif(n)

  # Perform interpolation using unique CDF values
  simulated_data <- approx(cdf_unique, x_unique, u)$y

  # Replace NA values with the median of the interpolated values
  median_value <- median(simulated_data, na.rm = TRUE)
  simulated_data[is.na(simulated_data)] <- median_value

  return(simulated_data)
}


# 3. KDE Estimation and Sampling using `kde1d` package
kde1d_sampling <- function(data, n = 1000) {
  kde_estimate <- kde1d::kde1d(data)
  simulated_data <- kde1d::rkde1d(n, kde_estimate)
  return(simulated_data)
}


# Function for improved tail fitting
improved_direct_sampling <- function(data,
                                      n = 1000,
                                      method = c("surrogate",
                                                 "kde",
                                                 "bootstrap"),
                                      kde = NULL,
                                      seed = NULL,
                                      ...) {

  method <- match.arg(method)
  num_tail_simulated <- 0
  num_tail_simulated_positive <- 0
  num_tail_simulated_negative <- 0
  gpd_simulated <- NULL

  # Fit GPD to the tails
  tail_data <- boxplot(data, plot=FALSE)$out
  tail_data_positive <- tail_data[tail_data > 0]
  tail_data_negative <- tail_data[tail_data < 0]
  tail_proportion <- length(tail_data) / length(data)
  tail_proportion_positive <- length(tail_data_positive) / length(data)
  tail_proportion_negative <- length(tail_data_negative) / length(data)

  if (tail_proportion_positive > 0)
  {
    fit_gev_positive <- as.list(evd::fgev(tail_data_positive,
                                          std.err=FALSE)$estimate)
    num_tail_simulated_positive <- ceiling(n * tail_proportion_positive)
  }

  if (tail_proportion_negative > 0)
  {
    fit_gev_negative <- as.list(evd::fgev(-tail_data_negative,
                                          std.err=FALSE)$estimate)
    num_tail_simulated_negative <- ceiling(n * tail_proportion_negative)
  }

  # Simulate the total number of data points required
  num_tail_simulated <- num_tail_simulated_positive + num_tail_simulated_negative
  num_kde_simulated <- n - max(num_tail_simulated, 0)

  # Generate data from GPD
  if (tail_proportion_positive > 0)
  {
    gpd_simulated_positive <- do.call(evd::rgev,
                           args = c(n=num_tail_simulated_positive,
                                               as.list(fit_gev_positive)))
    gpd_simulated <- c(gpd_simulated, gpd_simulated_positive)
   }

  if (tail_proportion_negative > 0)
  {
    gpd_simulated_negative <- do.call(evd::rgev,
                           args = c(n=num_tail_simulated_negative,
                                               as.list(fit_gev_negative)))
    gpd_simulated <- c(gpd_simulated, -gpd_simulated_negative)
  }

  # Generate data
  if (!is.null(gpd_simulated))
  {
    otherwise_simulated <- direct_sampling(setdiff(data, tail_data),
                                   n = num_kde_simulated,
                                   method = method,
                                   kde = kde,
                                   seed = seed,
                                   ...)
    # Combine KDE and GPD
    return(sample(c(otherwise_simulated, gpd_simulated)))
  } else {
    return(direct_sampling(data,
                          n = num_kde_simulated,
                          method = method,
                          kde = kde,
                          seed = seed,
                          ...))
  }
}
```


```r
%%R

# Sample data for testing
set.seed(12)
data <- rt(n = 1000, df=2)
#data <- rlnorm(n = 1000)
boxplot(data)

# Benchmark the three methods
benchmark_results <- microbenchmark(
  Direct_Sampling = direct_sampling(data, n = 1000),
  Direct_Sampling_Surrogate = direct_sampling(data, n = 1000, method="surrogate"),
  Direct_Sampling_Bootstrap = direct_sampling(data, n = 1000, method="bootstrap"),
  Inverse_Transform = inverse_transform_kde(data, n = 1000),
  Improved_Direct_Sampling = improved_direct_sampling(data, n = 1000),
  KDE1d_Sampling = kde1d_sampling(data, n = 1000),
  times = 100
)

# Print benchmark results
print(benchmark_results)

# Function to plot original vs simulated density
plot_density_comparison <- function(original_data, simulated_data, method_name) {
  original_density <- density(original_data)
  simulated_density <- density(simulated_data)

  # Create a dataframe for ggplot
  df <- data.frame(
    x = c(original_density$x, simulated_density$x),
    y = c(original_density$y, simulated_density$y),
    Type = rep(c("Original", "Simulated"), each = length(original_density$x))
  )

  ggplot(df, aes(x = x, y = y, color = Type)) +
    geom_line(size = 1.2) +
    labs(title = paste("Density Comparison:", method_name),
         x = "Value", y = "Density") +
    theme_minimal()
}

# Generate simulated data and plot comparisons
simulated_data_direct <- direct_sampling(data, n = 1000)
simulated_data_direct_surrogate <- direct_sampling(data, n = 1000, method="surrogate")
simulated_data_direct_bootstrap <- direct_sampling(data, n = 1000, method="bootstrap")
simulated_data_inverse <- inverse_transform_kde(data, n = 1000)
simulated_data_improved_inverse <- improved_direct_sampling(data, n = 1000)
simulated_data_kde1d <- kde1d_sampling(data, n = 1000)

# Plot density comparisons
plot1 <- plot_density_comparison(data, simulated_data_direct,
                                 "Direct Sampling")
plot5 <- plot_density_comparison(data, simulated_data_direct_surrogate,
                                 "Direct Sampling (Surrogate)")
plot6 <- plot_density_comparison(data, simulated_data_direct_bootstrap,
                                 "Direct Sampling (Bootstrap)")
plot2 <- plot_density_comparison(data, simulated_data_inverse,
                                 "Inverse Transform")
plot3 <- plot_density_comparison(data, simulated_data_kde1d,
                                 "KDE1d Sampling")
plot4 <- plot_density_comparison(data, simulated_data_improved_inverse,
                                 "Improved Direct Sampling")

# Display the plots
print(plot1)
print(plot2)
print(plot3)
print(plot5)
print(plot6)
print(plot4)

# Optional: plot the benchmark results for better visualization
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  autoplot(benchmark_results)
}

```

    Unit: microseconds
                          expr      min        lq      mean    median        uq
               Direct_Sampling 1418.931 1613.0310 1933.4443 1725.9520 1927.7870
     Direct_Sampling_Surrogate  198.815  234.3975  312.6513  289.6470  347.5540
     Direct_Sampling_Bootstrap  186.152  245.6500  312.5235  287.0165  371.2195
             Inverse_Transform 1591.152 1757.2420 1957.6380 1880.2680 2089.3670
      Improved_Direct_Sampling 5364.844 5821.7710 7249.4502 6322.6050 6838.8615
                KDE1d_Sampling 5058.386 5406.8820 5779.4887 5601.6085 5930.1325
           max neval
      7040.770   100
      1291.163   100
       609.353   100
      3180.180   100
     46254.558   100
      9136.384   100


    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image1.png){:class="img-responsive"}      
    



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image2.png){:class="img-responsive"}      
    



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image3.png){:class="img-responsive"}      
    



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image4.png){:class="img-responsive"}      
    



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image5.png){:class="img-responsive"}      
    



![xxx]({{base}}/images/2024-08-26/2024-08-26-image6.png){:class="img-responsive"}      
    


    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image7.png){:class="img-responsive"}      
    


##### confidence interval for the mean


```r
%%R

# 1 - functions -----------------------------------------------------------

# Define the function to split vector into three equal-sized parts
stratify_vector_equal_size <- function(vector, seed = 123) {
  # Set seed for reproducibility
  set.seed(seed)

  n <- length(vector)

  # First split: Create training and remaining (calibration + test)
  train_index <- drop(caret::createDataPartition(vector, p = 0.3, list = FALSE))
  train_data <- vector[train_index]
  remaining_data <- vector[-train_index]

  # Second split: Create calibration and test sets
  calib_index <- drop(caret::createDataPartition(remaining_data, p = 0.5, list = FALSE))
  calib_data <- remaining_data[calib_index]
  test_data <- remaining_data[-calib_index]

  # Return results as a list
  return(list(
    train = train_data,
    calib = calib_data,
    test = test_data
  ))
}

#' nonparametric confidence interval xx
#' no hypothesis is made about the distribution of xx
#' will probably fail if xx is not stationary
compute_ci_mean <- function(xx,
                            type_split = c("random",
                                         "sequential"),
                            method = c("kde",
                                       "surrogate",
                                       "bootstrap"),
                            level=95,
                            fit_tail=FALSE,
                            kde=NULL,
                            seed = 123)
{
  n <- length(xx)
  type_split <- match.arg(type_split)
  method <- match.arg(method)
  upper_prob <- 1 - 0.5*(1 - level/100)

  if (type_split == "random")
  {
    set.seed(seed)
    z <- stratify_vector_equal_size(xx, seed=seed)
    x_train <- z$train
    x_calib <- z$calib
    x_test <- z$test
    estimate_x <- base::mean(x_train)
    sd_x <- sd(x_train)

    calib_resids <- (x_calib - estimate_x)/sd_x # standardization => distro of gaps to the mean is homoscedastic and centered ('easier' to sample since stationary?)
    if (fit_tail)
    {
      if (!is.null(kde))
      {
        sim_calib_resids <- replicate(n=250L, improved_direct_sampling(calib_resids,
                                                    n=length(x_calib),
                                                    method = method,
                                                    kde = kde,
                                                    seed=NULL))
        } else {
        sim_calib_resids <- replicate(n=250L, improved_direct_sampling(calib_resids,
                                                    n=length(x_calib),
                                                    method = method,
                                                    seed=NULL))
      }
    } else {
      if (!is.null(kde))
      {
        sim_calib_resids <- replicate(n=250L, direct_sampling(calib_resids,
                                                              n=length(x_calib),
                                                              kde=kde,
                                                              seed=NULL))
      } else {
        sim_calib_resids <- replicate(n=250L, direct_sampling(calib_resids,
                                                              n=length(x_calib),
                                                              method = method,
                                                              seed=NULL))
      }
    }

pseudo_obs_x <- x_test + sd_x*sim_calib_resids
pseudo_means_x <- colMeans(pseudo_obs_x)
lower <- quantile(pseudo_means_x, probs = 1 - upper_prob)
upper <- quantile(pseudo_means_x, probs = upper_prob)
pvalue <- mean((lower > x_test) + (x_test > upper))

return(list(estimate = base::mean(x_test),
            lower = lower,
            upper = upper,
            pvalue = pvalue,
            pvalueboxtest = stats::Box.test(xx)$p.value))
  }
}

```

##### Examples


```r
%%R
z <- c(50, 100, 250, 500,
       750, 1000, 1500, 2000, 2500, 5000,
       7500, 10000, 25000, 50000, 100000,
       250000, 500000, 1000000, 1500000, 2000000)

n_iter <- length(z)

level <- 95

(lower_prob <- 0.5*(1 - level/100))

(upper_prob <- 1 - 0.5*(1 - level/100))

method <- "kde"

set.seed(123)

par(mfrow=c(3, 1))
i <- 1
lowers <- rep(0, n_iter)
uppers <- rep(0, n_iter)
pvalues <- rep(0, n_iter)
pb <- utils::txtProgressBar(min=1, max=n_iter, style=3)
for (n_sims in z)
{
   x <- rnorm(n_sims)
   res <- compute_ci_mean(x, level=level, method=method)
   uppers[i] <- res$upper
   lowers[i] <- res$lower
   pvalues[i] <- res$pvalue
   i <- i + 1
   utils::setTxtProgressBar(pb, i)
 }
close(pb)

plot(log(z), lowers, type='l', main="convergence towards \n lower bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), uppers, type='l', main="convergence towards \n upper bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), pvalues, type='l', main="convergence towards \n 'p-value'",
      xlab="log(number of simulations)")
abline(h = 1 - level/100, col="red", lty=2)

```

      |======================================================================| 100%



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image8.png){:class="img-responsive"}      
    



```r
%%R

method <- "surrogate"

set.seed(123)

par(mfrow=c(3, 1))
i <- 1
lowers <- rep(0, n_iter)
uppers <- rep(0, n_iter)
pvalues <- rep(0, n_iter)
pb <- utils::txtProgressBar(min=1, max=n_iter, style=3)
for (n_sims in z)
{
   x <- rlnorm(n_sims)
   res <- compute_ci_mean(x, level=level, method=method)
   uppers[i] <- res$upper
   lowers[i] <- res$lower
   pvalues[i] <- res$pvalue
   i <- i + 1
   utils::setTxtProgressBar(pb, i)
 }
close(pb)

plot(log(z), lowers, type='l', main="convergence towards \n lower bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), uppers, type='l', main="convergence towards \n upper bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), pvalues, type='l', main="convergence towards \n 'p-value'",
      xlab="log(number of simulations)")
abline(h = 1 - level/100, col="red", lty=2)

```

      |======================================================================| 100%



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image9.png){:class="img-responsive"}      
    



```r
%%R

method <- "bootstrap"

set.seed(123)

par(mfrow=c(3, 1))
i <- 1
lowers <- rep(0, n_iter)
uppers <- rep(0, n_iter)
pvalues <- rep(0, n_iter)
pb <- utils::txtProgressBar(min=1, max=n_iter, style=3)
for (n_sims in z)
{
   x <- rexp(n_sims)
   res <- compute_ci_mean(x, level=level, method=method)
   uppers[i] <- res$upper
   lowers[i] <- res$lower
   pvalues[i] <- res$pvalue
   i <- i + 1
   utils::setTxtProgressBar(pb, i)
 }
close(pb)

plot(log(z), lowers, type='l', main="convergence towards \n lower bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), uppers, type='l', main="convergence towards \n upper bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), pvalues, type='l', main="convergence towards \n 'p-value'",
      xlab="log(number of simulations)")
abline(h = 1 - level/100, col="red", lty=2)

```

      |======================================================================| 100%



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image10.png){:class="img-responsive"}      
    



```r
%%R

method <- "surrogate"

set.seed(123)

par(mfrow=c(3, 1))

i <- 1
lowers <- rep(0, n_iter)
uppers <- rep(0, n_iter)
pvalues <- rep(0, n_iter)
pb <- utils::txtProgressBar(min=1, max=n_iter, style=3)
for (n_sims in z)
{
   x <- 2 + rt(n_sims, df=3)
   res <- compute_ci_mean(x, level=level, method=method)
   uppers[i] <- res$upper
   lowers[i] <- res$lower
   pvalues[i] <- res$pvalue
   i <- i + 1
   utils::setTxtProgressBar(pb, i)
 }
close(pb)

plot(log(z), lowers, type='l', main="convergence towards \n lower bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), uppers, type='l', main="convergence towards \n upper bound value",
      xlab="log(number of simulations)")
abline(h = mean(x), col="red", lty=2)
plot(log(z), pvalues, type='l', main="convergence towards \n 'p-value'",
      xlab="log(number of simulations)")
abline(h = 1 - level/100, col="red", lty=2)

```

      |======================================================================| 100%



    
![xxx]({{base}}/images/2024-08-26/2024-08-26-image11.png){:class="img-responsive"}      
    

### Conclusion

This nonparametric method for constructing a confidence interval for the mean is powerful because it does not rely on the assumption of normality or other specific distributions. By using stratified sampling, standardization, and the generation of pseudo-observations, we can obtain a confidence interval that accurately reflects the uncertainty in our estimate of the mean, even in complex, non-standard situations.
