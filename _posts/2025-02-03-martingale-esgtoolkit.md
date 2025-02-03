---
layout: post
title: "A simple test of the martingale hypothesis in esgtoolkit"
description: "Details and examples of a simple test of the martingale hypothesis in esgtoolkit"
date: 2025-02-03
categories: R
comments: true
---

**Disclaimer**: This is a work in progress.

R code at the end. 

## Introduction

A fundamental utility in financial econometrics and stochastic processes is the **martingale property**, which implies that the best approximation of the future value of a time series or stochastic process, based on its historical values, is its present value. This property is critical in the efficient market hypothesis, risk-neutral pricing models. Testing whether a given time series satisfies the martingale hypothesis involves examining whether past values significantly predict future changes. This blog post outlines a formalized statistical test implemented in the [esgtoolkit](https://github.com/techtonique/esgtoolkit) package, and leveraging multiple linear regression, F-statistics, and residual diagnostics to determine whether a time series follows a martingale process.

## Martingale Hypothesis Test

Let $X=\left\{X_{1}, X_{2}, \ldots, X_{n}\right\}$ be a time series, where each $X_{t}$ represents a multivariate observation at time $t$. We are interested in testing the martingale hypothesis, which posits that the future value $X_{t+1}$ is unpredictable given the past values. That is:

$$
\begin{equation*}
\mathbb{E}\left[X_{n+1} \mid \sigma\left(X_{n}, X_{n-1}, \ldots, X_{1}\right)\right]=X_{n} \tag{1}
\end{equation*}
$$

Where $\sigma\left(X_{n}, X_{n-1}, \ldots, X_{1}\right)$ is the sigma-algebra containing all the information about $X^{\prime}$ 's past values.

## 1. Regression Model

One way to conduct such a test of the Martingale Hypothesis, is to adjust a multiple linear regression of the change in the series, $\Delta X_{t+1}=X_{t+1}-X_{t}$, on the past values $X_{t}, X_{t-1}, \ldots, X_{1}$, for all $t>0$ :

$$
\Delta X_{t+1} \approx \beta_{0}+\beta_{1} X_{t}+\beta_{2} X_{t-1}+\cdots+\beta_{p} X_{1}+\epsilon_{t+1}
$$

where $\epsilon_{t+1}$ are the (centered and homoskedastic) residuals of the regression, because, under the assumption that $\hat{\beta}_{1}=\hat{\beta}_{2}=\cdots=\hat{\beta}_{p}=$ 0 , we'd have:

$$
\begin{equation*}
\mathbb{E}\left[X_{t+1}-X_{t} \mid \sigma\left(X_{t}, X_{t-1}, \ldots, X_{1}\right)\right]=0 \tag{2}
\end{equation*}
$$

That's one of the simplest way to do it, and although we could think of other expressions of the conditional expectation, these would require more engineering.

## 2. F-statistic

The significance of the regression model is tested using the Fisher F-statistic. The null hypothesis $H_{0}$ is that none of the past values $X_{1}, X_{2}, \ldots, X_{t}$ significantly explains the change $\Delta X_{t+1}$ :

$$
H_{0}: \beta_{1}=\beta_{2}=\cdots=\beta_{p}=0
$$

The F-statistic is computed as:

$$
F=\frac{R^{2} / p}{\left(1-R^{2}\right) /(n-p-1)}
$$

where $R^{2}$ is the coefficient of determination, and $p$ is the number of predictors (lags). A significantly large value of $F$ would lead to reject $H_{0}$.

## 3. Critical Value and p-value

The critical value for the Fisher-Snedecor statistic is obtained from the F-distribution with $p$ and $n-p-1$ degrees of freedom at a chosen significance level $\alpha$ :

$$
F_{\text {critical }}=F_{\alpha}(p, n-p-1)
$$

The p -value of the F-statistic is computed as:

$$
p \text {-value }=P\left(F \geq F_{\text {observed }} \mid H_{0} \text { is true }\right)
$$

If the p -value is less than the chosen significance level $\alpha$, we'll reject the null hypothesis and conclude that the past values explain the change in the time series.

## 4. Stationarity of Residuals

To test the stationarity of the residuals $\epsilon_{t+1}$, we perform the Augmented Dickey-Fuller (ADF) test. The null hypothesis $H_{0}$ is that the residuals are non-stationary (i.e., they follow a random walk):

$$
H_{0}: \text { Residuals are non-stationary }
$$

The test statistic is the $t$-statistic of the lagged residuals, and the p -value indicates whether we can reject the null hypothesis. If the p -value is less than $\alpha$, we reject the null hypothesis and conclude that the residuals are stationary.

## 5. Autocorrelation of Residuals

In addition to the F-test, and in order to ensure no autocorrelation remains in the residuals, we perform the Ljung-Box test. The null hypothesis $H_{0}$ is that there is no autocorrelation in the residuals:

$$
H_{0} \text { : No autocorrelation in residuals }
$$

The Ljung-Box test statistic is computed as:

$$
Q=n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_{k}^{2}}{n-k}
$$

where $\hat{\rho}_{k}$ is the sample autocorrelation at lag $k$, and $m$ is the maximum lag. The p -value is computed based on this statistic.

## Examples

See: [https://techtonique.github.io/ESGtoolkit/articles/martingale_test.html](https://techtonique.github.io/ESGtoolkit/articles/martingale_test.html)

![image-title-here]({{base}}/images/2023-10-09/2023-10-09-image1.png){:class="img-responsive"}