---
layout: post
title: "mlreserve: machine-learning loss reserving on 'real-world' triangles (based on a 'ChainLadder' fork)"
description: "This post presents a function `mlReserve()` for machine-learning loss reserving on 'real-world' triangles, based on a fork of the 'ChainLadder' package. It includes two experiments, each scored against a known true reserve: (A) Two triangles from an individual-claims simulator (Wang & Wuthrich), embedded below: chain ladder vs mlReserve, in detail; (B) A mini-benchmark on freshly simulated triangles from a simpler aggregate generator, with tunable distortions."
date: 2026-10-03
categories: R
comments: true
---

This post presents a function `mlReserve()` for machine-learning loss reserving on 'real-world' triangles, **based on a fork of R's 'ChainLadder' package** ([https://github.com/thierrymoudiki/ChainLadder/tree/master](https://github.com/thierrymoudiki/ChainLadder/tree/master)). It includes two experiments, each scored against a known true reserve: (A) Two triangles from an individual-claims simulator (Wang & Wüthrich, [https://github.com/actuarial-data-science/PackageIndividualClaimsSimulator](https://github.com/actuarial-data-science/PackageIndividualClaimsSimulator)), embedded below: chain ladder vs mlReserve, in detail; (B) A mini-benchmark on freshly simulated triangles from a simpler aggregate generator, with tunable distortions.

```R
# =============================================================================
# mlReserve: machine-learning loss reserving on "real-world" triangles
#
# Colab: Runtime > Change runtime type > R
#
# Two experiments, each scored against a known true reserve:
#   A. Two triangles from an individual-claims simulator (Wang & Wüthrich),
#      embedded below: chain ladder vs mlReserve, in detail.
#   B. A mini-benchmark on freshly simulated triangles from a simpler aggregate
#      generator (defined in step 4), with tunable distortions.
# =============================================================================

# ---- 0. Setup -------------------------------------------------------------------

options(Ncpus = 2, repos = "https://cloud.r-project.org",
        repr.plot.width = 12, repr.plot.height = 5.5)
for (pkg in c("remotes", "ggplot2", "ranger", "e1071"))
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
if (!requireNamespace("ChainLadder", quietly = TRUE) ||
    !"mlReserve" %in% getNamespaceExports("ChainLadder"))
  remotes::install_github("thierrymoudiki/ChainLadder", upgrade = "never")

suppressPackageStartupMessages({
  library(ChainLadder)
  library(ggplot2)
})

n      <- 10                                   # triangle size (accident years)
models <- c("Chain ladder", "mlReserve: SVM", "mlReserve: random forest")
ink    <- "#0b0b0b"
colours <- c("Chain ladder"             = "#eb6834",
             "mlReserve: SVM"           = "#2a78d6",
             "mlReserve: random forest" = "#1baf7a",
             "Truth"                    = ink)
theme_set(theme_minimal(base_size = 13) +
            theme(legend.position = "top", legend.title = element_blank(),
                  panel.grid.minor = element_blank(), plot.title.position = "plot",
                  strip.text = element_text(face = "bold")))


# ---- 1. Helpers -----------------------------------------------------------------

# A full n x n square of incremental payments -> observed (cumulative) upper
# triangle + true outstanding reserve per accident year (sum of the lower triangle)
make_case <- function(full) {
  dimnames(full) <- list(origin = 1:n, dev = 1:n)
  future <- row(full) + col(full) > n + 1
  upper  <- full; upper[future] <- NA
  list(full      = full,
       upper     = incr2cum(as.triangle(upper)),
       true_ibnr = rowSums(full * future))
}

# Fit the three models; `uncertainty = FALSE` gives point estimates only
fit_models <- function(triangle, uncertainty = TRUE, nsim = 300) {
  mse <- if (uncertainty) "bootstrap" else "none"
  fits <- list(
    MackChainLadder(triangle, est.sigma = "Mack"),
    mlReserve(triangle, e1071::svm, features = "numeric", transform = "asinh",
              mse.method = mse, nsim = nsim, seed = 1),
    mlReserve(triangle, ranger::ranger, num.trees = 300, num.threads = 1,
              features = "both", transform = "asinh",
              mse.method = mse, nsim = nsim, seed = 1))
  setNames(fits, models)
}

# Reserve (IBNR) and standard error by accident year, whatever the model
reserve_by_origin <- function(fit) {
  if (inherits(fit, "MackChainLadder")) {
    s <- summary(fit)$ByOrigin
    data.frame(origin = 1:n, IBNR = s$IBNR, SE = s$Mack.S.E)
  } else {
    s <- fit$summary[rownames(fit$summary) != "total", ]
    data.frame(origin = as.integer(rownames(s)), IBNR = s$IBNR, SE = s$S.E)
  }
}
total_reserve <- function(fit) sum(reserve_by_origin(fit)$IBNR)


# ---- 2. Data for experiment A ---------------------------------------------------

# Incremental paid (thousands), aggregated from the Wang & Wuthrich individual
# claims simulator (github.com/actuarial-data-science/PackageIndividualClaimsSimulator):
#   "Speed-up"     : settlement accelerates for recent accident years (delays -45%)
#   "All combined" : speed-up + calendar inflation shock (2% -> 15% a year from
#                    calendar month 78) + a few large, slowly paid claims
squares <- list(
  "Speed-up" = c(4172.1, 16153.4, 14800.2, 7561.1, 3275, 1856.3, 726.7, 410.7, 35.8, 140, 4983.2, 18064.1, 13574.5, 8350.7, 3954.9, 1499, 510.3, 9.5, 0, 11.3, 4698.1, 19428.8, 14864, 7343, 3710.5, 1378.6, 384.7, 92.2, 6.8, 27.1, 5467.1, 18481, 14847.6, 8089.7, 2794.1, 1379.2, 549.5, 152.2, 6.8, 0, 5985.2, 21516.6, 15980.6, 8353, 2954.2, 1655.5, 636.4, 191.3, 211.3, 28.9, 8274.1, 22014.4, 15981.4, 7425.8, 2362.7, 731, 242.1, 41.6, 52.8, 0, 8148.4, 24712.9, 16329.1, 6587.5, 2804.9, 753.6, 49.5, 0, 0, 0, 9974.2, 27573.4, 14523, 5141.6, 1123.8, 350, 73.3, 0, 0, 0, 10698.8, 29877.8, 14303.3, 3200.3, 978.3, 239.1, 0, 0, 0, 0, 13669.1, 30541.7, 10984, 1837.5, 157.3, 0, 0, 0, 0, 0),
  "All combined" = c(4236.8, 16766.9, 15671.5, 8229, 3582.8, 2143.9, 838.8, 548.6, 57.6, 261.6, 5195.7, 19008.8, 15926.9, 9857.6, 4643.8, 1746.5, 712.4, 26.4, 0, 22.5, 4956.8, 20861.6, 16283.4, 8704.2, 4321.7, 1876.2, 666.5, 161.8, 14.7, 61.4, 5908.2, 20336.4, 16580.1, 9331.1, 3673.8, 2276.2, 1205.8, 332.2, 15.4, 0, 6586.1, 24045.6, 18499, 11218.3, 4576, 3174.8, 1307, 460.6, 665, 96.3, 9292, 26084.3, 21055.7, 11396.7, 4203.8, 1531.8, 585.4, 115, 163.7, 0, 9702.9, 33010, 25202.7, 12133, 6404.4, 1956.8, 385.9, 0, 0, 0, 13752.5, 42686.8, 25857.7, 10698.4, 2706.5, 3213, 4232.5, 0, 0, 0, 17094.3, 53962.4, 30132.7, 8527.6, 3087.9, 746.2, 0, 0, 0, 0, 25375.8, 63473.7, 26195.3, 5737.2, 507, 674.2, 0, 0, 0, 0))
cases <- lapply(squares, function(v) make_case(matrix(v, n, n, byrow = TRUE)))
combo <- cases[["All combined"]]

cat("Observed cumulative paid, 'All combined' (thousands):\n")
print(round(combo$upper))
fits <- suppressWarnings(fit_models(combo$upper))
cat("\nmlReserve with a random forest; true total reserve =",
    format(round(sum(combo$true_ibnr)), big.mark = ","), "\n")
print(fits[["mlReserve: random forest"]])


# ---- 3. Experiment A: one triangle in detail ------------------------------------

# 3a. Reserve by accident year vs the truth
by_origin <- do.call(rbind, lapply(models, function(m)
  data.frame(model = m, reserve_by_origin(fits[[m]]))))
by_origin <- subset(by_origin, origin >= 2)             # AY 1 is fully developed
by_origin$model <- factor(by_origin$model, levels = models)
truth_points <- data.frame(origin = 2:n, IBNR = combo$true_ibnr[2:n])

print(
  ggplot(by_origin, aes(origin, IBNR / 1000, colour = model)) +
    geom_linerange(aes(ymin = (IBNR - SE) / 1000, ymax = (IBNR + SE) / 1000),
                   position = position_dodge(0.6), linewidth = 0.9) +
    geom_point(position = position_dodge(0.6), size = 2.6) +
    geom_point(data = truth_points, aes(origin, IBNR / 1000), inherit.aes = FALSE,
               shape = 23, size = 3.4, fill = ink, colour = "white") +
    scale_colour_manual(values = colours) +
    scale_x_continuous(breaks = 2:n) +
    labs(title = "Reserve by accident year, +/- one standard error (black diamonds: truth)",
         x = "Accident year", y = "Reserve (millions)"))

# 3b. Predictive distribution of the total reserve. Each model uses its own
#     resampling scheme: ODP bootstrap (BootChainLadder) for chain ladder,
#     residual bootstrap (mlReserve) for the learners.
set.seed(1)
boot <- BootChainLadder(combo$upper, R = 2000, process.distr = "od.pois")
draws <- rbind(
  data.frame(model = models[1], total = boot$IBNR.Totals),
  data.frame(model = models[2], total = rowSums(fits[[2]]$sims.reserve.pred)),
  data.frame(model = models[3], total = rowSums(fits[[3]]$sims.reserve.pred)))
draws$model <- factor(draws$model, levels = models)

print(
  ggplot(draws, aes(total / 1000, colour = model)) +
    geom_density(linewidth = 1, adjust = 1.2) +
    geom_vline(xintercept = sum(combo$true_ibnr) / 1000, colour = ink, linewidth = 0.8) +
    annotate("text", x = sum(combo$true_ibnr) / 1000, y = Inf, label = " true reserve",
             hjust = 0, vjust = 1.5, colour = ink) +
    scale_colour_manual(values = colours) +
    labs(title = "Model-specific predictive distributions of the total reserve",
         x = "Total reserve (millions)", y = "Density"))

cat("\nPredictive quantiles of the total reserve (thousands):\n")
quants <- t(sapply(split(draws$total, draws$model), quantile, c(0.025, 0.5, 0.975)))
print(noquote(formatC(round(quants), format = "d", big.mark = ",")))

# 3c. Projected cumulative payments vs the truth, accident years 7-10
cumulative_paths <- function(case, label) {
  cl <- MackChainLadder(case$upper, est.sigma = "Mack")
  rf <- mlReserve(case$upper, ranger::ranger, num.trees = 300, num.threads = 1,
                  features = "both", transform = "asinh", mse.method = "none", seed = 1)
  projections <- list("Truth"                    = t(apply(case$full, 1, cumsum)),
                      "Chain ladder"             = unclass(cl$FullTriangle),
                      "mlReserve: random forest" = unclass(rf$FullTriangle))
  out <- expand.grid(dev = 1:n, origin = 7:n, model = names(projections),
                     stringsAsFactors = FALSE)
  out$value <- mapply(function(o, d, m) projections[[m]][o, d],
                      out$origin, out$dev, out$model)
  out$panel <- paste0(label, " · AY ", out$origin)
  subset(out, dev >= n + 1 - origin)      # from the last observed diagonal onwards
}
paths <- do.call(rbind, Map(cumulative_paths, cases, names(cases)))
paths$model <- factor(paths$model, levels = c("Truth", models[c(1, 3)]))
paths$panel <- factor(paths$panel, levels = unique(paths$panel))

options(repr.plot.height = 7)
print(
  ggplot(paths, aes(dev, value / 1000, colour = model, linetype = model)) +
    geom_line(linewidth = 1) +
    facet_wrap(~ panel, nrow = 2, scales = "free_y") +
    scale_colour_manual(values = colours) +
    scale_linetype_manual(values = c("solid", "22", "solid")) +
    scale_x_continuous(breaks = seq(2, n, 2)) +
    labs(title = "Projected cumulative payments from the last observed diagonal",
         x = "Development year", y = "Cumulative paid (millions)"))
options(repr.plot.height = 5.5)


# ---- 4. Experiment B: mini-benchmark on fresh triangles -------------------------

# Aggregate generator: gamma payment pattern with over-dispersed Poisson noise.
#   speedup   : mean payment delay shrinks by up to this share for recent AYs
#   inflation : calendar-year log-growth rate after calendar period 7 (2% before)
simulate_case <- function(seed, speedup = 0.3, inflation = 0.08, phi = 5) {
  set.seed(seed)
  ultimate <- 5e4 * (1 + 0.03 * (0:(n - 1))) * exp(rnorm(n, 0, 0.05))
  delay    <- 3 * (1 - speedup * pmax(0, (1:n - 4) / (n - 4)))
  pattern  <- t(sapply(delay, function(m) diff(pgamma(0:n, shape = 2, scale = m / 2))))
  calendar <- outer(1:n, 1:n, "+") - 1
  infl     <- exp(0.02 * pmin(calendar, 7) + inflation * pmax(calendar - 7, 0))
  mean_incr <- ultimate * pattern * infl
  make_case(phi * matrix(rpois(n * n, mean_incr / phi), n))
}

n_rep <- 20                                   # triangles per scenario
scenarios <- list(
  # "Baseline (no distortion)" = c(speedup = 0,   inflation = 0.02),  # control
  "Settlement speed-up"      = c(speedup = 0.3, inflation = 0.02),
  "Speed-up + inflation"     = c(speedup = 0.3, inflation = 0.08))

bench <- do.call(rbind, lapply(names(scenarios), function(sc) {
  do.call(rbind, lapply(seq_len(n_rep), function(seed) {
    case <- simulate_case(seed, speedup   = scenarios[[sc]][["speedup"]],
                                inflation = scenarios[[sc]][["inflation"]])
    point_fits <- suppressWarnings(fit_models(case$upper, uncertainty = FALSE))
    data.frame(scenario = sc, seed = seed, model = models,
               error = sapply(point_fits, total_reserve) - sum(case$true_ibnr))
  }))
}))

score <- aggregate(error ~ scenario + model, data = bench, FUN = function(e)
  c(bias = mean(e), RMSE = sqrt(mean(e^2)), MAE = mean(abs(e))))
score <- do.call(data.frame, score)
names(score) <- c("scenario", "model", "bias", "RMSE", "MAE")
score <- score[order(score$scenario, score$RMSE), ]

cat("\nError of the total reserve over", n_rep, "triangles per scenario (thousands):\n")
fmt <- function(x) formatC(round(x), format = "d", big.mark = ",")
print(transform(score, bias = fmt(bias), RMSE = fmt(RMSE), MAE = fmt(MAE)),
      row.names = FALSE)

long <- rbind(data.frame(score[c("scenario", "model")], metric = "RMSE", value = score$RMSE),
              data.frame(score[c("scenario", "model")], metric = "MAE",  value = score$MAE))
long$model <- factor(long$model, levels = rev(models))
print(
  ggplot(long, aes(value / 1000, model, fill = model)) +
    geom_col(width = 0.6) +
    facet_grid(metric ~ scenario, scales = "free_x") +
    scale_fill_manual(values = colours, guide = "none") +
    labs(title = "Out-of-sample error of the total reserve (lower is better)",
         x = "Millions", y = NULL))
```

    Observed cumulative paid, 'All combined' (thousands):
          dev
    origin     1     2     3     4     5     6     7     8     9    10
        1   4237 21004 36675 44904 48487 50631 51470 52018 52076 52338
        2   5196 24204 40131 49989 54633 56379 57092 57118 57118    NA
        3   4957 25818 42102 50806 55128 57004 57670 57832    NA    NA
        4   5908 26245 42825 52156 55830 58106 59312    NA    NA    NA
        5   6586 30632 49131 60349 64925 68100    NA    NA    NA    NA
        6   9292 35376 56432 67829 72032    NA    NA    NA    NA    NA
        7   9703 42713 67916 80049    NA    NA    NA    NA    NA    NA
        8  13752 56439 82297    NA    NA    NA    NA    NA    NA    NA
        9  17094 71057    NA    NA    NA    NA    NA    NA    NA    NA
        10 25376    NA    NA    NA    NA    NA    NA    NA    NA    NA
    
    mlReserve with a random forest; true total reserve = 174,050 
    mlReserve (formula interface)
    
            Latest Dev.To.Date Ultimate   IBNR         S.E        CV
    2      57118.1   0.9994925  57147.1     29    39.76128 1.3710786
    3      57832.2   0.9982222  57935.2    103    91.80065 0.8912685
    4      59311.6   0.9935674  59695.6    384   244.97640 0.6379594
    5      68099.8   0.9732881  69968.8   1869   888.80702 0.4755522
    6      72032.5   0.9286787  77564.5   5532  1605.55443 0.2902304
    7      80048.6   0.8558364  93532.6  13484  2443.15573 0.1811892
    8      82297.0   0.7087359 116118.0  33821  7158.48746 0.2116581
    9      71056.7   0.5017785 141609.7  70553 13547.27988 0.1920156
    10     25375.8   0.1881838 134845.8 109470 23912.17707 0.2184359
    total 573172.3   0.7090055 808417.3 235245 37338.60673 0.1587222



    
![image-title-here]({{base}}/images/2026-10-03/2026-10-03-mlreserve_0_1.png){:class="img-responsive"}
    


    
    Predictive quantiles of the total reserve (thousands):
                             2.5%    50%     97.5%  
    Chain ladder             316,100 356,087 397,519
    mlReserve: SVM           178,587 252,035 371,639
    mlReserve: random forest 162,579 217,369 301,946



    
![image-title-here]({{base}}/images/2026-10-03/2026-10-03-mlreserve_0_3.png){:class="img-responsive"}
    


    
    Error of the total reserve over 20 triangles per scenario (thousands):
                 scenario                    model    bias    RMSE     MAE
      Settlement speed-up mlReserve: random forest  67,436  67,519  67,436
      Settlement speed-up             Chain ladder  94,022  94,097  94,022
      Settlement speed-up           mlReserve: SVM  93,888  94,318  93,888
     Speed-up + inflation mlReserve: random forest  58,834  58,975  58,834
     Speed-up + inflation           mlReserve: SVM  79,344  79,800  79,344
     Speed-up + inflation             Chain ladder 100,217 100,311 100,217



    
![image-title-here]({{base}}/images/2026-10-03/2026-10-03-mlreserve_0_5.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-10-03/2026-10-03-mlreserve_0_6.png){:class="img-responsive"}
    

The experiment is not a universal proof that random forests are better. It is a demonstration that `mlReserve()` can be used to explore the performance of machine-learning models on loss reserving problems, and that it can outperform the classical chain ladder method in some scenarios.