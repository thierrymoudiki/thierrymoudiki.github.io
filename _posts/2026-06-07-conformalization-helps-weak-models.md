---
layout: post
title: "How Conformal Prediction Makes Linear Models Good Enough — An Example Using R Package mlS3"
description: "A comparison of split conformal prediction across several predictive models, using R package mlS3, on the BostonHousing dataset"
date: 2026-06-07
categories: R
comments: true
---

In this post, we compare [split conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction)
across several predictive models, using [R package mlS3](https://cran.r-project.org/web/packages/mlS3/index.html).

**The results are deliberately thought-provoking**, though limited to one dataset, so treat them
as a starting point rather than a general verdict (I [noticed something similar before](https://thierrymoudiki.github.io/blog/2026/05/21/r/python/Conformalized-TabICL-nnetsauce)...):
Linear regression, albeit generally misspecified, requires no hyperparameter search, produces
a fully interpretable white-box model, and under split conformal prediction achieves coverage
_statistically indistinguishable_ from that of gradient boosting ensembles. The residual gap in
[Winkler score](https://otexts.com/fpp3/distaccuracy.html#winkler-score) — approximately 5 units, or 19% relative to LightGBM (see Section 2) — reflects
the cost of a wider predictive interval, not a coverage failure. In enterprise settings where
auditability, regulatory traceability, speed, and operational reproducibility carry weight
alongside predictive performance, this trade-off would'nt be a concession but indeed, a considered
engineering choice.

We start by installing the necessary packages, then run a single-seed experiment to illustrate the process and results. In the second part, we run a multi-seed stability analysis.

# 0 - Installing packages


```R
# --- 0. Install / load packages ----------------------------------------------

if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak")

pak::pak(c(
  "mlbench",
  "mlS3",          # on CRAN — pak handles it cleanly
  "glmnet",
  "ranger",
  "e1071",
  "lightgbm",
  "caret",
  "randomForest",
  "gbm",
  "kknn",
  "nnet",
  "ggplot2",
  "tidyr"
))
```

# 1 - With one seed (not stable)


```R
# =============================================================================
# Split Conformal Prediction as a field-leveler for tabular regression
# Models: wrap_glmnet, wrap_ranger, wrap_svm, wrap_lightgbm, wrap_caret (x4)
# Dataset: BostonHousing (mlbench)
# Coverage target: 95%
# No tuning — all default hyperparameters
# Metrics: empirical coverage, mean interval length, Winkler score
# =============================================================================

# --- 0. Install / load packages ----------------------------------------------

library(mlS3)
library(mlbench)
library(ggplot2)
library(scales)

# --- 1. Data -----------------------------------------------------------------

data(BostonHousing)
df      <- BostonHousing
df$chas <- as.numeric(df$chas)

set.seed(42) # in the next section, we run multiple seeds to stabilize the results
n     <- nrow(df)
y_all <- df$medv
X_all <- as.data.frame(df[, names(df) != "medv"])

# 50% train | 30% calibration | 20% test
# larger calibration set → stable quantile estimate
idx_test  <- sample(n, round(0.20 * n))
rest      <- setdiff(seq_len(n), idx_test)
idx_cal   <- sample(rest, round(0.30 * n))
idx_train <- setdiff(rest, idx_cal)

X_train <- X_all[idx_train, ]; y_train <- y_all[idx_train]
X_cal   <- X_all[idx_cal,   ]; y_cal   <- y_all[idx_cal]
X_test  <- X_all[idx_test,  ]; y_test  <- y_all[idx_test]

cat(sprintf("Train: %d | Cal: %d | Test: %d\n",
            length(idx_train), length(idx_cal), length(idx_test)))

# --- 2. Model zoo ------------------------------------------------------------

models <- list(

  list(label    = "glmnet (ridge)",
       fit_fn   = wrap_glmnet,
       fit_args = list(alpha = 0)),

  list(label    = "glmnet (lasso)",
       fit_fn   = wrap_glmnet,
       fit_args = list(alpha = 1)),

  list(label    = "ranger (RF)",
       fit_fn   = wrap_ranger,
       fit_args = list(num.trees = 500L)),

  list(label    = "SVM (radial)",
       fit_fn   = wrap_svm,
       fit_args = list(kernel = "radial")),

  list(label    = "LightGBM",
       fit_fn   = wrap_lightgbm,
       fit_args = list(
         params  = list(objective = "regression", verbose = -1),
         nrounds = 100)),

  list(label    = "caret: lm",
       fit_fn   = wrap_caret,
       fit_args = list(method = "lm")),

  list(label    = "caret: knn",
       fit_fn   = wrap_caret,
       fit_args = list(method = "kknn")),

  list(label    = "caret: rf",
       fit_fn   = wrap_caret,
       fit_args = list(method = "rf", mtry = 3L))
)

# --- 3. Run ------------------------------------------------------------------
# Stores: results (summary metrics), cal_residuals_all, test_preds_all

alpha_main        <- 0.05
results           <- vector("list", length(models))
cal_residuals_all <- list()
test_preds_all    <- list()

for (i in seq_along(models)) {
  m <- models[[i]]
  cat(sprintf("\n[%d/%d] Fitting: %s ...\n", i, length(models), m$label))

  res <- tryCatch({

    mod <- do.call(m$fit_fn, c(list(X_train, y_train), m$fit_args))

    # calibration nonconformity scores
    cal_preds  <- predict(mod, newx = X_cal)
    scores     <- as.numeric(abs(cal_preds - y_cal))

    # finite-sample corrected conformal quantile
    n_cal   <- length(scores)
    q_level <- min(ceiling((n_cal + 1) * (1 - alpha_main)) / n_cal, 1)
    q_hat   <- as.numeric(quantile(scores, probs = q_level, type = 1))

    # test predictions and intervals
    test_preds <- as.numeric(predict(mod, newx = X_test))
    lower      <- test_preds - q_hat
    upper      <- test_preds + q_hat
    covered    <- (y_test >= lower) & (y_test <= upper)
    width      <- upper - lower

    # Winkler score — penalty is distance to nearest bound, always positive
    penalty <- ifelse(!covered,
                      (2 / alpha_main) * ifelse(y_test < lower,
                                                lower - y_test,
                                                y_test - upper),
                      0)

    list(
      scores     = scores,
      test_preds = test_preds,
      q_hat      = q_hat,
      coverage   = mean(covered),
      mean_len   = mean(width),
      winkler    = mean(width + penalty)
    )
  },
  error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL })

  if (!is.null(res)) {
    results[[i]] <- data.frame(
      Model       = m$label,
      Coverage    = round(res$coverage, 3),
      Mean_Length = round(res$mean_len, 3),
      Winkler     = round(res$winkler,  3),
      q_hat       = round(res$q_hat,    3),
      stringsAsFactors = FALSE
    )
    cal_residuals_all[[m$label]] <- data.frame(
      Model = m$label,
      resid = res$scores,
      q_hat = res$q_hat,
      row.names = NULL
    )
    test_preds_all[[m$label]] <- res$test_preds
  }
}

# --- 4. Summary table --------------------------------------------------------

summary_df         <- do.call(rbind, Filter(Negate(is.null), results))
summary_df$Cov_Dev <- round(summary_df$Coverage - (1 - alpha_main), 3)
summary_df         <- summary_df[order(summary_df$Winkler), ]
rownames(summary_df) <- NULL

cat("\n\n========== Split Conformal Prediction — BostonHousing ==========\n")
cat(sprintf("Nominal coverage: %.0f%%  |  alpha = %.2f\n\n",
            100 * (1 - alpha_main), alpha_main))
print(summary_df, row.names = FALSE)
cat("=================================================================\n\n")

# --- 5. Plots — main metrics -------------------------------------------------

# 5a. Coverage with deviation annotation
p_cov <- ggplot(summary_df,
                aes(x = reorder(Model, Coverage), y = Coverage)) +
  geom_col(fill = "#4C72B0", alpha = 0.85) +
  geom_hline(yintercept = 1 - alpha_main, linetype = "dashed",
             colour = "firebrick", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%+.1f%%", Cov_Dev * 100)),
            hjust = -0.15, size = 3.2, colour = "grey30") +
  annotate("text", x = 0.6, y = 1 - alpha_main + 0.005,
           label = "95% target", colour = "firebrick", hjust = 0, size = 3.5) +
  coord_flip(ylim = c(0.80, 1.02)) +
  labs(title    = "Empirical Coverage — the leveled field",
       subtitle = "All conformalized models should hover near 95%\nAnnotation = deviation from nominal coverage",
       x = NULL, y = "Empirical coverage") +
  theme_minimal(base_size = 12)

print(p_cov)

# 5b. Mean interval length
p_len <- ggplot(summary_df,
                aes(x = reorder(Model, -Mean_Length), y = Mean_Length)) +
  geom_col(fill = "#55A868", alpha = 0.85) +
  coord_flip() +
  labs(title    = "Mean Interval Length — where models diverge",
       subtitle = "Sharper (shorter) intervals → better-calibrated base model",
       x = NULL, y = "Mean interval length (same units as medv)") +
  theme_minimal(base_size = 12)

print(p_len)

# 5c. Winkler score
p_wink <- ggplot(summary_df,
                 aes(x = reorder(Model, -Winkler), y = Winkler)) +
  geom_col(fill = "#C44E52", alpha = 0.85) +
  coord_flip() +
  labs(title    = "Winkler Score — the composite verdict",
       subtitle = "Lower is better (penalises width and coverage failures)",
       x = NULL, y = "Winkler score") +
  theme_minimal(base_size = 12)

print(p_wink)

# 5d. Coverage vs interval length scatter
p_scatter <- ggplot(summary_df,
                    aes(x = Mean_Length, y = Coverage, label = Model)) +
  geom_point(aes(colour = Winkler), size = 4) +
  geom_text(vjust = -0.8, size = 3, check_overlap = TRUE) +
  geom_hline(yintercept = 1 - alpha_main, linetype = "dashed",
             colour = "firebrick", linewidth = 0.7) +
  scale_colour_gradient(low = "#55A868", high = "#C44E52",
                        name = "Winkler\nscore") +
  labs(title    = "Coverage vs Interval Length",
       subtitle = "Coverage converges; length is where quality shows",
       x = "Mean interval length", y = "Empirical coverage") +
  theme_minimal(base_size = 12)

print(p_scatter)

# 5e. Winkler lollipop coloured by coverage
p_tradeoff <- ggplot(summary_df,
                     aes(x = reorder(Model, Winkler))) +
  geom_segment(aes(xend = reorder(Model, Winkler),
                   y = 0, yend = Winkler),
               colour = "grey70", linewidth = 0.8) +
  geom_point(aes(y = Winkler, colour = Coverage), size = 5) +
  scale_colour_gradient2(
    low      = "#C44E52",
    mid      = "#f7f7f7",
    high     = "#4C72B0",
    midpoint = 1 - alpha_main,
    name     = "Empirical\ncoverage"
  ) +
  coord_flip() +
  labs(title    = "Winkler Score vs Coverage — the lm surprise",
       subtitle = paste0(
         "lm hits exactly 95% coverage; tree ensembles are sharper but under-cover,\n",
         "which the Winkler penalty exposes"),
       x = NULL, y = "Winkler score (lower = better)") +
  theme_minimal(base_size = 12)

print(p_tradeoff)

# --- 6. Calibration residual distributions -----------------------------------

cal_df       <- do.call(rbind, cal_residuals_all)
rownames(cal_df) <- NULL
cal_df$Model <- factor(cal_df$Model, levels = rev(summary_df$Model))
q_hat_df     <- unique(cal_df[, c("Model", "q_hat")])

p_resid <- ggplot(cal_df, aes(x = resid)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30, fill = "#4C72B0", alpha = 0.6) +
  geom_density(colour = "#4C72B0", linewidth = 0.7) +
  geom_vline(data = q_hat_df,
             aes(xintercept = q_hat),
             colour = "firebrick", linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~ Model, scales = "free_y", ncol = 3) +
  labs(
    title    = "Calibration absolute residuals per model",
    subtitle = paste0(
      "Red dashed line = q̂ (conformal quantile = half interval width)\n",
      "lm residuals are compact and well-concentrated → small q̂ relative to its coverage\n",
      "Tree ensembles: tight residuals but right-tail outliers push q̂ enough to under-cover"),
    x = "| y_cal − ŷ_cal |",
    y = "Density"
  ) +
  theme_minimal(base_size = 11) +
  theme(strip.text = element_text(face = "bold", size = 9))

print(p_resid)

# --- 7. Winkler vs alpha sweep -----------------------------------------------
# No refitting: reuses cal_residuals_all and test_preds_all from section 3

alpha_grid   <- c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
model_labels <- names(test_preds_all)

sweep_rows <- vector("list", length(alpha_grid) * length(model_labels))
k <- 1L

for (a in alpha_grid) {
  for (lab in model_labels) {

    scores     <- as.numeric(cal_residuals_all[[lab]]$resid)
    test_preds <- as.numeric(test_preds_all[[lab]])
    y          <- as.numeric(y_test)

    n_cal   <- length(scores)
    q_level <- min(ceiling((n_cal + 1) * (1 - a)) / n_cal, 1)
    q_hat   <- as.numeric(quantile(scores, probs = q_level, type = 1))

    lower   <- test_preds - q_hat
    upper   <- test_preds + q_hat
    covered <- (y >= lower) & (y <= upper)
    width   <- upper - lower
    penalty <- ifelse(!covered,
                      (2 / a) * ifelse(y < lower, lower - y, y - upper),
                      0)

    sweep_rows[[k]] <- data.frame(
      Model    = as.character(lab),
      alpha    = as.numeric(a),
      nominal  = as.numeric(1 - a),
      Coverage = as.numeric(mean(covered)),
      Winkler  = as.numeric(mean(width + penalty)),
      stringsAsFactors = FALSE,
      row.names = NULL
    )
    k <- k + 1L
  }
}

sweep_df <- do.call(rbind.data.frame, sweep_rows)
rownames(sweep_df) <- NULL
sweep_df$Model <- factor(sweep_df$Model, levels = rev(as.character(summary_df$Model)))

# 7a. Winkler score vs nominal coverage level
p7a <- ggplot(sweep_df,
              aes(x = nominal, y = Winkler, colour = Model, group = Model)) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_point(size = 2.5) +
  geom_vline(xintercept = 1 - alpha_main,
             linetype = "dashed", colour = "grey40", linewidth = 0.6) +
  annotate("text",
           x     = 1 - alpha_main + 0.004,
           y     = max(sweep_df$Winkler) * 0.98,
           label = sprintf("main α (%.0f%%)", 100 * (1 - alpha_main)),
           colour = "grey40", hjust = 0, size = 3.2) +
  scale_x_continuous(name   = "Nominal coverage (1 − α)",
                     breaks = 1 - alpha_grid,
                     labels = percent_format(accuracy = 1)) +
  scale_y_continuous(name = "Winkler score (lower = better)") +
  scale_colour_brewer(palette = "Dark2", name = NULL) +
  labs(title    = "Winkler score across coverage targets",
       subtitle = paste0(
         "Tight targets (99%): miss penalty 2/α = 200 → calibration dominates, lm wins\n",
         "Loose targets (70%): penalty shrinks to 6.7 → sharpness dominates, ensembles win")) +
  theme_minimal(base_size = 12) +
  theme(legend.position  = "right",
        legend.key.width = unit(1.5, "lines"))

print(p7a)

# 7b. Empirical vs nominal coverage — conformal guarantee check
p7b <- ggplot(sweep_df,
              aes(x = nominal, y = Coverage, colour = Model, group = Model)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "firebrick", linewidth = 0.8) +
  geom_line(linewidth = 0.9, alpha = 0.85) +
  geom_point(size = 2.5) +
  annotate("text",
           x = 0.715, y = 0.745,
           label  = "perfect calibration",
           colour = "firebrick", angle = 34, size = 3.2) +
  scale_x_continuous(name   = "Nominal coverage (1 − α)",
                     breaks = 1 - alpha_grid,
                     labels = percent_format(accuracy = 1),
                     limits = c(0.68, 1.00)) +
  scale_y_continuous(name   = "Empirical coverage",
                     labels = percent_format(accuracy = 1),
                     limits = c(0.68, 1.00)) +
  scale_colour_brewer(palette = "Dark2", name = NULL) +
  labs(title    = "Empirical vs nominal coverage across α values",
       subtitle = paste0(
         "All models should track or exceed the diagonal — the marginal conformal guarantee\n",
         "Dipping below the line would indicate a violation")) +
  theme_minimal(base_size = 12) +
  theme(legend.position  = "right",
        legend.key.width = unit(1.5, "lines"))

print(p7b)
```

    Train: 253 | Cal: 152 | Test: 101
    
    [1/8] Fitting: glmnet (ridge) ...
    
    [2/8] Fitting: glmnet (lasso) ...
    
    [3/8] Fitting: ranger (RF) ...
    
    [4/8] Fitting: SVM (radial) ...
    
    [5/8] Fitting: LightGBM ...
    
    [6/8] Fitting: caret: lm ...
    
    [7/8] Fitting: caret: knn ...
    
    [8/8] Fitting: caret: rf ...
    
    
    ========== Split Conformal Prediction — BostonHousing ==========
    Nominal coverage: 95%  |  alpha = 0.05
    
              Model Coverage Mean_Length Winkler  q_hat Cov_Dev
           LightGBM    0.901      13.708  37.531  6.854  -0.049
          caret: lm    0.950      21.305  38.969 10.653   0.000
     glmnet (lasso)    0.931      20.519  41.052 10.260  -0.019
        ranger (RF)    0.881      13.430  48.011  6.715  -0.069
       SVM (radial)    0.901      15.894  48.032  7.947  -0.049
          caret: rf    0.881      13.306  48.354  6.653  -0.069
         caret: knn    0.871      16.127  50.213  8.063  -0.079
     glmnet (ridge)    0.921      32.728  55.507 16.364  -0.029
    =================================================================
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_3.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_4.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_5.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_6.png){:class="img-responsive"}
    


    Warning message:
    “[1m[22mRemoved 5 rows containing missing values or values outside the scale range
    (`geom_line()`).”
    Warning message:
    “[1m[22mRemoved 5 rows containing missing values or values outside the scale range
    (`geom_point()`).”



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_8.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_4_9.png){:class="img-responsive"}
    


# 2 - With multiple seeds (more stable results)

The previous section used only one seed, which could make the result fragile. This section uses 10 different data splits to measure coverages and Winkler scores.


```R
# =============================================================================
# Split Conformal Prediction — multi-seed stability analysis
# Wraps the single-seed loop in an outer seed loop.
# Reports mean ± sd of Coverage and Winkler per model across seeds,
# so that single-split variance doesn't drive the ranking.
# =============================================================================

# --- 0. Load packages --------------------------------------------------------

library(mlS3)
library(mlbench)
library(ggplot2)
library(scales)

# --- 1. Data -----------------------------------------------------------------

data(BostonHousing)
df      <- BostonHousing
df$chas <- as.numeric(df$chas)   # coerce factor so all models see numeric input

n     <- nrow(df)
y_all <- df$medv
X_all <- as.data.frame(df[, names(df) != "medv"])

# --- 2. Model zoo ------------------------------------------------------------

models <- list(

  list(label    = "glmnet (ridge)",
       fit_fn   = wrap_glmnet,
       fit_args = list(alpha = 0)),

  list(label    = "glmnet (lasso)",
       fit_fn   = wrap_glmnet,
       fit_args = list(alpha = 1)),

  list(label    = "ranger (RF)",
       fit_fn   = wrap_ranger,
       fit_args = list(num.trees = 500L)),

  list(label    = "SVM (radial)",
       fit_fn   = wrap_svm,
       fit_args = list(kernel = "radial")),

  list(label    = "LightGBM",
       fit_fn   = wrap_lightgbm,
       fit_args = list(
         params  = list(objective = "regression", verbose = -1),
         nrounds = 100)),

  list(label    = "caret: lm",
       fit_fn   = wrap_caret,
       fit_args = list(method = "lm")),

  list(label    = "caret: knn",
       fit_fn   = wrap_caret,
       fit_args = list(method = "kknn")),

  list(label    = "caret: rf",
       fit_fn   = wrap_caret,
       fit_args = list(method = "rf", mtry = 3L))
)

# --- 3. Multi-seed conformal loop --------------------------------------------
# Each seed produces an independent 50/30/20 split; we refit every model and
# record Coverage and Winkler. Aggregating over seeds averages out the ~±5pp
# sampling noise that dominates single-split comparisons (n_test ≈ 100).

alpha_main <- 0.05
seeds      <- c(42, 123, 2024, 314, 999, 7, 2025, 1234, 77, 42000)

# all_runs: one row per (seed × model)
all_runs <- vector("list", length(seeds) * length(models))
k        <- 1L

for (s in seeds) {

  set.seed(s)
  idx_test  <- sample(n, round(0.20 * n))
  rest      <- setdiff(seq_len(n), idx_test)
  idx_cal   <- sample(rest, round(0.30 * n))
  idx_train <- setdiff(rest, idx_cal)

  X_train <- X_all[idx_train, ]; y_train <- y_all[idx_train]
  X_cal   <- X_all[idx_cal,   ]; y_cal   <- y_all[idx_cal]
  X_test  <- X_all[idx_test,  ]; y_test  <- y_all[idx_test]

  cat(sprintf("\nSeed %d — Train: %d | Cal: %d | Test: %d\n",
              s, length(idx_train), length(idx_cal), length(idx_test)))

  for (i in seq_along(models)) {
    m <- models[[i]]

    res <- tryCatch({

      mod       <- do.call(m$fit_fn, c(list(X_train, y_train), m$fit_args))
      cal_preds <- predict(mod, newx = X_cal)
      scores    <- as.numeric(abs(cal_preds - y_cal))

      # finite-sample corrected conformal quantile (Vovk et al. 2005)
      n_cal   <- length(scores)
      q_level <- min(ceiling((n_cal + 1) * (1 - alpha_main)) / n_cal, 1)
      q_hat   <- as.numeric(quantile(scores, probs = q_level, type = 1))

      test_preds <- as.numeric(predict(mod, newx = X_test))
      lower      <- test_preds - q_hat
      upper      <- test_preds + q_hat
      covered    <- (y_test >= lower) & (y_test <= upper)
      width      <- upper - lower

      penalty <- ifelse(!covered,
                        (2 / alpha_main) * ifelse(y_test < lower,
                                                  lower - y_test,
                                                  y_test - upper),
                        0)

      data.frame(
        Seed        = s,
        Model       = m$label,
        Coverage    = mean(covered),
        Mean_Length = mean(width),
        Winkler     = mean(width + penalty),
        q_hat       = q_hat,
        stringsAsFactors = FALSE
      )
    },
    error = function(e) {
      cat(sprintf("  ERROR [%s]: %s\n", m$label, conditionMessage(e)))
      NULL
    })

    if (!is.null(res)) { all_runs[[k]] <- res }
    k <- k + 1L
  }
}

runs_df <- do.call(rbind, Filter(Negate(is.null), all_runs))
rownames(runs_df) <- NULL

# --- 4. Aggregate: mean ± sd per model ---------------------------------------

agg    <- aggregate(cbind(Coverage, Mean_Length, Winkler, q_hat) ~ Model,
                    data = runs_df, FUN = mean)
agg_sd <- aggregate(cbind(Coverage, Mean_Length, Winkler, q_hat) ~ Model,
                    data = runs_df, FUN = sd)

names(agg_sd)[-1] <- paste0(names(agg_sd)[-1], "_sd")
summary_df <- merge(agg, agg_sd, by = "Model")

summary_df$Cov_Dev <- round(summary_df$Coverage - (1 - alpha_main), 3)

for (col in names(summary_df)[-1]) summary_df[[col]] <- round(summary_df[[col]], 3)

summary_df <- summary_df[order(summary_df$Winkler), ]
rownames(summary_df) <- NULL

cat(sprintf("\n\n===== Split Conformal — BostonHousing | %d seeds =====\n", length(seeds)))
cat(sprintf("Nominal coverage: 95%%  |  alpha = %.2f\n\n", alpha_main))
print(summary_df[, c("Model", "Coverage", "Coverage_sd",
                      "Mean_Length", "Mean_Length_sd",
                      "Winkler", "Winkler_sd", "Cov_Dev")],
      row.names = FALSE)
cat("=======================================================\n\n")

# --- 5. Plots ----------------------------------------------------------------

# 5a. Mean coverage ± 1 sd
# Models whose bar reliably clips the 95% line meet the guarantee across splits;
# those consistently below it have a structural under-coverage problem.
p_cov <- ggplot(summary_df,
                aes(x = reorder(Model, Coverage), y = Coverage)) +
  geom_col(fill = "#4C72B0", alpha = 0.85) +
  geom_errorbar(aes(ymin = Coverage - Coverage_sd,
                    ymax = Coverage + Coverage_sd),
                width = 0.3, colour = "grey30") +
  geom_hline(yintercept = 1 - alpha_main, linetype = "dashed",
             colour = "firebrick", linewidth = 0.8) +
  annotate("text", x = 0.6, y = 1 - alpha_main + 0.005,
           label = "95% target", colour = "firebrick", hjust = 0, size = 3.5) +
  coord_flip(ylim = c(0.75, 1.10)) +
  labs(title    = "Mean empirical coverage across seeds",
       subtitle = sprintf("Error bars = ±1 sd over %d seeds", length(seeds)),
       x = NULL, y = "Empirical coverage") +
  theme_minimal(base_size = 12)

print(p_cov)

# 5b. Mean Winkler ± 1 sd
# Overlapping error bars → ranking not stable across splits.
# Non-overlapping → difference is robust to split variance.
p_wink <- ggplot(summary_df,
                 aes(x = reorder(Model, -Winkler), y = Winkler)) +
  geom_col(fill = "#C44E52", alpha = 0.85) +
  geom_errorbar(aes(ymin = Winkler - Winkler_sd,
                    ymax = Winkler + Winkler_sd),
                width = 0.3, colour = "grey30") +
  coord_flip() +
  labs(title    = "Mean Winkler score across seeds",
       subtitle = sprintf("Error bars = ±1 sd over %d seeds | lower = better", length(seeds)),
       x = NULL, y = "Winkler score") +
  theme_minimal(base_size = 12)

print(p_wink)

# 5c. Per-seed coverage jitter
# Full distribution across seeds; reveals systematic under-coverage vs. high variance.
p_jit_cov <- ggplot(runs_df,
                    aes(x = reorder(Model, Coverage, FUN = mean),
                        y = Coverage)) +
  geom_jitter(width = 0.15, alpha = 0.6, colour = "#4C72B0", size = 2) +
  stat_summary(fun = mean, geom = "point", shape = 18,
               size = 4, colour = "black") +
  geom_hline(yintercept = 1 - alpha_main, linetype = "dashed",
             colour = "firebrick", linewidth = 0.8) +
  coord_flip() +
  labs(title    = "Per-seed coverage distribution",
       subtitle = "Diamond = mean; jitter = individual seeds",
       x = NULL, y = "Empirical coverage") +
  theme_minimal(base_size = 12)

print(p_jit_cov)

# 5d. Per-seed Winkler jitter
p_jit_wink <- ggplot(runs_df,
                     aes(x = reorder(Model, Winkler, FUN = mean),
                         y = Winkler)) +
  geom_jitter(width = 0.15, alpha = 0.6, colour = "#C44E52", size = 2) +
  stat_summary(fun = mean, geom = "point", shape = 18,
               size = 4, colour = "black") +
  coord_flip() +
  labs(title    = "Per-seed Winkler score distribution",
       subtitle = "Diamond = mean; jitter = individual seeds | lower = better",
       x = NULL, y = "Winkler score") +
  theme_minimal(base_size = 12)

print(p_jit_wink)
```

    
    Seed 42 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 123 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 2024 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 314 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 999 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 7 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 2025 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 1234 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 77 — Train: 253 | Cal: 152 | Test: 101
    
    Seed 42000 — Train: 253 | Cal: 152 | Test: 101
    
    
    ===== Split Conformal — BostonHousing | 10 seeds =====
    Nominal coverage: 95%  |  alpha = 0.05
    
              Model Coverage Coverage_sd Mean_Length Mean_Length_sd Winkler
           LightGBM    0.944       0.026      16.195          2.047  27.435
          caret: rf    0.940       0.029      15.421          2.723  28.460
        ranger (RF)    0.941       0.032      15.559          2.728  28.672
          caret: lm    0.948       0.033      20.258          3.879  32.668
       SVM (radial)    0.946       0.042      17.808          3.795  34.129
     glmnet (lasso)    0.946       0.030      21.361          3.950  34.222
         caret: knn    0.947       0.038      20.307          4.157  36.279
     glmnet (ridge)    0.948       0.017      35.807          6.309  49.028
     Winkler_sd Cov_Dev
          5.342  -0.006
          9.722  -0.010
          9.922  -0.009
          7.725  -0.002
         11.908  -0.004
          8.640  -0.004
          9.474  -0.003
          5.271  -0.002
    =======================================================
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_7_1.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_7_2.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_7_3.png){:class="img-responsive"}
    



    
![image-title-here]({{base}}/images/2026-06-07/2026-06-07-conformalization-helps-weak-models_7_4.png){:class="img-responsive"}
    

