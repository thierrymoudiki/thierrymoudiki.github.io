---
layout: post
title: "Techtonique dot net, the Machine Learning web API, is back online (but more like a passion project for now)"
description: "Techtonique dot net is back, but more like a passion project, with an API for machine learning tasks (classification, regression, survival analysis, reserving, forecasting etc.). Examples in R, Python are provided in the blog post."
date: 2026-05-31
categories: [R, Python, Techtonique]
comments: true
---


[https://www.techtonique.net](https://www.techtonique.net) contains tools for **Exploratory Data Analysis** (EDA), **editors** for R and Python code, tools for **data visualization**, **no-code web interfaces** for various data science tasks, **a language-agnostic API** for machine learning tasks (classification, regression, survival analysis, reserving, forecasting etc.).

I recently noticed a **spike** in sign-ups (not sure how they noticed it was up again, without official announcement...), and I thought it was a good opportunity to share that **[techtonique.net](https://www.techtonique.net) is back**. More like a passion project, or part of a portfolio (for now). The API is available and evolving. You can use it for free (with rate limiting), but please note that it is slightly slower than it used to be (indeed, I used to have Azure Credits thanks to Microsoft for Startups  => a faster server). It still serves the responses, in a reasonable time though, and I will try to optimize it as much as possible.

It is worth mentioning that I removed the stochastic simulation API for now, as it required (more technical) to run R in Python, through a Docker container.

Here are some examples of how to use the API for machine learning tasks in R and Python (there are also no-code interfaces for these tasks in the website, and the API is language-agnostic, so you can use it with any programming language that can make HTTP requests):

The starting point is to signup/login (if you're facing issues, contact support@techtonique.net) and get a token from: [https://www.techtonique.net/token](https://www.techtonique.net/token). Then, you can use the token in API requests for machine learning tasks. Each response is a prediction, as envisaged by the chosen model, and based on the input data.

# 1 - R example:

```R
#!/usr/bin/env Rscript

# Install httr if needed: install.packages("httr")
# All you need is a token from www.techtonique.net/token, then run the script
library(httr)

BASE_URL   <- "https://www.techtonique.net"
GITHUB_RAW <- "https://raw.githubusercontent.com/Techtonique/datasets/main"

DATASETS <- list(
  univariate     = paste0(GITHUB_RAW, "/time_series/univariate/a10.csv"),
  multivariate   = paste0(GITHUB_RAW, "/time_series/multivariate/ice_cream_vs_heater.csv"),
  classification = paste0(GITHUB_RAW, "/tabular/classification/breast_cancer_dataset2.csv"),
  regression     = paste0(GITHUB_RAW, "/tabular/regression/boston_dataset2.csv"),
  raa            = paste0(GITHUB_RAW, "/tabular/triangle/raa.csv"),
  abc            = paste0(GITHUB_RAW, "/tabular/triangle/abc.csv"),
  km             = paste0(GITHUB_RAW, "/tabular/survival/kidney.csv"),
  ridge_survival = paste0(GITHUB_RAW, "/tabular/survival/gbsg2_2.csv")
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

get_token <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  idx  <- which(args == "--token")
  if (length(idx) > 0 && idx < length(args)) return(args[idx + 1])
  cat("Please enter your JWT token: ")
  readLines(con = "stdin", n = 1)
}

# Download a GitHub dataset to a temp file and return its path.
# Using a real file is the only reliable way to do multipart uploads in httr.
fetch_dataset <- function(key) {
  url      <- DATASETS[[key]]
  filename <- basename(url)
  cat(sprintf("  Fetching %s from GitHub... ", filename))
  tmp <- tempfile(fileext = ".csv")
  resp <- GET(url, write_disk(tmp, overwrite = TRUE))
  stop_for_status(resp)
  cat("OK\n")
  tmp   # return the temp file path
}

make_request <- function(endpoint, token, dataset_key = NULL, params = list()) {
  url     <- paste0(BASE_URL, endpoint)
  headers <- add_headers(Authorization = paste("Bearer", token))

  if (!is.null(dataset_key)) {
    tmp_path <- fetch_dataset(dataset_key)
    on.exit(unlink(tmp_path), add = TRUE)   # clean up temp file when done
    body <- list(file = upload_file(tmp_path, type = "text/csv"))
    resp <- POST(url, headers, query = params, body = body, encode = "multipart")
  } else {
    resp <- GET(url, headers, query = params)
  }

  if (http_error(resp)) {
    cat(sprintf("  ERROR %s: %s\n", status_code(resp),
                content(resp, as = "text", encoding = "UTF-8")))
    return(NULL)
  }
  content(resp, as = "parsed", type = "application/json")
}

print_response <- function(resp) {
  if (is.null(resp)) return(invisible(NULL))
  for (key in names(resp)) {
    val <- resp[[key]]
    if (is.list(val) || length(val) > 6) {
      cat(sprintf("  %s: [%s ...]\n", key, paste(head(unlist(val), 6), collapse = ", ")))
    } else {
      cat(sprintf("  %s: %s\n", key, paste(val, collapse = ", ")))
    }
  }
}

# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

test_forecasting <- function(token) {
  cat("\n=== Testing Forecasting Endpoints ===\n")

  cat("\nTesting univariate forecasting...\n")
  resp <- make_request("/forecasting", token,
    dataset_key = "univariate",
    params = list(base_model = "RidgeCV", n_hidden_features = 5,
                  lags = 25, type_pi = "kde", replications = 4, h = 3))
  print_response(resp)

  cat("\nTesting multivariate forecasting...\n")
  resp <- make_request("/forecasting", token,
    dataset_key = "multivariate",
    params = list(base_model = "RidgeCV", n_hidden_features = 5, lags = 25, h = 3))
  print_response(resp)
}

test_ml <- function(token) {
  cat("\n=== Testing Machine Learning Endpoints ===\n")

  cat("\nTesting classification...\n")
  resp <- make_request("/mlclassification", token,
    dataset_key = "classification",
    params = list(base_model = "RandomForestClassifier",
                  n_hidden_features = 5, predict_proba = TRUE))
  print_response(resp)

  cat("\nTesting regression...\n")
  resp <- make_request("/mlregression", token,
    dataset_key = "regression",
    params = list(base_model = "RidgeCV", n_hidden_features = 5, return_pi = TRUE))
  print_response(resp)
}

test_reserving <- function(token) {
  cat("\n=== Testing Reserving Endpoints ===\n")

  cat("\nTesting RidgeCV...\n")
  resp <- make_request("/mlreserving", token,
    dataset_key = "raa",
    params = list(method = "RidgeCV"))
  print_response(resp)

  cat("\nTesting LassoCV...\n")
  resp <- make_request("/mlreserving", token,
    dataset_key = "abc",
    params = list(method = "LassoCV"))
  print_response(resp)
}

test_survival <- function(token) {
  cat("\n=== Testing Survival Analysis Endpoints ===\n")

  cat("\nTesting Kaplan-Meier...\n")
  resp <- make_request("/survivalcurve", token,
    dataset_key = "km",
    params = list(method = "km"))
  print_response(resp)

  cat("\nTesting Ridge survival...\n")
  resp <- make_request("/survivalcurve", token,
    dataset_key = "ridge_survival",
    params = list(method = "RidgeCV", patient_id = 0))
  print_response(resp)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  token <- get_token()
  test_forecasting(token)
  test_ml(token)
  test_reserving(token)
  test_survival(token)
}

main()
```

# 2 - Python example:

```python
import argparse
import requests
import os
import io
from typing import Dict, Any

# All you need is a token from www.techtonique.net/token, then run the script

# Base URL for API endpoints
base_url = "https://www.techtonique.net"

# Base URL for raw GitHub content
GITHUB_RAW = "https://raw.githubusercontent.com/Techtonique/datasets/main"

# Dataset paths within the GitHub repo
DATASETS = {
    "univariate":       f"{GITHUB_RAW}/time_series/univariate/a10.csv",
    "multivariate":     f"{GITHUB_RAW}/time_series/multivariate/ice_cream_vs_heater.csv",
    "classification":   f"{GITHUB_RAW}/tabular/classification/breast_cancer_dataset2.csv",
    "regression":       f"{GITHUB_RAW}/tabular/regression/boston_dataset2.csv",
    "raa":              f"{GITHUB_RAW}/tabular/triangle/raa.csv",
    "abc":              f"{GITHUB_RAW}/tabular/triangle/abc.csv",
    "km":               f"{GITHUB_RAW}/tabular/survival/kidney.csv",
    "ridge_survival":   f"{GITHUB_RAW}/tabular/survival/gbsg2_2.csv",
}


def get_token() -> str:
    """Get token from command line argument or prompt"""
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--token', help='JWT token for authentication')
    args = parser.parse_args()

    if args.token:
        return args.token

    return input("Please enter your JWT token: ")


def fetch_dataset(dataset_key: str) -> tuple[io.BytesIO, str]:
    """
    Download a dataset from GitHub and return it as an in-memory bytes buffer
    together with a filename, ready to be passed to requests as a file upload.
    """
    url = DATASETS[dataset_key]
    filename = url.split("/")[-1]
    print(f"  Fetching {filename} from GitHub...", end=" ")
    response = requests.get(url)
    response.raise_for_status()
    print("OK")
    return io.BytesIO(response.content), filename


def make_request(
    url: str,
    token: str,
    method: str = "POST",
    file_tuple: tuple = None,   # (BytesIO, filename)
    params: Dict = None,
) -> Dict[str, Any]:
    """Make an API request and return the response."""
    headers = {"Authorization": f"Bearer {token}"}

    try:
        if method == "POST":
            files = None
            if file_tuple:
                buf, filename = file_tuple
                files = {"file": (filename, buf, "text/csv")}
            response = requests.post(url, headers=headers, files=files, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}:")
        print(f"Status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        print(f"Response: {e.response.text if hasattr(e, 'response') else str(e)}")
        return None


def test_forecasting(token: str):
    print("\n=== Testing Forecasting Endpoints ===")

    print("\nTesting univariate forecasting...")
    response = make_request(
        f"{base_url}/forecasting", token,
        file_tuple=fetch_dataset("univariate"),
        params={"base_model": "RidgeCV", "n_hidden_features": 5,
                "lags": 25, "type_pi": "kde", "replications": 4, "h": 3},
    )
    if response:
        print("Response:", response)

    print("\nTesting multivariate forecasting...")
    response = make_request(
        f"{base_url}/forecasting", token,
        file_tuple=fetch_dataset("multivariate"),
        params={"base_model": "RidgeCV", "n_hidden_features": 5, "lags": 25, "h": 3},
    )
    if response:
        print("Response:", response)


def test_ml(token: str):
    print("\n=== Testing Machine Learning Endpoints ===")

    print("\nTesting classification...")
    response = make_request(
        f"{base_url}/mlclassification", token,
        file_tuple=fetch_dataset("classification"),
        params={"base_model": "RandomForestClassifier",
                "n_hidden_features": 5, "predict_proba": True},
    )
    if response:
        print("Response:", response)

    print("\nTesting regression...")
    response = make_request(
        f"{base_url}/mlregression", token,
        file_tuple=fetch_dataset("regression"),
        params={"base_model": "RidgeCV", "n_hidden_features": 5, "return_pi": True},
    )
    if response:
        print("Response:", response)


def test_reserving(token: str):
    print("\n=== Testing Reserving Endpoints ===")

    print("\nTesting RidgeCV...")
    response = make_request(
        f"{base_url}/mlreserving", token,
        file_tuple=fetch_dataset("raa"),
        params={"method": "RidgeCV"},
    )
    if response:
        print("Response:", response)

    print("\nTesting LassoCV...")
    response = make_request(
        f"{base_url}/mlreserving", token,
        file_tuple=fetch_dataset("abc"),
        params={"method": "LassoCV"},
    )
    if response:
        print("Response:", response)


def test_survival(token: str):
    print("\n=== Testing Survival Analysis Endpoints ===")

    print("\nTesting Kaplan-Meier...")
    response = make_request(
        f"{base_url}/survivalcurve", token,
        file_tuple=fetch_dataset("km"),
        params={"method": "km"},
    )
    if response:
        print("Response:", response)

    print("\nTesting Ridge survival...")
    response = make_request(
        f"{base_url}/survivalcurve", token,
        file_tuple=fetch_dataset("ridge_survival"),
        params={"method": "RidgeCV", "patient_id": 0},
    )
    if response:
        print("Response:", response)


def main():
    token = get_token()
    test_forecasting(token)
    test_ml(token)
    test_reserving(token)
    test_survival(token)


if __name__ == "__main__":
    main()
```

![image-title-here]({{base}}/images/2025-06-09/2025-06-09-image1.gif){:class="img-responsive"}    

PS: Do not use models ending with `CV` for time series forecasting, as they are not designed for that (they are for tabular data, and use a cross-validation which is not suitable for time series). The API will still run the request, but the results may not be good. For time series forecasting, it is better to use models like `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor` etc.