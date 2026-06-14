---
layout: post
title: "No-Code Machine Learning in Excel with the Techtonique API"
description: "Using the Techtonique API in Excel without writing any code"
date: 2026-06-14
categories: Techtonique
comments: true
---

[Many newcomers](https://thierrymoudiki.github.io/blog/2026/05/31/r/python/techtonique/techtonique-dot-net-is-back) in the [Techtonique API](https://www.techtonique.net/) these days (10x more subscribers in one month). Hope you are enjoying it. 

The occasion for me to write (again) a short post about how to use the Techtonique API in Excel, without writing any code. A lot of people are indeed familiar with Excel, but not everyone is comfortable writing Python or R code (or [else](https://curlconverter.com/)). 

Under the hood, the [xlwings lite](https://lite.xlwings.org/installation) Excel add-in (that you can install from the link) exposes [Techtonique API's](https://www.techtonique.net/) forecasting, regression, classification, reserving, and survival analysis functions as native worksheet functions -- type a formula, select a data range, get results. That's it.

---

## Setup

**Step 1 — Get a token from Techtonique API**

Go to [https://www.techtonique.net/token](https://www.techtonique.net/token), sign up or log in, and copy your token. Valid for one hour. Then you can renew it as many times as you want.

**Step 2 — Store the token in Excel**

In xlwings lite, open the **dropdown menu** → **Environment variables** (see a detailed example file at [https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx](https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx)), and add:

| Name | Value |
|---|---|
| `TECHTONIQUE_TOKEN` | *(paste your token here)* |

**Step 3 — Restart**

Click **Restart** in the dropdown menu. The token is not active until after a restart. You only need to do this once, for one hour. 

---

## Using the functions

In any cell, type `=TECHTO_` — Excel autocomplete lists everything available. The main functions are:

| Function | Task |
|---|---|
| `=TECHTO_FORECAST(...)` | Probabilistic time series forecasting |
| `=TECHTO_MLREGRESSION(...)` | Regression with prediction intervals |
| `=TECHTO_MLCLASSIFICATION(...)` | Classification |
| `=TECHTO_MLRESERVING(...)` | Claims reserving (loss triangles) |
| `=TECHTO_SURVIVAL(...)` | Survival analysis |

Each function takes a **data range** as its first argument — just select your data in the sheet — followed by optional parameters such as the model, forecast horizon, or number of hidden features. Results are returned as a table directly in the spreadsheet.

If something goes wrong, the error message from the API is displayed in the cell, which makes debugging straightforward without leaving Excel.

---

## Help

- **In Excel**: click the function name in the formula bar to open the built-in help for that function.
- **Online reference**: [https://docs.techtonique.net/techtonique_api_py/techtonique_apis.html](https://docs.techtonique.net/techtonique_api_py/techtonique_apis.html)
- **Example workbook** with pre-filled formulas: [https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx](https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx)


![image-title-here]({{base}}/images/2025-07-07/2025-07-07-image1.gif){:class="img-responsive"}
