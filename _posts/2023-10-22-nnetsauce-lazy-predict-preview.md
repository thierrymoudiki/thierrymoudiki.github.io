---
layout: post
title: "AutoML in nnetsauce (randomized and quasi-randomized nnetworks)"
description: "nnetsauce implements randomized and quasi-randomized 'neural' networks for supervised learning and time series forecasting"
date: 2023-10-22
categories: [Python, QuasiRandomizedNN]
comments: true
---

**Content:**

<ol>
  <li> Intro </li>
  <li> Installing nnetsauce for Python </li>
  <li> Classification </li>
  <li> Regression </li>
</ol>

# Intro

A new version of [nnetsauce](https://github.com/Techtonique/nnetsauce), v0.12.0, is available on PyPI and for conda. It's been mostly tested on Linux and macOS platforms. For **Windows users**: you can use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about) in case it doesn't work directly on your computer.

{% jupyter_notebook "thierrymoudiki_221023_LazyPredict.ipynb" %}