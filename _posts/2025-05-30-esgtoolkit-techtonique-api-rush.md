---
layout: post
title: "A Guide to Using techtonique.net's API and rush for simulating and plotting Stochastic Scenarios"
description: "How to make API calls to techtonique.net for stochastic simulation using diffusion models, and plot the results with rush"
date: 2025-05-30
categories: [R, Python, Techtonique]
comments: true
---

Yesterday's blog post demonstrated how to use the (work in progress) stochastic simulation API provided by [techtonique.net](https://www.techtonique.net), to generate scenarios. We explored how to simulate paths using the popular:

1. Geometric Brownian Motion (GBM)
2. Cox-Ingersoll-Ross (CIR) process
3. Ornstein-Uhlenbeck (OU) process
4. Gaussian _Shocks_ scenarios
   
In today's post, we'll see how to use [_rush_](https://jeroenjanssens.com/dsatcl/chapter-7-exploring-data) and the stochastic simulation API provided by [techtonique.net](https://www.techtonique.net) to plot simulation results.

First, you need to install rush. Here is how I did it: 

```bash
cd /Users/t/Documents/Python_Packages
git clone https://github.com/jeroenjanssens/rush.git 
export PATH="/Users/t/Documents/Python_Packages/rush/exec:$PATH"
source ~/.zshrc # or source ~/.bashrc
rush --help # check if rush is installed
```

Now, download and save the following script in your current directory (note that there's nothing malicious in it). Replace AUTH_TOKEN by a token that can be found at [techtonique.net/token](https://www.techtonique.net/token): 

<script src="https://gist.github.com/thierrymoudiki/026834be69dfbc034ba05ee27e338ddf.js"></script>

Then, at the command line, run: 

```bash 
./fetch_and_plot_rush_techtonique.sh
```

The result plot can be found in your current directory as a PNG file named `plot.png`.

![image-title-here]({{base}}/images/2025-05-30/2025-05-30-image1.png){:class="img-responsive"}
