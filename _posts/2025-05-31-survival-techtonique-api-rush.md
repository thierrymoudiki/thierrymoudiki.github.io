---
layout: post
title: "Which patient is going to survive longer? Another guide to using techtonique dot net's API (with R + Python + the command line) for survival analysis"
description: "How to make API calls to techtonique.net for survival analysis and plot the results with rush"
date: 2025-05-31
categories: [R, Python, Techtonique]
comments: true
---

   
In today's post, we'll see how to use [_rush_](https://jeroenjanssens.com/dsatcl/chapter-7-exploring-data) and the probabilistic survival analysis API provided by [techtonique.net](https://www.techtonique.net) (along with R and Python) to plot survival curves results. Note the that, [the web app](https://www.techtonique.net) also contains a page for plotting these curves,  in 1 click. 

First, you'd need to install _rush_. Here is how I did it: 

```bash
cd /Users/t/Documents/Python_Packages
git clone https://github.com/jeroenjanssens/rush.git 
export PATH="/Users/t/Documents/Python_Packages/rush/exec:$PATH"
source ~/.zshrc # or source ~/.bashrc
rush --help # check if rush is installed
```

Now, download and save the following script in your current directory (note that there's nothing malicious in it). Replace AUTH_TOKEN below by a token that can be found at [techtonique.net/token](https://www.techtonique.net/token): 

<script src="https://gist.github.com/thierrymoudiki/2cba38ca1a6155a3ff4ecef493d86648.js"></script>

Then, at the command line, run: 

```bash 
./2025-05-31-survival.sh
```

The result plot can be found in your current directory as a PNG file.

![image-title-here]({{base}}/images/2025-05-31/2025-05-31-image1.png){:class="img-responsive"}
