---
layout: post
title: "R package development workflow (assuming you're using macOS or Linux)"
description: "R package development workflow (assuming you're using macOS or Linux)"
date: 2024-08-27
categories: R
comments: true
---

I'm using VS Code on macOS for Python, R, Javascript (etc.) 
development. I needed a quick/automated workflow for my R 
package development **at the command line**, so I created 
this Makefile (if necessary [install Python](https://www.python.org/downloads/)): 

[https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb](https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb)

All you need to do is: **store the Makefile at the root of the package folder**. 
Type `make` or `make help` at the command line to see all the commands available. You can start with `make initialize`, that will install `devtools` and `rmarkdown`, if they're not available yet. Here's what you can do so far (as of 2024-08-27): 

- `make clean`: remove all R artifacts
- `make start`: start or restart R session
- `make setwd`: set working directory to current directory
- `make docs`: generate package documentation
- `make check`: check package for potential errors
- `make install`: install package
- `make initialize`: initialize environment (install packages)
- `make load`: load all (when developing the package)
- `make render`: run R markdown file in `/vignettes` (you'll be prompted to give the file name, without extension)

Of course, **work in progress** (no package creation, or running tests, etc.). And also, nothing malicious about the script (ask an LLM to break it down for you if necessary :) ). Feel free to **comment the [GitHub Gist](https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb)** if you have a suggestion.


<script src="https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb.js"></script>

