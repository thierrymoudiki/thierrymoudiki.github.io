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


```bash
.PHONY: clean docs start setwd check install load render initialize
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url

# The input is expected to be the full HTML filename
filename = sys.argv[1]
filepath = os.path.abspath(os.path.join("./vignettes/", filename))
webbrowser.open("file://" + pathname2url(filepath))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ## remove all build, test, coverage and Python artifacts
	rm -f .Rbuildignore
	rm -f .Rhistory
	rm -f *.RData
	rm -f *.Rproj
	rm -rf .Rproj.user

start: ## start or restart R session
	Rscript -e "system('R')"

setwd: ## set working directory
	Rscript -e "setwd(getwd())"

docs: clean setwd ## generate docs		
	Rscript -e "devtools::document('.')"

check: clean setwd ## check package
	Rscript -e "devtools::check('.')"

install: clean setwd ## install package
	Rscript -e "devtools::install('.')"	

initialize: setwd ## initialize environment (install packages)
	Rscript -e "utils::install.packages(c('devtools', 'rmarkdown'), repos='https://cloud.r-project.org')"		

load: clean setwd ## load all (when developing the package)
	Rscript -e "devtools::load_all('.')"

render: ## run markdown file in /vignettes
	@read -p "Enter the name of the Rmd file (without extension): " filename; \
	Rscript -e "rmarkdown::render(paste0('./vignettes/', '$$filename', '.Rmd'))"; \
	python3 -c "$$BROWSER_PYSCRIPT" "$$filename.html"
```


