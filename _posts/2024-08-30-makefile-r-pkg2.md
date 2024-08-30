---
layout: post
title: "Quick/automated R package development workflow (assuming you're using macOS or Linux) Part2"
description: "Quick/automated R package development workflow (assuming you're using macOS or Linux) Part2"
date: 2024-08-30
categories: R
comments: true
---

**Disclaimer**: I have no affiliation with the [VS Code](https://code.visualstudio.com/) team, just a user who likes the product.

Earlier this week in [#155](https://thierrymoudiki.github.io/blog/2024/08/27/r/makefile-r-pkg), I posted about a quick/automated workflow for R package development **at the command line**. Using this workflow along with VS Code Editor -- after experimenting it myself -- is **a breeze**... Interested in using VS Code for your R package development? Read this resource: [https://code.visualstudio.com/docs/languages/r](https://code.visualstudio.com/docs/languages/r).

Here's the updated Makefile (as of 2024-08-30): [https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb](https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb)

All you need to do is **store it at the root of your package folder**. Type `make` or `make help` at the command line to see all the commands available. You can start with `make initialize`, that will install `devtools`, `usethis` and `rmarkdown`, if they're not available yet. Here's what you can do so far (as of 2024-08-30): 

- `buildsite`: create a website for your package
- `check`: check package
- `clean`: remove all build, and artifacts
- `coverage`: get test coverage
- `create`: create a new package in current directory
- `docs`: generate docs
- `getwd`: get current directory
- `install`: install package
- `initialize`: initialize: install packages devtools, usethis, pkgdown and rmarkdown
- `help`: print menu with all options
- `load`: load all (when developing the package)
- `render`: run R markdown file in /vignettes, open rendered HTML file in the browser
- `setwd`: set working directory to current directory
- `start`: start or restart R session
- `test`: runs the the package tests
- `usegit`: initialize Git repo and initial commit

You can even **chain operations by doing**: 

```bash
make check&&make install
```

Feel free to fork or **comment the [GitHub Gist](https://gist.github.com/thierrymoudiki/3bd7cfa099aef0c64eb5f91138d8cedb)** if you have a suggestion like [Prof. Rob J Hyndman](https://robjhyndman.com/) who found it useful (like me). The more feedback, the better the experience for everyone ;) 

![idontowntherightsofthispic]({{base}}/images/2024-08-30/2024-08-30-image1.png){:class="img-responsive"} 

