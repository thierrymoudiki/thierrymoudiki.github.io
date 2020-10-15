---
layout: post
title: "Submitting R package to CRAN"
description: Submitting R package to CRAN.
date: 2020-10-16
categories: R
---

_Disclaimer:_ I have no affiliation with Microsoft Corp. or Revolution Analytics.


For the n-th time in x years, submitting an R package to CRAN ended up 
like comedy. This time for **one anecdotal note** (a kind of warning), 
whereas the previous accepted version of [ESGtoolkit](https://techtonique.github.io/ESGtoolkit/) has had, 
for 5 years, 12 warnings and notes combined. No error, nothing's broken. And I'm not even comparing 
it to some other packages' results.


![package-results-yeah]({{base}}/images/2020-10-16/2020-10-16-image1.png){:class="img-responsive"}


Literally no online repo (PyPi, conda forge, submit a Julia package, etc.) works 
this way these days, with a censor sending emails back and forth to a student receiving an examination. The validation process is automated (sure, you can sometimes contact someone). It's not as if all of this was about controlling the calculation of EBITDA. **It's just code**. And perfection is unattainable. One. Warning. Man. Sighs. 

It seems to me like, under the hood and 
after years and years of observation and curious, repeated subliminal (and not so  subliminal) threats, it isn't just about code on CRAN, and more about **a lot of obscure politics and private interests**. Even if it was about code, _censor_, why not submitting a pull request/issue to my GitHub repo directly as everyone does these days, instead of continuously shooting shots for nothing? It also happens that, when you submit something there, your package is _kind of_ confiscated and you can **never (ever)** take it back. So definitely, and again, [think twice (or more)](https://cran.r-project.org/web/packages/policies.html) [before submitting](https://choosealicense.com/). 


I know exactly what to do to remove the note (`xlab` and `ylab` passed to `...` in a 
plotting function, and causing the "not-error" `no visible global function definition 
for ‘xlab’`). Which wasn't suggested by _censor_. But that's when I was reminded that in 2020, there are thousands of ways to circumvent reactionaries (you should use GitHub/Gitlab search sometimes, your whole world and certainties will fall apart) on the internet, and of 
this **gem of a package** I once heard about: 
[miniCRAN](https://github.com/andrie/miniCRAN). 



How does [miniCRAN](https://github.com/andrie/miniCRAN) feels to me? Just like when I want 
to create a virtual environment in Python with only packages that I'd like to use, 
isolated from other packages (well, for actually using R in a virtual environment, I don't know yet). And there's more to it than that, as shown 
[in this presentation](https://blog.revolutionanalytics.com/2014/10/introducing-minicran.html), including the invaluable ability to use a CRAN-like system of R packages behind a firewall, in an entreprise -- or wherever -- ecosystem. With [miniCRAN](https://github.com/andrie/miniCRAN), I was able to **create an [online repo](https://techtonique.github.io/r-techtonique-forge/)**  just like what I'm used to, with only the set of packages I'd like to  use.


And **for someone who's been using R for 15 + years (I wonder why it's studied and taught in university though), it was delightful to be able to do this for the first time**: 

```r
# my mirror of packages created with miniCRAN
repo <- "https://techtonique.github.io/r-techtonique-forge/"

# list of packages currently available in the repo for Linux/macOS and Windows
print(miniCRAN::pkgAvail(repos = repo, type = "source"))
print(miniCRAN::pkgAvail(repos = repo, type = "win.binary"))

# installing `pkg_name` from my mirror using the old faithful `install.packages`
install.packages("pkg_name", repos=repo, type="source") 

# using the package as usual 
library(pkg_name)    
```