---
layout: post
title: "Technical documentation"
description: Technical documentation.
date: 2020-09-18
categories: [Misc, Python, QuasiRandomizedNN, DataBases, mlsauce, ExplainableML]
---

All the Python packages presented in this blog:

<ul>
  
  <li> <a href="http://127.0.0.1:4000/blog/index.html#QuasiRandomizedNN">nnetsauce</a> &nbsp; Statistical/Machine Learning using Randomized and Quasi-Randomized (neural) networks &nbsp;|&nbsp; <a href="https://forms.gle/HQVbrUsvZE7o8xco8">feedback form</a></li>
  
  <li> <a href="https://thierrymoudiki.github.io/blog/#DataBases">querier</a> &nbsp; A query language for Python Data Frames &nbsp;|&nbsp; <a href="https://forms.gle/uStfcXJjtGki2R3s6">feedback form</a></li>
  
  <li> <a href="https://thierrymoudiki.github.io/blog/#mlsauce">mlsauce</a> &nbsp; Miscellaneous Statistical/Machine Learning stuff &nbsp;|&nbsp; <a href="https://forms.gle/tm7dxP1jSc75puAb9">feedback form</a></li>

  <li> <a href="https://thierrymoudiki.github.io/blog/#ExplainableML">teller</a> &nbsp; Model-agnostic Statistical/Machine Learning explainability &nbsp;|&nbsp; <a href="https://forms.gle/Y18xaEHL78Fvci7r8">feedback form</a></li>                

</ul>   

Now have a common home for their documentation, available 
on [Techtonique website](https://techtonique.github.io/) (it's a work in progress). 


![new-techtonique-website]({{base}}/images/2020-09-11/2020-09-11-image1.png){:class="img-responsive"}
_Figure: [New Techtonique Website](https://techtonique.github.io/)_


For this documentation, I'm using [**MkDocs**](https://www.mkdocs.org/) in conjunction with [
**keras-autodoc**](https://github.com/keras-team/keras-autodoc). With MkDocs, 
I found out that you can create a static website rapidly using [Markdown](https://en.wikipedia.org/wiki/Markdown). Regarding package technical 
documentation in particular, one thing that I find useful and that I've been 
searching for a while, is the ability for a tool to loop and read Python [docstrings](https://en.wikipedia.org/wiki/Docstring): that's what 
keras-autodoc allowed me to do. I've **heard of a way of doing such a thing for R documentation** recently: 


![new-techtonique-website]({{base}}/images/2020-09-18/2020-09-18-image1.png){:class="img-responsive"}


 So, [cross_val](https://github.com/Techtonique/cross_val) and [ESGtoolkit](https://github.com/Techtonique/ESGtoolkit), two 
 R members of [Techtonique](https://github.com/Techtonique), are next in line for [Techtonique website](https://techtonique.github.io/). 
 I may also give [pkgdown](https://pkgdown.r-lib.org/) a try, depending on how 
 I'm able to tweak the styles, and all. 