---
layout: post
title: "Analytics Pipeline for Dashboards, with Python, R and Javascript"
description: "I had a lot of fun this morning, brainstorming and assembling this _Analytics Pipeline_ for Dashboards with Claude (and yes it takes much, much more than only 10 prompts)."
date: 2026-09-27
categories: [Python, R, Javascript]
comments: true
---

I had a lot of fun this morning, brainstorming and assembling this _Analytics Pipeline_ for Dashboards with Claude (and yes it takes much, much more than only 10 prompts). 

The philosophy is borrowed from Observable Framework's data loader, but implemented from scratch with voluntarily opinionated choices of libraries for Python, R and Javascript (and no Markdown): pre-compute data with **Python + Polars** (Polars is my new _crush_) or **R + dplyr**, publish it as static files, and explore them with **JavaScript** in the browser ([Arquero](https://github.com/uwdata/arquero) for data wrangling, [Observable Plot](https://observablehq.github.io/plot/) or [Highcharts](https://www.highcharts.com/) for charts)

The repository is available on [GitHub](https://github.com/thierrymoudiki/analytics-pipeline) and here's the quick start guide to run it locally (you need to have [Python](https://www.python.org/downloads/) and [R](https://www.r-project.org/) installed on your machine):


```bash
make                        # list all targets
uv venv venv                # or: make venv (python -m venv venv without uv)
source venv/bin/activate    # Windows: venv\Scripts\activate
make install                # Polars into venv/, dplyr into .uvr/library/
make dev                    # starts http://localhost:3000 and opens your browser
```

The repository's README file is also informative and contains a few more details about the pipeline. 

![image-title-here]({{base}}/images/2026-09-27/2026-09-27-image1.png){:class="img-responsive"}
    

