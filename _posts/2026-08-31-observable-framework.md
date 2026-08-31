---
layout: post
title: "Skip the R/Python runtime: fast tabular dashboards with Observable Framework"
description: "Shiny and Streamlit dashboards depend on a live R or Python process, which makes vanilla deployments slow to load. Observable Framework compiles your data pipeline at build time and ships a static, JavaScript-only site — here's how it works, with a small restaurant-tips dashboard as an example."
date: 2026-08-31
categories: [Python, R, JavaScript]
comments: true
---

If you've built dashboards with **Shiny** or **Streamlit**, you know the pattern: a Python or R
process runs on a server, holds your data in memory, and re-executes app logic on every
interaction. That's powerful, but it also means the app is only as fast as its backend.

**Observable Framework** (and I'm not being paid for saying what I say in this post) takes a 
different approach. Instead of running R/Python at *request*
time, it could run your data pipeline once at *build* time, then ship a fully static site — HTML,
JS, and pre-processed data files. No backend process serves the page; the browser does the
rendering. With a bit of JavaScript, you get dashboards that load like a static webpage because,
after build, that's exactly what they are.

The full source code is available at
[github.com/thierrymoudiki/tips-dashboard](https://github.com/thierrymoudiki/tips-dashboard).

## The core building blocks

**1. Data loaders run once, at build time.**

Any file under `src/data/` named `<name>.<ext>.py` (or `.R`, `.js`, `.sh`, etc.) is executed
during `npm run dev` / `npm run build`, and its stdout becomes a static data file:

```python
import sys, pandas as pd

df = pd.read_csv(SOURCE_URL)
df["tip_pct"] = (df["tip"] / df["total_bill"] * 100).round(2)
df.to_csv(sys.stdout, index=False)
```

Pandas (or R, or a database query) can do the heavy lifting *once*. 
The browser never runs Python — it just fetches the resulting CSV/Parquet file.

**2. Pages are Markdown with embedded reactive JavaScript.**

```js
const tips = FileAttachment("data/tips.csv").csv({typed: true});
```

```js
const filtered = tips.filter((d) => day.includes(d.day));
```

Every JS code block is reactive: change an input, and every block referencing it re-runs
automatically — no manual event wiring, no full-page reload.

**3. Inputs drive state, Generators expose it as a reactive value.**

```js
const dayInput = Inputs.checkbox(["Thur", "Fri", "Sat", "Sun"], {value: ["Thur", "Fri", "Sat", "Sun"]});
const day = Generators.input(dayInput);
```

**4. Observable Plot renders charts declaratively**, similar in spirit to ggplot2's grammar of
graphics, but native to JS:

```js
Plot.plot({
  marks: [
    Plot.dot(filtered, {x: "total_bill", y: "tip", fill: "smoker", tip: true}),
    Plot.linearRegressionY(filtered, {x: "total_bill", y: "tip"})
  ]
});
```

**5. `npm run build` produces a static `dist/`.** Loaders re-run once, outputs get
content-hashed for cache-busting, and the result deploys anywhere static files are served —
GitHub Pages, Netlify, S3, no server process required.

## Why this matters for load times

In a typical Shiny/Streamlit app, the first paint waits on: server boot, R/Python session init,
and often a full data load into memory — every time a new session starts, unless you've invested
in caching infrastructure. In Observable Framework, that entire pipeline already happened at
build time. The visitor's browser downloads static assets and a pre-processed data file, then
JS takes over — comparable to loading any other static site.

The trade-off is real: you lose the ability to run arbitrary server-side computation per request
(live model inference, per-user database queries) unless you add a separate backend. But for the
very common case — take some tabular data, clean it, let users filter and explore it — Framework
gets you a snappier result with less infrastructure.

## Try it

A minimal example: a restaurant-tips dashboard with a Python loader (pandas), reactive filters
(`Inputs.checkbox`/`Inputs.radio`), and four Plot charts, built and deployed as a static site with
`npm run build`. The whole interactive layer is under 100 lines of Markdown + JS — no server to
manage after deploy.

The full source code is available at
[github.com/thierrymoudiki/tips-dashboard](https://github.com/thierrymoudiki/tips-dashboard).

![xxx]({{base}}/images/2026-08-31/2026-08-31-image1.png){:class="img-responsive"}  