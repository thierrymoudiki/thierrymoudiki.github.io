---
layout: post
title: "Transfer Learning using ahead::ridge2f on synthetic stocks returns"
description: "I pretrain ahead::ridge2f on 1000 synthetic stock returns using Bayesian Optimization, and test its performance on real market data."
date: 2025-09-08
categories: R
comments: true
---

In [https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f](https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f), I pretrain `ahead::ridge2f` on 1000 synthetic stock returns using Bayesian Optimization, and test its performance on real market data. 

In order to reproduce results from [https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f](https://github.com/thierrymoudiki/2025-09-05-transfer-learning-ridge2f), either: 

Run `2025-09-07-transfer-learning-stock-returns.Rmd` 

or

Execute the `.R` files in the order in which they appear. 

Results on 4 major European indices: 

```R
[1] "\n=== MEDIAN PERFORMANCE ACROSS ALL SERIES ==="
   Method    Winkler Coverage Interval_Width
1  fgarch 0.05925044     93.5     0.04735842
2  ridge2 0.06024835     94.5     0.04753165
3 rugarch 0.05919827     93.5     0.04753477
```

Results on 10 CAC40 stocks: 

```R
  Method    Winkler Coverage Interval_Width
1  fgarch 0.09799624 96.42857     0.06746349
2  ridge2 0.09573495 95.71429     0.07500853
3 rugarch 0.09797592 97.14286     0.06758644
```

More details about this model (actually used in an industrial setting):

- [https://thierrymoudiki.github.io/blog/2025/07/01/r/python/ridge2-bayesian](https://thierrymoudiki.github.io/blog/2025/07/01/r/python/ridge2-bayesian)
- [https://www.mdpi.com/2227-9091/6/1/22](https://www.mdpi.com/2227-9091/6/1/22)
- [https://thierrymoudiki.github.io/blog/2024/02/26/python/r/julia/ahead-v0100](https://thierrymoudiki.github.io/blog/2024/02/26/python/r/julia/ahead-v0100)
- [Doc for R](https://docs.techtonique.net/ahead/index.html)
- [Doc for Python](https://docs.techtonique.net/ahead_python/ahead.html#Ridge2Regressor)
