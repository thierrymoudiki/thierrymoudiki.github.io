---
layout: post
title: "ESGtoolkit, a tool for Monte Carlo simulation (v0.2.0)"
description: ESGtoolkit, a tool for Monte Carlo simulation (v0.2.0)
date: 2020-01-17
categories: R
---


I'm still receiving questions about [ESGtoolkit](https://github.com/Techtonique/ESGtoolkit) -- a tool that I developped in 2014 for stochastic simulation --  from time to time, even if it's not really my current focus. As I also noticed recently, `ESGtoolkit` is downloaded a few times each month: 

![image-title-here]({{base}}/images/2020-01-17/2020-01-17-image1.png){:class="img-responsive"} 

If you recognize yourself in these numbers, I'd be really happy to hear from you (_via_ __thierry dot moudiki at pm dot me__ or a pull request [on Github](https://github.com/Techtonique/ESGtoolkit)), about your use of `ESGtoolkit`. This will prevent me from accidentally breaking the interface,  and you won't ask yourself "it worked perfectly yesterday, why is it broken today?". 

I decided to _declutter_ it a little bit [on Github](https://github.com/Techtonique/ESGtoolkit), so that it can host your future contributions/remarks/suggestions. The new version, v0.2.0, contains an updated [vignette](https://www.researchgate.net/publication/338549100_ESGtoolkit_a_tool_for_stochastic_simulation_v020) and a refactored code. Simulation functions `simdiff`  and `simshocks` include a reproducibility seed (random state) now, for you to be able to... reproduce your simulations from one function call to the other. 

Package [vignette](https://www.researchgate.net/publication/338549100_ESGtoolkit_a_tool_for_stochastic_simulation_v020) should be a good introduction. Otherwise, if this vignette is too technical, please refer to these blog posts, [here](https://thierrymoudiki.wordpress.com/2014/12/24/calibrated-hull-and-white-short-rates-with-rquantlib-and-esgtoolkit/) and [here](https://thierrymoudiki.wordpress.com/2016/01/20/heston-model-for-options-pricing-with-esgtoolkit/), or [this presentation](https://fr.slideshare.net/thierrymoudiki/isfa23092014-es-gtoolkit).

![image-title-here]({{base}}/images/2020-01-17/2020-01-17-image2.png){:class="img-responsive"} 


Contributions/remarks/suggestions are welcome as usual. You can submit a pull request [on Github](https://github.com/Techtonique/ESGtoolkit) or send me an email: __thierry dot moudiki at pm dot me__.


__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!



