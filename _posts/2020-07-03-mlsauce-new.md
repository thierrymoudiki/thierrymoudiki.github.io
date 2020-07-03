---
layout: post
title: "New version of mlsauce, with Gradient Boosted randomized networks and stump decision trees"
description: Announcements.
date: 2020-07-03
categories: [Misc]
---


Last week, I announced a new version of [mlsauce](https://github.com/thierrymoudiki/mlsauce) (see [AdaOpt classification on MNIST handwritten digits (without preprocessing)]({% post_url 2020-05-29-adaopt-classifier-3 %}), [AdaOpt (a probabilistic classifier based on a mix of multivariable optimization and nearest neighbors) for R]({% post_url 2020-05-22-adaopt-classifier-2 %}), and [AdaOpt]({% post_url 2020-05-15-adaopt-classifier-1 %})), with new features. Well... We are still next week! 

This __new version contains__:

- A [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) algorithm using [randomized networks](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN) as weak learners
- An implementation of [stump decision trees](https://en.wikipedia.org/wiki/Decision_stump) 

I've been putting work in this new version like... __crazy__. Because it contains some compiled code in C, ported to Python _via_ Cython. Hence, so far, version `0.4.0` is only available through a Git branch called `refactor`. If you're interested in a sneak peek, you can install `0.4.0` by doing: 

- [Clone the `refactor` branch](https://stackoverflow.com/questions/1911109/how-do-i-clone-a-specific-git-branch)
- When cloned, run `make install` into the repo
- Play with the examples stored in `/examples` (tune the parameters)

Check [the repo](https://github.com/thierrymoudiki/mlsauce) next week, and you'll see the new changes merged (at least in Python for now, and hopefully for R). More insights on the Gradient Boosting algorithm will be unveiled in a few weeks. I will also talk about the stump decision trees in more details.

__Stay tuned!__

