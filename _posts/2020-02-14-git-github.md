---
layout: post
title: "Git/Github for contributing to package development"
description: Git/Github
date: 2020-02-14
categories: [Misc]
---

__Disclaimer__: I have no affiliation with Microsoft's GitHub, GitLab, CodeCademy or D2L team. 

[Last week]({% post_url 2020-02-07-forms %}), I presented __feedback forms__ for the tools I'm actively maintaining. Forms are the best way to interact with, or contribute to these projects, if you're not familiar with Git/GitHub yet. A link to these forms can be found in each package's GitHub README, @ section __Contributing__. 

This week I present Git/GitHub, two useful and complementary tools for online collaboration, and especially for submitting changes to [nnetsauce](https://github.com/Techtonique/nnetsauce), [teller](https://github.com/Techtonique/teller), [querier](https://github.com/Techtonique/querier), [ESGtoolkit](https://github.com/Techtonique/ESGtoolkit) or [crossval](https://github.com/Techtonique/crossval). __Git__ is a [__command line__](https://www.codecademy.com/articles/command-line-commands) version control tool that allows you to track changes to a project's repository. [__GitHub__](https://github.com/) is, among other things, a hub for Git with a user-friendly web interface. With GitHub (and I guess it's the same for [GitLab](https://about.gitlab.com/)), you can work remotely with many different contributors on the same project and keep a _journal_ of who did what, where and when.

The best resources that I found to learn Git basics were the first three chapters of [Pro Git](https://git-scm.com/book/en/v2),  available online for free. Of course, you can read the entire book for a deeper dive into Git's intricacies, but I found these chapters to be sufficient to start doing a lot of interesting things. 

If you want to contribute to a __repository__ (_repo_), you'll first need to [install Git and create a Github account](https://help.github.com/en/github/getting-started-with-github/set-up-git). Once you're done with installing Git and creating a Github account, you can [__fork__ + __clone__](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) [nnetsauce](https://github.com/Techtonique/nnetsauce), [teller](https://github.com/Techtonique/teller), [querier](https://github.com/Techtonique/querier), [ESGtoolkit](https://github.com/Techtonique/ESGtoolkit) or [crossval](https://github.com/Techtonique/crossval). That means: __creating your own local copy of these repos__ (note that if the source repo is dropped, then your forked repo will be dropped too), so that you can work on them without affecting the source repo. Your local changes to __forked__ + __cloned__ repos are not visible to the source repo until you send a proposal for changes -- which can be accepted or rejected -- i.e [_submit a pull request_](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

![image-title-here]({{base}}/images/2020-02-14/2020-02-14-image1.png){:class="img-responsive"}

To finish, I found [this resource](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html) to be very informative in describing the process __fork__ + __clone__ + __submit a pull request__. You can replace all references to _the book_ in there, by references to [nnetsauce](https://github.com/Techtonique/nnetsauce), [teller](https://github.com/Techtonique/teller), [querier](https://github.com/Techtonique/querier), [ESGtoolkit](https://github.com/Techtonique/ESGtoolkit) or [crossval](https://github.com/Techtonique/crossval). 

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!



