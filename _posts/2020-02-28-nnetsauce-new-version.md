---
layout: post
title: "Version 0.4.0 of nnetsauce, with fruits and breast cancer classification "
description: Version 0.4.0 of nnetsauce, with fruits and breast cancer classification 
date: 2020-02-28
categories: [Python, QuasiRandomizedNN, R]
---

[English version](#english-version) / [Version en français](#french-version)

<hr>

# English version

A new version of [nnetsauce](https://github.com/Techtonique/nnetsauce), version `0.4.0`, is now available on Pypi and for __R__. As usual, you can install it on __Python__ by using the following commands (command line):

{% highlight bash %}
pip install nnetsauce
{% endhighlight %}

And if you're using __R__, it's still (R console):

{% highlight R %}
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
{% endhighlight %}

The __R__ version may be slightly lagging behind on some features; feel free to signal it on GitHub or [contact me](https://thierrymoudiki.github.io/#contact) directly. This new release is accompanied by a few goodies: 

1) __New features__, detailed in the [changelog](https://github.com/Techtonique/nnetsauce/blob/master/CHANGES.md).

2) A refreshed [__web page__](https://thierrymoudiki.github.io/software/nnetsauce/) containing all the information about package installation, use, interface's work-in-progress documentation, and contribution to package development. 

3) A specific [RSS feed](https://thierrymoudiki.github.io/feed_nnetsauce.xml) related to nnetsauce on this blog (there's still a general feed containing everything [here](https://thierrymoudiki.github.io/feed.xml)).

4) A working paper related to `Bayesianrvfl2Regressor`, `Ridge2Regressor`, `Ridge2Classifier`, and `Ridge2MultitaskClassifier` : [_Quasi-randomized networks for regression and classification, with two shrinkage parameters_](https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters). About `Ridge2Classifier` specifically, you can also consult [this other paper](https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks). 

Among nnetsauce's new features, there's a new model class called `MultitaskClassifier`, briefly described in the first paper from point 4). It's a __multitask__ classification model based on regression models, with shared covariates.  __What does that mean?__ We use the figure below to start the  explanation: 

![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image1.png){:class="img-responsive"} 

Imagine that we have 4 fruits at our disposal, and we would like to classify them as avocados (is an avocado a fruit?), apples or tomatoes, by looking at their color and shapes. What we called __covariates__ before in model description are `color` and `shape`, also known as explanatory variables or predictors. The column containing fruit names in the figure -- on the left -- is a so-called __response__; a variable that `MultitaskClassifier` must learn to classify (which is typically much larger). This raw response is transformed into a one-hot encoded one -- on the right. 

Instead of one response vector, we now have three different responses. And instead of one classification problem on one response, three different two-class classification problems on three responses: __is this fruit an apple or not? Is this fruit a tomato or not? Is this fruit an avocado or not?__ All these three problems share the same covariates: `color` and `shape`. 

`MultitaskClassifier` can use any regressor (meaning, a statistical learning model for continuous responses) to solve these three problems; with the same regressor being used for all three of them -- which is _a priori_ a relatively strong hypothesis. Regressor's predictions on each response are interpreted as raw probabilities that the fruit is either one of them or not. 

We now use `MultitaskClassifier` on breast cancer data, as we did in [this post]({% post_url 2019-09-18-nnetsauce-adaboost-1 %}) for `AdaBoostClassifier`, to illustrate how it works. The __R__ version for this code would be almost identical, replacing "."'s by "$"'s.

__Import packages:__

{% highlight python %}
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
{% endhighlight %}

__Model fitting on training set:__

{% highlight python %}
breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

# Training/testing datasets (using reproducibility seed)
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# Linear Regression is used (can be anything, but must be a regression model)
regr = LinearRegression()
fit_obj = ns.MultitaskClassifier(regr, n_hidden_features=5, 
                             n_clusters=2, type_clust="gmm")

# Model fitting on training set 
start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# Model accuracy on test set
print(fit_obj.score(X_test, y_test))

# Area under the curve
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))
{% endhighlight %}

These results can be found in [nnetsauce/demo/](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_200220_multitask_classification.ipynb). `MultitaskClassifier`'s  accuracy on this dataset is 99.1%, and other indicators such as precision are equal to 99% on this dataset too. Let's visualize the __missclassification results__, as we did in a [previous post]({% post_url 2019-09-18-nnetsauce-adaboost-1 %}), on the same dataset. 

![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image2.png){:class="img-responsive"} 

In this case with `MultitaskClassifier`, and no advanced hyperparameter tweaking, there is one patient out of 114 who is missclassified. A robust way to understand `MultitaskClassifier`'s accuracy on this dataset using these same parameters, could be to repeat the same procedure for multiple random reproducibility seeds (see code, the training and testing sets randomly change when we change the `seed`, and we change `MultitaskClassifier`'s seed too). 	

We obtain the results below for 100 reproducibility seeds. The accuracy is always at least 90%, mostly 95% and quite sometimes, higher than 98% (with no advanced hyperparameter tweaking).


![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image3.png){:class="img-responsive"}

<hr>

# French version

Une nouvelle version de nnetsauce, la version 0.4.0, est maintenant disponible sur Pypi et pour R. Comme d'habitude, vous pouvez l'installer sur Python en utilisant les commandes suivantes (ligne de commande):

{% highlight bash %}
pip install nnetsauce
{% endhighlight %}

Et si vous utilisez R, c'est toujours (console R):

{% highlight R %}
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
{% endhighlight %}

La version R peut être légèrement en retard sur certaines fonctionnalités; n'hésitez pas à le signaler sur GitHub ou à me [contacter](https://thierrymoudiki.github.io/#contact) directement en cas de problème. Cette nouvelle version s'accompagne de quelques _goodies_:

1) Nouvelles fonctionnalités, détaillées dans le fichier [changelog](https://github.com/Techtonique/nnetsauce/blob/master/CHANGES.md).

2) Une [__page Web__](https://thierrymoudiki.github.io/software/nnetsauce/)  actualisée contenant toutes les informations sur l'installation, l'utilisation, la documentation de travail en cours, et la contribution au développement de l'outil.

3) Un flux [RSS](https://thierrymoudiki.github.io/feed_nnetsauce.xml) spécifique lié à nnetsauce sur ce blog (il existe toujours un flux général contenant tous les autres sujets, [ici](https://thierrymoudiki.github.io/feed.xml)).

4) Un document de travail relatif à Bayesianrvfl2Regressor, Ridge2Regressor, Ridge2Classifier et Ridge2MultitaskClassifier: [_Quasi-randomized networks for regression and classification, with two shrinkage parameters_](https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters). À propos de Ridge2Classifier en particulier, vous pouvez également consulter [cet autre article](https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks).

Parmi les nouvelles fonctionnalités, il existe une nouvelle classe de modèle appelée MultitaskClassifier, brièvement décrite dans le premier article du point 4). Il s'agit d'un modèle de classification multitâche basé sur des modèles de régression, avec des covariables partagées. Qu'est-ce que cela signifie? Nous utilisons la figure ci-dessous pour commencer l'explication:

![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image1.png){:class="img-responsive"} 

Imaginez que nous ayons 4 fruits à notre disposition, et nous aimerions les classer comme avocats (un avocat est-il un fruit?), pommes ou tomates, en regardant leur couleur et leur forme. Ce que nous appelions auparavant les covariables dans la description du modèle sont la couleur et la forme, également appelées variables explicatives ou prédicteurs. La colonne contenant les noms des fruits sur la figure -- à gauche -- est la réponse; la variable que MultitaskClassifier doit apprendre à classer (qui contient généralement beaucoup plus d'observations). Cette réponse brute est transformée en une réponse codée -- à droite.

Au lieu d'une réponse, nous avons maintenant trois réponses différentes. Et au lieu d'un problème de classification sur une réponse, trois problèmes de classification à deux classes différents, sur trois réponses: ce fruit est-il ou non une pomme? Ce fruit est-il ou non une tomate? Ce fruit est-il ou non un avocat? Ces trois problèmes partagent les mêmes covariables: la couleur et la forme.

MultitaskClassifier peut utiliser n'importe quel modèle de régression (c'est-à-dire un modèle d'apprentissage statistique pour des réponses continues) pour résoudre ces trois problèmes simultanément; avec le même modèle de régression utilisé pour les trois - ce qui est a priori une hypothèse relativement forte. Les prédictions du modèle de régression sur chaque réponse sont alors interprétées comme des probabilités brutes que le fruit soit ou non, dans l'une ou l'autre des classes.

Nous utilisons maintenant MultitaskClassifier sur des données de cancer du sein, comme nous l'avions fait dans [cet article]({% post_url 2019-09-18-nnetsauce-adaboost-1 %}) pour AdaBoostClassifier. La version R de ce code serait presque identique, il s'agirait essentiellement de remplacer les «.» ’S par des « $ »’ s.

__Import des packages__:

{% highlight python %}
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
{% endhighlight %}


__Ajustement du modèle sur l'ensemble d'entraînement__:

{% highlight python %}
breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

# Training/testing datasets (using reproducibility seed)
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# Linear Regression is used (can be anything, but must be a regression model)
regr = LinearRegression()
fit_obj = ns.MultitaskClassifier(regr, n_hidden_features=5, 
                             n_clusters=2, type_clust="gmm")

# Model fitting on training set 
start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# Model accuracy on test set
print(fit_obj.score(X_test, y_test))

# Area under the curve
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))
{% endhighlight %}

Les résultats de ce traitement se trouvent dans [ce notebook](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_200220_multitask_classification.ipynb). La précision du MultitaskClassifier sur cet ensemble de données est de 99,1%, et d'autres indicateurs sont également de l'ordre de 99% sur ces données. Visualisons maintenant comment les observations sont bien ou mal classées en fonction de leur classe réelle, comme nous l'avions fait dans un [_post_]({% post_url 2019-09-18-nnetsauce-adaboost-1 %}) précédent, sur le même ensemble de données.

![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image2.png){:class="img-responsive"} 

Dans ce cas, avec MultitaskClassifier et aucun ajustement avancé des hyperparamètres, il n'y a qu'un patient sur 114 qui est mal classé. Un moyen robuste de comprendre la précision de MultitaskClassifier sur cet ensemble de données en utilisant ces mêmes paramètres, pourrait être de répéter la même procédure pour plusieurs graines de reproductibilité aléatoire (voir le code, les ensembles d'apprentissage et de test changent de manière aléatoire lorsque nous changeons la graine `seed`, et nous changeons aussi la graine de MultitaskClassifier ).


Nous obtenons les résultats ci-dessous pour 100 graines de reproductibilité. La précision est toujours d'au moins 90%, la plupart du temps égale à 95% et assez souvent, supérieure à 98% (sans ajustement avancé des hyperparamètres).

![image-title-here]({{base}}/images/2020-02-28/2020-02-28-image3.png){:class="img-responsive"}

__Note:__ I am currently looking for a _gig_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Licence Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />Under License <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>.


