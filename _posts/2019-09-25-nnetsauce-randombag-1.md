---
layout: post
title: "Bagging in the nnetsauce"
description: Examples of use of Bagging in Python package nnetsauce
date: 2019-09-25
categories: QuasiRandomizedNN
---

In this post, I will show you how to use a _bootstrap aggregating_  classification algorithm (do not leave yet, I will explain it with apples and tomatoes!). This algorithm is implemented in the new version of [nnetsauce](https://github.com/Techtonique/nnetsauce) (v0.2.0) and is called `randomBag`. The complete list of changes included in this new version can be found [here](https://github.com/Techtonique/nnetsauce/blob/master/CHANGES.md). 

nnetsauce (v0.2.0) can be installed (using the command line) from [Pypi](https://pypi.org/project/nnetsauce/) as: 

```bash
pip install nnetsauce
```

The development/cutting-edge version can still be installed from Github, by using: 

```bash
pip install git+https://github.com/Techtonique/nnetsauce.git
```

Let's start with an introduction on how `randomBag` works, with apple and tomatoes. Like we've seen for [Adaboost]({{base}}/blog/2019/09/18/nnetsauce-adaboost-1), `randomBag` is an ensemble learning method. It __combines multiple individual statistical/machine learning (ML) models into one__, with the hope of achieving a greater performance. We consider the following dataset, containing 2 apples, 3 tomatoes, and a few characteristics describing them: __shape__, __color__, __size of the leaves__. 

![image-title-here]({{base}}/images/2019-09-25/2019-09-25-image1.png){:class="img-responsive"}

3 individual ML models are tasked to classify these fruits (is tomato... a fruit?). That is, to say: __"given that the shape is x and the size of leaves is y, observation 1 is an apple"__. For all the 5 fruits. In the `randomBag`, these individual ML models are __quasi-randomized networks__, and they typically classify fruits by choosing a part of each fruit's characteristics, and a part of these 5 fruits with repetition allowed. For example: 

-ML model __#1__ uses __shape__ and __size of the leaves__, and fruits 1, 2, 3, 4 

-ML model __#2__ uses __shape__ and __color__ (yes, here apples are always green!), and fruits 1, 3, 3, 4 

-ML model __#3__ uses __color__ and __size of the leaves__, and fruits 1, 2, 3, 5 

Then, each individual trained model provides __probabilities to say, for each fruit, "it's an apple"__ (or not):

Fruit# | ML __model #1__ | ML __model #2__ | ML __model #3__ | is an apple?
------------ | ------------- | ------------- | -------------
1 			 | __79.9%__     | 48.5%      | 35.0%  | __yes__
2 			 | 51.5%     | __11.1%__  | __20.0%__  | no
3 			 | 26.1%         | 5.8%       | __90.0%__  | __yes__
4 			 | 85.5%    | 70.5%  | 51.0%  | no
5 			 | __22.5%__         | 61.3%  | 55.0%  | no

__How to read this table?__ Model #1 estimates that fruit #2 has 51% of chances of being an apple (thus, 49% of being a tomato). Similarly, model #3 estimates that, given its characteristics, fruit #5 has 55% of chances of being an apple. When a probability in the previous table is __> 50%__, the model decides __"it's an apple"__. Therefore, here are ML models' classification accuracies:

__Accuracy__ | ML __model #1__ | ML __model #2__ | ML __model #3__ | 
------------ | ------------- | ------------- | -------------
 	 | __40.0%__         | __20.0%__  | __40.0%__  | 


If we calculate a standard deviation of decision probabilities per model, we can obtain an _ad hoc_ - reaaally _ad hoc_ - measure of their uncertainty around their decisions. These uncertainties are respectively __29.3%__, __29.4%__, __26.2%__ for the 3 ML models we estimated. `randomBag` will now take each fruit, and calculate an average probability over the 3 ML models that, __"it's an apple"__. The ensemble's decision probabilities are:

Observation# | `randomBag` ensemble | is an apple? | ensemble decision
------------ | ------------- | ------------- | ------------- | -------------
1 			 | 54.4%     | __yes__ | __yes__
2 			 | 27.5%         | __no__ | __no__
3 			 | 40.6%         | yes | no
4 			 | 69.0%         | no | yes
5 			 | 46.3%     | __no__ | __no__

Doing this, the accuracy of the ensemble increases to __60.0%__ (compared to __40.0%__, __20.0%__, __40.0%__ for individual ML models), and the ensemble's _ad hoc_ uncertainty about its decisions is now __15.5%__ (compared to __29.3%__, __29.4%__, __26.2%__ for individual ML models).

How does it work in the `nnetsauce`? As mentioned before, the individual models are quasi-randomized networks (__deterministic__, cf. [here](https://www.mdpi.com/2227-9091/6/1/22/htm) for details). At each bootstrapping repeat, a fraction of dataset's observations and columns are randomly chosen (with replacement for observations), in order to increase diversity within the ensemble and reduce its variance.

We use the `wine` dataset from [UCI repository](http://archive.ics.uci.edu/ml/index.php) for training `nnetsauce`'s `randomBag`. This dataset contains information about wines' quality, depending on their characteristics. With ML applied to this dataset, we can deduce the quality of a wine - previously unseen - by using its characteristics.


```python
# Load dataset

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)


# Define the ensemble model, with 100 individual models (n_estimators)
# that are decision tree classifiers with depth=2

# One half of the observations and one half of the columns are considered 
# at each repeat (`col_sample`, `row_sample`)

clf = DecisionTreeClassifier(max_depth=2, random_state=123)
fit_obj = ns.RandomBagClassifier(clf, n_hidden_features=5,
                                direct_link=True,
                                n_estimators=100, 
                                col_sample=0.5, row_sample=0.5,
                                dropout=0.1, n_clusters=3, 
                                type_clust="gmm", verbose=1)


# Fitting the model to the training set

fit_obj.fit(Z_train, y_train)
print(fit_obj.score(Z_test, y_test))


# Obtain model predictions on test set's unseen data 

preds = fit_obj.predict(Z_test)


# Obtain classifiction report on test set

print(metrics.classification_report(preds, y_test))
```

The `randomBag` is ridiculously accurate on this dataset. So, you might have some fun trying [these other examples](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/thierrymoudiki_250919_randombag_classification.ipynb), or any other real world example of your choice! If you happen to create a notebook, it will find its home [here](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/demo/) (naming convention: yourgithubname_ddmmyy_shortdescriptionofdemo.ipynb).

__Note:__ I am currently looking for a _side hustle_. You can hire me on [Malt](https://www.malt.fr/profile/thierrymoudiki) or send me an email: __thierry dot moudiki at pm dot me__. I can do descriptive statistics, data preparation, feature engineering, model calibration, training and validation, and model outputs' interpretation. I am fluent in Python, R, SQL, Microsoft Excel, Visual Basic (among others) and French. My résumé? [Here]({{base}}/cv/thierry-moudiki.pdf)!

