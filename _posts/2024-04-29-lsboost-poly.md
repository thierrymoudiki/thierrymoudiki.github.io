---
layout: post
title: "mlsauce's `v0.17.0`: boosting with polynomials and heterogeneity in explanatory variables"
description: "LSBoost's boosting with polynomials and heterogeneity in explanatory variables."
date: 2024-04-29
categories: Python
comments: true
---


Last week [in #135](https://thierrymoudiki.github.io/blog/), I talked about `mlsauce`'s `v0.13.0`, and [`LSBoost`](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) in particular. When using `LSBoost`, it's now possible to: 

- Obtain **prediction intervals** for regression, notably by employing Split Conformal Prediction. 
  
- Take into account an _a priori_ heterogeneity in explanatory variables through clustering.

In `v0.17.0`, I added a **2 new features** to `LSBoost`:

- The possibility to add polynomial interaction functions of explanatory variables to the mix (see  [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) for more details). This is done by setting the `degree` parameter of `LSBoostRegressor` or `LSBoostClassifier` to a positive integer value.
  
- The possibility to use [Elastic Net](https://glmnet.stanford.edu/articles/glmnet.html#linear-regression-family-gaussian-default) as a `solver` (a _base learner_), in addition to `ridge`, and `lasso`. /!\ `enet` (for Elastic Net) **will become the default value** for the `solver` parameter next week, as it's _fast_ (uses [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)) and gracefully combines both ridge regression and the lasso. `ridge`, and `lasso` will remain available to avoid breaking existing pipelines, but you'd have to specify them explicitly as `solver`s. For `enet`, `reg_lambda` is still used as a regularization parameter, and an `alpha` (in [0, 1]) parameter defines a compromise between lasso (`alpha = 1`) and ridge (`alpha = 0`) penalties. The default value for `alpha` is `0.5`.

 This week, I'll show how to use the new features, to **gain an intuition** of how they work. Keep in mind however: these examples only show that it's possible to overfit the training set (hence reducing the loss function's magnitude) by adding some clusters. The whole model's hyperparameters need to be 'fine-tuned', the `learning_rate` and `n_iterations` in particular, for example by using  [GPopt](https://thierrymoudiki.github.io/blog/2023/11/05/python/r/adaopt/lsboost/mlsauce_classification). Next week, I'll update the documentation and notably [this working paper](https://www.researchgate.net/publication/346059361_LSBoost_gradient_boosted_penalized_nonlinear_least_squares) in a more comprehensive way. 


The _best_ way (feel free to answer [this question](https://stackoverflow.com/questions/78397654/getting-an-importerror-when-packaging-a-cython-extension) on stackoverflow) to install the package is still to use the development version (tested in colab...):

```bash
pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```

You can reproduce the results with [this notebook](https://github.com/Techtonique/mlsauce/blob/master/mlsauce/demo/thierrymoudiki-2024-04-29-LSBoost.ipynb).


 # 0 - Install and import data


```python
!pip uninstall mlsauce --yes
```


```python
!pip install git+https://github.com/Techtonique/mlsauce.git --verbose
```


```python
import mlsauce as ms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt


dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)
```

# 1 - `solver = 'ridge'` and polynomial degree >= 2


```python
obj1 = ms.LSBoostRegressor()
print(obj1.get_params())
start = time()
obj1.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj1.obj['loss']}")

obj2 = ms.LSBoostRegressor(n_clusters=2, learning_rate=0.2)
print(obj2.get_params())
start = time()
obj2.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj2.obj['loss']}")

obj3 = ms.LSBoostRegressor(n_clusters=2, degree=2)
print(obj3.get_params())
start = time()
obj3.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj3.obj['loss']}")

obj4 = ms.LSBoostRegressor(n_clusters=3, degree=3)
print(obj4.get_params())
start = time()
obj4.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj4.obj['loss']}")

```

    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_clusters': 0, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:01<00:00, 79.75it/s]


    
     Elapsed: 1.2682864665985107
    loss: [139.32111847614541, 131.70308253447635, 125.19499783631174, 119.59820193020641, 114.93497551784296, 110.8527083582196, 107.54354587416081, 104.7134674297217, 102.39708217776042, 100.42933365689483, 98.83144460196893, 97.52844685361345, 96.30852369223383, 95.39003032176558, 94.54352885226272, 93.96512308838865, 93.47502447209713, 93.03977520683647, 92.65829051516376, 92.2802164369876, 92.01046242081102, 91.74049787360917, 91.52039293368806, 91.35783846316978, 91.18787261074638, 91.04354476856568, 90.94852971198672, 90.86008655659633, 90.78674222651686, 90.74000692736819, 90.63215837544449, 90.52950586401757, 90.48233169529428, 90.37423492871945, 90.32472893384154, 90.21890835467052, 90.21111046873041, 90.13909134552976, 90.0139953107389, 89.97612995083936, 89.94348148438813, 89.93304401122035, 89.89959616857168, 89.85807584682216, 89.8392966602071, 89.80667250002107, 89.77924928861444, 89.77367456574875, 89.75559610481965, 89.71397020176407, 89.59175806869736, 89.56664373311831, 89.54091445963215, 89.46168703639418, 89.45189787914167, 89.42845994767242, 89.36573344897069, 89.34909990107334, 89.31251207605847, 89.30960378231673, 89.2900193168802, 89.15571629926559, 89.1332794039042, 89.06143615145245, 89.04323756985643, 89.00592023420243, 89.0020187793033, 88.96391864010155, 88.94535574532689, 88.93949958765987, 88.91891525045274, 88.90760445979019, 88.89798522168671, 88.88338871701343, 88.85385469298946, 88.82386465259158, 88.81106654969633, 88.7874323259479, 88.7748363821829, 88.74795322218621, 88.74667050075695, 88.68556071633893, 88.67176501250366, 88.66029100611858, 88.65336168694326, 88.62276144125545, 88.6201402067312, 88.60335631240642, 88.57280352448245, 88.5594483820922, 88.49023189096201, 88.44785594877571, 88.43544524381494, 88.40251935329863, 88.39618944976945, 88.37910622847218, 88.37611957899422, 88.3640605723657, 88.33872924544201, 88.33664108588486]
    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.2, 'n_clusters': 2, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:01<00:00, 73.32it/s]


    
     Elapsed: 1.5329296588897705
    loss: [130.83367893567655, 118.36978853470147, 109.70263745563274, 103.66906516909664, 99.41491268844976, 96.62374388556682, 94.83298574697442, 93.68222947764735, 92.9153846160194, 92.31803473535508, 91.97115696869, 91.71083711206805, 91.51983855278394, 91.38012699363394, 91.28552640926058, 91.23390055534463, 91.18292155812665, 91.13266090009023, 91.01351872357111, 90.94152103319904, 90.88256741236287, 90.76917649363118, 90.68916515483345, 90.6241463204776, 90.54149853760747, 90.50983879162368, 90.43465235971378, 90.33848797211253, 90.2680297351062, 90.24095813688729, 90.18312455033704, 90.05454272754294, 90.03731488611776, 89.9471623200098, 89.89669185915396, 89.75343778455495, 89.70676861370553, 89.56565026994494, 89.45908888711088, 89.36680063879811, 89.34911086865165, 89.31088815546408, 89.17808729324963, 89.1125761522851, 89.09901562204598, 89.01167429798109, 88.95076279488383, 88.92192476032442, 88.89143552678257, 88.8399353076311, 88.7733300141213, 88.76804706309517, 88.69814287249048, 88.5788453268133, 88.57819734317886, 88.48007430990472, 88.38166860953304, 88.35506600185093, 88.33691014634925, 88.32996570082982, 88.22934992687436, 88.10191428809424, 88.08363243633644, 88.05564920162551, 88.03926225732519, 88.01212148638815, 87.97514703192572, 87.93122870673841, 87.92194926097288, 87.88218854141395, 87.86070378212733, 87.85385147379762, 87.75543031113773, 87.73143263493998, 87.69174842759524, 87.67965321647605, 87.673454211299, 87.63804462078319, 87.5854261384591, 87.54857396987046, 87.5293009106566, 87.5269864514613, 87.52241658670458, 87.46538794704216, 87.3685761090272, 87.24895344995836, 87.23889062286989, 87.22965005021842, 87.19078151706046, 87.17004581387963, 87.08081926597161, 87.04897060083277, 87.02453881475769, 87.00386156116492, 86.99266101056178, 86.98340206274179, 86.93831520947467, 86.93420075031149, 86.86839781973956, 86.78001242587659]
    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 2, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_clusters': 2, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:03<00:00, 31.79it/s]


    
     Elapsed: 3.2886624336242676
    loss: [138.46406679289797, 130.02911687784407, 122.74598097793128, 116.47657742727863, 111.08524444228846, 106.55680329178956, 102.72017473071023, 99.47042863602837, 96.75628668834405, 94.50553551404269, 92.63949144013449, 91.0691722107126, 89.80317774899414, 88.74124411409235, 87.87069719885474, 87.16695784585367, 86.55517523465173, 86.02400455678165, 85.62603500112755, 85.27070382572722, 85.00266337877103, 84.76565021927969, 84.58713376109829, 84.40933853463417, 84.28716539837815, 84.18976818357947, 84.10173059703584, 84.03046303014399, 83.97274729370876, 83.92075520312329, 83.88192039529922, 83.84111176784998, 83.79511863502773, 83.76136287330272, 83.73337968833519, 83.71127142629713, 83.6918744941698, 83.64818431151006, 83.63210752232445, 83.62237186924207, 83.59768445289845, 83.57998394117516, 83.56990256435788, 83.56076546063005, 83.54908605112888, 83.54167871651399, 83.52967231659126, 83.51665242112935, 83.49544601625972, 83.47893395060991, 83.45981431662462, 83.45316456129692, 83.42405210925749, 83.41671254058762, 83.41456973986233, 83.40860977678491, 83.40464082684298, 83.40020895250508, 83.39602524105483, 83.3853736635742, 83.37780008790719, 83.37597304768038, 83.3687469839559, 83.36297069936599, 83.36029035126958, 83.35203272637273, 83.35059500671038, 83.34967485591902, 83.34000115609149, 83.32722314668666, 83.32358467659589, 83.31136126470331, 83.29259945057044, 83.28914212751293, 83.28521039760028, 83.28481746246688, 83.27460743074492, 83.26962034466729, 83.2628548274941, 83.26036082589354, 83.25827409034957, 83.25321243448998, 83.24859393673994, 83.24568366451265, 83.23549679513344, 83.23430138376206, 83.22909756587205, 83.22599323519538, 83.22550682810837, 83.21019176234417, 83.20724081337342, 83.20574882859364, 83.2042111859721, 83.19948926198819, 83.193568246941, 83.15961915958252, 83.1577172169777, 83.15182952964035, 83.15132244242335, 83.14764227408152]
    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 3, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_clusters': 3, 'n_estimators': 100, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'ridge', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 100/100 [00:10<00:00,  9.89it/s]

    
     Elapsed: 10.495529413223267
    loss: [138.14931908588397, 129.4027435243869, 121.88694206439065, 115.41145625256686, 109.87726578132731, 105.14963912878896, 101.17914759601891, 97.83191553199156, 95.0289111825631, 92.67406781644411, 90.73041397940855, 89.11116067394822, 87.70840560127974, 86.60244317194072, 85.6780514819482, 84.92440650331801, 84.29736111426118, 83.79278115563103, 83.35892332229895, 83.00162692480497, 82.72002881320502, 82.47126331888968, 82.2781242828601, 82.12232589978221, 81.98856994702976, 81.87415215829667, 81.77693471717099, 81.6961133540323, 81.6090635806128, 81.54414873216453, 81.49038650109277, 81.43905501208901, 81.39981895004144, 81.37411574944969, 81.33550631900393, 81.30587634281169, 81.28408474336716, 81.25579561110607, 81.23994744567902, 81.21942411130279, 81.20594539477105, 81.19324314448927, 81.17555560987032, 81.16151415480925, 81.1482104356931, 81.12852873652301, 81.11831172643035, 81.07918029614552, 81.06147632347029, 81.0524168083218, 81.04378841205214, 81.02537802368047, 80.9925138598936, 80.97930632272424, 80.96601996808367, 80.95349353716256, 80.94038307845716, 80.91914687808031, 80.88390169235261, 80.86194574294117, 80.85448718971496, 80.8423225915586, 80.83661148464881, 80.82743562581062, 80.8187318876223, 80.8122028241299, 80.80500996384802, 80.79800245234749, 80.79156632394738, 80.78025350191442, 80.77210849733146, 80.76065721716013, 80.75626236456611, 80.74383580649274, 80.7337232134181, 80.72810744836396, 80.72038672045795, 80.71322853377548, 80.70518351193886, 80.69680451558467, 80.69294035424154, 80.68319307292629, 80.67887178809434, 80.66527265856205, 80.65506840664501, 80.6477872646086, 80.63140309323022, 80.6203078711991, 80.60896984379703, 80.60379161057065, 80.59523925524817, 80.57821467893916, 80.57061121781464, 80.56348346600964, 80.55992990139632, 80.55021895814008, 80.5440861559823, 80.5409277611609, 80.53768386936636, 80.53084993480212]


    



```python
# Plotting the lines with labels
plt.plot(obj1.obj['loss'], label='Loss - learning_rate=0.1')
plt.plot(obj2.obj['loss'], label='Loss - n_clusters=2, learning_rate=0.2')
plt.plot(obj3.obj['loss'], label='Loss - n_clusters=2, degree=2')
plt.plot(obj4.obj['loss'], label='Loss - n_clusters=3, degree=3')

# Displaying the legend
plt.legend()

# Show the plot
plt.show()

```

![pres-image]({{base}}/images/2024-04-29/2024-04-29-image1.png){:class="img-responsive"}    
    


# 2 - `solver = 'elasticnet'`


```python
obj1 = ms.LSBoostRegressor(solver="enet", n_estimators=25)
print(obj1.get_params())
start = time()
obj1.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj1.obj['loss']}")

obj2 = ms.LSBoostRegressor(n_clusters=2, learning_rate=0.2, solver="enet",
                           n_estimators=25)
print(obj2.get_params())
start = time()
obj2.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj2.obj['loss']}")

obj3 = ms.LSBoostRegressor(n_clusters=2, learning_rate=0.2, solver="enet",
                           n_estimators=25, alpha=0)
print(obj3.get_params())
start = time()
obj3.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj3.obj['loss']}")

obj4 = ms.LSBoostRegressor(n_clusters=2, learning_rate=0.2, solver="enet",
                           n_estimators=25, alpha=1)
print(obj4.get_params())
start = time()
obj4.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj4.obj['loss']}")

obj5 = ms.LSBoostRegressor(n_clusters=2, learning_rate=0.2, solver="enet",
                           n_estimators=25, alpha=1, degree=2)
print(obj5.get_params())
start = time()
obj5.fit(X_train, y_train)
print(f"\n Elapsed: {time()-start}")
print(f"loss: {obj5.obj['loss']}")


# Plotting the lines with labels
plt.plot(obj1.obj['loss'], label='Loss - learning_rate=0.1, alpha=0.5 (L1+L2)')
plt.plot(obj2.obj['loss'], label='Loss - n_clusters=2, learning_rate=0.2, alpha=0.5 (L1+L2)')
plt.plot(obj3.obj['loss'], label='Loss - learning_rate=0.1, alpha=0 (L2 pen.)')
plt.plot(obj4.obj['loss'], label='Loss - n_clusters=2, learning_rate=0.2, alpha=1 (L1 pen.)')
plt.plot(obj5.obj['loss'], label='Loss - n_clusters=2, learning_rate=0.2, alpha=1, degree=2')

# Displaying the legend
plt.legend()

# Show the plot
plt.show()

```

    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.1, 'n_clusters': 0, 'n_estimators': 25, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'enet', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 25/25 [00:00<00:00, 47.55it/s]


    
     Elapsed: 0.5511448383331299
    loss: [139.4354291181404, 134.04860172435463, 129.03704407555875, 124.61992090034947, 121.40657221231528, 118.29722826228726, 116.0586796182525, 114.03187153272418, 112.46589723152815, 110.91617268708922, 109.83817740965685, 108.83140089329949, 108.03339225985236, 107.31107218851054, 106.74696561474326, 106.24728675392203, 105.83624710221372, 105.43061331833108, 105.10885864438998, 104.84726156958487, 104.6004194476885, 104.39348029601547, 104.1902981736055, 103.98754676300594, 103.83640154671413]
    {'activation': 'relu', 'alpha': 0.5, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.2, 'n_clusters': 2, 'n_estimators': 25, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'enet', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 25/25 [00:00<00:00, 38.39it/s]


    
     Elapsed: 0.818737268447876
    loss: [133.99068400971234, 125.34899781758122, 118.81439186169462, 114.2065839785141, 110.89199116823127, 108.6632906200425, 107.28352102111089, 106.27779161258653, 105.37816545333398, 104.83746195637701, 104.3924327210331, 103.9605256119815, 103.69362395896887, 103.44599872477694, 103.25816556316416, 103.1113827199278, 102.98555567782437, 102.81463416446454, 102.72467961038245, 102.56794496516358, 102.50207696487394, 102.43635148400048, 102.38351774624878, 102.3353390515508, 102.28040658406701]
    {'activation': 'relu', 'alpha': 0, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.2, 'n_clusters': 2, 'n_estimators': 25, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'enet', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 25/25 [00:00<00:00, 35.58it/s]


    
     Elapsed: 0.8718903064727783
    loss: [131.86362748424415, 121.78836293839404, 114.7942241393109, 109.81305727402513, 106.52982461570632, 104.53698341110132, 103.43711830704014, 102.66014534085303, 102.0290328230347, 101.53196714619348, 101.13653642251832, 100.76244643310258, 100.43993149453821, 100.14792721545851, 99.86115860023098, 99.60671481524858, 99.37868525070988, 99.12166989810993, 98.89375784569046, 98.65892652244924, 98.4390070995436, 98.2396329270718, 98.04168541594527, 97.85287791146655, 97.67911238188505]
    {'activation': 'relu', 'alpha': 1, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 0, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.2, 'n_clusters': 2, 'n_estimators': 25, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'enet', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 25/25 [00:02<00:00, 12.44it/s]


    
     Elapsed: 2.0565598011016846
    loss: [136.0548156094128, 128.3375760646074, 122.60976291353313, 118.38212020636537, 115.15294063696274, 112.80382509739829, 111.33788072698884, 110.23901729385085, 109.28636442755926, 108.64103084049539, 108.12424759573145, 107.60104359686447, 107.28948200200561, 106.96416400066711, 106.72476067118077, 106.56320649382118, 106.43213481870976, 106.25997068482286, 106.17285564398698, 105.90038447774012, 105.84192168484296, 105.760298648814, 105.71968675085114, 105.68599468010764, 105.63893336547038]
    {'activation': 'relu', 'alpha': 1, 'backend': 'cpu', 'cluster_scaling': 'standard', 'clustering_method': 'kmeans', 'col_sample': 1, 'degree': 2, 'direct_link': 1, 'dropout': 0, 'kernel': None, 'learning_rate': 0.2, 'n_clusters': 2, 'n_estimators': 25, 'n_hidden_features': 5, 'reg_lambda': 0.1, 'replications': None, 'row_sample': 1, 'seed': 123, 'solver': 'enet', 'tolerance': 0.0001, 'type_pi': None, 'verbose': 1}


    100%|██████████| 25/25 [00:03<00:00,  6.73it/s]


    
     Elapsed: 3.9757931232452393
    loss: [111.4866474356533, 106.6426178890959, 105.96652736220898, 105.73027817932079, 105.61130062816889, 105.53571011105976, 105.4761027412948, 105.42890209756975, 105.39139060173062, 105.36148684966663, 105.33758493904953, 105.31829398244282, 105.3026655765702, 105.28998257594097, 105.27967513834588, 105.27128836109804, 105.26445764217611, 105.2588897553494, 105.25434819349432, 105.25064173117914, 105.24761543824263, 105.24514357611025, 105.24312395236538, 105.24147341339146, 105.24012423128893]


![pres-image]({{base}}/images/2024-04-29/2024-04-29-image2.png){:class="img-responsive"}        


