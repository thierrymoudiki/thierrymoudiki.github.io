---
layout: post
title: "Quasi-randomized nnetworks in Julia, Python and R"
description: "Machine Learning Classification with quasi-randomized nnetworks in Julia, Python and R"
date: 2023-11-27
categories: [Python, R, Julia, QuasiRandomizedNN]
comments: true
---

`nnetsauce`, a package for quasi-randomized supervised learning (classification and regression), is currently available for [R and Python](https://github.com/Techtonique/nnetsauce). For more details on `nnetsauce`, you can read [these posts](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN). 

I've always wanted to port `nnetsauce` to the Julia language. However, in the past few years, there was a little timing _overhead_ (more precisely, a lag) when I tried to do that with Julia's `PyCall`, based on my Python source code. This _overhead_ seems to have 'disappeared' now. 

Julia language's `nnetsauce` is not a package **yet**, but you can already use `nnetsauce` in Julia. 

Here's how I did it on Ubuntu Linux: 

**Contents**

<ul>
 <li> <a href="#1---install-julia">1 - Install Julia</a> </li>
 <li> <a href="#2---example-using-a-nnetsauce-classifier-in-julia-language">2 - Example using a nnetsauce classifier in Julia language</a> </li>
</ul>


# 1 - Install Julia

See also: [https://www.digitalocean.com/community/tutorials/how-to-install-julia-programming-language-on-ubuntu-22-04](https://www.digitalocean.com/community/tutorials/how-to-install-julia-programming-language-on-ubuntu-22-04). 


Run (terminal):
```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
```

Run (terminal):
```bash
tar zxvf julia-1.9.4-linux-x86_64.tar.gz
```
Run (terminal)(This is VSCode, but use your favorite editor here):
```bash
code ~/.bashrc
```

Add to `.bashrc` (last line): 
```bash
export PATH="$PATH:julia-1.9.4/bin"
```

Run (terminal): 
```bash
source ~/.bashrc
```

Run (terminal): 
```bash
julia nnetsauce_example.jl
```

# 2 - Example using a nnetsauce classifier in Julia language

For Python user's, notice that this is _basically_ Python ^^

```julia
using Pkg
ENV["PYTHON"] = ""  # replace with your Python path
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.add("Conda")
Pkg.build("Conda")

using PyCall
using Conda

Conda.add("pip")  # Ensure pip is installed
Conda.pip_interop(true)  # Enable pip interop
Conda.pip("install", "scikit-learn")  # Install scikit-learn
Conda.pip("install", "jax")  # /!\ Only on Linux or macOS: Install jax
Conda.pip("install", "jaxlib")  # /!\ Only on Linux or macOS: Install jaxlib
Conda.pip("install", "nnetsauce")  # Install nnetsauce
Conda.add("numpy")

np = pyimport("numpy")
ns = pyimport("nnetsauce")
sklearn = pyimport("sklearn")


# 1 - breast cancer dataset

dataset = sklearn.datasets.load_breast_cancer()

X = dataset["data"]
y = dataset["target"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
test_size=0.2, random_state=123)

clf = ns.Ridge2MultitaskClassifier(n_hidden_features=9, dropout=0.43, n_clusters=1, 
lambda1=1.24023438e+01, lambda2=7.30263672e+03)

@time clf.fit(X=X_train, y=y_train) # timing?

print("\n\n Model parameters: \n\n")
print(clf.get_params())

print("\n\n Testing score: \n\n") # Classifier's accuracy
print(clf.score(X_test, y_test)) # Must be: 0.9824561403508771
print("\n\n")
```

```julia
# 2 - wine dataset

dataset = sklearn.datasets.load_wine()

X = dataset["data"]
y = dataset["target"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
test_size=0.2, random_state=123)

clf = ns.Ridge2MultitaskClassifier(n_hidden_features=15,
dropout=0.1, n_clusters=3, 
type_clust="gmm")

@time clf.fit(X=X_train, y=y_train) # timing?

print("\n\n Model parameters: \n\n")
print(clf.get_params())

print("\n\n Testing score: \n\n") # Classifier's accuracy
print(clf.score(X_test, y_test)) # Must be 1.0
print("\n\n")
```