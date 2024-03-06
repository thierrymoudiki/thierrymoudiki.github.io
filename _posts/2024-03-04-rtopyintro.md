---
layout: post
title: "rtopy (v0.1.1): calling R functions in Python"
description: "rtopy (v0.1.1): calling R functions in Python, based on the command line and 
text mining and caching tools."
date: 2024-03-04
categories: [Python, R]
comments: true
---

R code is not rendered properly below, but you can [open this notebook](https://colab.research.google.com/github/Techtonique/rtopy/blob/main/rtopy/demo/thierrymoudiki_20240304_rtopyintro.ipynb) or 

<span>
<a target="_blank" rel="noreferrer noopener" href="https://colab.research.google.com/github/Techtonique/rtopy/blob/main/rtopy/demo/thierrymoudiki_20240304_rtopyintro.ipynb">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</span>

[`rtopy`](https://github.com/Techtonique/rtopy) is a Python package that allows you to call R functions in Python. 
There are other packages doing something similar, but in addition to have a different interface, under the hood,  `rtopy` explicitly uses R at the command line, plus **text mining** and **caching** tools. It's a work in progress, and you can find some examples of use below. 

# 1 - Install and import `rtopy`

R code is not rendered properly below, but you can [open this notebook](https://colab.research.google.com/github/Techtonique/rtopy/blob/main/rtopy/demo/thierrymoudiki_20240304_rtopyintro.ipynb) or 

<span>
<a target="_blank" rel="noreferrer noopener" href="https://colab.research.google.com/github/Techtonique/rtopy/blob/main/rtopy/demo/thierrymoudiki_20240304_rtopyintro.ipynb">
  <img style="width: inherit;" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</span>


```python
!pip install rtopy
```


```python
import rtopy as rp
```

# 2 - R codes


```python
# an R function that returns the product of an arbitrary number of arguments
# notice the (mandatory) double braces around the R function's code
# and the a semi-colon (';') after each instruction
r_code1 = f'''my_func <- function(arg1=NULL, arg2=NULL, arg3=NULL, arg4=NULL, arg5=NULL) {{
                args <- c(arg1, arg2, arg3, arg4, arg5);
                args <- args[!sapply(args, is.null)];
                result <- prod(args);
                return(result)
              }}
              '''

# an R function that returns the sum of an arbitrary number of arguments
r_code2 = f'''my_func <- function(arg1=NULL, arg2=NULL, arg3=NULL, arg4=NULL, arg5=NULL) {{
            args <- c(arg1, arg2, arg3, arg4, arg5);
            args <- args[!sapply(args, is.null)];
            result <- sum(args);
            return(result)
          }}
         '''

# an R function that returns a list of vectors
r_code3 = f'''my_func <- function(arg1, arg2) {{
            list(x = mtcars[, 'mpg'], y = mtcars[, arg1], z = mtcars[, arg2])
          }}
         '''

# an R function that returns a vector
r_code4 = f'''my_func <- function(arg1=NULL, arg2=NULL, arg3=NULL) {{
            args <- c(arg1, arg2, arg3);
            args <- args[!sapply(args, is.null)];
            print(args);
            return(as.vector(args))
          }}
         '''

# an R function that returns a list of matrices
# won't work for named rows
r_code5 = f'''my_func <- function(arg1, arg2) {{
            X <- as.matrix(mtcars);
            colnames(X) <- NULL;
            rownames(X) <- NULL;
            list(x = X[, 1], y = X[, c(arg1, arg2)])
          }}
         '''

# an R function that returns a list of vector, matrix and scalar
r_code6 = f'''my_func <- function(arg1, arg2) {{
            X <- as.matrix(mtcars);
            colnames(X) <- NULL;
            rownames(X) <- NULL;
            list(x = X[, 1], y = X[, c(arg1, arg2)], z = 5)
          }}
         '''
```

# 3 - Examples


```python
print(rp.callfunc(r_code=r_code1, type_return="int", arg1=3, arg2=5, arg3=2))
print(rp.callfunc(r_code=r_code2, type_return="float", arg1=1.5, arg2=2.5, arg4=4.5))
print(rp.callfunc(r_code=r_code2, type_return="float", arg1=3.5, arg3=5.3, arg4=4.2))
```

    30
    8.5
    13.0



```python
res = rp.callfunc(r_code=r_code3, type_return="dict", arg1=2, arg2=3)
print(res)
```

    {'x': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4], 'y': [6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 6.0, 8.0, 4.0], 'z': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0]}



```python
res2 = rp.callfunc(r_code=r_code3, type_return="dict", arg1="cyl", arg2="disp")
print(res2)
```

    {'x': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4], 'y': [6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 6.0, 8.0, 4.0], 'z': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0]}



```python
res3 = rp.callfunc(r_code=r_code3, type_return="dict", arg1="cyl", arg2=3)
print(res3)
```

    {'x': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4], 'y': [6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 6.0, 8.0, 4.0], 'z': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0]}



```python
res4 = rp.callfunc(r_code=r_code5, type_return="dict", arg1=2, arg2=3)
print(res4)
```

    {'x': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4], 'y': [[6.0, 160.0], [6.0, 160.0], [4.0, 108.0], [6.0, 258.0], [8.0, 360.0], [6.0, 225.0], [8.0, 360.0], [4.0, 146.7], [4.0, 140.8], [6.0, 167.6], [6.0, 167.6], [8.0, 275.8], [8.0, 275.8], [8.0, 275.8], [8.0, 472.0], [8.0, 460.0], [8.0, 440.0], [4.0, 78.7], [4.0, 75.7], [4.0, 71.1], [4.0, 120.1], [8.0, 318.0], [8.0, 304.0], [8.0, 350.0], [8.0, 400.0], [4.0, 79.0], [4.0, 120.3], [4.0, 95.1], [8.0, 351.0], [6.0, 145.0], [8.0, 301.0], [4.0, 121.0]]}



```python
print(rp.callfunc(r_code=r_code4, type_return="list", arg1=3.5, arg2=5.3))
print(rp.callfunc(r_code=r_code4, type_return="list", arg1=3.5, arg2=5.3, arg3=4.1))
```

    [3.5, 5.3]
    [3.5, 5.3, 4.1]



```python
res5 = rp.callfunc(r_code=r_code6, type_return="dict", arg1=2, arg2=3)
print(res5)
```

    {'x': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4], 'y': [[6.0, 160.0], [6.0, 160.0], [4.0, 108.0], [6.0, 258.0], [8.0, 360.0], [6.0, 225.0], [8.0, 360.0], [4.0, 146.7], [4.0, 140.8], [6.0, 167.6], [6.0, 167.6], [8.0, 275.8], [8.0, 275.8], [8.0, 275.8], [8.0, 472.0], [8.0, 460.0], [8.0, 440.0], [4.0, 78.7], [4.0, 75.7], [4.0, 71.1], [4.0, 120.1], [8.0, 318.0], [8.0, 304.0], [8.0, 350.0], [8.0, 400.0], [4.0, 79.0], [4.0, 120.3], [4.0, 95.1], [8.0, 351.0], [6.0, 145.0], [8.0, 301.0], [4.0, 121.0]], 'z': 5.0}


![xxx]({{base}}/images/2024-03-04/2024-03-04-image1.png){:class="img-responsive"}  