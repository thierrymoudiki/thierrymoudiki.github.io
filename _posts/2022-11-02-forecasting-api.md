---
layout: post
title: "A web application for forecasting in Python, R, Ruby, C#, JavaScript, PHP, Go, Rust, Java, MATLAB, etc."
description: "A Forecasting API, with examples"
date: 2022-11-02
categories: [Python, R, Forecasting, Misc]
---

**Content**

<ul>
  <li> 0 - Intro </li>
  <li> 1 - Create an account </li>
  <li> 2 - Get a token for authentication </li>
  <li> 3 - Requests for forecasts </li>
  <li> 4 - On model calibration and cross-validation </li>
</ul>

## 0 - Intro

In this post, I'll describe an (work-in-progress) Application Programming Interface (API) for time series forecasting. An API is a system that can receive requests from your computer, to carry out given tasks on given resources, and return a response. This type of system is programming-language-agnostic. That means: it can be used with Python, JavaScript, PHP, R, Go, C#, Ruby, Rust, Java, MATLAB, Julia, and any other programming language speaking http. And therefore, it could be relatively easily integrated into existing workflows for uncertainty forecasting. I've used the following tools for building it: 

<ul>
  <li> Python's <a href="https://flask.palletsprojects.com/en/2.2.x/">Flask</a> for backend development </li>
  <li>  <a href="https://bootswatch.com/">Bootswatch</a> for the HTML/CSS theme </li>
  <li>  <a href="https://plotly.com/python/">Plotly</a> for interactive graphs </li>
  <li>  <a href="https://www.sqlalchemy.org/">SQLAlchemy</a>| <a href="https://www.postgresql.org/">PostgreSQL</a>  for database management </li>
  <li>  <a href="https://swagger.io/">Swagger</a> for API documentation </li>
  <li>  Salesforce's <a href="https://www.heroku.com/">Heroku</a> (Cloud Application Platform) for deploying the application </li>
</ul>


**The application is here:**

  [https://techtonique2.herokuapp.com/](https://techtonique2.herokuapp.com/)

![app's homepage]({{base}}/images/2022-11-02/2022-11-02-image1.png){:class="img-responsive"}

In the homepage ("/"), you can plot a time series by uploading a csv file, and pushing the button "Plot!". Some examples of input files are stored on GitHub, at [https://github.com/Techtonique/datasets/tree/main/time_series/univariate](https://github.com/Techtonique/datasets/tree/main/time_series/univariate). Hover your cursor over the graph to see the options available, like downloading as png, zooming in and out, etc. Let's describe the API now.

## 1 - Create an account:

In order to sign up, you can use your username or an email address. A valid email address is preferable, because usernames  duplicates aren't authorized in the database. You don't want to spend your precious time trying to figure out which username hasn't been registered yet! In addition, without a valid email address, you won't be notified for changes and improvements in the API (e.g new forecasting models added, bugs fixes...). 


**Using  `curl`**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"username":"tester_curl@example.com","password":"pwd"}' https://techtonique2.herokuapp.com/api/users
```


If you want to translate these commands from `curl` to your favorite programming language (Python, JavaScript, PHP, R, Go, C#, Ruby, Rust, Elixir, Java, MATLAB, Dart, CFML, Ansible URI, Strest), you can simply use the following website: [https://curlconverter.com/](https://curlconverter.com/). Of course, it's important to [choose a better password](https://www.lifewire.com/strong-password-examples-2483118)!


**Using  Python**

In the near future, there will be a user-friendly Python package encapsulating these steps.

```Python
import requests

headers = {
    # Already added when you pass json=
    # 'Content-Type': 'application/json',
}

json_data = {
    'username': 'tester_py@example.com',
    'password': 'pwd',
}

response = requests.post('https://techtonique2.herokuapp.com/api/users', headers=headers, json=json_data)
```

**Using  R**

In the near future, there will be a user-friendly R package encapsulating these steps.

```R
require(httr)

headers = c(
  `Content-Type` = 'application/json'
)

data = '{"username":"tester_r@example.com","password":"pwd"}'

res <- httr::POST(url = 'https://techtonique2.herokuapp.com/api/users', 
                  httr::add_headers(.headers=headers), body = data)

print(res)
```

Now that you have an account, you'll need a token to obtain time series forecasts. 
The username and password could be used for that purpose, but it's better to avoid 
sending them in every request. In any case, make sure that you're always sending 
requests to `https://` and not `http://`.

## 2 - Get a token for authentication

**Using  `curl`**

```bash
curl -u tester_curl@example.com:pwd -X GET https://techtonique2.herokuapp.com/api/token 
```

If you want to translate these commands from `curl` to your favorite programming language (Python, JavaScript, PHP, R, Go, C#, Ruby, Rust, Elixir, Java, MATLAB, Dart, CFML, Ansible URI, Strest), you can simply use the following website: [https://curlconverter.com/](https://curlconverter.com/).  


**Using  Python**

In the near future, there will be a user-friendly Python package encapsulating these steps.

```Python
response_token = requests.get('https://techtonique2.herokuapp.com/api/token', 
auth=('tester_py@example.com', 'pwd'))

token = response_token.json()['token']

print("\n")
print(f"token: {token}")
```


**Using  R**

In the near future, there will be a user-friendly R package encapsulating these steps.

```R
res_token <- httr::GET(url = 'https://techtonique2.herokuapp.com/api/token', 
                 httr::authenticate('tester_r@example.com', 'python22orpython33'))

print(res_token)

(token <- httr::content(res_token)$token)
```

## 3 - Requests for forecasts

We want to obtain 10 months-ahead forecasts 
for the number of accidental Deaths in the US from 1973 to 1978
, and 
a confidence level of 95% for prediction intervals. The forecasting method is 
Theta from [1] and [2], winner of the M3 competition. 

The token from section 2 (valid for 5 minutes)
will be used here for authentication. You should read [the API's  documentation](https://techtonique2.herokuapp.com/apidocs/) to understand each forecasting model's parameters.

**Using  `curl`**

```bash
curl -u eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpOCsOC8-I:x -F 'file=@/Users/t/Documents/datasets/time_series/univariate/USAccDeaths.csv' "https://techtonique2.herokuapp.com/api/theta?h=10&level=95"
```
If you want to translate these commands from `curl` to your favorite programming language (Python, JavaScript, PHP, R, Go, C#, Ruby, Rust, Elixir, Java, MATLAB, Dart, CFML, Ansible URI, Strest), you can simply use the following website: [https://curlconverter.com/](https://curlconverter.com/).  

`eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpOCsOC8-I` is a simplified example of the type of token which can be obtained in step 2. The csv file sent in the request must be stored on your computer. Examples of such files can be found [here](https://github.com/Techtonique/datasets/tree/main/time_series/univariate).

**Using  Python**

In the near future, there will be a user-friendly Python package encapsulating these steps.

```Python
params = {
    'h': '10',
    'level': '95',
}


files = {
    'file': open('./USAccDeaths.csv', 'rb') # File available at https://github.com/Techtonique/datasets/tree/main/time_series/univariate
}

response_forecast = requests.post('https://techtonique2.herokuapp.com/api/theta', 
files=files, params=params, auth=(token, 'x'))

print(response_forecast.json())

```

**Using  R**

In the near future, there will be a user-friendly R package encapsulating these steps.

```R
params = list(
  `h` = '10', # horizon
  `level` = '95' # level of confidence for prediction intervals
)

files = list(
  `file` = upload_file('./USAccDeaths.csv') # File available at https://github.com/Techtonique/datasets/tree/main/time_series/univariate
)

ptm <- proc.time()[3]
res_forecast <- httr::POST(url = 'https://techtonique2.herokuapp.com/api/theta', 
                           query = params, body = files, encode = 'multipart', 
                           httr::authenticate(token, 'x'))
proc.time()[3] - ptm

list_res <- httr::content(res_forecast)

# Plot results

# 1 - Results From R package forecast -----

require(forecast)

(forecast_object_R <- forecast::thetaf(USAccDeaths, h=10, level = 95))


# 2 - Results From a Python implementation (in the API) -----

h <- length(list_res$ranges)

forecast_object_api <- list()
forecast_object_api$mean <- forecast_object_api$upper <- forecast_object_api$lower <- ts(rep(0, h), 
                                                                             start = start(forecast_object_R$mean),
                                                                             frequency = frequency(forecast_object_R$x))

for (i in 1:h)
{
  forecast_object_api$mean[i] <- list_res$averages[[i]][[2]]
  forecast_object_api$lower[i] <- list_res$ranges[[i]][[2]]
  forecast_object_api$upper[i] <- list_res$ranges[[i]][[3]]
}

forecast_object_api$x <- forecast_object_R$x
forecast_object_api$method <- paste0(forecast_object_R$method, " (API)")
forecast_object_api$level <- forecast_object_R$level
forecast_object_api <- structure(forecast_object_api, class = "forecast")

print(forecast_object_api)
print(forecast_object_R)

# graphs

par(mfrow=c(1, 2))
plot(forecast_object_R)
plot(forecast_object_api)
```
![api responses]({{base}}/images/2022-11-02/2022-11-02-image2.png){:class="img-responsive"}

## 4 - On model calibration and cross-validation

Each model [in the API](https://techtonique2.herokuapp.com/apidocs/) has 2 additional parameters that we haven't discussed yet: 

  - `start_training`: Start training index for cross-validation
  - `n_training`: Size of training set window for cross-validation

Both of these parameters are to be used in a loop, in your favorite programming language, 
when you want to compare models' performance, or tune their hyperparameters (model calibration). 
You'd code a loop (with 3-seconds delays between each API call in the loop, because you're nice!) in which: 

<ul>
  <li><code>start_training</code> is incremented of 1 at each iteration, and <code>n_training</code>
  remains constant. 
  </li>
  <li> <code>n_training</code> is incremented of 1 at each iteration, and <code>start_training</code>
  remains constant. 
  </li>
</ul>
  
More on this (cross-validation and model calibration) in a future post. Stay tuned. 

<hr>

[1] Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition approach to forecasting. International journal of forecasting, 16(4), 521-530.

[2] Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method. International Journal of Forecasting, 19(2), 287-290.
 
