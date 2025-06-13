# 0 - Download data


```python
!wget https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv
```

    --2025-06-13 19:41:16--  https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 995 [text/plain]
    Saving to: ‘raa.csv.1’
    
    raa.csv.1           100%[===================>]     995  --.-KB/s    in 0s      
    
    2025-06-13 19:41:16 (54.7 MB/s) - ‘raa.csv.1’ saved [995/995]
    



```python
%load_ext rpy2.ipython
```

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython


Note that you can use https://curlconverter.com/ to translate the following request in your favorite programming language.




```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDk4NDczMDF9.UF3Ob0HgnykT-mFaGqtTFB6bu9QDTfc6wOsjzewzoiQ" \
-F "file=@./raa.csv;type=text/csv" \
"https://www.techtonique.net/mlreserving?method=RidgeCV" > ridge_ibnr.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  2071  100   893  100  1178    896   1182 --:--:-- --:--:-- --:--:--  2077


Note that you can use https://curlconverter.com/ to translate the following request in your favorite programming language.




```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDk4NDczMDF9.UF3Ob0HgnykT-mFaGqtTFB6bu9QDTfc6wOsjzewzoiQ" \
-F "file=@./raa.csv;type=text/csv" \
"https://www.techtonique.net/mlreserving?method=lightgbm" > lightgbm_ibnr.json

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  2069  100   891  100  1178   1267   1675 --:--:-- --:--:-- --:--:--  2938


Note that you can use https://curlconverter.com/ to translate the following request in your favorite programming language.




```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDk4NDczMDF9.UF3Ob0HgnykT-mFaGqtTFB6bu9QDTfc6wOsjzewzoiQ" \
-F "file=@./raa.csv;type=text/csv" \
"https://www.techtonique.net/mlreserving?method=xgboost" > xgboost_ibnr.json

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  2010  100   832  100  1178   1123   1591 --:--:-- --:--:-- --:--:--  2712


Note that you can use https://curlconverter.com/ to translate the following request in your favorite programming language.




```python
!curl -X POST \
-H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NGY3ZDE3Ny05OWQ0LTQzNDktOTc1OC0zZTBkOGVkYWZkYWUiLCJlbWFpbCI6InRoaWVycnkubW91ZGlraS50ZWNodG9uaXF1ZUBnbWFpbC5jb20iLCJleHAiOjE3NDk4NDczMDF9.UF3Ob0HgnykT-mFaGqtTFB6bu9QDTfc6wOsjzewzoiQ" \
-F "file=@./raa.csv;type=text/csv" \
"https://www.techtonique.net/mlreserving?method=ExtraTreesRegressor" > et_ibnr.json

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  2076  100   898  100  1178   1071   1405 --:--:-- --:--:-- --:--:--  2477



```r
%%R

install.packages("jsonlite")
library("jsonlite")
```


    Installing package into ‘/usr/local/lib/R/site-library’
    (as ‘lib’ is unspecified)
    trying URL 'https://cran.rstudio.com/src/contrib/jsonlite_2.0.0.tar.gz'
    Content type 'application/x-gzip' length 1055849 bytes (1.0 MB)
    ==================================================
    downloaded 1.0 MB
    
    
    The downloaded source packages are in
    	‘/tmp/Rtmp4Rnpdc/downloaded_packages’




```r
%%R

(ridge_ibnr <- fromJSON("./ridge_ibnr.json"))
(lightgbm_ibnr <- fromJSON("./lightgbm_ibnr.json"))
(xgboost_ibnr <- fromJSON("./xgboost_ibnr.json"))
(et_ibnr <- fromJSON("./et_ibnr.json"))
```

    $Origin
     [1] 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990
    
    $Ultimate
     [1] 17079.00 21189.25 26681.43 28958.43 35834.39 23260.73 21003.31 24944.61
     [9] 14434.00 15609.66
    
    $IBNR
    [1]   412.2454  1043.4278  1646.4278  2273.3870  3989.7307  5783.3109  7587.6089
    [8]  9910.0009 15609.6626
    
    $`IBNR 95%`
    [1]  1163.717  2944.499  4645.851  6414.730 11253.996 16310.950 21398.112
    [8] 27945.501 44013.410
    
    $`Ultimate Lo95`
     [1] 17079.000 20922.619 26006.919 27894.220 34365.021 20683.340 17268.063
     [8] 20044.589  8034.937  5531.906
    
    $`Mean Ultimate`
    [1] 228994.8
    
    $`Mean IBNR`
    [1] 48255.8
    
    $`Total IBNR 95%`
    [1] 136090.8
    



```r
%%R

plot_ibnr <- function(ibnr_data, ml_model="")
{
  # Remove the first origin year
origin_years <- ibnr_data$Origin[-1]

# Extract IBNR (mean) and remove the first value
(ibnr_mean <- ibnr_data$IBNR)

# Extract IBNR 95% upper bound and remove the first value
(ibnr_upper <- ibnr_data$`IBNR 95%`)  # assuming [[3]] is "IBNR 95%"

# Plotting - initialize empty plot with correct dimensions
plot(origin_years, ibnr_upper, type = "l",
     ylim = c(min(ibnr_mean), max(ibnr_upper)),
     col="red", lwd=2,
     xlab = "Origin Year", ylab = "IBNR",
     main = paste0("IBNR - Mean and Upper 95% Bound \n for", ml_model))
# Draw shaded area between mean and upper bound
polygon(x = c(origin_years, rev(origin_years)),
        y = c(ibnr_mean, rev(ibnr_upper)),
        col = rgb(0, 0, 1, 0.2), border = NA)
# Add mean IBNR line
lines(origin_years, ibnr_mean, col = "blue", lwd = 2)
# Optional: add grid
grid()
# Optional: add legend
legend("topleft", legend = c("Mean IBNR", "Upper 95%"),
       col = c("blue", "red"),
       lty = 1, pch = c(NA, NA), lwd = 2)
}
```


```r
%%R

par(mfrow=c(2, 2))
plot_ibnr(ridge_ibnr, ml_model="Ridge")
plot_ibnr(lightgbm_ibnr, ml_model="LightGBM")
plot_ibnr(xgboost_ibnr, ml_model="XGBoost")
plot_ibnr(et_ibnr, ml_model="ExtraTrees")
```


    
![png](2025_06_13_techtonique_triangle_files/2025_06_13_techtonique_triangle_14_0.png)
    

