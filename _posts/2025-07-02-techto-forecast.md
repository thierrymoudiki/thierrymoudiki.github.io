---
layout: post
title: "Calling =TECHTO_FORECAST for forecasting in Excel is just a matter of copying and pasting"
description: "Calling =TECHTO_FORECAST for forecasting in Excel is just a matter of copying and pasting"
date: 2025-07-02
categories: [R, Python]
comments: true
---

`=TECHTO_FORECAST` in Excel is based on R and Python code from the [Techtonique](https://github.com/Techtonique) project's univariate and multivariate time series forecasting functions.

Here's a unified way to use `=TECHTO_FORECAST` directly and easily in Excel with xlwings Lite:


- If you haven't done it yet, install xlwings Lite [https://lite.xlwings.org/installation](https://lite.xlwings.org/installation) and click on its icon in the top right corner to open the editor
  
- Get a token from [https://www.techtonique.net/token](https://www.techtonique.net/token) 
  
-  In the xlwings Lite editor, click on the dropdown menu > Environment variables 
  
- Name the environment variable `TECHTONIQUE_TOKEN`, provide its token value, and **push "Save"** 
  
- In the main xlwings Lite code editor (`main.py`), **paste**:

```python
from techtonique_apis import TECHTO_FORECAST
```



- In xlwings Lite's requirements.txt, **paste**:

```bash
xlwings==0.33.14  # required
python-dotenv==1.1.0  # required
pyodide-http  # required
black  # required
techtonique_apis
```



- Click Restart (or click "Restart" in the dropdown menu) 
  
- Now you can close the xlwings Lite task pane, and **Type =TECHTO_FORECAST in a cell** and select an input range (link to examples [https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx](https://github.com/Techtonique/techtonique-apis/blob/main/examples/excel_formulas.xlsx)). as long as the token is valid, it will work. It's valid for 1 hour. 
  
- Help for each function is here: https://docs.techtonique.net/techtonique_api_py/techtonique_apis.html. Or use Excel help for the function 
  
- If you suddenly get VALUE error every time, check if the token is still valid (https://www.techtonique.net/token). It's valid for 1 hour 

</ol>

![image-title-here]({{base}}/images/2025-07-02/2025-07-02-image1.gif){:class="img-responsive"}    

