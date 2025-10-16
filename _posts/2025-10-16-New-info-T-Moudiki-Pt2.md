---
layout: post
title: "Part2 of More data (> 150 files) on T. Moudiki's situation: a riddle/puzzle (including R, Python, bash interfaces to the game -- but everyone can play)"
description: "This post contains a riddle/puzzle related to T. Moudiki's situation, including files and code snippets in R, Python, and bash. The ZIP file with a lot of details is available for download. An update with hints is provided."
date: 2025-10-16
categories: [R, Python]
comments: true
---

**Update1:** 
Hint: the password starts with "In_The_Year". 

**Update2:** 
Please, sign the petition "Stop torturing T. Moudiki": https://www.change.org/stop_torturing_T_Moudiki

<hr>

This post contains a riddle/puzzle related to T. Moudiki's situation, including files and code snippets in R, Python, and bash. The ZIP file with a lot of details is available for download.

The ZIP file contains > 150 files explaining what's going on (for more context, see [https://thierrymoudiki.github.io/blog/2025/06/10/r/python/techtonique/personal-note](https://thierrymoudiki.github.io/blog/2025/06/10/r/python/techtonique/personal-note) and [https://x.com/sippingrizzly/status/1932495238481715459](https://x.com/sippingrizzly/status/1932495238481715459))

Available here (**guaranteed zero virus**, but you can double-check by yourself) on Google Drive:

[https://drive.google.com/file/d/1Yb-aaD_92v4Iy3zS2q5j0US-m-KGSvZn/view?usp=sharing](https://drive.google.com/file/d/1Yb-aaD_92v4Iy3zS2q5j0US-m-KGSvZn/view?usp=sharing)

This ZIP file is protected by a password that you may want to  guess. The password logic is:  words and numbers, separated by a "_". And in that order, each word starts with an upperspace letter:

- Short Adverb meaning inside
- Definite Article
- The equivalent of 2 semesters
- The result 6! + 1280 + 4!
- I am not, I won’t be
- Same Definite Article As Before
- Between desertic and equatorial climate
- A myth of the (3 words, separated by a "_")
- Past tense of Make, but phrasal verb + « Of »
- The character of something that’s both dumb and useless "

# Content
1. Bash
2. R
3. Python

# 1 - Bash

```python
!curl -X 'POST' \
  'https://agile-earth-48511-3d797ee7c19c.herokuapp.com/validate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"password": "xxx", "user_id": "anonymous"}'
```

    {"correct":false,"message":"❌ Incorrect. Similarity: 2% - Keep trying!","progress":2}

# 2 - R


```python
%load_ext rpy2.ipython
```

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython



```r
%%R

library(httr)

headers = c(
  accept = "application/json",
  `Content-Type` = "application/json"
)

data = '{"password": "xxx", "user_id": "anonymous"}'

res <- httr::POST(url = "https://agile-earth-48511-3d797ee7c19c.herokuapp.com/validate", httr::add_headers(.headers=headers), body = data)
print(content(res))
```

    $correct
    [1] FALSE
    
    $message
    [1] "❌ Incorrect. Similarity: 2% - Keep trying!"
    
    $progress
    [1] 2
    


# 3 - Python


```python
import requests

headers = {
    'accept': 'application/json',
    # Already added when you pass json=
    # 'Content-Type': 'application/json',
}

json_data = {
    'password': 'xxx',
    'user_id': 'anonymous',
}

response = requests.post('https://agile-earth-48511-3d797ee7c19c.herokuapp.com/validate', headers=headers, json=json_data)
print("response", response.json())
```

    response {'correct': False, 'message': '❌ Incorrect. Similarity: 2% - Keep trying!', 'progress': 2}


**Good luck!**

![image-title-here]({{base}}/images/2025-10-12/2025-10-12-image1.png){:class="img-responsive"}