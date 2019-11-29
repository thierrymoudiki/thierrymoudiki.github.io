---
layout: post
title: "Benchmarking the querier's verbs"
description: Benchmarking the verbs from Python package querier
date: 2019-11-29
---



You may have noticed the instruction `qr.join.cache.clear()`. This, is because each `querier` verb has a cache, a dictionary (Python `dict`) that you can _manipulate_ accordingly. Only the first function call might be time-consuming (or not!), but subsequent calls will be much, much faster. Thus, memory usage must be monitored in some cases, such as here. 
