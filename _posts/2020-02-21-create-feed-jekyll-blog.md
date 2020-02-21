---
layout: post
title: "Create a specific feed in your Jekyll blog"
description: How to create a specific feed in your Jekyll blog
date: 2020-02-21
categories: [Misc]
---

To those who are using [_Jekyll_](https://jekyllrb.com/) for blogging on their websites: this post is about __how I created a specific blog feed__ in mine. Indeed sometimes, you may want to __subscribe to a blog aggregator that only accepts a given subject__, and still be able to blog about other subjects. 

I found _Jekyll_ to be helpful for adding numerous features on my website from scratch. Among those features, I [created this blog ]({% post_url 2019-09-04-using-jekyll %}), which used to be a monolithic HTML page. Of course, if you're not into _programming_, there are many commercial drag-and-drop alternatives out there.

_Liquid_ is the programming language used for our purpose, in combination with [xml](https://www.w3schools.com/xml/). _Liquid_ looks and _feels_ like [Ruby](https://www.ruby-lang.org/en/), a programming language that I love, and a concise documentation of this dialect can be found [online](https://shopify.github.io/liquid/). 

The first step is to __create blog categories__ in each post available in `_posts/` that you'd want to be categorized. For the post you're reading now, it's category `Misc`. 

![image-title-here]({{base}}/images/2020-02-21/2020-02-21-image1.png){:class="img-responsive"}

If there are more than one categories for a given post, it would be instead something as:

```ruby
categories: [cat1, cat2]
```
Then in your blog's configuration file `_config.yml`, you'll need to declare that __links to each post contain information about their categories__ too:

![image-title-here]({{base}}/images/2020-02-21/2020-02-21-image2.png){:class="img-responsive"}

Having added categories in posts and modified the configuration file, now create a file `feed_cat.xml` in website's root, where `cat` is the __category of interest__ for the aggregator. In my case, I created [`feed_R.xml`](https://github.com/thierrymoudiki/thierrymoudiki.github.io/blob/master/feed_R.xml) manually, with the following content excerpt:

![image-title-here]({{base}}/images/2020-02-21/2020-02-21-image3.png){:class="img-responsive"}

In the source file `feed_R.xml`, `<xml>` tags indicates that it's a `.xml` file, with english characters encoding. Our feed is delimited by `<feed>` tags, and each post content by `<entry>` tags. In the file's body, I __loop through all the categories__ available on my website, filter on category "R", and fetch blog posts available in this category -- limited to 5 posts fetched. I stop looping on categories as soon as the category of interest is found and its 5 most recent posts are fetched.

For more details into what each instruction in [`feed_R.xml`](https://github.com/thierrymoudiki/thierrymoudiki.github.io/blob/master/feed_R.xml) does, you can refer to [Liquid docs](https://shopify.github.io/liquid/) and [xml tutorial](https://www.w3schools.com/xml/). And if there's __any other way of doing what we just did__, I'd be happy to hear about it. Once `feed_cat.xml`'s been created in website's root, you should verify your feed's syntax with this online tool -- otherwise, it may not work: 

[https://validator.w3.org/feed/](https://validator.w3.org/feed/)

If __everything in the syntax is fine__, you should get a message like the one below: 

![image-title-here]({{base}}/images/2020-02-21/2020-02-21-image4.png){:class="img-responsive"}

You can now submit `feed_cat.xml` to the aggregator interested in `cat`. 