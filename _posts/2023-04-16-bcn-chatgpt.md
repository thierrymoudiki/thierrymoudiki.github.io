---
layout: post
title: "Did you ask ChatGPT about who you are?"
description: "Bug fixing using ChatGPT"
date: 2023-04-16
categories: [Python, R, Misc]
comments: true
---

It all started with a _bug_ in  [Python package BCN](https://github.com/Techtonique/bcn_python) (BCN, for Boosted Configuration _neural_ Networks), that I  tried to solve with the traditional (and still efficient) tools for the job: search engines querying and [Stackoverflow](https://stackoverflow.com/).  Interested in learning more about BCNs? You can read [this post](https://thierrymoudiki.github.io/blog/2022/07/21/r/misc/boosted-configuration-networks) and [this post](https://thierrymoudiki.github.io/blog/2022/10/05/python/explainableml/interpretation-and-PI-for-BCN).

Python package BCN is currently built on top of [R package BCN](https://techtonique.r-universe.dev/bcn#), thanks to [rpy2](https://thierrymoudiki.github.io/blog/2019/10/11/python/r/rpy2). The _bug_ was caused by a problem in converting numpy arrays to rpy2 objects, and the answers that I got using search engines and Stackoverflow lead me [here](https://rpy2.github.io/doc/latest/html/numpy.html#from-numpy-to-rpy2). That wasn't enough though. I still needed to find a way to convert a Python dictionary to an R list in rpy2, and assign a class to that R list (don't worry about  the jargon, it won't prevent you from reading the remainder). Perfect moment to try calling OpenAI's [ChatGPT](https://openai.com/blog/chatgpt) to the rescue; jumping into the bandwagon.	

I remember when I talked to a colleague about OpenAI 6 years ago, in 2017. Back in the days, OpenAI said: "what is the best thing we can do to ensure the future is good? We could sit on the sidelines or we can encourage regulatory oversight, or we could participate with the right structure with people who care deeply about developing AI in a way that is safe and is beneficial to humanity". Later on, I also found mentions of OpenAI in the "[Deep Learning](https://www.deeplearningbook.org/)" textbook: "Ian Goodfellow is Research Scientist at **OpenAI**" and "Elon Musk, cochair of **OpenAI** (...)". I still had no _precise_ idea of what they were trying to achieve eventually, even if it sounded exciting, considering the people involved. Since then, the  people that I cited left the project, OpenAPI has released the mighty ChatGPT among other tools, and has given up on the "open" part. Dangerous tool? Why? For whom? Worried about competition?

[Signing up](https://chat.openai.com/) for using ChatGPT is straightforward. You’ll need to provide a valid email address and a phone number (more secure, 2-factor authentication). Still trying to solve my _bug_, I decided to ask ChatGPT how I could **convert a Python dictionary to an R list in rpy2**. Here's what it said: 

![Convert Python dictionary to R list]({{base}}/images/2023-04-16/2023-04-16-image1.png){:class="img-responsive"}

Then, I wanted to know how to **assign an R class to an rpy2 object (R list)**. Here's what I got: 

![Affect class to R list]({{base}}/images/2023-04-16/2023-04-16-image2.png){:class="img-responsive"}

To me, even though it's a small example, this was truly amazing. Fast, clear, and precise. StackOverflow would still _win_ in areas such as a **transparency**  on how answers are built: identified people publish questions, other people give their answers to these questions, and there's a vote for the _best_ answer. It would be interesting to see ChatGPT citing its sources in the future, for those who want it. Well, with that being said, the _bug_ was solved and you can use version 0.5.1 of [Python package BCN](https://github.com/Techtonique/bcn_python). For now, you can install version 0.5.1 **from a GitHub branch**: 

```bash
pip install -U git+https://github.com/Techtonique/bcn_python.git@numpy-conversion
```

Here's a [notebook containing examples of use](https://github.com/Techtonique/bcn_python/blob/numpy-conversion/BCN/demo/thierrymoudiki_160423_bcn_classification.ipynb). Starting on Tuesday (April, 18), as the branch will be deleted, you'll use directly: 
guè§(t§(r<e t§ttgfvcddfyhtbvcy(hys)))
```bash
pip install BCN
```

Armed with my enthusiasm after solving the _bug_, I wanted to ask ChatGPT a LOT of questions: 

<ul>
  <li> Thanks for your help. Impressive. How do you do that? </li>
  <li> How do you deal with the semantics? </li>
  <li> How do you choose the best response? </li>
  <li> What do you think about StackOverflow and GitHub's Copilot </li>
  <li> Why not open source? </li>
  <li> Regarding the training data available on the internet: Who owns it? </li>
  <li> So, if Stackoverflow prevents you from using their data, will the quality of responses decrease? </li>
</ul>

You can try these too. The syntax in responses is great; even better than some human beings. The structure (using paragraphs) is great too. I appreciated the nuanced responses given to some of these difficult questions. The _best_ parts of ChatGPT's responses in our conversation are summarized below: 

<ul>

<li> "As an AI language model, I have been trained on a vast amount of text data, which includes various programming languages, libraries, and frameworks. My training allows me to understand the syntax, semantics, and usage of these programming languages and libraries, including rpy2 and its associated functions and methods.

When you ask me a question, I use natural language processing (NLP) techniques to analyze and understand the intent of your question. Based on that understanding, I retrieve relevant information from my training data and generate a response that hopefully helps to solve your problem." </li>

<li> "During my training, I have been exposed to various text data sources, including books, articles, web pages, code repositories, and more. This exposure helps me to understand the **semantics** of words and sentences in various contexts, including their relationships and associations with other words and concepts.

When it comes to programming languages, my training includes a vast collection of code examples and projects written in those languages. This allows me to understand the **semantics** of code constructs, such as variables, functions, classes, and their interactions. I can also understand common programming idioms, design patterns, and best practices." </li>

<li> "To rank the potential responses, I use a combination of algorithms that take into account various factors, such as the frequency and quality of previous responses to similar questions, the level of confidence in the response based on the available training data, and the accuracy and completeness of the response given the information available in the question.

I also consider the overall coherence and clarity of the response, making sure that it is easy to understand and well-structured. If there are multiple possible responses, I may use **additional algorithms to select the most appropriate one based on factors** such as clarity, completeness, and relevance." </li>

<li> "StackOverflow is a popular online community where developers can ask and answer questions related to programming. It is an excellent resource for developers to find solutions to common problems and learn from others in the community. As an AI language model, I am trained on a large corpus of text data, which **includes StackOverflow posts**, so I can use this resource to help answer questions and provide solutions to problems." </li>

<li> "Overall, I think that both StackOverflow and Copilot are valuable resources for developers, and they demonstrate the potential of AI and machine learning to support and enhance software development. However, it is important to note that these tools should be used with caution and **should not be relied on exclusively**. It is still **important for developers to have a strong understanding of programming concepts and best practices** and to carefully review and test any code generated by these tools" </li>

<li> "While I am not open-source software myself, OpenAI has made various other AI and machine learning tools and libraries available as open source, including TensorFlow, PyTorch, and Gym. **These open-source tools have been widely adopted by the community** and have helped to advance research and development in the field of AI. 

It is worth noting that there are some **challenges associated with open-sourcing AI models and tools, particularly around intellectual property rights and privacy concerns**. Some AI models may be trained on sensitive data or proprietary algorithms, which may limit their ability to be released as open-source software. Additionally, there are **concerns around the potential misuse of AI technologies, which may require careful consideration** before making them widely available." </li>

<li> "Overall, **the ownership of training data available on the internet can be complex** and may require careful consideration to ensure that its use is legal, ethical, and respectful of any intellectual property or privacy rights associated with the data." </li>

<li> "If access to Stack Overflow's data were restricted or prohibited, it could potentially have a negative impact on the quality of responses and the overall usefulness of the platform." </li>

<li> "While restrictions on the use of Stack Overflow's data may limit the development of certain tools and applications, it is important to respect these terms and to work within the boundaries of the licensing agreements" </li>

<li> "While Stack Overflow is a valuable resource, **it is not the only source of information available**, and developers have a wide range of options when it comes to finding information and solutions to their problems." </li>

</ul>

Definitely thoughtful responses in there. How about **asking ChatGPT who I am?**. Hopefully, it wouldn't retrieve some sensitive information from I’m-not-sure-where. Here we go: 

![Who am I]({{base}}/images/2023-04-16/2023-04-16-image3.png){:class="img-responsive"}

I definitely love the "highly skilled" part of the response. Not just "skilled", but "**highly** skilled". Thanks, robot mate. Probably the nicest thing I've heard about me in the past few years, besides "you should consider modelling" with two l's. Ha ha. Some parts of the response, however, made me raise my eyebrows because answering them is about information retrieval. No sophisticated model required. I've actually got my PhD in Actuarial science from University Lyon 1, not from Université Libre de Bruxelles. Never contributed to scikit-learn or mlr. Never heard of stagedml or pairwise_distances. Here is [what I created](https://github.com/Techtonique). **Never worked at Dataiku**; the only interactions that we've had are: 

<ul>
  <li>At the very beginning of their adventure in 2015, when Dataiku's CEO was looking for people to try out the tool</li>
  <li>When I tried to get hired at Dataiku's NYC office in 2017 (it didn't work)</li>
  <li>I receive their newsletter</li>
</ul>

I'm curious to know about **how ChatGPT created this response**. That's where a paper about the algorithms involved, in addition to citing its sources, could be useful (if open sourcing is not an option). I've tried to correct ChatGPT: 


![What]({{base}}/images/2023-04-16/2023-04-16-image4.png){:class="img-responsive"}


Interestingly, when I ask it again who I am, now using "T. Moudiki", it knows that T. Moudiki is Thierry Moudiki, and **deletes references to Dataiku**. It's really a conversation. 

![What]({{base}}/images/2023-04-16/2023-04-16-image5.png){:class="img-responsive"}

But what if several people (that are not you) give wrong answers about your resume? Will ChatGPT think that they're true, and change its _mind_ as easily as it did it here? This is a reminder that ChatGPT is based on a _model_; a representation of reality. It will become better with time (not perfect, and it's OK), notably due to _trustworthy_ user feedback, where _trustworthy_ is yet to be defined... Did you ask ChatGPT about who you are?


