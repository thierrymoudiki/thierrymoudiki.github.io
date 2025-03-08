---
layout: post
title: "Word-Online: recreating Karpathy's char-RNN (with supervised linear online learning of word embeddings) for text completion" 
description: "R and Python implementations of word completion"
date: 2025-03-08
categories: [R, Python, LLMs]
comments: true
---

In this post, I implement a simple word completion model, based on [Karpathy's char-RNN](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), but using **supervised linear online learning of word embeddings**. More precisely, I use the [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) from `scikit-learn`, which is **a simple linear classifier that can be updated incrementally**.

Keep in mind that this is an illustrative example, based on a few words and small vocabulary. There are many, many ways to improve the model, and many other configurations could be envisaged. So, feel free to experiment and [extend this example](https://github.com/thierrymoudiki/word-online). Nonetheless, the grammatical structure of the generated text (don't generalize this result yet) is surprisingly good.

My 2-cents-non-scientific (?) extrapolation about this is, is that artificial _neural_ networks are not intrinsically better than other methods: it takes a model with high capacity, capable of learning and generalize well. 

Here is how to reproduce the example, assuming you named the file `word-online.py` (the repository is named [`word-online`](https://github.com/thierrymoudiki/word-online)):

```
uv venv venv --python=3.11
source venv/bin/activate
uv pip install -r requirements.txt
```


```
python word-online.py
```

`word-online.py` contains the following code:

# Python version 

```Python
import numpy as np
import gensim
import time  # Added for the delay parameter

from collections import deque
from tqdm import tqdm
from scipy.special import softmax
from sklearn.linear_model import SGDClassifier


# Sample text 
text = """Hello world, this is an online learning example with word embeddings.
          It learns words and generates text incrementally using an SGD classifier."""

def debug_print(x):
    print(f"{x}")

# Tokenization (simple space-based)
words = text.lower().split()
vocab = sorted(set(words))
vocab.append("<UNK>")  # Add unknown token for OOV words

# Train Word2Vec model (or load pretrained embeddings)
embedding_dim = 50  # Change to 100/300 if using a larger model
word2vec = gensim.models.Word2Vec([words], vector_size=embedding_dim, window=5, min_count=1, sg=0)

# Create word-to-index mapping
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Hyperparameters
context_size = 12  # Default 10, Words used for prediction context
learning_rate = 0.005
epochs = 10

# Prepare training data
X_train, y_train = [], []

for i in tqdm(range(len(words) - context_size)):
    context = words[i:i + context_size]
    target = words[i + context_size]
    # Convert context words to embeddings
    context_embedding = np.concatenate([word2vec.wv[word] for word in context])
    X_train.append(context_embedding)
    y_train.append(word_to_idx[target])

X_train, y_train = np.array(X_train), np.array(y_train)

# Initialize SGD-based classifier
clf = SGDClassifier(loss="hinge", max_iter=1, learning_rate="constant", eta0=learning_rate)

# Online training (stochastic updates, multiple passes)
for epoch in tqdm(range(epochs)):
    for i in range(len(X_train)):
        clf.partial_fit([X_train[i]], [y_train[i]], classes=np.arange(len(vocab)))

# 🔥 **Softmax function for probability scaling**
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stability trick
    return exp_logits / np.sum(exp_logits)


def sample_from_logits(logits, k=5, temperature=1.0, random_seed=123):
    """ Applies Top-K sampling & Temperature scaling """
    logits = np.array(logits) / temperature  # Apply temperature scaling
    probs = softmax(logits)  # Convert logits to probabilities
    # Select top-K indices
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= top_k_probs.sum()  # Normalize
    # Sample from Top-K distribution
    np.random.seed(random_seed)
    return np.random.choice(top_k_indices, p=top_k_probs)


def generate_text(seed="this is", length=20, k=5, temperature=1.0, random_state=123, delay=3):
    seed_words = seed.lower().split()

    # Ensure context has `context_size` words (pad with zero vectors if needed)
    while len(seed_words) < context_size:
        seed_words.insert(0, "<PAD>")

    context = deque(
        [word_to_idx[word] if word in word_to_idx else -1 for word in seed_words[-context_size:]],
        maxlen=context_size
    )

    generated = seed
    previous_word = seed

    for _ in range(length):
        # Generate embeddings, use a zero vector if word is missing
        context_embedding = np.concatenate([
            word2vec.wv[idx_to_word[idx]] if idx in idx_to_word else np.zeros(embedding_dim)
            for idx in context
        ])
        logits = clf.decision_function([context_embedding])[0]  # Get raw scores
        # Sample next word using Top-K & Temperature scaling
        pred_idx = sample_from_logits(logits, k=k, temperature=temperature)
        next_word = idx_to_word.get(pred_idx, "<PAD>")
        
        print(f"Generating next word: {next_word}")  # Added this line
        time.sleep(delay)  # Added this line
        
        if previous_word[-1] == "." and previous_word[-1] != "" and previous_word[-1] != seed:
          generated += " " + next_word.capitalize()
        else: 
          generated += " " + next_word
        previous_word = next_word
        context.append(pred_idx)

    return generated

# 🔥 Generate text
print("\n\n Generated Text:")
seed = "This is a"
print(seed)
print(generate_text(seed, length=12, k=1, delay=0)) # delay seconds for next word generation, optimal for delay=0 seconds 
```
```
100%|████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 12164.45it/s]
100%|███████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00,  8.34it/s]


 Generated Text:
This is a
Generating next word: classifier.
Generating next word: an
Generating next word: sgd
Generating next word: classifier.
Generating next word: and
Generating next word: generates
Generating next word: text
Generating next word: incrementally
Generating next word: using
Generating next word: an
Generating next word: sgd
Generating next word: classifier.
This is a classifier. An sgd classifier. And generates text incrementally using an sgd classifier.
```

![image-title-here]({{base}}/images/2025-03-08/2025-03-08-image1.gif)

# R version 

```R
%%R
library(reticulate)
library(progress)
library(stats)

# Initialize Python modules through reticulate
np <- import("numpy")
gensim <- import("gensim")
time <- import("time")  # Added for the delay parameter

# Sample text 
text <- "This is a model used for classification purposes. It applies continuous learning on word vectors, converting words into embeddings, learning from those embeddings, and gradually producing text through the iterative process of an SGD classifier."

debug_print <- function(x) {
  print(paste0(x))
}

# Tokenization (simple space-based)
words <- strsplit(tolower(text), "\\s+")[[1L]]
vocab <- sort(unique(words))
vocab <- c(vocab, "<UNK>")  # Add unknown token for OOV words

# Train Word2Vec model (or load pretrained embeddings)
embedding_dim <- 50L  # Change to 100/300 if using a larger model
word2vec <- gensim$models$Word2Vec(list(words), vector_size=embedding_dim, window=5L, min_count=1L, sg=0L)

# Ensure "<UNK>" is in the Word2Vec vocabulary
# This is the crucial step to fix the KeyError
if (!("<UNK>" %in% word2vec$wv$index_to_key)) {
  word2vec$wv$add_vector("<UNK>", rep(0, embedding_dim))  # Add "<UNK>" with a zero vector
}


# Create word-to-index mapping
word_to_idx <- setNames(seq_along(vocab) - 1L, vocab)  # 0-based indexing to match Python
idx_to_word <- setNames(vocab, as.character(word_to_idx))

# Hyperparameters
context_size <- 12L  # Default 10, Words used for prediction context
learning_rate <- 0.005
epochs <- 10L
    
# Prepare training data
X_train <- list()
y_train <- list()

pb <- progress_bar$new(total = length(words) - context_size)
for (i in 1L:(length(words) - context_size)) {
  context <- words[i:(i + context_size - 1L)]
  target <- words[i + context_size]
  # Convert context words to embeddings
  context_vectors <- lapply(context, function(word) as.array(word2vec$wv[word]))
  context_embedding <- np$concatenate(context_vectors)
  X_train[[i]] <- context_embedding
  y_train[[i]] <- word_to_idx[target]
  pb$tick()
}

# Initialize SGD-based classifier
sklearn <- import("sklearn.linear_model")
clf <- sklearn$SGDClassifier(loss="hinge", max_iter=1L, learning_rate="constant", eta0=learning_rate)

# Online training (stochastic updates, multiple passes)
pb <- progress_bar$new(total = epochs)
for (epoch in 1L:epochs) {
  for (i in 1L:length(X_train)) {
    # Use the list version for indexing individual samples
    clf$partial_fit(
      np$array(list(X_train[[i]])), 
      np$array(list(y_train[[i]])), 
      classes=np$arange(length(vocab))
    )
  }
  pb$tick()
}

# Softmax function for probability scaling
softmax_fn <- function(logits) {
  exp_logits <- exp(logits - max(logits))  # Stability trick
  return(exp_logits / sum(exp_logits))
}

sample_from_logits <- function(logits, k=5L, temperature=1.0, random_seed=123L) {
  # Applies Top-K sampling & Temperature scaling
  logits <- as.numeric(logits) / temperature  # Apply temperature scaling
  probs <- softmax_fn(logits)  # Convert logits to probabilities
  
  # Select top-K indices - ensure k doesn't exceed the length of logits
  k <- min(k, length(logits))
  sorted_indices <- order(probs)
  top_k_indices <- sorted_indices[(length(sorted_indices) - k + 1L):length(sorted_indices)]
  
  # Handle case where k=1 specially
  if (k == 1L) {
    return(top_k_indices)
  }
  
  top_k_probs <- probs[top_k_indices]
  # Ensure probabilities sum to 1
  top_k_probs <- top_k_probs / sum(top_k_probs)
  
  # Check if all probabilities are valid
  if (any(is.na(top_k_probs)) || length(top_k_probs) != length(top_k_indices)) {
    # If there are issues with probabilities, just return the highest probability item
    return(top_k_indices[which.max(probs[top_k_indices])])
  }
  
  # Sample from Top-K distribution
  set.seed(random_seed)
  return(sample(top_k_indices, size=1L, prob=top_k_probs))
}

generate_text <- function(seed="this is", length=20L, k=5L, temperature=1.0, random_state=123L, delay=3L) {
  seed_words <- strsplit(tolower(seed), "\\s+")[[1L]]
  
  # Ensure context has `context_size` words (pad with zero vectors if needed)
  while (length(seed_words) < context_size) {
    seed_words <- c("<PAD>", seed_words)
  }
  
  # Use a fixed-size list as a ring buffer
  context <- vector("list", context_size)
  for (i in 1L:context_size) {
    word <- tail(seed_words, context_size)[i]
    if (word %in% names(word_to_idx)) {
      context[[i]] <- word_to_idx[word]
    } else {
      context[[i]] <- -1L
    }
  }
  
  # Track position in the ring buffer
  context_pos <- 1L
  
  generated <- seed
  previous_word <- seed
  
  for (i in 1L:length) {
    # Generate embeddings, use a zero vector if word is missing
    context_vectors <- list()
    for (idx in unlist(context)) {
      if (as.character(idx) %in% names(idx_to_word)) {
        word <- idx_to_word[as.character(idx)]
        context_vectors <- c(context_vectors, list(as.array(word2vec$wv[word])))
      } else {
        context_vectors <- c(context_vectors, list(np$zeros(embedding_dim)))
      }
    }
    
    context_embedding <- np$concatenate(context_vectors)
    logits <- clf$decision_function(np$array(list(context_embedding)))[1L,]
    
    # Sample next word using Top-K & Temperature scaling
    pred_idx <- sample_from_logits(logits, k=k, temperature=temperature, random_seed=random_state+i)
    next_word <- if (as.character(pred_idx) %in% names(idx_to_word)) {
      idx_to_word[as.character(pred_idx)]
    } else {
      "<PAD>"
    }
    
    print(paste0("Generating next word: ", next_word))
    if (delay > 0) {
      time$sleep(delay)  # Added delay
    }
    
    if (substr(previous_word, nchar(previous_word), nchar(previous_word)) == "." && 
        previous_word != "" && previous_word != seed) {
      generated <- paste0(generated, " ", toupper(substr(next_word, 1, 1)), substr(next_word, 2, nchar(next_word)))
    } else {
      generated <- paste0(generated, " ", next_word)
    }
    
    previous_word <- next_word
    
    # Update context (ring buffer style)
    context[[context_pos]] <- pred_idx
    context_pos <- (context_pos %% context_size) + 1L
  }
  
  return(generated)
}
    
cat("\n\n Generated Text:\n")
seed <- "This classifier is"
cat(seed, "\n")
result <- generate_text(seed, length=2L, k=3L, delay=0L)  # delay seconds for next word generation
print(result)    
```
```R
Generated Text:
This classifier is 
[1] "Generating next word: for"
[1] "Generating next word: text"
[1] "This classifier is for text"
```

