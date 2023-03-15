# Sentiment Analysis on Large Movie Review Dataset

This project demonstrates sentiment analysis on the Large Movie Review Dataset using two different approaches: Bag of Words and Word Embeddings with LSTM.

## Citation:
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142-150). Portland, Oregon, USA: Association for Computational Linguistics. Retrieved from http://www.aclweb.org/anthology/P11-1015

## Overview

The Large Movie Review Dataset contains movie reviews labeled as either positive or negative. The goal of this project is to build models that can accurately classify a review as positive or negative based on its text content.

### Approach 1: Bag of Words

In the first approach, the Bag of Words model is used to create a representation of the text data. The text is preprocessed, tokenized, and vectorized using the CountVectorizer from the scikit-learn library. A Logistic Regression model is then trained on the vectorized data and used to make predictions on the test set.

### Approach 2: Word Embeddings with LSTM

In the second approach, Word Embeddings and LSTM (Long Short-Term Memory) networks are used. The text data is preprocessed and tokenized using the Keras Tokenizer. The sequences are then padded to a fixed length. An LSTM model is built with an Embedding layer, two stacked Bidirectional LSTM layers, a Dense layer with ReLU activation, and a Dropout layer to avoid overfitting. The model is compiled with binary crossentropy loss and the Adam optimizer, and trained on the padded sequences.

## Dataset

The Large Movie Review Dataset can be downloaded from [this link](http://ai.stanford.edu/~amaas/data/sentiment/).

## Results

- Approach 1: Test accuracy - 87%
- Approach 2: test accuracy - 82%
