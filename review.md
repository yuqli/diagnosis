
##### Document Classification

As mentioned in the description, this project can be viewed as a text classification task, with the following proposed workflow :

1. Data Preprocessing. In the context of text classification, this include data cleaning and learning word representation.
2. Model Fitting. Can fit "simple" supervised learning models particularly for high dimensional data (SVM, K-nearest neighbor, Naive Bayes, Ridge Regression), or neural network models. All these models mentioned here has been tried, as reviewed in the literature review section.
3. Model Evaluation and re-fitting. Tuning hyperparameters.

I propose the following highly cited papers in document classification task might be helpful. Some of them are authored by NYU researchers.

Sentence Level, CNNs:

- Yoon Kim (2014): Convolutional Neural Networks for Sentence Classification. Comment: Applying a CNN to word vectors. Paper: https://arxiv.org/abs/1408.5882 PyTorch Code: https://github.com/Shawn1993/cnn-text-classification-pytorch
- Zhang el al (2016): Character-level Convolutional Networks for Text Classification. Comment: treat sentences like images and apply CNN to character set of length $m$. Paper: https://arxiv.org/abs/1509.01626 Code:
https://github.com/zhangxiangxiao/Crepe

Reviews on non-neural network models:
- Vijayan el al (2017): A Comprehensive Study of
Text Classification Algorithms. Reviews KNN, Naive Bayes, Regression, Rule-based Algorithms for text classification


#### ICD9 label prediction

- Rule based systems, such as Goldstein et al. (2007), have consisted mainly of hand-crafted rules to capture lexical elements (e.g. short, meaningful sets of words).
- Simple machine learning techniques. Larkey and Croft (1995) used a combination of K-nearest neighbor and Bayesian classifiers to predict ICD9 codes, while Lita et. (2008) and Perotte et. (2014) used ridge regression and support vector machines (SVM).
- Neural network models. Baumel et al (2017) used SVM, CBOW, CNN, and Hierarchical GRU model for ICD9 label classification on the MIMIC III dataset

In general, rule based methods have usually outperformed machine learning classifiers, although comparisons between experiments are tricky because different studies may not use the same subset of ICD9 codes (Baumel et al 2017).


#### Specific challenges
In particular, my previous work with a colleague on diagnoses code prediction on the MIMIC III Dataset (https://mimic.physionet.org/) has identified the following challenges for medical summaries :

- Large Label Space : In MIMIC II dataset, there are over 5000 unique five-digit ICD9 codes. Even after rolling up to three-digit codes, there are still 970 of them. I would suspect for the "intake clinical note" in this project this would be similar. Large label space would translate to a large number of binary classification problems, and post computational constraints. Possible solutions can be found in this NIPS 2016 workshop on extreme classification. https://nips.cc/Conferences/2016/Schedule?showEvent=6211

- Long text input: regular RNN models (GRU, LSTM) work at sequences. In the case of sentences this corresponds to the number of tokens in a sentence. But the average length of an input summary for MIMIC dataset is ~2000 tokens, thus, regular GRU architecture does not work. The solution is a hierarchical model, one example is implemented here: http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

- Out-of-vocabulary words: Many NLP models rely on a fixed vocabulary consist of words that appear at least X times in the corpus, but such fixed vocabulary might be problematic for medical text because they exclude rare diseases that do not occur in existing summaries. Also, the MIMIC III dataset consists of many spelling errors and typos. If similar conditions exist for this project, I propose the techniques proposed in the following paper: https://dspace.mit.edu/handle/1721.1/29241
