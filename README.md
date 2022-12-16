# Tweet-Sentiment-Analysis
Tweet sentiment analysis with machine learning and deep learning methods. The used embedding methods are TF-IDF and Word2Vec. The trained classic machine learning models are Bernoulli Naive Bayes, Logistic Regression, SVM. The deep learning models are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), [C-LSTM](https://arxiv.org/abs/1511.08630) and [Bi-LSTM-CNN](https://ieeexplore.ieee.org/abstract/document/8589431).

### Prerequisite

- Git clone this project
- Download dataset from [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140), and put the csv file under <u>Twitter-Sentiment-Analysis\dataset\</u>

## Training

The main process includes:

- Data loading
- Data preprocessing
- Text embedding/vectorizing
- Model defining and training
- Model testing

Please refer to `ml_training.ipynb` and `nn_training.ipynb` for details.

## Results

| Vectorizing Method | Model                 | Accuracy |
| ------------------ | --------------------- | -------- |
| TF-IDF             | Bernoulli Naive Bayes | 0.78     |
| TF-IDF             | Logistic Regression   | 0.80     |
| TF-IDF             | SVM                   | 0.80     |
| Word2Vec           | LSTM                  | 0.828    |
| Word2Vec           | C-LSTM                | 0.836    |
| Word2Vec           | Bi-LSTM               | 0.846    |