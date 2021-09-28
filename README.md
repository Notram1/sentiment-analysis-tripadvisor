# Sentiment Analysis based on the TripAdvisor review dataset

## Download Dataset
The dataset can be downloaded from: https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
Save the _tripadvisor_hotel_reviews.csv_ file to the [data/](https://gitlab.com/Notram1/sentiment-analysis-tripadvisor/-/tree/master/data/) folder.

## Set Up Environment
Requirements can be found in [docs/requirements.txt](https://gitlab.com/Notram1/sentiment-analysis-tripadvisor/-/tree/master/docs/requirements.txt)

To create an environment corresponding to the requirements follow the code below:

```
conda create -n <myenv> python=3.7
conda activate <myenv>
cd path/to/git/repo/
cd docs/
pip install -r requirements.txt
```

## Notebooks

Notebook [1_DataPreproc_SentimentAnalysis](https://gitlab.com/Notram1/sentiment-analysis-tripadvisor/-/tree/master/1_DataPreproc_SentimentAnalysis.ipynb) contains the exploration of the dataset, the prepocessing of the input data (text normalization, sentiment encoding, dataset dividing, etc.), and finally, the comparison of different machine learning methods (Decision tree, Naive Bayes, kNN, Logistic regression, SVM, etc.) based on the sentiment analysis and rating prediction task.

Notebook [2_SentimentAnalysis_PyTorch](https://gitlab.com/Notram1/sentiment-analysis-tripadvisor/-/tree/master/2_SentimentAnalysis_PyTorch.ipynb) uses the preprocessed data from the previous notebook, prepares the data in a suitable input form for the `RNNClassifier` (RNN, LSTM or GRU), and in the last step the sentiment analysis task is being tackled with the help of the PyTorch framework. The model consists of (multiple) recurrent layers and dropout regularization with randomly initialized embedding layers. The model has the option to use bidirectional recurrent layers and it has proper weight initialization. The evaluation metrics used are accuracy, RMSE, f1-score, precision and recall. The notebook also contains an inference demo to check model prediction for arbitrary input.

The training process uses the macro accuracy (mean of accuracy per class) as validation metric, and a simple decay as learning rate scheduling. Since the dataset being used has unbalanced classes, the model tuned for accuracy might discriminate classes with smaller sample numbers. To counter this, the terms in the Cross Entropy loss function are weighted with the inverse of the corresponding class proportion compared to the whole dataset.

## Further development
Several ideas have come up to increase model performance. First of all, the automatisation of the complex task of hyperparameter tuning is in order with the help of [TensorBoard](https://www.tensorflow.org/tensorboard). Second, the use of pretrained word embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/), [Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)) on larger datasets could have a significant impact on the performance in sentiment analysis.
