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

Notebook [2_SentimentAnalysis_PyTorch](https://gitlab.com/Notram1/sentiment-analysis-tripadvisor/-/tree/master/2_SentimentAnalysis_PyTorch.ipynb) uses the preprocessed data from the previous notebook, prepares the data in a suitable input form for the `RNNClassifier` (RNN, LSTM or GRU), and in the last step the sentiment analysis task is being tackled with the help of the PyTorch framework. The model consists of (multiple) recurrent layers and dropout regularization with randomly initialized or pre-trained embedding layers. The model has the option to use bidirectional recurrent layers and it has proper weight initialization. The evaluation metrics used are accuracy, RMSE, f1-score, precision and recall. The notebook also contains an inference demo to check model prediction for arbitrary input.

The training process uses the mean of accuracy per class as validation metric, and a simple decay as learning rate scheduling. Since the dataset being used has unbalanced classes, the model tuned for accuracy might discriminate classes with smaller sample numbers. To counter this, the terms in the Cross Entropy loss function are weighted with the inverse of the corresponding class proportion compared to the whole dataset. Gradient clipping is used to avoid the exploding gradient issue.

To prevent overfitting, the training process takes advantage of the early stopping procedure. This stops the training process if the validation accuracy has not improved for a specific number of epochs. Additionally, the model uses learning rate scheduling, by reducing it in case the validation loss has stopped decreasing for a specific number of steps.

The notebook also includes an option to use pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings. These were trained on much larger datasets, hence this could have a significant impact on the performance in sentiment analysis. In this case, the weights of the embedding layer are not trained. For every word in the vocabulary we assign the weights from GloVe if the word is present in the pre-trained embeddings. Otherwise, a random weight vector of the same length is created from a normal distribution with the same standard deviation as for the GloVe vectors.

Performance of models achieved with current parameters (using GloVe embeddings with a length of 300, if applicable):
<table class="center">
    <tr>
        <td></td>
        <td>Bidirectional</td>
        <td>GloVe</td>
        <td>Accuracy</td>
        <td>F1-Score</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>RMSE</td>
    </tr>
    <tr>
        <td rowspan="4">RNN</td>
        <td rowspan="2">no</td>
        <td>no</td>
        <td>0.4963</td>
        <td>0.4024</td>
        <td>0.4754</td>
        <td>0.5435</td>
        <td>0.8973</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.4224</td>
        <td>0.3384</td>
        <td>0.2701</td>
        <td>0.4898</td>
        <td>1.0503</td>
    </tr>
    <tr>
        <td rowspan="2">yes</td>
        <td>no</td>
        <td>0.6076</td>
        <td>0.5950</td>
        <td>0.5952</td>
        <td>0.6458</td>
        <td>0.7648</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.6315</td>
        <td>0.6251</td>
        <td>0.6145</td>
        <td>0.6652</td>
        <td>0.7140</td>
    </tr>
    <tr>
        <td rowspan="4">LSTM</td>
        <td rowspan="2">no</td>
        <td>no</td>
        <td>0.7243</td>
        <td>0.7275</td>
        <td>0.7296</td>
        <td>0.7341</td>
        <td>0.5427</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.7299</td>
        <td>0.7365</td>
        <td>0.7240</td>
        <td>0.7542</td>
        <td>0.5398</td>
    </tr>
    <tr>
        <td rowspan="2">yes</td>
        <td>no</td>
        <td>0.7177</td>
        <td>0.7174</td>
        <td>0.7186</td>
        <td>0.7310</td>
        <td>0.5657</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.7311</td>
        <td>0.7375</td>
        <td>0.7314</td>
        <td>0.7452</td>
        <td>0.5394</td>
    </tr>
    <tr>
        <td rowspan="4"> GRU</td>
        <td rowspan="2">no</td>
        <td>no</td>
        <td>0.6994</td>
        <td>0.7014</td>
        <td>0.7072</td>
        <td>0.7050</td>
        <td>0.5949</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.7352</td>
        <td>0.7460</td>
        <td>0.7362</td>
        <td>0.7583</td>
        <td>0.5362</td>
    </tr>
    <tr>
        <td rowspan="2">yes</td>
        <td>no</td>
        <td>0.7184</td>
        <td>0.7214</td>
        <td>0.7141</td>
        <td>0.7403</td>
        <td>0.5606</td>
    </tr>
    <tr>
        <td>yes</td>
        <td>0.7194</td>
        <td>0.7276</td>
        <td>0.7128</td>
        <td>0.7505</td>
        <td>0.5579</td>
    </tr>
</table>

## Further development
The automatisation of the complex task of hyperparameter tuning is in order with the help of [TensorBoard](https://www.tensorflow.org/tensorboard). 
