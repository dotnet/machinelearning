## Goal
This is a getting started example that shows the simplest way of using ML.NET APIs for solving a binary classification problem on sentiment analysis example.

## Problem
The task is to build and train ML model (machine learning model) that will predict if a text has positive or negative sentiment. For training and evaluating the model we used imdb and yelp comments with known sentiments.

## Problem Class - Binary Classification
The described task is an example of a binary classification problem. 
> In machine learning, `binary classification` is the problem of classifying instances into one of a two classes. (Classifying instances into more than two classes is called `multiclass classification`.)

Machine learning engineering process includes three steps: training ML model, evaluating how good it is, and if the quality is acceptable, using this model for predictions. If the quality of the model is not good enough, different algorithms and/or additional data transformations can be applied and the model should be trained and evaluated again.

1. **Training** the ML model is implemented in `TrainAsync()` method that constructs `LearningPipeline`, trains it and saves the trained model as a .zip file.
2. **Evaluating** the ML model is implemented in `Evaluate()` method which runs the model against a test data (new data with known answers, that was not involved in training). As a result it produces a set of metrics describing the quality of the model.
3. **Predicting** the sentiment is performed in the `Main()` method:
```CSharp
var predictions = model.Predict(TestSentimentData.Sentiments);
```
where you send a text as a `SentimentData` object. As a result you receive `SentimentPrediction` object that contains a boolean field `Sentiment`: true for positive, false for negative sentiments.