## Goal
This is a getting started example that shows the simplest way of using ML.NET APIs for solving a regression problem on taxi fare prediction example.

## Problem
The task is to build and train ML model (machine learning model) that will predict a taxi fare based on taxi vendor id, passengers count, rate code, etc.

## Problem Class - Regression
The described task is an example of regression problem. 
> In machine learning, `regression` is the problem of predicting a continuous quantity for a given parameters. Unlike `classification` problems where the result we want to predict can have one of a few values (classes), the regression problem is predicting a continuous value, such as an integer or floating point value. These are often amounts, prices, sizes, etc.

Machine learning engineering process includes three steps: training ML model, evaluating how good it is, and if the quality is acceptable, using this model for predictions. If the quality of the model is not good enough, different algorithms and/or additional data transformations can be applied and the model should be trained and evaluated again.

1. **Training** the ML model is implemented in `TrainAsync()` method that constructs `LearningPipeline`, trains it and saves the trained model as a .zip file.
2. **Evaluating** the ML model is implemented in `Evaluate()` method which runs the model against a test data (new data with known answers, that was not involved in training). As a result it produces a set of metrics describing the quality of the model.
3. **Predicting** the predicting of the taxi fare amount is performed in the `Main()` method:
```CSharp
var prediction = model.Predict(TestTaxiTrips.Trip1);
```
where you send an instance of the `TaxiTrip` - the type that contains all features:  taxi vendor id, passengers count, etc. As a result you receive `TaxiTripFarePrediction` object that has predicted `FareAmount`.