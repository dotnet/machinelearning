## Goal
This is a getting started example that shows the simplest way of using ML.NET APIs for solving a multiclass classification problem on iris flower classification example.

## Problem
The task is to build and train ML model (machine learning model) that will predict the type of iris flower (setosa, versicolor, or virginica) based on four features: petal length, petal width, sepal length, and sepal width.

## Problem Class - Multiclass Classification
The described task is an example of multiclass classification problem. 
> In machine learning, `multiclass classification` is the problem of classifying instances into one of three or more classes. (Classifying instances into one of the two classes is called `binary classification`.)

Machine learning engineering process includes three steps: training ML model, evaluating how good it is, and if the quality is acceptable, using this model for predictions. If the quality of the model is not good enough, different algorithms and/or additional data transformations can be applied and the model should be trained and evaluated again.

1. **Training** the ML model is implemented in `TrainAsync()` method that constructs `LearningPipeline`, trains it and saves the trained model as a .zip file.
2. **Evaluating** the ML model is implemented in `Evaluate()` method which runs the model against a test data (new data with known answers, that was not involved in training). As a result it produces a set of metrics describing the quality of the model. 
3. **Predicting** the type of the flower is performed in the `Main()` method:
```CSharp
var prediction = model.Predict(TestIrisData.Iris1);
```
where you send an instance of the `TestIrisData` - the type that contains all features:  petal length, width, etc. As a result you receive `IrisPrediction` with `Score` - array of probabilities that the given flower belongs to the type setosa, versicolor, or virginica correspondently.