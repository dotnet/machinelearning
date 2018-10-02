# ML.NET 0.1 Release Notes

ML.NET 0.1 is the first preview release of ML.NET. Thank you for trying it out and we look forward to your feedback! Try training, scoring, and using machine learning models in your app and tell us how it goes.

### Installation

ML.NET works on any platform that supports [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started/windows). It also works on the .NET Framework.

You can install ML.NET NuGet from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

Or from within Visual Studio's NuGet package manager.

### Release Notes

This initial release contains core ML.NET components for enabling machine learning pipelines:

* ML Data Structures (e.g. `IDataView`, `LearningPipeline`)

* TextLoader (loading data from a delimited text file into a `LearningPipeline`)

* Transforms (to get data in the correct format for training):
    * Processing/featurizing text: `TextFeaturizer`
    * Schema modifcation: `ColumnConcatenator`, `ColumnSelector`, and `ColumnDropper`
    * Working with categorical features: `CategoricalOneHotVectorizer` and `CategoricalHashOneHotVectorizer`
    * Dealing with missing data: `MissingValueHandler`
    * Filters: `RowTakeFilter`, `RowSkipFilter`, `RowRangeFilter`
    * Feature selection: `FeatureSelectorByCount` and `FeatureSelectorByMutualInformation`
    
* Learners (to train machine learning models) for a variety of tasks:
    * Binary classification: `FastTreeBinaryClassifier`, `StochasticDualCoordinateAscentBinaryClassifier`, `AveragedPerceptronBinaryClassifier`, `BinaryLogisticRegressor`, `FastForestBinaryClassifier`,  `LinearSvmBinaryClassifier`, and `GeneralizedAdditiveModelBinaryClassifier`
    * Multiclass classification: `StochasticDualCoordinateAscentClassifier`, `LogisticRegressor`, and`NaiveBayesClassifier`
    * Regression: `FastTreeRegressor`, `FastTreeTweedieRegressor`, `StochasticDualCoordinateAscentRegressor`, `OrdinaryLeastSquaresRegressor`, `OnlineGradientDescentRegressor`, `PoissonRegressor`, and `GeneralizedAdditiveModelRegressor`
    
* Evaluators (to check how well the model works):
    * For Binary classification: `BinaryClassificationEvaluator`
    * For Multiclass classification: `ClassificationEvaluator`
    * For Regression: `RegressionEvaluator`

Additional components have been included in the repository but cannot be used in the `LearningPipeline` yet (this will be updated in future releases).