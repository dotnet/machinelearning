# ML.NET 0.1 Release Notes

ML.NET 0.1 is the first preview release of ML.NET. Thank you for trying it out and we look forward to your feedback!

### Installation

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

### Release Notes

This initial release contains core ML.NET components for enabling machine learning pipelines:

* ML Data Structures (e.g. `IDataView`, `LearningPipeline`)
* TextLoader (loading data from a text file into a `LearningPipeline`)
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
* Evaluators (to check the model works well):
    * For Binary classification: `BinaryClassificationEvaluator`
    * For Multiclass classification: `ClassificationEvaluator`
    * For Regression: `RegressionEvaluator`

Additional components have been included in the repository but cannot be used in `LearningPipeline` yet (this will be updated in future releases).