// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Defines static extension methods that allow operations like train-test split, cross-validate,
    /// sampling etc. with the <see cref="TrainCatalogBase"/>.
    /// </summary>
    public static class TrainingStaticExtensions
    {
        /// <summary>
        /// Split the dataset into the train set and test set according to the given fraction.
        /// Respects the <paramref name="stratificationColumn"/> if provided.
        /// </summary>
        /// <typeparam name="T">The tuple describing the data schema.</typeparam>
        /// <param name="catalog">The training catalog.</param>
        /// <param name="data">The dataset to split.</param>
        /// <param name="testFraction">The fraction of data to go into the test set.</param>
        /// <param name="stratificationColumn">Optional selector for the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>A pair of datasets, for the train and test set.</returns>
        public static (DataView<T> trainSet, DataView<T> testSet) TrainTestSplit<T>(this DataOperationsCatalog catalog,
            DataView<T> data, double testFraction = 0.1, Func<T, PipelineColumn> stratificationColumn = null, int? seed = null)
        {
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckParam(0 < testFraction && testFraction < 1, nameof(testFraction), "Must be between 0 and 1 exclusive");
            env.CheckValueOrNull(stratificationColumn);

            string stratName = null;

            if (stratificationColumn != null)
            {
                var indexer = StaticPipeUtils.GetIndexer(data);
                var column = stratificationColumn(indexer.Indices);
                env.CheckParam(column != null, nameof(stratificationColumn), "Stratification column not found");
                stratName = indexer.Get(column);
            }

            var split = catalog.TrainTestSplit(data.AsDynamic, testFraction, stratName, seed);
            return (new DataView<T>(env, split.TrainSet, data.Shape), new DataView<T>(env, split.TestSet, data.Shape));
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="label"/> and return metrics.
        /// </summary>
        /// <typeparam name="TInShape">The input schema shape.</typeparam>
        /// <typeparam name="TOutShape">The output schema shape.</typeparam>
        /// <typeparam name="TTransformer">The type of the trained model.</typeparam>
        /// <param name="catalog">The training catalog.</param>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="label">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional selector for the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public static (RegressionMetrics metrics, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TOutShape> scoredTestData)[] CrossValidate<TInShape, TOutShape, TTransformer>(
            this RegressionCatalog catalog,
            DataView<TInShape> data,
            Estimator<TInShape, TOutShape, TTransformer> estimator,
            Func<TOutShape, Scalar<float>> label,
            int numFolds = 5,
            Func<TInShape, PipelineColumn> stratificationColumn = null, int? seed = null)
            where TTransformer : class, ITransformer
        {
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            env.CheckValue(label, nameof(label));
            env.CheckValueOrNull(stratificationColumn);

            var outIndexer = StaticPipeUtils.GetIndexer(estimator);
            var labelColumn = label(outIndexer.Indices);
            env.CheckParam(labelColumn != null, nameof(stratificationColumn), "Stratification column not found");
            var labelName = outIndexer.Get(labelColumn);

            string stratName = null;
            if (stratificationColumn != null)
            {
                var indexer = StaticPipeUtils.GetIndexer(data);
                var column = stratificationColumn(indexer.Indices);
                env.CheckParam(column != null, nameof(stratificationColumn), "Stratification column not found");
                stratName = indexer.Get(column);
            }

            var results = catalog.CrossValidate(data.AsDynamic, estimator.AsDynamic, numFolds, labelName, stratName, seed);

            return results.Select(x => (
                    x.Metrics,
                    new Transformer<TInShape, TOutShape, TTransformer>(env, (TTransformer)x.Model, data.Shape, estimator.Shape),
                    new DataView<TOutShape>(env, x.ScoredHoldOutSet, estimator.Shape)))
                .ToArray();
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="label"/> and return metrics.
        /// </summary>
        /// <typeparam name="TInShape">The input schema shape.</typeparam>
        /// <typeparam name="TOutShape">The output schema shape.</typeparam>
        /// <typeparam name="TTransformer">The type of the trained model.</typeparam>
        /// <param name="catalog">The training catalog.</param>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="label">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional selector for the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public static (MulticlassClassificationMetrics metrics, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TOutShape> scoredTestData)[] CrossValidate<TInShape, TOutShape, TTransformer>(
            this MulticlassClassificationCatalog catalog,
            DataView<TInShape> data,
            Estimator<TInShape, TOutShape, TTransformer> estimator,
            Func<TOutShape, Key<uint>> label,
            int numFolds = 5,
            Func<TInShape, PipelineColumn> stratificationColumn = null, int? seed = null)
            where TTransformer : class, ITransformer
        {
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            env.CheckValue(label, nameof(label));
            env.CheckValueOrNull(stratificationColumn);

            var outputIndexer = StaticPipeUtils.GetIndexer(estimator);
            var labelColumn = label(outputIndexer.Indices);
            env.CheckParam(labelColumn != null, nameof(stratificationColumn), "Stratification column not found");
            var labelName = outputIndexer.Get(labelColumn);

            string stratName = null;
            if (stratificationColumn != null)
            {
                var indexer = StaticPipeUtils.GetIndexer(data);
                var column = stratificationColumn(indexer.Indices);
                env.CheckParam(column != null, nameof(stratificationColumn), "Stratification column not found");
                stratName = indexer.Get(column);
            }

            var results = catalog.CrossValidate(data.AsDynamic, estimator.AsDynamic, numFolds, labelName, stratName, seed);

            return results.Select(x => (
                    x.Metrics,
                    new Transformer<TInShape, TOutShape, TTransformer>(env, (TTransformer)x.Model, data.Shape, estimator.Shape),
                    new DataView<TOutShape>(env, x.ScoredHoldOutSet, estimator.Shape)))
                .ToArray();
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="label"/> and return metrics.
        /// </summary>
        /// <typeparam name="TInShape">The input schema shape.</typeparam>
        /// <typeparam name="TOutShape">The output schema shape.</typeparam>
        /// <typeparam name="TTransformer">The type of the trained model.</typeparam>
        /// <param name="catalog">The training catalog.</param>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="label">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional selector for the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public static (BinaryClassificationMetrics metrics, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TOutShape> scoredTestData)[] CrossValidateNonCalibrated<TInShape, TOutShape, TTransformer>(
            this BinaryClassificationCatalog catalog,
            DataView<TInShape> data,
            Estimator<TInShape, TOutShape, TTransformer> estimator,
            Func<TOutShape, Scalar<bool>> label,
            int numFolds = 5,
            Func<TInShape, PipelineColumn> stratificationColumn = null, int? seed = null)
            where TTransformer : class, ITransformer
        {
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            env.CheckValue(label, nameof(label));
            env.CheckValueOrNull(stratificationColumn);

            var outputIndexer = StaticPipeUtils.GetIndexer(estimator);
            var labelColumn = label(outputIndexer.Indices);
            env.CheckParam(labelColumn != null, nameof(stratificationColumn), "Stratification column not found");
            var labelName = outputIndexer.Get(labelColumn);

            string stratName = null;
            if (stratificationColumn != null)
            {
                var indexer = StaticPipeUtils.GetIndexer(data);
                var column = stratificationColumn(indexer.Indices);
                env.CheckParam(column != null, nameof(stratificationColumn), "Stratification column not found");
                stratName = indexer.Get(column);
            }

            var results = catalog.CrossValidateNonCalibrated(data.AsDynamic, estimator.AsDynamic, numFolds, labelName, stratName, seed);

            return results.Select(x => (
                    x.Metrics,
                    new Transformer<TInShape, TOutShape, TTransformer>(env, (TTransformer)x.Model, data.Shape, estimator.Shape),
                    new DataView<TOutShape>(env, x.ScoredHoldOutSet, estimator.Shape)))
                .ToArray();
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="label"/> and return metrics.
        /// </summary>
        /// <typeparam name="TInShape">The input schema shape.</typeparam>
        /// <typeparam name="TOutShape">The output schema shape.</typeparam>
        /// <typeparam name="TTransformer">The type of the trained model.</typeparam>
        /// <param name="catalog">The training catalog.</param>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="label">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional selector for the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public static (CalibratedBinaryClassificationMetrics metrics, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TOutShape> scoredTestData)[] CrossValidate<TInShape, TOutShape, TTransformer>(
            this BinaryClassificationCatalog catalog,
            DataView<TInShape> data,
            Estimator<TInShape, TOutShape, TTransformer> estimator,
            Func<TOutShape, Scalar<bool>> label,
            int numFolds = 5,
            Func<TInShape, PipelineColumn> stratificationColumn = null, int? seed = null)
            where TTransformer : class, ITransformer
        {
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            env.CheckValue(label, nameof(label));
            env.CheckValueOrNull(stratificationColumn);

            var outputIndexer = StaticPipeUtils.GetIndexer(estimator);
            var labelColumn = label(outputIndexer.Indices);
            env.CheckParam(labelColumn != null, nameof(stratificationColumn), "Stratification column not found");
            var labelName = outputIndexer.Get(labelColumn);

            string stratName = null;
            if (stratificationColumn != null)
            {
                var indexer = StaticPipeUtils.GetIndexer(data);
                var column = stratificationColumn(indexer.Indices);
                env.CheckParam(column != null, nameof(stratificationColumn), "Stratification column not found");
                stratName = indexer.Get(column);
            }

            var results = catalog.CrossValidate(data.AsDynamic, estimator.AsDynamic, numFolds, labelName, stratName, seed);

            return results.Select(x => (
                    x.Metrics,
                    new Transformer<TInShape, TOutShape, TTransformer>(env, (TTransformer)x.Model, data.Shape, estimator.Shape),
                    new DataView<TOutShape>(env, x.ScoredHoldOutSet, estimator.Shape)))
                .ToArray();
        }
    }
}
