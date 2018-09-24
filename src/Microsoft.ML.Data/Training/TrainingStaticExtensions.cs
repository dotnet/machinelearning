// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML
{
    /// <summary>
    /// Defines static extension methods that allow operations like train-test split, cross-validate,
    /// sampling etc. with the <see cref="TrainContextBase"/>.
    /// </summary>
    public static class TrainingStaticExtensions
    {
        /// <summary>
        /// Split the dataset into the train set and test set according to the given fraction.
        /// Respects the <paramref name="stratificationColumn"/> if provided.
        /// </summary>
        /// <typeparam name="T">The tuple describing the data schema.</typeparam>
        /// <param name="context">The training context.</param>
        /// <param name="data">The dataset to split.</param>
        /// <param name="testFraction">The fraction of data to go into the test set.</param>
        /// <param name="stratificationColumn">Optional selector for the stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <returns>A pair of datasets, for the train and test set.</returns>
        public static (DataView<T> trainSet, DataView<T> testSet) TrainTestSplit<T>(this TrainContextBase context,
            DataView<T> data, double testFraction = 0.1, Func<T, PipelineColumn> stratificationColumn = null)
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

            var (trainData, testData) = context.TrainTestSplit(data.AsDynamic, testFraction, stratName);
            return (new DataView<T>(env, trainData, data.Shape), new DataView<T>(env, testData, data.Shape));
        }

        /// <summary>
        /// Runs a sequential cross-validation by training <paramref name="estimator"/>
        /// on <paramref name="data"/> in <paramref name="numFolds"/> folds.
        /// </summary>
        /// <typeparam name="TInShape">The input schema shape.</typeparam>
        /// <typeparam name="TOutShape">The output schema shape.</typeparam>
        /// <typeparam name="TTransformer">The type of the trained model.</typeparam>
        /// <param name="context"></param>
        /// <param name="data"></param>
        /// <param name="estimator"></param>
        /// <param name="label"></param>
        /// <param name="numFolds"></param>
        /// <param name="stratificationColumn"></param>
        /// <returns></returns>
        public static (RegressionEvaluator.Result metrics, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TOutShape> scoredTestData)[] CrossValidate<TInShape, TOutShape, TTransformer>(
            this RegressionContext context,
            DataView<TInShape> data,
            Estimator<TInShape, TOutShape, TTransformer> estimator,
            Func<TOutShape, Scalar<float>> label,
            int numFolds = 5,
            Func<TOutShape, PipelineColumn> stratificationColumn = null)
            where TTransformer : class, ITransformer
        {

        }
    }
}
