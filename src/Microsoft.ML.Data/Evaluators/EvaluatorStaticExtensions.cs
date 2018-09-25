// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Extension methods for evaluation.
    /// </summary>
    public static class EvaluatorStaticExtensions
    {
        /// <summary>
        /// Evaluates scored binary classification data.
        /// </summary>
        /// <typeparam name="T">The shape type for the input data.</typeparam>
        /// <param name="ctx">The binary classification context.</param>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The index delegate for the label column.</param>
        /// <param name="pred">The index delegate for columns from calibrated prediction of a binary classifier.
        /// Under typical scenarios, this will just be the same tuple of results returned from the trainer.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public static BinaryClassifierEvaluator.CalibratedResult Evaluate<T>(
            this BinaryClassificationContext ctx,
            DataView<T> data,
            Func<T, Scalar<bool>> label,
            Func<T, (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel)> pred)
        {
            Contracts.CheckValue(data, nameof(data));
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckValue(label, nameof(label));
            env.CheckValue(pred, nameof(pred));

            var indexer = StaticPipeUtils.GetIndexer(data);
            string labelName = indexer.Get(label(indexer.Indices));
            (var scoreCol, var probCol, var predCol) = pred(indexer.Indices);
            env.CheckParam(scoreCol != null, nameof(pred), "Indexing delegate resulted in null score column.");
            env.CheckParam(probCol != null, nameof(pred), "Indexing delegate resulted in null probability column.");
            env.CheckParam(predCol != null, nameof(pred), "Indexing delegate resulted in null predicted label column.");
            string scoreName = indexer.Get(scoreCol);
            string probName = indexer.Get(probCol);
            string predName = indexer.Get(predCol);

            var eval = new BinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data.AsDynamic, labelName, scoreName, probName, predName);
        }

        /// <summary>
        /// Evaluates scored binary classification data, if the predictions are not calibrated.
        /// </summary>
        /// <typeparam name="T">The shape type for the input data.</typeparam>
        /// <param name="ctx">The binary classification context.</param>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The index delegate for the label column.</param>
        /// <param name="pred">The index delegate for columns from uncalibrated prediction of a binary classifier.
        /// Under typical scenarios, this will just be the same tuple of results returned from the trainer.</param>
        /// <returns>The evaluation results for these uncalibrated outputs.</returns>
        public static BinaryClassifierEvaluator.Result Evaluate<T>(
            this BinaryClassificationContext ctx,
            DataView<T> data,
            Func<T, Scalar<bool>> label,
            Func<T, (Scalar<float> score, Scalar<bool> predictedLabel)> pred)
        {
            Contracts.CheckValue(data, nameof(data));
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckValue(label, nameof(label));
            env.CheckValue(pred, nameof(pred));

            var indexer = StaticPipeUtils.GetIndexer(data);
            string labelName = indexer.Get(label(indexer.Indices));
            (var scoreCol, var predCol) = pred(indexer.Indices);
            Contracts.CheckParam(scoreCol != null, nameof(pred), "Indexing delegate resulted in null score column.");
            Contracts.CheckParam(predCol != null, nameof(pred), "Indexing delegate resulted in null predicted label column.");
            string scoreName = indexer.Get(scoreCol);
            string predName = indexer.Get(predCol);

            var eval = new BinaryClassifierEvaluator(env, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data.AsDynamic, labelName, scoreName, predName);
        }

        /// <summary>
        /// Evaluates scored multiclass classification data.
        /// </summary>
        /// <typeparam name="T">The shape type for the input data.</typeparam>
        /// <typeparam name="TKey">The value type for the key label.</typeparam>
        /// <param name="ctx">The multiclass classification context.</param>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The index delegate for the label column.</param>
        /// <param name="pred">The index delegate for columns from the prediction of a multiclass classifier.
        /// Under typical scenarios, this will just be the same tuple of results returned from the trainer.</param>
        /// <param name="topK">If given a positive value, the <see cref="MultiClassClassifierEvaluator.Result.TopKAccuracy"/> will be filled with
        /// the top-K accuracy, that is, the accuracy assuming we consider an example with the correct class within
        /// the top-K values as being stored "correctly."</param>
        /// <returns>The evaluation metrics.</returns>
        public static MultiClassClassifierEvaluator.Result Evaluate<T, TKey>(
            this MulticlassClassificationContext ctx,
            DataView<T> data,
            Func<T, Key<uint, TKey>> label,
            Func<T, (Vector<float> score, Key<uint, TKey> predictedLabel)> pred,
            int topK = 0)
        {
            Contracts.CheckValue(data, nameof(data));
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckValue(label, nameof(label));
            env.CheckValue(pred, nameof(pred));
            env.CheckParam(topK >= 0, nameof(topK), "Must not be negative.");

            var indexer = StaticPipeUtils.GetIndexer(data);
            string labelName = indexer.Get(label(indexer.Indices));
            (var scoreCol, var predCol) = pred(indexer.Indices);
            Contracts.CheckParam(scoreCol != null, nameof(pred), "Indexing delegate resulted in null score column.");
            Contracts.CheckParam(predCol != null, nameof(pred), "Indexing delegate resulted in null predicted label column.");
            string scoreName = indexer.Get(scoreCol);
            string predName = indexer.Get(predCol);

            var args = new MultiClassClassifierEvaluator.Arguments() { };
            if (topK > 0)
                args.OutputTopKAcc = topK;

            var eval = new MultiClassClassifierEvaluator(env, args);
            return eval.Evaluate(data.AsDynamic, labelName, scoreName, predName);
        }

        private sealed class TrivialRegressionLossFactory : ISupportRegressionLossFactory
        {
            private readonly IRegressionLoss _loss;
            public TrivialRegressionLossFactory(IRegressionLoss loss) => _loss = loss;
            public IRegressionLoss CreateComponent(IHostEnvironment env) => _loss;
        }

        /// <summary>
        /// Evaluates scored multiclass classification data.
        /// </summary>
        /// <typeparam name="T">The shape type for the input data.</typeparam>
        /// <param name="ctx">The regression context.</param>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The index delegate for the label column.</param>
        /// <param name="score">The index delegate for predicted score column.</param>
        /// <param name="loss">Potentially custom loss function. If left unspecified defaults to <see cref="SquaredLoss"/>.</param>
        /// <returns>The evaluation metrics.</returns>
        public static RegressionEvaluator.Result Evaluate<T>(
            this RegressionContext ctx,
            DataView<T> data,
            Func<T, Scalar<float>> label,
            Func<T, Scalar<float>> score,
            IRegressionLoss loss = null)
        {
            Contracts.CheckValue(data, nameof(data));
            var env = StaticPipeUtils.GetEnvironment(data);
            Contracts.AssertValue(env);
            env.CheckValue(label, nameof(label));
            env.CheckValue(score, nameof(score));

            var indexer = StaticPipeUtils.GetIndexer(data);
            string labelName = indexer.Get(label(indexer.Indices));
            string scoreName = indexer.Get(score(indexer.Indices));

            var args = new RegressionEvaluator.Arguments() { };
            if (loss != null)
                args.LossFunction = new TrivialRegressionLossFactory(loss);
            return new RegressionEvaluator(env, args).Evaluate(data.AsDynamic, labelName, scoreName);
        }
    }
}
