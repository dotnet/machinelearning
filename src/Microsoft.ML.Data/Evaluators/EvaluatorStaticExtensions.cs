// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime.Training;

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
        /// Evaluates scored binary classification data.
        /// </summary>
        /// <typeparam name="T">The shape type for the input data.</typeparam>
        /// <param name="ctx">The binary classification context.</param>
        /// <param name="data">The data to evaluate.</param>
        /// <param name="label">The index delegate for the label column.</param>
        /// <param name="pred">The index delegate for columns from calibrated prediction of a binary classifier.
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
    }
}
