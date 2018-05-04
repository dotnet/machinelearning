// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// Common output classes for trainers and transforms.
    /// </summary>
    public static class CommonOutputs
    {
        /// <summary>
        /// The common output class for all transforms.
        /// The output consists of the transformed dataset and the transformation model.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(ITransformOutput))]
        public sealed class TransformOutput
        {
            [TlcModule.Output(Desc = "Transformed dataset", SortOrder = 1)]
            public IDataView OutputData;

            [TlcModule.Output(Desc = "Transform model", SortOrder = 2)]
            public ITransformModel Model;
        }

        /// <summary>
        /// Interface that all API transform output classes will implement.
        /// </summary>
        public interface ITransformOutput
        {
            Var<IDataView> OutputData { get; }
            Var<ITransformModel> Model { get; }
        }

        /// <summary>
        /// The common output class for all trainers.
        /// The output is a trained predictor model.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(ITrainerOutput))]
        public abstract class TrainerOutput
        {
            [TlcModule.Output(Desc = "The trained model", SortOrder = 1)]
            public IPredictorModel PredictorModel;
        }

        /// <summary>
        /// The common output for calibrators.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(ICalibratorOutput))]
        public sealed class CalibratorOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for calibrators output.
        /// </summary>
        public interface ICalibratorOutput
        {
        }

        /// <summary>
        /// The common output for binary classification trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IBinaryClassificationOutput))]
        public sealed class BinaryClassificationOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for binary classification trainer output.
        /// </summary>
        public interface IBinaryClassificationOutput
        {
        }

        /// <summary>
        /// The common output for multiclass classification trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IMulticlassClassificationOutput))]
        public sealed class MulticlassClassificationOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for multiclass classification trainer output.
        /// </summary>
        public interface IMulticlassClassificationOutput
        {
        }

        /// <summary>
        /// The common output for regression trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IRegressionOutput))]
        public sealed class RegressionOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for regression trainer output.
        /// </summary>
        public interface IRegressionOutput
        {
        }

        /// <summary>
        /// The common output for multi regression trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IMultiRegressionOutput))]
        public sealed class MultiRegressionOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for multi regression trainer output.
        /// </summary>
        public interface IMultiRegressionOutput
        {
        }

        /// <summary>
        /// The common output for clustering trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IClusteringOutput))]
        public sealed class ClusteringOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for clustering trainer output.
        /// </summary>
        public interface IClusteringOutput
        {
        }

        /// <summary>
        /// The common output for anomaly detection trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IAnomalyDetectionOutput))]
        public sealed class AnomalyDetectionOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for anomaly detection trainer output.
        /// </summary>
        public interface IAnomalyDetectionOutput
        {
        }

        /// <summary>
        /// The common output for ranking trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IRankingOutput))]
        public sealed class RankingOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for ranking trainer output.
        /// </summary>
        public interface IRankingOutput
        {
        }

        /// <summary>
        /// The common output for sequence prediction trainers.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(ISequencePredictionOutput))]
        public sealed class SequencePredictionOutput : TrainerOutput
        {
        }

        /// <summary>
        /// Marker interface for sequence prediction trainer output.
        /// </summary>
        public interface ISequencePredictionOutput
        {
        }

        /// <summary>
        /// Interface that all API trainer output classes will implement.
        /// </summary>
        public interface ITrainerOutput
        {
            Var<IPredictorModel> PredictorModel { get; }
        }

        /// <summary>
        /// Macro output class base. 
        /// </summary>
        public abstract class MacroOutput
        {
            public IEnumerable<EntryPointNode> Nodes;
        }

        /// <summary>
        /// The common output class for all macro entry points.
        /// The output class is the type parameter. The expansion must guarantee
        /// that the generated graph will generate all the outputs.
        /// </summary>
        /// <typeparam name="TOut">The output class of the macro.</typeparam>
        public sealed class MacroOutput<TOut> : MacroOutput
        {}

        /// <summary>
        /// The common output class for all evaluators.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IEvaluatorOutput))]
        public abstract class EvaluateOutputBase
        {
            [TlcModule.Output(Desc = "Warning dataset", SortOrder = 1)]
            public IDataView Warnings;

            [TlcModule.Output(Desc = "Overall metrics dataset", SortOrder = 2)]
            public IDataView OverallMetrics;

            [TlcModule.Output(Desc = "Per instance metrics dataset", SortOrder = 3)]
            public IDataView PerInstanceMetrics;
        }

        /// <summary>
        /// The output class for classification evaluators.
        /// </summary>
        [TlcModule.EntryPointKind(typeof(IClassificationEvaluatorOutput))]
        public sealed class ClassificationEvaluateOutput : EvaluateOutputBase
        {
            [TlcModule.Output(Desc = "Confusion matrix dataset", SortOrder = 4)]
            public IDataView ConfusionMatrix;
        }

        /// <summary>
        /// The output class for regression evaluators.
        /// </summary>
        public sealed class CommonEvaluateOutput : EvaluateOutputBase
        {
        }

        /// <summary>
        /// Interface that all API evaluator output classes will implement.
        /// </summary>
        public interface IEvaluatorOutput
        {
            Var<IDataView> Warnings { get; }
            Var<IDataView> OverallMetrics { get; }
            Var<IDataView> PerInstanceMetrics { get; }
        }

        /// <summary>
        /// Interface that all API evaluator output classes will implement.
        /// </summary>
        public interface IClassificationEvaluatorOutput : IEvaluatorOutput
        {
            Var<IDataView> ConfusionMatrix { get; }
        }

        public sealed class SummaryOutput
        {
            [TlcModule.Output(Desc = "The summary of a predictor")]
            public IDataView Summary;

            [TlcModule.Output(Desc = "The training set statistics. Note that this output can be null.")]
            public IDataView Stats;
        }
    }
}
