// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;

namespace Microsoft.ML.EntryPoints
{
    [BestFriend]
    internal enum CachingOptions
    {
        Auto,
        Memory,
        Disk,
        None
    }

    /// <summary>
    /// The base class for all evaluators inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.IEvaluatorInput))]
    [BestFriend]
    internal abstract class EvaluateInputBase
    {
        [Argument(ArgumentType.Required, ShortName = "data", HelpText = "The data to be used for evaluation.", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name.", ShortName = "name", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string NameColumn = DefaultColumnNames.Name;
    }

    /// <summary>
    /// Common input interfaces for TLC components.
    /// </summary>
    [BestFriend]
    internal static class CommonInputs
    {
        /// <summary>
        /// Interface that all API transform input classes will implement.
        /// </summary>
        public interface ITransformInput
        {
            Var<IDataView> Data { get; set; }
        }

        /// <summary>
        /// Interface that all API trainable featurizers will implement.
        /// </summary>
        public interface IFeaturizerInput : ITransformInput
        {
            Var<PredictorModel> PredictorModel { get; set; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInput
        {
            Var<IDataView> TrainingData { get; set; }
            string FeatureColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInputWithLabel : ITrainerInput
        {
            string LabelColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface IUnsupervisedTrainerWithWeight : ITrainerInput
        {
            string WeightColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInputWithWeight : ITrainerInputWithLabel
        {
            string WeightColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInputWithGroupId : ITrainerInputWithWeight
        {
            string GroupIdColumn { get; }
        }

        /// <summary>
        /// Interface that all API calibrator input classes will implement.
        /// </summary>
        public interface ICalibratorInput : ITransformInput
        {
            Var<PredictorModel> UncalibratedPredictorModel { get; }
            int MaxRows { get; }
        }

        /// <summary>
        /// Interface that all API evaluator input classes will implement.
        /// </summary>
        public interface IEvaluatorInput
        {
            Var<IDataView> Data { get; }
            string NameColumn { get; }
        }
    }
}
