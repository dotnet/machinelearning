// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// The base class for all trainer inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInput))]
    public abstract class TrainerInputBase
    {
        private protected TrainerInputBase() { }

        /// <summary>
        /// The data to be used for training. Used only in entry-points, since in the API the expected mechanism is
        /// that the user will use the <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> or some other train
        /// method.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.Required, ShortName = "data", HelpText = "The data to be used for training", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        internal IDataView TrainingData;

        /// <summary>
        /// Column to use for features.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for features", ShortName = "feat", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string FeatureColumnName = DefaultColumnNames.Features;

        /// <summary>
        /// Normalize option for the feature column. Used only in entry-points, since in the API the user is expected to do this themselves.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize option for the feature column", ShortName = "norm", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        internal NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

        /// <summary>
        /// Whether trainer should cache input training data. Used only in entry-points, since the intended API mechanism
        /// is that the user will use the <see cref="DataOperationsCatalog.Cache(IDataView, string[])"/> or other method
        /// like <see cref="EstimatorChain{TLastTransformer}.AppendCacheCheckpoint(IHostEnvironment)"/>.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether trainer should cache input training data", ShortName = "cache", SortOrder = 6, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        internal CachingOptions Caching = CachingOptions.Auto;
    }

    /// <summary>
    /// The base class for all trainer inputs that support a Label column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithLabel))]
    public abstract class TrainerInputBaseWithLabel : TrainerInputBase
    {
        private protected TrainerInputBaseWithLabel() { }

        /// <summary>
        /// Column to use for labels.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string LabelColumnName = DefaultColumnNames.Label;
    }

    // REVIEW: This is a known antipattern, but the solution involves the decorator pattern which can't be used in this case.
    /// <summary>
    /// The base class for all trainer inputs that support a weight column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithWeight))]
    public abstract class TrainerInputBaseWithWeight : TrainerInputBaseWithLabel
    {
        private protected TrainerInputBaseWithWeight() { }

        /// <summary>
        /// Column to use for example weight.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string ExampleWeightColumnName = null;
    }

    /// <summary>
    /// The base class for all unsupervised trainer inputs that support a weight column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.IUnsupervisedTrainerWithWeight))]
    public abstract class UnsupervisedTrainerInputBaseWithWeight : TrainerInputBase
    {
        private protected UnsupervisedTrainerInputBaseWithWeight() { }

        /// <summary>
        /// Column to use for example weight.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string ExampleWeightColumnName = null;
    }

    /// <summary>
    /// The base class for all trainer inputs that support a group column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithGroupId))]
    public abstract class TrainerInputBaseWithGroupId : TrainerInputBaseWithWeight
    {
        private protected TrainerInputBaseWithGroupId() { }

        /// <summary>
        /// Column to use for example groupId.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example groupId", ShortName = "groupId", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string RowGroupColumnName = null;
    }
}
