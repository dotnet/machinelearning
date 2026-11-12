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
        /// The default value is <c>Features</c>.
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
        [Argument(ArgumentType.LastOccurrenceWins, HelpText = "Whether trainer should cache input training data", ShortName = "cache", SortOrder = 6, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
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
        /// The default value is <c>Label</c>.
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
        /// The default value is <see langword="null"/>.
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
        /// The name of the column containing a weight for each training example, or <see langword="null"/>
        /// to give all examples equal weight.
        /// The default value is <see langword="null"/>.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The column must contain a single <see cref="System.Single"/> value per row.
        /// Weights control the relative influence of training examples, not individual features.
        /// How weights affect training depends on the trainer.
        /// If no weight column is specified, each example has a weight of 1.
        /// </para>
        /// <para>
        /// Only one column name can be specified. To calculate weights from multiple input columns,
        /// first combine their values into a single weight column before training. For example, a row
        /// with a quantity of 3 and a confidence of 0.5 could be assigned a weight of 1.5 by multiplying
        /// those values.
        /// </para>
        /// </remarks>
        /// <example>
        /// The following options configure K-means to use a column named "Weight" containing the
        /// per-example weights. The training data must contain that column in addition to "Features".
        /// <code>
        /// var options = new KMeansTrainer.Options
        /// {
        ///     FeatureColumnName = "Features",
        ///     ExampleWeightColumnName = "Weight",
        ///     NumberOfClusters = 2
        /// };
        /// var trainer = mlContext.Clustering.Trainers.KMeans(options);
        /// var model = trainer.Fit(trainingData);
        /// </code>
        /// </example>
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
        /// The default is <see langword="null"/>; FastTree and LightGBM ranking options use <c>GroupId</c>.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example groupId", ShortName = "groupId", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string RowGroupColumnName = null;
    }
}
