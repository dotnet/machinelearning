// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Calibration;

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// The base class for all transform inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITransformInput))]
    public abstract class TransformInputBase
    {
        /// <summary>
        /// The input dataset. Used only in entry-point methods, since the normal API mechanism for feeding in a dataset to
        /// create an <see cref="ITransformer"/> is to use the <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> method.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.Required, HelpText = "Input dataset", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, SortOrder = 1)]
        internal IDataView Data;
    }

    [BestFriend]
    internal enum CachingOptions
    {
        Auto,
        Memory,
        Disk,
        None
    }

    /// <summary>
    /// The base class for all learner inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInput))]
    public abstract class LearnerInputBase
    {
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
        public string FeatureColumn = DefaultColumnNames.Features;

        /// <summary>
        /// Normalize option for the feature column. Used only in entry-points, since in the API the user is expected to do this themselves.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize option for the feature column", ShortName = "norm", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        internal NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

        /// <summary>
        /// Whether learner should cache input training data. Used only in entry-points, since the intended API mechanism
        /// is that the user will use the <see cref="DataOperationsCatalog.Cache(IDataView, string[])"/> or other method
        /// like <see cref="EstimatorChain{TLastTransformer}.AppendCacheCheckpoint(IHostEnvironment)"/>.
        /// </summary>
        [BestFriend]
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether learner should cache input training data", ShortName = "cache", SortOrder = 6, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        internal CachingOptions Caching = CachingOptions.Auto;
    }

    /// <summary>
    /// The base class for all learner inputs that support a Label column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithLabel))]
    public abstract class LearnerInputBaseWithLabel : LearnerInputBase
    {
        /// <summary>
        /// Column to use for labels.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string LabelColumn = DefaultColumnNames.Label;
    }

    // REVIEW: This is a known antipattern, but the solution involves the decorator pattern which can't be used in this case.
    /// <summary>
    /// The base class for all learner inputs that support a weight column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithWeight))]
    public abstract class LearnerInputBaseWithWeight : LearnerInputBaseWithLabel
    {
        /// <summary>
        /// Column to use for example weight.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public Optional<string> WeightColumn = Optional<string>.Implicit(DefaultColumnNames.Weight);
    }

    /// <summary>
    /// The base class for all unsupervised learner inputs that support a weight column.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.IUnsupervisedTrainerWithWeight))]
    public abstract class UnsupervisedLearnerInputBaseWithWeight : LearnerInputBase
    {
        /// <summary>
        /// Column to use for example weight.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public Optional<string> WeightColumn = Optional<string>.Implicit(DefaultColumnNames.Weight);
    }

    /// <summary>
    /// The base class for all evaluators inputs.
    /// </summary>
    [TlcModule.EntryPointKind(typeof(CommonInputs.IEvaluatorInput))]
    public abstract class EvaluateInputBase
    {
        [Argument(ArgumentType.Required, ShortName = "data", HelpText = "The data to be used for evaluation.", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IDataView Data;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name.", ShortName = "name", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string NameColumn = DefaultColumnNames.Name;
    }

    [TlcModule.EntryPointKind(typeof(CommonInputs.ITrainerInputWithGroupId))]
    public abstract class LearnerInputBaseWithGroupId : LearnerInputBaseWithWeight
    {
        /// <summary>
        /// Column to use for example groupId.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example groupId", ShortName = "groupId", SortOrder = 5, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public Optional<string> GroupIdColumn = Optional<string>.Implicit(DefaultColumnNames.GroupId);
    }

    [BestFriend]
    internal static class LearnerEntryPointsUtils
    {
        public static string FindColumn(IExceptionContext ectx, DataViewSchema schema, Optional<string> value)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValue(value, nameof(value));

            if (string.IsNullOrEmpty(value?.Value))
                return null;
            if (!schema.TryGetColumnIndex(value, out int col))
            {
                if (value.IsExplicit)
                    throw ectx.Except("Column '{0}' not found", value);
                return null;
            }
            return value;
        }

        public static TOut Train<TArg, TOut>(IHost host, TArg input,
            Func<ITrainer> createTrainer,
            Func<string> getLabel = null,
            Func<string> getWeight = null,
            Func<string> getGroup = null,
            Func<string> getName = null,
            Func<IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>>> getCustom = null,
            ICalibratorTrainerFactory calibrator = null,
            int maxCalibrationExamples = 0)
            where TArg : LearnerInputBase
            where TOut : CommonOutputs.TrainerOutput, new()
        {
            using (var ch = host.Start("Training"))
            {
                var schema = input.TrainingData.Schema;
                var feature = FindColumn(ch, schema, input.FeatureColumn);
                var label = getLabel?.Invoke();
                var weight = getWeight?.Invoke();
                var group = getGroup?.Invoke();
                var name = getName?.Invoke();
                var custom = getCustom?.Invoke();

                var trainer = createTrainer();

                IDataView view = input.TrainingData;
                TrainUtils.AddNormalizerIfNeeded(host, ch, trainer, ref view, feature, input.NormalizeFeatures);

                ch.Trace("Binding columns");
                var roleMappedData = new RoleMappedData(view, label, feature, group, weight, name, custom);

                RoleMappedData cachedRoleMappedData = roleMappedData;
                Cache.CachingType? cachingType = null;
                switch (input.Caching)
                {
                    case CachingOptions.Memory:
                        {
                            cachingType = Cache.CachingType.Memory;
                            break;
                        }
                    case CachingOptions.Disk:
                        {
                            cachingType = Cache.CachingType.Disk;
                            break;
                        }
                    case CachingOptions.Auto:
                        {
                            // REVIEW: we should switch to hybrid caching in future.
                            if (!(input.TrainingData is BinaryLoader) && trainer.Info.WantCaching)
                                // default to Memory so mml is on par with maml
                                cachingType = Cache.CachingType.Memory;
                            break;
                        }
                    case CachingOptions.None:
                        break;
                    default:
                        throw ch.ExceptParam(nameof(input.Caching), "Unknown option for caching: '{0}'", input.Caching);
                }

                if (cachingType.HasValue)
                {
                    var cacheView = Cache.CacheData(host, new Cache.CacheInput()
                    {
                        Data = roleMappedData.Data,
                        Caching = cachingType.Value
                    }).OutputData;
                    cachedRoleMappedData = new RoleMappedData(cacheView, roleMappedData.Schema.GetColumnRoleNames());
                }

                var predictor = TrainUtils.Train(host, ch, cachedRoleMappedData, trainer, calibrator, maxCalibrationExamples);
                return new TOut() { PredictorModel = new PredictorModelImpl(host, roleMappedData, input.TrainingData, predictor) };
            }
        }
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
            Optional<string> WeightColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInputWithWeight : ITrainerInputWithLabel
        {
            Optional<string> WeightColumn { get; }
        }

        /// <summary>
        /// Interface that all API trainer input classes will implement.
        /// </summary>
        public interface ITrainerInputWithGroupId : ITrainerInputWithWeight
        {
            Optional<string> GroupIdColumn { get; }
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
