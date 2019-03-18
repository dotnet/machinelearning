// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.EntryPoints
{
    [BestFriend]
    internal enum CachingOptions
    {
        Auto,
        Memory,
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

    [BestFriend]
    internal static class TrainerEntryPointsUtils
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
            where TArg : TrainerInputBase
            where TOut : CommonOutputs.TrainerOutput, new()
        {
            using (var ch = host.Start("Training"))
            {
                var schema = input.TrainingData.Schema;
                var feature = FindColumn(ch, schema, input.FeatureColumnName);
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
                const string registrationName = "CreateCache";
                var createCacheHost = host.Register(registrationName);
                IDataView outputData = null;

                switch (input.Caching)
                {
                    case CachingOptions.Memory:
                        {
                            outputData = new CacheDataView(host, roleMappedData.Data, null);
                            break;
                        }
                    case CachingOptions.Auto:
                        {
                            // REVIEW: we should switch to hybrid caching in future.
                            if (!(input.TrainingData is BinaryLoader) && trainer.Info.WantCaching)
                                // default to Memory so mml is on par with maml
                                outputData = new CacheDataView(host, roleMappedData.Data, null);
                            break;
                        }
                    case CachingOptions.None:
                        break;
                    default:
                        throw ch.ExceptParam(nameof(input.Caching), "Unknown option for caching: '{0}'", input.Caching);
                }

                if (outputData != null)
                {
                    cachedRoleMappedData = new RoleMappedData(outputData, roleMappedData.Schema.GetColumnRoleNames());
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
