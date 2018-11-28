// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

// The warning #612 is disabled because the following code uses a lot of things in Legacy.Models while Legacy.Model is marked as obsolete.
// Because that dependency will be removed form ML.NET, one needs to rewrite all places where legacy APIs are used.
#pragma warning disable 612
namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class MacroUtils
    {
        /// <summary>
        /// Lists the types of trainer signatures. Used by entry points and autoML system
        /// to know what types of evaluators to use for the train test / pipeline sweeper.
        /// </summary>
        public enum TrainerKinds
        {
            SignatureBinaryClassifierTrainer,
            SignatureMultiClassClassifierTrainer,
            SignatureRankerTrainer,
            SignatureRegressorTrainer,
            SignatureMultiOutputRegressorTrainer,
            SignatureAnomalyDetectorTrainer,
            SignatureClusteringTrainer,
        }

        public sealed class EvaluatorSettings
        {
            public string LabelColumn { get; set; }
            public string NameColumn { get; set; }
            public string WeightColumn { get; set; }
            public string GroupColumn { get; set; }
            public string FeatureColumn { get; set; }

            public EvaluatorSettings()
            {
                LabelColumn = DefaultColumnNames.Label;
            }
        }

        private sealed class TaskInformationBundle
        {
            public string TrainerFunctionName { get; set; }
            public Type TrainerSignatureType { get; set; }
            public Func<EvaluatorSettings, CommonInputs.IEvaluatorInput> EvaluatorInput { get; set; }
            public Func<CommonOutputs.IEvaluatorOutput> EvaluatorOutput { get; set; }
        }

        private static Dictionary<TrainerKinds, TaskInformationBundle>
            TrainerKindDict => new Dictionary<TrainerKinds, TaskInformationBundle>
            {
                {
                    TrainerKinds.SignatureBinaryClassifierTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "BinaryClassifier",
                        TrainerSignatureType = typeof(SignatureBinaryClassifierTrainer),
                        EvaluatorInput = settings => new Legacy.Models.BinaryClassificationEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.BinaryClassificationEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureMultiClassClassifierTrainer,
                    new TaskInformationBundle{
                        TrainerFunctionName = "Classifier",
                        TrainerSignatureType = typeof(SignatureMultiClassClassifierTrainer),
                        EvaluatorInput = settings => new Legacy.Models.ClassificationEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.ClassificationEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureRankerTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "Ranker",
                        TrainerSignatureType = typeof(SignatureRankerTrainer),
                        EvaluatorInput = settings => new Legacy.Models.RankerEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn,
                            GroupIdColumn = settings.GroupColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.RankerEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureRegressorTrainer,
                    new TaskInformationBundle{
                        TrainerFunctionName = "Regressor",
                        TrainerSignatureType = typeof(SignatureRegressorTrainer),
                        EvaluatorInput = settings => new Legacy.Models.RegressionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.RegressionEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureMultiOutputRegressorTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "MultiOutputRegressor",
                        TrainerSignatureType = typeof(SignatureMultiOutputRegressorTrainer),
                        EvaluatorInput = settings => new Legacy.Models.MultiOutputRegressionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn,
                        },
                        EvaluatorOutput = () => new Legacy.Models.MultiOutputRegressionEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureAnomalyDetectorTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "AnomalyDetector",
                        TrainerSignatureType = typeof(SignatureAnomalyDetectorTrainer),
                        EvaluatorInput = settings => new Legacy.Models.AnomalyDetectionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.AnomalyDetectionEvaluator.Output()
                        }
                },
                {
                    TrainerKinds.SignatureClusteringTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "Clusterer",
                        TrainerSignatureType = typeof(SignatureClusteringTrainer),
                        EvaluatorInput = settings => new Legacy.Models.ClusterEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            WeightColumn = settings.WeightColumn,
                            FeatureColumn = settings.FeatureColumn
                        },
                        EvaluatorOutput = () => new Legacy.Models.ClusterEvaluator.Output()
                    }
                },
            };

        public static Tuple<CommonInputs.IEvaluatorInput, CommonOutputs.IEvaluatorOutput> GetEvaluatorInputOutput(
            TrainerKinds kind, EvaluatorSettings settings = null) => new Tuple<CommonInputs.IEvaluatorInput, CommonOutputs.IEvaluatorOutput>
            (TrainerKindDict[kind].EvaluatorInput(settings), TrainerKindDict[kind].EvaluatorOutput());

        public static Type[] PredictorTypes = TrainerKindDict.Select(kvp => kvp.Value.TrainerSignatureType).ToArray();

        public static Type TrainerKindToType(TrainerKinds kind) => TrainerKindDict[kind].TrainerSignatureType;

        public static TrainerKinds SignatureTypeToTrainerKind(Type sigType)
        {
            foreach (var kvp in TrainerKindDict)
                if (sigType == kvp.Value.TrainerSignatureType)
                    return kvp.Key;
            throw new NotSupportedException($"Signature type {sigType} unsupported.");
        }

        public static TrainerKinds[] SignatureTypesToTrainerKinds(IEnumerable<Type> sigTypes) =>
            sigTypes.Select(SignatureTypeToTrainerKind).ToArray();

        private static string GetTrainerName(TrainerKinds kind) => TrainerKindDict[kind].TrainerFunctionName;

        public static T TrainerKindApiValue<T>(TrainerKinds trainerKind)
        {
            if (Enum.GetName(typeof(TrainerKinds), trainerKind) is string name)
                return (T)Enum.Parse(typeof(T), name);
            throw new Exception($"Could not interpret enum value: {trainerKind}");
        }

        public static bool IsTrainerOfKind(Type type, TrainerKinds trainerKind)
        {
            if (trainerKind != TrainerKinds.SignatureMultiClassClassifierTrainer && trainerKind != TrainerKinds.SignatureMultiOutputRegressorTrainer)
                return type.Name.EndsWith(GetTrainerName(trainerKind));

            if (trainerKind == TrainerKinds.SignatureMultiClassClassifierTrainer)
                return type.Name.EndsWith(GetTrainerName(trainerKind)) && !type.Name.EndsWith(GetTrainerName(TrainerKinds.SignatureBinaryClassifierTrainer));

            return type.Name.EndsWith(GetTrainerName(trainerKind)) && !type.Name.EndsWith(GetTrainerName(TrainerKinds.SignatureRegressorTrainer));
        }
    }
}
#pragma warning restore 612
