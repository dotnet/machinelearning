// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

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
            public string ScoreColumn { get; set; }
            public string[] StratColumn { get; set; }
            public string WeightColumn { get; set; }
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
                        EvaluatorInput = settings => new Models.BinaryClassificationEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Models.BinaryClassificationEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureMultiClassClassifierTrainer,
                    new TaskInformationBundle{
                        TrainerFunctionName = "Classifier",
                        TrainerSignatureType = typeof(SignatureMultiClassClassifierTrainer),
                        EvaluatorInput = settings => new Models.ClassificationEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Models.ClassificationEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureRankerTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "Ranker",
                        TrainerSignatureType = typeof(SignatureRankerTrainer),
                        EvaluatorInput = settings => new Models.RankerEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Models.RankerEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureRegressorTrainer,
                    new TaskInformationBundle{
                        TrainerFunctionName = "Regressor",
                        TrainerSignatureType = typeof(SignatureRegressorTrainer),
                        EvaluatorInput = settings => new Models.RegressionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Models.RegressionEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureMultiOutputRegressorTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "MultiOutputRegressor",
                        TrainerSignatureType = typeof(SignatureMultiOutputRegressorTrainer),
                        EvaluatorInput = settings => new Models.MultiOutputRegressionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn,
                        },
                        EvaluatorOutput = () => new Models.MultiOutputRegressionEvaluator.Output()
                    }
                },
                {
                    TrainerKinds.SignatureAnomalyDetectorTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "AnomalyDetector",
                        TrainerSignatureType = typeof(SignatureAnomalyDetectorTrainer),
                        EvaluatorInput = settings => new Models.AnomalyDetectionEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn
                        },
                        EvaluatorOutput = () => new Models.AnomalyDetectionEvaluator.Output()
                        }
                },
                {
                    TrainerKinds.SignatureClusteringTrainer,
                    new TaskInformationBundle {
                        TrainerFunctionName = "Clusterer",
                        TrainerSignatureType = typeof(SignatureClusteringTrainer),
                        EvaluatorInput = settings => new Models.ClusterEvaluator
                        {
                            LabelColumn = settings.LabelColumn,
                            NameColumn = settings.NameColumn,
                            ScoreColumn = settings.ScoreColumn,
                            StratColumn = settings.StratColumn,
                            WeightColumn = settings.WeightColumn,
                            FeatureColumn = settings.FeatureColumn
                        },
                        EvaluatorOutput = () => new Models.ClusterEvaluator.Output()
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
