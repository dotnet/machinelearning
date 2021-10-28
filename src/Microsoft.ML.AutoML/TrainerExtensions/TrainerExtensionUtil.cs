// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    internal enum TrainerName
    {
        AveragedPerceptronBinary,
        AveragedPerceptronOva,
        FastForestBinary,
        FastForestOva,
        FastForestRegression,
        FastTreeBinary,
        FastTreeOva,
        FastTreeRegression,
        FastTreeTweedieRegression,
        LightGbmBinary,
        LightGbmMulti,
        LightGbmRegression,
        LinearSvmBinary,
        LinearSvmOva,
        LbfgsLogisticRegressionBinary,
        LbfgsLogisticRegressionOva,
        LbfgsMaximumEntropyMulti,
        OnlineGradientDescentRegression,
        OlsRegression,
        Ova,
        LbfgsPoissonRegression,
        SdcaLogisticRegressionBinary,
        SdcaMaximumEntropyMulti,
        SdcaRegression,
        SgdCalibratedBinary,
        SgdCalibratedOva,
        SymbolicSgdLogisticRegressionBinary,
        SymbolicSgdLogisticRegressionOva,
        MatrixFactorization,
        ImageClassification,
        LightGbmRanking,
        FastTreeRanking
    }

    internal static class TrainerExtensionUtil
    {
        private const string WeightColumn = "ExampleWeightColumnName";
        private const string LabelColumn = "LabelColumnName";
        private const string GroupColumn = "GroupColumnName";

        public static T CreateOptions<T>(IEnumerable<SweepableParam> sweepParams, string labelColumn) where T : TrainerInputBaseWithLabel
        {
            var options = Activator.CreateInstance<T>();
            options.LabelColumnName = labelColumn;
            if (sweepParams != null)
            {
                UpdateFields(options, sweepParams);
            }
            return options;
        }

        public static T CreateOptions<T>(IEnumerable<SweepableParam> sweepParams) where T : class
        {
            var options = Activator.CreateInstance<T>();
            if (sweepParams != null)
            {
                UpdateFields(options, sweepParams);
            }
            return options;
        }

        private static readonly string[] _lightGbmBoosterParamNames = new[] { "L2Regularization", "L1Regularization" };
        private const string LightGbmBoosterPropName = "Booster";

        public static TOptions CreateLightGbmOptions<TOptions, TOutput, TTransformer, TModel>(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
            where TOptions : LightGbmTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase, new()
            where TTransformer : ISingleFeaturePredictionTransformer<TModel>
            where TModel : class
        {
            var options = new TOptions();
            options.LabelColumnName = columnInfo.LabelColumnName;
            options.ExampleWeightColumnName = columnInfo.ExampleWeightColumnName;
            options.Booster = new GradientBooster.Options();
            if (sweepParams != null)
            {
                var boosterParams = sweepParams.Where(p => _lightGbmBoosterParamNames.Contains(p.Name));
                var parentArgParams = sweepParams.Except(boosterParams);
                UpdateFields(options, parentArgParams);
                UpdateFields(options.Booster, boosterParams);
            }
            return options;
        }

        public static PipelineNode BuildOvaPipelineNode(ITrainerExtension multiExtension, ITrainerExtension binaryExtension,
            IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var ovaNode = new PipelineNode()
            {
                Name = TrainerName.Ova.ToString(),
                NodeType = PipelineNodeType.Trainer,
                Properties = new Dictionary<string, object>()
                {
                    { LabelColumn, columnInfo.LabelColumnName }
                }
            };
            var binaryNode = binaryExtension.CreatePipelineNode(sweepParams, columnInfo);
            ovaNode.Properties["BinaryTrainer"] = binaryNode;
            return ovaNode;
        }

        public static PipelineNode BuildPipelineNode(TrainerName trainerName, IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn = null, IDictionary<string, object> additionalProperties = null)
        {
            var properties = BuildBasePipelineNodeProps(sweepParams, labelColumn, weightColumn);

            if (additionalProperties != null)
            {
                foreach (var property in additionalProperties)
                {
                    properties[property.Key] = property.Value;
                }
            }

            return new PipelineNode(trainerName.ToString(), PipelineNodeType.Trainer, DefaultColumnNames.Features,
                DefaultColumnNames.Score, properties);
        }

        public static PipelineNode BuildLightGbmPipelineNode(TrainerName trainerName, IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn, string groupColumn)
        {
            return new PipelineNode(trainerName.ToString(), PipelineNodeType.Trainer, DefaultColumnNames.Features,
                DefaultColumnNames.Score, BuildLightGbmPipelineNodeProps(sweepParams, labelColumn, weightColumn, groupColumn));
        }

        private static IDictionary<string, object> BuildBasePipelineNodeProps(IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn)
        {
            var props = new Dictionary<string, object>();
            if (sweepParams != null)
            {
                foreach (var sweepParam in sweepParams)
                {
                    props[sweepParam.Name] = sweepParam.ProcessedValue();
                }
            }
            props[LabelColumn] = labelColumn;
            if (weightColumn != null)
            {
                props[WeightColumn] = weightColumn;
            }
            return props;
        }

        private static IDictionary<string, object> BuildLightGbmPipelineNodeProps(IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn, string groupColumn)
        {
            Dictionary<string, object> props = null;
            if (sweepParams == null || !sweepParams.Any())
            {
                props = new Dictionary<string, object>();
            }
            else
            {
                var boosterParams = sweepParams.Where(p => _lightGbmBoosterParamNames.Contains(p.Name));
                var parentArgParams = sweepParams.Except(boosterParams);

                var boosterProps = boosterParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
                var boosterCustomProp = new CustomProperty("GradientBooster.Options", boosterProps);

                props = parentArgParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
                props[LightGbmBoosterPropName] = boosterCustomProp;
            }

            props[LabelColumn] = labelColumn;
            if (weightColumn != null)
            {
                props[WeightColumn] = weightColumn;
            }
            if (groupColumn != null)
            {
                props[GroupColumn] = groupColumn;
            }

            return props;
        }

        public static ParameterSet BuildParameterSet(TrainerName trainerName, IDictionary<string, object> props)
        {
            props = props.Where(p => p.Key != LabelColumn && p.Key != WeightColumn)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            if (trainerName == TrainerName.LightGbmBinary || trainerName == TrainerName.LightGbmMulti ||
                trainerName == TrainerName.LightGbmRegression || trainerName == TrainerName.LightGbmRanking)
            {
                return BuildLightGbmParameterSet(props);
            }

            var paramVals = props.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            return new ParameterSet(paramVals);
        }

        public static ColumnInformation BuildColumnInfo(IDictionary<string, object> props)
        {
            var columnInfo = new ColumnInformation();

            columnInfo.LabelColumnName = props[LabelColumn] as string;

            props.TryGetValue(WeightColumn, out var weightColumn);
            columnInfo.ExampleWeightColumnName = weightColumn as string;

            return columnInfo;
        }

        private static ParameterSet BuildLightGbmParameterSet(IDictionary<string, object> props)
        {
            IEnumerable<IParameterValue> parameters;
            if (props == null || !props.Any())
            {
                parameters = new List<IParameterValue>();
            }
            else
            {
                var parentProps = props.Where(p => p.Key != LightGbmBoosterPropName);
                var treeProps = ((CustomProperty)props[LightGbmBoosterPropName]).Properties;
                var allProps = parentProps.Union(treeProps);
                parameters = allProps.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            }
            return new ParameterSet(parameters);
        }

        private static void SetValue(FieldInfo fi, IComparable value, object obj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                fi.SetValue(obj, value);
            else if (propertyType == typeof(double) && value is float)
                fi.SetValue(obj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                fi.SetValue(obj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                fi.SetValue(obj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of object instance based on the values in sweepParams
        /// </summary>
        public static void UpdateFields(object obj, IEnumerable<SweepableParam> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    if (param.RawValue == null)
                    {
                        continue;
                    }
                    var fi = obj.GetType().GetField(param.Name);
                    var propType = Nullable.GetUnderlyingType(fi.FieldType) ?? fi.FieldType;

                    if (param is SweepableDiscreteParam dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(fi.FieldType) != null)
                                fi.SetValue(obj, null);
                            else if (fi.FieldType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = fi.FieldType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(fi.FieldType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    fi.SetValue(obj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(fi, (IComparable)dp.Options[optIndex], obj, propType);
                    }
                    else
                        SetValue(fi, param.RawValue, obj, propType);
                }
                catch (Exception)
                {
                    throw new InvalidOperationException($"Cannot set parameter {param.Name} for {obj.GetType()}");
                }
            }
        }

        public static TrainerName GetTrainerName(BinaryClassificationTrainer binaryTrainer)
        {
            switch (binaryTrainer)
            {
                case BinaryClassificationTrainer.AveragedPerceptron:
                    return TrainerName.AveragedPerceptronBinary;
                case BinaryClassificationTrainer.FastForest:
                    return TrainerName.FastForestBinary;
                case BinaryClassificationTrainer.FastTree:
                    return TrainerName.FastTreeBinary;
                case BinaryClassificationTrainer.LightGbm:
                    return TrainerName.LightGbmBinary;
                case BinaryClassificationTrainer.LinearSvm:
                    return TrainerName.LinearSvmBinary;
                case BinaryClassificationTrainer.LbfgsLogisticRegression:
                    return TrainerName.LbfgsLogisticRegressionBinary;
                case BinaryClassificationTrainer.SdcaLogisticRegression:
                    return TrainerName.SdcaLogisticRegressionBinary;
                case BinaryClassificationTrainer.SgdCalibrated:
                    return TrainerName.SgdCalibratedBinary;
                case BinaryClassificationTrainer.SymbolicSgdLogisticRegression:
                    return TrainerName.SymbolicSgdLogisticRegressionBinary;
            }

            // never expected to reach here
            throw new NotSupportedException($"{binaryTrainer} not supported");
        }

        public static TrainerName GetTrainerName(MulticlassClassificationTrainer multiTrainer)
        {
            switch (multiTrainer)
            {
                case MulticlassClassificationTrainer.AveragedPerceptronOva:
                    return TrainerName.AveragedPerceptronOva;
                case MulticlassClassificationTrainer.FastForestOva:
                    return TrainerName.FastForestOva;
                case MulticlassClassificationTrainer.FastTreeOva:
                    return TrainerName.FastTreeOva;
                case MulticlassClassificationTrainer.LightGbm:
                    return TrainerName.LightGbmMulti;
                case MulticlassClassificationTrainer.LinearSupportVectorMachinesOva:
                    return TrainerName.LinearSvmOva;
                case MulticlassClassificationTrainer.LbfgsMaximumEntropy:
                    return TrainerName.LbfgsMaximumEntropyMulti;
                case MulticlassClassificationTrainer.LbfgsLogisticRegressionOva:
                    return TrainerName.LbfgsLogisticRegressionOva;
                case MulticlassClassificationTrainer.SdcaMaximumEntropy:
                    return TrainerName.SdcaMaximumEntropyMulti;
                case MulticlassClassificationTrainer.SgdCalibratedOva:
                    return TrainerName.SgdCalibratedOva;
                case MulticlassClassificationTrainer.SymbolicSgdLogisticRegressionOva:
                    return TrainerName.SymbolicSgdLogisticRegressionOva;
            }

            // never expected to reach here
            throw new NotSupportedException($"{multiTrainer} not supported");
        }

        public static TrainerName GetTrainerName(RegressionTrainer regressionTrainer)
        {
            switch (regressionTrainer)
            {
                case RegressionTrainer.FastForest:
                    return TrainerName.FastForestRegression;
                case RegressionTrainer.FastTree:
                    return TrainerName.FastTreeRegression;
                case RegressionTrainer.FastTreeTweedie:
                    return TrainerName.FastTreeTweedieRegression;
                case RegressionTrainer.LightGbm:
                    return TrainerName.LightGbmRegression;
                case RegressionTrainer.OnlineGradientDescent:
                    return TrainerName.OnlineGradientDescentRegression;
                case RegressionTrainer.Ols:
                    return TrainerName.OlsRegression;
                case RegressionTrainer.LbfgsPoissonRegression:
                    return TrainerName.LbfgsPoissonRegression;
                case RegressionTrainer.StochasticDualCoordinateAscent:
                    return TrainerName.SdcaRegression;
            }

            // never expected to reach here
            throw new NotSupportedException($"{regressionTrainer} not supported");
        }

        public static TrainerName GetTrainerName(RankingTrainer rankingTrainer)
        {
            switch (rankingTrainer)
            {
                case RankingTrainer.FastTreeRanking:
                    return TrainerName.FastTreeRanking;
                case RankingTrainer.LightGbmRanking:
                    return TrainerName.LightGbmRanking;
            }

            // never expected to reach here
            throw new NotSupportedException($"{rankingTrainer} not supported");
        }

        public static TrainerName GetTrainerName(RecommendationTrainer recommendationTrainer)
        {
            switch (recommendationTrainer)
            {
                case RecommendationTrainer.MatrixFactorization:
                    return TrainerName.MatrixFactorization;
            }

            // never expected to reach here
            throw new NotSupportedException($"{recommendationTrainer} not supported");
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<BinaryClassificationTrainer> binaryTrainers)
        {
            return binaryTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<MulticlassClassificationTrainer> multiTrainers)
        {
            return multiTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<RegressionTrainer> regressionTrainers)
        {
            return regressionTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<RecommendationTrainer> recommendationTrainers)
        {
            return recommendationTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<RankingTrainer> rankingTrainers)
        {
            return rankingTrainers?.Select(t => GetTrainerName(t));
        }
    }
}
