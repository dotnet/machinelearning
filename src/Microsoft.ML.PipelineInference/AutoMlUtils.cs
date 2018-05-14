// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.Sweeper;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class AutoMlUtils
    {
        public static AutoInference.RunSummary ExtractRunSummary(IHostEnvironment env, IDataView result, string metricColumnName, IDataView trainResult = null)
        {
            double testingMetricValue = 0;
            double trainingMetricValue = -1d;
            int numRows = 0;
            var schema = result.Schema;
            bool hasIndex = schema.TryGetColumnIndex(metricColumnName, out var metricCol);
            env.Check(hasIndex);

            using (var cursor = result.GetRowCursor(col => col == metricCol))
            {
                var getter = cursor.GetGetter<double>(metricCol);
                bool moved = cursor.MoveNext();
                env.Check(moved);
                getter(ref testingMetricValue);
            }

            if (trainResult != null)
            {
                var trainSchema = trainResult.Schema;
                env.Check(trainSchema.TryGetColumnIndex(metricColumnName, out var trainingMetricCol));

                using (var cursor = trainResult.GetRowCursor(col => col == trainingMetricCol))
                {
                    var getter = cursor.GetGetter<double>(trainingMetricCol);
                    bool moved = cursor.MoveNext();
                    env.Check(moved);
                    getter(ref trainingMetricValue);
                }
            }

            return new AutoInference.RunSummary(testingMetricValue, numRows, 0, trainingMetricValue);
        }

        public static CommonInputs.IEvaluatorInput CloneEvaluatorInstance(CommonInputs.IEvaluatorInput evalInput) =>
            CloneEvaluatorInstance<CommonInputs.IEvaluatorInput>(evalInput);

        public static CommonOutputs.IEvaluatorOutput CloneEvaluatorInstance(CommonOutputs.IEvaluatorOutput evalOutput) =>
            CloneEvaluatorInstance<CommonOutputs.IEvaluatorOutput>(evalOutput);

        private static T CloneEvaluatorInstance<T>(T evaler)
        {
            Type instanceType = evaler.GetType();
            T newInstance = (T)Activator.CreateInstance(instanceType);
            if (evaler is CommonOutputs.IEvaluatorOutput) return newInstance;
            foreach (var prop in instanceType.GetProperties(BindingFlags.Instance | BindingFlags.Public))
                prop.SetValue(newInstance, prop.GetValue(evaler));
            return newInstance;
        }

        /// <summary>
        /// Using the dependencyMapping and included transforms, determines whether every
        /// transform present only consumes columns produced by a lower- or same-level transform, 
        /// or existed in the original dataset. Note, a column could be produced by a 
        /// transform on the same level, such as in multipart (atomic group) transforms.
        /// </summary>
        public static bool AreColumnsConsistent(TransformInference.SuggestedTransform[] includedTransforms,
            AutoInference.DependencyMap dependencyMapping)
        {
            foreach (var transform in includedTransforms)
                foreach (var colConsumed in transform.RoutingStructure.ColumnsConsumed)
                {
                    AutoInference.LevelDependencyMap ldm = dependencyMapping[transform.RoutingStructure.Level];
                    var colInfo = ldm.Keys.FirstOrDefault(k => k.Name == colConsumed.Name);

                    // Consumed column does not exist at this sublevel. Since we never drop columns
                    // it will not exist at any lower levels, either. Thus, problem with column consumption.
                    if (colInfo.Name == null)
                        return false;

                    // If this column could have been produced by a transform, make sure at least one
                    // of the possible producer transforms in in our included transforms list.
                    if (ldm[colInfo].Count > 0 && !ldm[colInfo].Any(t => includedTransforms.Contains(t)))
                        return false;
                }

            // Passed all tests
            return true;
        }

        public static bool IsValidTransformsPipeline(long transformsBitMask, TransformInference.SuggestedTransform[] selectedAndFinalTransforms,
            TransformInference.SuggestedTransform[] allTransforms, AutoInference.DependencyMap dependencyMapping)
        {
            // If no transforms and none selected, valid.
            if (transformsBitMask == 0 && allTransforms.Length == 0)
                return true;

            // If including transforms that aren't there, invalid pipeline
            if (transformsBitMask > 0 && allTransforms.Length == 0)
                return false;

            var graph = BuildAtomicIdDependencyGraph(allTransforms);
            var selectedInitialTransforms =
                allTransforms.Where(t => AtomicGroupPresent(transformsBitMask, t.AtomicGroupId)).ToArray();

            // Make sure all necessary atomic groups are present, beginning with last level
            for (int l = allTransforms.Select(t => t.RoutingStructure.Level).DefaultIfEmpty(0).Max(); l > 0; l--)
            {
                int level = l; // To avoid complaint about access to modified closure
                var subset = allTransforms.Where(t => t.RoutingStructure.Level == level);
                var atomicIdsForLevel = subset.Select(t => t.AtomicGroupId).Distinct().ToArray();
                if (atomicIdsForLevel.Any(a =>
                    AtomicGroupPresent(transformsBitMask, a) &&
                    !graph[a].All(r => AtomicGroupPresent(transformsBitMask, r))))
                    return false;
            }

            // Make sure each transform only consumes columns actually produced by
            // a lower-level transform, or existed in original dataset.
            if (!AreColumnsConsistent(selectedInitialTransforms, dependencyMapping))
                return false;

            // Make sure has numeric vector Features column
            if (!HasFinalFeatures(selectedAndFinalTransforms, dependencyMapping))
                return false;

            // Passed all tests
            return true;
        }

        private static bool HasFinalFeatures(TransformInference.SuggestedTransform[] transforms,
            AutoInference.DependencyMap dependencyMapping) => HasFinalFeaturesColumnTransform(transforms) || HasInitialNumericFeatures(dependencyMapping);

        private static bool HasFinalFeaturesColumnTransform(TransformInference.SuggestedTransform[] transforms) =>
            transforms.Any(t => t.ExpertType == typeof(TransformInference.Experts.FeaturesColumnConcatRenameNumericOnly));

        private static bool HasInitialNumericFeatures(AutoInference.DependencyMap dependencyMapping)
        {
            if (dependencyMapping.Count == 0)
                return false;
            foreach (var info in dependencyMapping[0])
            {
                if (info.Key.Name == DefaultColumnNames.Features &&
                    !info.Key.IsHidden &&
                    info.Key.ItemType.IsNumber &&
                    info.Value.Count == 0)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Simple wrapper which allows the call signature to match the signature needed for the PipelineOptimizerBase interface.
        /// </summary>
        public static Func<PipelinePattern, long, bool> ValidationWrapper(TransformInference.SuggestedTransform[] allTransforms, AutoInference.DependencyMap dependencyMapping)
        {
            return (p, b) => IsValidTransformsPipeline(b, p.Transforms, allTransforms.Union(p.Transforms).ToArray(), dependencyMapping);
        }

        /// <summary>
        /// Using the dependencyMapping and included transforms, computes which subset of columns in dataSample
        /// will be present in the final transformed dataset when only the transforms present are applied.
        /// </summary>
        private static int[] GetExcludedColumnIndices(TransformInference.SuggestedTransform[] includedTransforms, IDataView dataSample,
            AutoInference.DependencyMap dependencyMapping)
        {
            List<int> includedColumnIndices = new List<int>();

            // For every column, see if either present in initial dataset, or 
            // produced by a transform used in current pipeline.               
            for (int columnIndex = 0; columnIndex < dataSample.Schema.ColumnCount; columnIndex++)
            {
                // Create ColumnInfo object for indexing dictionary
                var colInfo = new AutoInference.ColumnInfo
                {
                    Name = dataSample.Schema.GetColumnName(columnIndex),
                    ItemType = dataSample.Schema.GetColumnType(columnIndex).ItemType,
                    IsHidden = dataSample.Schema.IsHidden(columnIndex)
                };

                // Exclude all hidden and non-numeric columns 
                if (colInfo.IsHidden || !colInfo.ItemType.IsNumber)
                    continue;

                foreach (var level in dependencyMapping.Keys.Reverse())
                {
                    var levelResponsibilities = dependencyMapping[level];

                    if (!levelResponsibilities.ContainsKey(colInfo))
                        continue;

                    // Include any numeric column present in initial dataset. Does not need
                    // any transforms applied to be present in final dataset.
                    if (level == 0 && colInfo.ItemType.IsNumber && levelResponsibilities[colInfo].Count == 0)
                    {
                        includedColumnIndices.Add(columnIndex);
                        break;
                    }

                    // If column could not have been produced by transforms at this level, move down to the next level.
                    if (levelResponsibilities[colInfo].Count == 0)
                        continue;

                    // Check if could have been produced by any transform in this pipeline
                    if (levelResponsibilities[colInfo].Any(t => includedTransforms.Contains(t)))
                        includedColumnIndices.Add(columnIndex);
                }
            }

            // Exclude all columns not discovered by our inclusion process
            return Enumerable.Range(0, dataSample.Schema.ColumnCount).Except(includedColumnIndices).ToArray();
        }

        /// <summary>
        /// Builds dependency dictionary of which atomic groups depend on which. Assumes that
        /// a column will be produced by the highest level possible producing transform.
        /// </summary>
        /// <param name="allTransforms">The set of possible transforms for a dataset for all levels.</param>
        private static Dictionary<int, List<int>> BuildAtomicIdDependencyGraph(TransformInference.SuggestedTransform[] allTransforms)
        {
            var graph = new Dictionary<int, List<int>>();

            foreach (var transform in allTransforms)
            {
                var route = transform.RoutingStructure;

                foreach (var columnConsumedName in route.ColumnsConsumed)
                {
                    if (!graph.ContainsKey(transform.AtomicGroupId))
                        graph.Add(transform.AtomicGroupId, new List<int>());
                    var possibleProducers =
                        allTransforms.Where(t => t.RoutingStructure.ColumnsProduced.Contains(columnConsumedName) &&
                            !t.Equals(transform)).ToList();
                    if (possibleProducers.Count == 0)
                        continue;
                    var bestCandidate = possibleProducers.OrderByDescending(t => t.RoutingStructure.Level).First();
                    var index = allTransforms.ToList().IndexOf(bestCandidate);
                    var atomicId = allTransforms[index].AtomicGroupId;
                    graph[transform.AtomicGroupId].Add(atomicId);
                }
            }

            return graph;
        }

        public static bool AtomicGroupPresent(long bitmask, int atomicGroupId) => (bitmask & (1 << atomicGroupId)) > 0;

        public static long TransformsToBitmask(TransformInference.SuggestedTransform[] transforms) =>
            transforms.Aggregate(0, (current, t) => current | 1 << t.AtomicGroupId);

        /// <summary>
        /// Gets a final transform to concatenate all numeric columns into a "Features" vector column.
        /// Note: May return empty set if Features column already present and is only relevant numeric column.
        /// (In other words, if there would be nothing for that concatenate transform to do.)
        /// </summary>
        private static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(IHostEnvironment env,
            IDataView dataSample, int[] excludedColumnIndices, int level, int atomicIdOffset)
        {
            var finalArgs = new TransformInference.Arguments
            {
                EstimatedSampleFraction = 1.0,
                ExcludeFeaturesConcatTransforms = false,
                ExcludedColumnIndices = excludedColumnIndices
            };

            var featuresConcatTransforms = TransformInference.InferConcatNumericFeatures(env, dataSample, finalArgs);

            for (int i = 0; i < featuresConcatTransforms.Length; i++)
            {
                featuresConcatTransforms[i].RoutingStructure.Level = level;
                featuresConcatTransforms[i].AtomicGroupId += atomicIdOffset;
            }

            return featuresConcatTransforms.ToArray();
        }

        /// <summary>
        /// Exposed version of the method.
        /// </summary>
        public static TransformInference.SuggestedTransform[] GetFinalFeatureConcat(IHostEnvironment env, IDataView data,
            AutoInference.DependencyMap dependencyMapping, TransformInference.SuggestedTransform[] selectedTransforms,
            TransformInference.SuggestedTransform[] allTransforms)
        {
            int level = 1;
            int atomicGroupLimit = 0;
            if (allTransforms.Length != 0)
            {
                level = allTransforms.Max(t => t.RoutingStructure.Level) + 1;
                atomicGroupLimit = allTransforms.Max(t => t.AtomicGroupId) + 1;
            }
            var excludedColumnIndices = GetExcludedColumnIndices(selectedTransforms, data, dependencyMapping);
            return GetFinalFeatureConcat(env, data, excludedColumnIndices, level, atomicGroupLimit);
        }

        public static IDataView ApplyTransformSet(IHostEnvironment env, IDataView data, TransformInference.SuggestedTransform[] transforms)
        {
            // Double-check all transforms are supported (i.e., have pipleline nodes)
            transforms = transforms.Where(t => t.PipelineNode != null).ToArray();
            // Build experiment graph and run on data, using transforms
            var recipe = new RecipeInference.SuggestedRecipe("dummy", transforms,
                new RecipeInference.SuggestedRecipe.SuggestedLearner[0]);
            var epGraph = recipe.ToEntryPointGraph(env);
            epGraph.Graph.Compile();
            epGraph.Graph.SetInput(epGraph.GetSubgraphFirstNodeDataVarName(env), data);
            epGraph.Graph.Run();
            // Get dataview output after all transforms applied
            return epGraph.Graph.GetOutput(epGraph.TransformsOutputData);
        }

        /// <summary>
        /// Creates a dictionary mapping column names to the transforms which could have produced them.
        /// </summary>
        public static AutoInference.LevelDependencyMap ComputeColumnResponsibilities(IDataView transformedData,
            TransformInference.SuggestedTransform[] appliedTransforms)
        {
            var mapping = new AutoInference.LevelDependencyMap();
            for (int i = 0; i < transformedData.Schema.ColumnCount; i++)
            {
                if (transformedData.Schema.IsHidden(i))
                    continue;
                var colInfo = new AutoInference.ColumnInfo
                {
                    IsHidden = false,
                    ItemType = transformedData.Schema.GetColumnType(i).ItemType,
                    Name = transformedData.Schema.GetColumnName(i)
                };
                mapping.Add(colInfo, appliedTransforms.Where(t =>
                    t.RoutingStructure.ColumnsProduced.Any(o => o.Name == colInfo.Name &&
                    o.IsNumeric == transformedData.Schema.GetColumnType(i).ItemType.IsNumber)).ToList());
            }
            return mapping;
        }

        public static TlcModule.SweepableParamAttribute[] GetSweepRanges(Type learnerInputType)
        {
            var paramSet = new List<TlcModule.SweepableParamAttribute>();
            foreach (var prop in learnerInputType.GetProperties(BindingFlags.Instance |
                                                                BindingFlags.Static |
                                                                BindingFlags.Public))
            {
                if (prop.GetCustomAttributes(typeof(TlcModule.SweepableLongParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableLongParamAttribute lpAttr)
                {
                    lpAttr.Name = lpAttr.Name ?? prop.Name;
                    paramSet.Add(lpAttr);
                }

                if (prop.GetCustomAttributes(typeof(TlcModule.SweepableFloatParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableFloatParamAttribute fpAttr)
                {
                    fpAttr.Name = fpAttr.Name ?? prop.Name;
                    paramSet.Add(fpAttr);
                }

                if (prop.GetCustomAttributes(typeof(TlcModule.SweepableDiscreteParamAttribute), true).FirstOrDefault()
                    is TlcModule.SweepableDiscreteParamAttribute dpAttr)
                {
                    dpAttr.Name = dpAttr.Name ?? prop.Name;
                    paramSet.Add(dpAttr);
                }
            }

            return paramSet.ToArray();
        }

        public static IValueGenerator ToIValueGenerator(TlcModule.SweepableParamAttribute attr)
        {
            if (attr is TlcModule.SweepableLongParamAttribute sweepableLongParamAttr)
            {
                var args = new LongParamArguments
                {
                    Min = sweepableLongParamAttr.Min,
                    Max = sweepableLongParamAttr.Max,
                    LogBase = sweepableLongParamAttr.IsLogScale,
                    Name = sweepableLongParamAttr.Name,
                    StepSize = sweepableLongParamAttr.StepSize
                };
                if (sweepableLongParamAttr.NumSteps != null)
                    args.NumSteps = (int)sweepableLongParamAttr.NumSteps;
                return new LongValueGenerator(args);
            }

            if (attr is TlcModule.SweepableFloatParamAttribute sweepableFloatParamAttr)
            {
                var args = new FloatParamArguments
                {
                    Min = sweepableFloatParamAttr.Min,
                    Max = sweepableFloatParamAttr.Max,
                    LogBase = sweepableFloatParamAttr.IsLogScale,
                    Name = sweepableFloatParamAttr.Name,
                    StepSize = sweepableFloatParamAttr.StepSize
                };
                if (sweepableFloatParamAttr.NumSteps != null)
                    args.NumSteps = (int)sweepableFloatParamAttr.NumSteps;
                return new FloatValueGenerator(args);
            }

            if (attr is TlcModule.SweepableDiscreteParamAttribute sweepableDiscreteParamAttr)
            {
                var args = new DiscreteParamArguments
                {
                    Name = sweepableDiscreteParamAttr.Name,
                    Values = sweepableDiscreteParamAttr.Options.Select(o => o.ToString()).ToArray()
                };
                return new DiscreteValueGenerator(args);
            }

            throw new Exception($"Sweeping only supported for Discrete, Long, and Float parameter types. Unrecognized type {attr.GetType()}");
        }

        private static void SetValue(PropertyInfo pi, IComparable value, object entryPointObj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                pi.SetValue(entryPointObj, value);
            else if (propertyType == typeof(double) && value is float)
                pi.SetValue(entryPointObj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                pi.SetValue(entryPointObj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                pi.SetValue(entryPointObj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>        
        public static bool UpdateProperties(object entryPointObj, TlcModule.SweepableParamAttribute[] sweepParams)
        {
            bool result = true;
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    var pi = entryPointObj.GetType().GetProperty(param.Name);
                    if (pi is null || param.RawValue == null)
                        continue;
                    var propType = Nullable.GetUnderlyingType(pi.PropertyType) ?? pi.PropertyType;

                    if (param is TlcModule.SweepableDiscreteParamAttribute dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(pi.PropertyType) != null)
                                pi.SetValue(entryPointObj, null);
                            else if (pi.PropertyType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = pi.PropertyType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(pi.PropertyType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    pi.SetValue(entryPointObj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(pi, (IComparable)dp.Options[optIndex], entryPointObj, propType);
                    }
                    else
                        SetValue(pi, param.RawValue, entryPointObj, propType);
                }
                catch (Exception)
                {
                    // Could not update param
                    result = false;
                }
            }

            // Make sure all changes were saved.
            return result && CheckEntryPointStateMatchesParamValues(entryPointObj, sweepParams);
        }

        /// <summary>
        /// Updates properties of entryPointObj instance based on the values in sweepParams
        /// </summary>        
        public static void PopulateSweepableParams(RecipeInference.SuggestedRecipe.SuggestedLearner learner)
        {
            foreach (var param in learner.PipelineNode.SweepParams)
            {
                if (param is TlcModule.SweepableDiscreteParamAttribute dp)
                {
                    var learnerVal = learner.PipelineNode.GetPropertyValueByName(dp.Name, (IComparable)dp.Options[0]);
                    param.RawValue = dp.IndexOf(learnerVal);
                }
                else if (param is TlcModule.SweepableFloatParamAttribute fp)
                    param.RawValue = learner.PipelineNode.GetPropertyValueByName(fp.Name, 0f);
                else if (param is TlcModule.SweepableLongParamAttribute lp)
                    param.RawValue = learner.PipelineNode.GetPropertyValueByName(lp.Name, 0L);
            }
        }

        public static bool CheckEntryPointStateMatchesParamValues(object entryPointObj,
            TlcModule.SweepableParamAttribute[] sweepParams)
        {
            foreach (var param in sweepParams)
            {
                var pi = entryPointObj.GetType().GetProperty(param.Name);
                if (pi is null)
                    continue;

                // Make sure the value matches
                var epVal = pi.GetValue(entryPointObj);
                if (param.RawValue != null
                    && (!param.ProcessedValue().ToString().ToLower().Contains("auto") || epVal != null)
                    && !epVal.Equals(param.ProcessedValue()))
                    return false;
            }
            return true;
        }

        public static double ProcessWeight(double weight, double maxWeight, bool isMaximizingMetric) =>
            isMaximizingMetric ? weight : maxWeight - weight;

        public static long IncludeMandatoryTransforms(List<TransformInference.SuggestedTransform> availableTransforms) =>
            TransformsToBitmask(GetMandatoryTransforms(availableTransforms.ToArray()));

        public static TransformInference.SuggestedTransform[] GetMandatoryTransforms(
            TransformInference.SuggestedTransform[] availableTransforms) =>
            availableTransforms.Where(t => t.AlwaysInclude).ToArray();

        private static ParameterSet ConvertToParameterSet(TlcModule.SweepableParamAttribute[] hps,
            RecipeInference.SuggestedRecipe.SuggestedLearner learner)
        {
            if (learner.PipelineNode.HyperSweeperParamSet != null)
                return learner.PipelineNode.HyperSweeperParamSet;

            var paramValues = new IParameterValue[hps.Length];

            if (hps.Any(p => p.RawValue == null))
                PopulateSweepableParams(learner);

            for (int i = 0; i < hps.Length; i++)
            {
                Contracts.CheckValue(hps[i].RawValue, nameof(TlcModule.SweepableParamAttribute.RawValue));

                switch (hps[i])
                {
                    case TlcModule.SweepableDiscreteParamAttribute dp:
                        var learnerVal =
                            learner.PipelineNode.GetPropertyValueByName(dp.Name, (IComparable)dp.Options[0]);
                        var optionIndex = (int)(dp.RawValue ?? dp.IndexOf(learnerVal));
                        paramValues[i] = new StringParameterValue(dp.Name, dp.Options[optionIndex].ToString());
                        break;
                    case TlcModule.SweepableFloatParamAttribute fp:
                        paramValues[i] =
                            new FloatParameterValue(fp.Name,
                                (float)(fp.RawValue ?? learner.PipelineNode.GetPropertyValueByName(fp.Name, 0f)));
                        break;
                    case TlcModule.SweepableLongParamAttribute lp:
                        paramValues[i] =
                            new LongParameterValue(lp.Name,
                                (long)(lp.RawValue ?? learner.PipelineNode.GetPropertyValueByName(lp.Name, 0L)));
                        break;
                }
            }

            learner.PipelineNode.HyperSweeperParamSet = new ParameterSet(paramValues);
            return learner.PipelineNode.HyperSweeperParamSet;
        }

        public static IRunResult ConvertToRunResult(RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            AutoInference.RunSummary rs, bool isMetricMaximizing) =>
                new RunResult(ConvertToParameterSet(learner.PipelineNode.SweepParams, learner), rs.MetricValue, isMetricMaximizing);

        public static IRunResult[] ConvertToRunResults(PipelinePattern[] history, bool isMetricMaximizing) =>
            history.Select(h =>
                ConvertToRunResult(h.Learner, h.PerformanceSummary, isMetricMaximizing)).ToArray();

        /// <summary>
        /// Method to convert set of sweepable hyperparameters into strings of a format understood
        /// by the current smart hyperparameter sweepers.
        /// </summary>
        public static Tuple<string, string[]>[] ConvertToSweepArgumentStrings(TlcModule.SweepableParamAttribute[] hps)
        {
            var results = new Tuple<string, string[]>[hps.Length];

            for (int i = 0; i < hps.Length; i++)
            {
                string logSetting;
                string numStepsSetting;
                string stepSizeSetting;
                switch (hps[i])
                {
                    case TlcModule.SweepableDiscreteParamAttribute dp:
                        results[i] = new Tuple<string, string[]>("dp",
                            new[] { $"name={dp.Name}", $"{string.Join(" ", dp.Options.Select(o => $"v={o}"))}" });
                        break;
                    case TlcModule.SweepableFloatParamAttribute fp:
                        logSetting = fp.IsLogScale ? "log+" : "";
                        numStepsSetting = fp.NumSteps != null ? $"numsteps={fp.NumSteps}" : "";
                        stepSizeSetting = fp.StepSize != null ? $"stepsize={fp.StepSize}" : "";

                        results[i] =
                            new Tuple<string, string[]>("fp",
                                new[]
                                {
                                    $"name={fp.Name}",
                                    $"min={fp.Min}",
                                    $"max={fp.Max}",
                                    logSetting,
                                    numStepsSetting,
                                    stepSizeSetting
                                });
                        break;
                    case TlcModule.SweepableLongParamAttribute lp:
                        logSetting = lp.IsLogScale ? "logbase+" : "";
                        numStepsSetting = lp.NumSteps != null ? $"numsteps={lp.NumSteps}" : "";
                        stepSizeSetting = lp.StepSize != null ? $"stepsize={lp.StepSize}" : "";

                        results[i] =
                            new Tuple<string, string[]>("lp",
                                new[]
                                {
                                    $"name={lp.Name}",
                                    $"min={lp.Min}",
                                    $"max={lp.Max}",
                                    logSetting,
                                    numStepsSetting,
                                    stepSizeSetting
                                });
                        break;
                }
            }
            return results;
        }

        public static string GenerateOverallTrainingMetricVarName(Guid id) => $"Var_Training_OM_{id:N}";
    }
}
