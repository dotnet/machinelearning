// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public abstract class BaseBestPerformanceSelector<TOutput> : SubModelDataSelector<TOutput>
    {
        protected abstract string MetricName { get; }

        protected virtual bool IsAscMetric => true;

        protected BaseBestPerformanceSelector(ArgumentsBase args, IHostEnvironment env, string name)
            : base(args, env, name)
        {
        }

        public override void CalculateMetrics(FeatureSubsetModel<IPredictorProducing<TOutput>> model,
            ISubsetSelector subsetSelector, Subset subset, Batch batch, bool needMetrics)
        {
            base.CalculateMetrics(model, subsetSelector, subset, batch, true);
        }

        public override IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> Prune(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models)
        {
            using (var ch = Host.Start("Pruning"))
            {
                var sortedModels = models.ToArray();
                Array.Sort(sortedModels, new ModelPerformanceComparer(MetricName, IsAscMetric));
                Print(ch, sortedModels, MetricName);
                int modelCountToBeSelected = (int)(models.Count * LearnersSelectionProportion);
                if (modelCountToBeSelected == 0)
                    modelCountToBeSelected = 1;

                return sortedModels.Where(m => m != null).Take(modelCountToBeSelected).ToList();
            }
        }

        protected static string FindMetricName(Type type, object value)
        {
            Contracts.Assert(type.IsEnum);
            Contracts.Assert(value.GetType() == type);

            foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                if (field.FieldType != type)
                    continue;
                if (field.GetCustomAttribute<HideEnumValueAttribute>() != null)
                    continue;
                var displayAttr = field.GetCustomAttribute<EnumValueDisplayAttribute>();
                if (displayAttr != null)
                {
                    var valCur = field.GetValue(null);
                    if (value.Equals(valCur))
                        return displayAttr.Name;
                }
            }
            Contracts.Assert(false);
            return null;
        }

        private sealed class ModelPerformanceComparer : IComparer<FeatureSubsetModel<IPredictorProducing<TOutput>>>
        {
            private readonly string _metricName;
            private readonly bool _isAscMetric;

            public ModelPerformanceComparer(string metricName, bool isAscMetric)
            {
                Contracts.AssertValue(metricName);

                _metricName = metricName;
                _isAscMetric = isAscMetric;
            }

            public int Compare(FeatureSubsetModel<IPredictorProducing<TOutput>> x, FeatureSubsetModel<IPredictorProducing<TOutput>> y)
            {
                if (x == null || y == null)
                    return (x == null ? 0 : 1) - (y == null ? 0 : 1);
                double xValue = 0;
                var found = false;
                foreach (var kvp in x.Metrics)
                {
                    if (_metricName == kvp.Key)
                    {
                        xValue = kvp.Value;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw Contracts.Except("Metrics did not contain the requested metric '{0}'", _metricName);
                double yValue = 0;
                found = false;
                foreach (var kvp in y.Metrics)
                {
                    if (_metricName == kvp.Key)
                    {
                        yValue = kvp.Value;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw Contracts.Except("Metrics did not contain the requested metric '{0}'", _metricName);
                if (xValue > yValue)
                    return _isAscMetric ? -1 : 1;
                if (yValue > xValue)
                    return _isAscMetric ? 1 : -1;
                return 0;
            }
        }
    }
}
