// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public abstract class BaseDiverseSelector<TOutput, TDiversityMetric> : SubModelDataSelector<TOutput>
        where TDiversityMetric : class, IDiversityMeasure<TOutput>
    {
        public abstract class DiverseSelectorArguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "The metric type to be used to find the diversity among base learners", ShortName = "dm", SortOrder = 50)]
            [TGUI(Label = "Diversity Measure Type")]
            public ISupportDiversityMeasureFactory<TOutput> DiversityMetricType;
        }

        private readonly ISupportDiversityMeasureFactory<TOutput> _diversityMetricType;
        private ConcurrentDictionary<FeatureSubsetModel<IPredictorProducing<TOutput>>, TOutput[]> _predictions;

        protected abstract ISupportDiversityMeasureFactory<TOutput> DefaultDiversityMetricType { get; }

        protected internal BaseDiverseSelector(IHostEnvironment env, DiverseSelectorArguments args, string name)
            : base(args, env, name)
        {
            _diversityMetricType = args.DiversityMetricType;
            _predictions = new ConcurrentDictionary<FeatureSubsetModel<IPredictorProducing<TOutput>>, TOutput[]>();
        }

        protected IDiversityMeasure<TOutput> CreateDiversityMetric()
        {
            if (_diversityMetricType == null)
                return DefaultDiversityMetricType.CreateComponent(Host);
            return _diversityMetricType.CreateComponent(Host);
        }

        public override void CalculateMetrics(FeatureSubsetModel<IPredictorProducing<TOutput>> model,
            ISubsetSelector subsetSelector, Subset subset, Batch batch, bool needMetrics)
        {
            base.CalculateMetrics(model, subsetSelector, subset, batch, needMetrics);

            var vm = model.Predictor as IValueMapper;
            Host.Check(vm != null, "Predictor doesn't implement the expected interface");
            var map = vm.GetMapper<VBuffer<Single>, TOutput>();

            TOutput[] preds = new TOutput[100];
            int count = 0;
            var data = subsetSelector.GetTestData(subset, batch);
            using (var cursor = new FeatureFloatVectorCursor(data, CursOpt.AllFeatures))
            {
                while (cursor.MoveNext())
                {
                    Utils.EnsureSize(ref preds, count + 1);
                    map(ref cursor.Features, ref preds[count]);
                    count++;
                }
            }
            Array.Resize(ref preds, count);
            _predictions[model] = preds;
        }

        /// <summary>
        /// This calculates the diversity by calculating the disagreement measure which is defined as the sum of number of instances correctly(incorrectly)
        /// classified by first classifier and incorrectly(correctly) classified by the second classifier over the total number of instances.
        /// All the pairwise classifiers are sorted out to take the most divers classifiers.
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public override IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> Prune(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models)
        {
            if (models.Count <= 1)
                return models;

            // 1. Find the disagreement number
            List<ModelDiversityMetric<TOutput>> diversityValues = CalculateDiversityMeasure(models, _predictions);
            _predictions.Clear();

            // 2. Sort all the pairwise classifiers
            var sortedModels = diversityValues.ToArray();
            Array.Sort(sortedModels, new ModelDiversityComparer());
            var modelCountToBeSelected = (int)(models.Count * LearnersSelectionProportion);

            if (modelCountToBeSelected == 0)
                modelCountToBeSelected++;

            // 3. Take the most diverse classifiers
            var selectedModels = new List<FeatureSubsetModel<IPredictorProducing<TOutput>>>();
            foreach (var item in sortedModels)
            {
                if (selectedModels.Count < modelCountToBeSelected)
                {
                    if (!selectedModels.Contains(item.ModelX))
                    {
                        selectedModels.Add(item.ModelX);
                    }
                }

                if (selectedModels.Count < modelCountToBeSelected)
                {
                    if (!selectedModels.Contains(item.ModelY))
                    {
                        selectedModels.Add(item.ModelY);
                        continue;
                    }
                }
                else
                {
                    break;
                }
            }

            return selectedModels;
        }

        public abstract List<ModelDiversityMetric<TOutput>> CalculateDiversityMeasure(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models,
            ConcurrentDictionary<FeatureSubsetModel<IPredictorProducing<TOutput>>, TOutput[]> predictions);

        public class ModelDiversityComparer : IComparer<ModelDiversityMetric<TOutput>>
        {
            public int Compare(ModelDiversityMetric<TOutput> x, ModelDiversityMetric<TOutput> y)
            {
                if (x == null || y == null)
                    return 0;
                if (x.DiversityNumber > y.DiversityNumber)
                    return -1;
                if (y.DiversityNumber > x.DiversityNumber)
                    return 1;
                return 0;
            }
        }
    }
}
