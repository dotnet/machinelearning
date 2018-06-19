// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Runtime.Ensemble
{
    using Stopwatch = System.Diagnostics.Stopwatch;
    public abstract class EnsembleTrainerBase<TOutput, TPredictor, TSelector, TCombiner, TSig> : TrainerBase<RoleMappedData, TPredictor>
         where TPredictor : class, IPredictorProducing<TOutput>
         where TSelector : class, ISubModelSelector<TOutput>
         where TCombiner : class, IOutputCombiner<TOutput>
    {
        public abstract class ArgumentsBase : LearnerInputBaseWithLabel
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of models per batch. If not specified, will default to 50 if there is only one base predictor, " +
                "or the number of base predictors otherwise.", ShortName = "nm", SortOrder = 3)]
            [TGUI(Label = "Number of Models per batch")]
            public int? NumModels;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Batch size", ShortName = "bs", SortOrder = 107)]
            [TGUI(Label = "Batch Size",
                Description =
                "Number of instances to be loaded in memory to create an ensemble out of it. All the instances will be loaded if the value is -1.")]
            public int BatchSize = -1;

            [Argument(ArgumentType.Multiple, HelpText = "Sampling Type", ShortName = "st", SortOrder = 2)]
            [TGUI(Label = "Sampling Type", Description = "Subset Selection Algorithm to induce the base learner.Sub-settings can be used to select the features")]
            public ISupportSubsetSelectorFactory SamplingType = new BootstrapSelector.Arguments();

            [Argument(ArgumentType.AtMostOnce, HelpText = "All the base learners will run asynchronously if the value is true", ShortName = "tp", SortOrder = 106)]
            [TGUI(Label = "Train parallel", Description = "All the base learners will run asynchronously if the value is true")]
            public bool TrainParallel;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "True, if metrics for each model need to be evaluated and shown in comparison table. This is done by using validation set if available or the training set",
                ShortName = "sm", SortOrder = 108)]
            [TGUI(Label = "Show Sub-Model Metrics")]
            public bool ShowMetrics;

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportOutputCombinerFactory <TOutput> OutputCombiner;

            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type",
                Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportSubModelSelectorFactory<TOutput> SubModelSelectorType;

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility =ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public SubComponent<ITrainer<RoleMappedData, IPredictorProducing<TOutput>>, TSig>[] BasePredictors;

            public const int DefaultNumModels = 50;
        }

        /// <summary> Command-line arguments </summary>
        protected readonly ArgumentsBase Args;
        protected readonly int NumModels;

        /// <summary> Ensemble members </summary>
        protected readonly ITrainer<RoleMappedData, IPredictorProducing<TOutput>>[] Trainers;

        private readonly ISubsetSelector _subsetSelector;
        private readonly ISubModelSelector<TOutput> _subModelSelector;

        protected readonly IOutputCombiner<TOutput> Combiner;

        protected List<FeatureSubsetModel<IPredictorProducing<TOutput>>> Models;

        private readonly bool _needNorm;
        private readonly bool _needCalibration;

        internal EnsembleTrainerBase(ArgumentsBase args, IHostEnvironment env, string name)
            : base(env, name)
        {
            Args = args;

            using (var ch = Host.Start("Init"))
            {
                ch.CheckUserArg(Utils.Size(Args.BasePredictors) > 0, nameof(Args.BasePredictors), "This should have at-least one value");

                NumModels = Args.NumModels ??
                    (Args.BasePredictors.Length == 1 ? ArgumentsBase.DefaultNumModels : Args.BasePredictors.Length);

                ch.CheckUserArg(NumModels > 0, nameof(Args.NumModels), "Must be positive, or null to indicate numModels is the number of base predictors");

                if (Utils.Size(Args.BasePredictors) > NumModels)
                    ch.Warning("The base predictor count is greater than models count. Some of the base predictors will be ignored.");

                _subsetSelector = Args.SamplingType.CreateComponent(Host);
                _subModelSelector = Args.SubModelSelectorType.CreateComponent(Host);
                Combiner = Args.OutputCombiner.CreateComponent(Host);

                Trainers = new ITrainer<RoleMappedData, IPredictorProducing<TOutput>>[NumModels];
                for (int i = 0; i < Trainers.Length; i++)
                    Trainers[i] = Args.BasePredictors[i % Args.BasePredictors.Length].CreateInstance(Host);
                _needNorm = Trainers.Any(
                    t =>
                    {
                        return t is ITrainerEx nn && nn.NeedNormalization;
                    });
                _needCalibration = Trainers.Any(
                    t =>
                    {
                        return t is ITrainerEx nn && nn.NeedCalibration;
                    });
                ch.Done();
            }
        }

        public override bool NeedNormalization { get { return _needNorm; } }

        public override bool NeedCalibration { get { return _needCalibration; } }

        // No matter the internal predictors, we are performing multiple passes over the data
        // so it is probably appropriate to always cache.
        public override bool WantCaching { get { return true; } }

        public override void Train(RoleMappedData data)
        {
            using (var ch = Host.Start("Training"))
            {
                TrainCore(ch, data);
                ch.Done();
            }
        }

        private void TrainCore(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            ch.AssertValue(data);

            // 1. Subset Selection
            var stackingTrainer = Combiner as IStackingTrainer<TOutput>;

            //REVIEW ansarim: Implement stacking for Batch mode.
            ch.CheckUserArg(stackingTrainer == null || Args.BatchSize <= 0, nameof(Args.BatchSize), "Stacking works only with Non-batch mode");

            var validationDataSetProportion = _subModelSelector.ValidationDatasetProportion;
            if (stackingTrainer != null)
                validationDataSetProportion = Math.Max(validationDataSetProportion, stackingTrainer.ValidationDatasetProportion);

            var needMetrics = Args.ShowMetrics || Combiner is IWeightedAverager;

            _subsetSelector.Initialize(data, NumModels, Args.BatchSize, validationDataSetProportion);
            int batchNumber = 1;
            foreach (var batch in _subsetSelector.GetBatches(Host.Rand))
            {
                // 2. Core train
                ch.Info("Training {0} learners for the batch {1}", Trainers.Length, batchNumber++);
                var models = new FeatureSubsetModel<IPredictorProducing<TOutput>>[Trainers.Length];

                Parallel.ForEach(_subsetSelector.GetSubsets(batch, Host.Rand),
                    new ParallelOptions() { MaxDegreeOfParallelism = Args.TrainParallel ? -1 : 1 },
                    (subset, state, index) =>
                    {
                        ch.Info("Beginning training model {0} of {1}", index + 1, Trainers.Length);
                        Stopwatch sw = Stopwatch.StartNew();
                        try
                        {
                            if (EnsureMinimumFeaturesSelected(subset))
                            {
                                Trainers[(int)index].Train(subset.Data);

                                var model = new FeatureSubsetModel<IPredictorProducing<TOutput>>(
                                    Trainers[(int)index].CreatePredictor(),
                                    subset.SelectedFeatures,
                                    null);
                                _subModelSelector.CalculateMetrics(model, _subsetSelector, subset, batch, needMetrics);
                                models[(int)index] = model;
                            }
                        }
                        catch (Exception ex)
                        {
                            ch.Assert(models[(int)index] == null);
                            ch.Warning(ex.Sensitivity(), "Trainer {0} of {1} was not learned properly due to the exception '{2}' and will not be added to models.",
                                index + 1, Trainers.Length, ex.Message);
                        }
                        ch.Info("Trainer {0} of {1} finished in {2}", index + 1, Trainers.Length, sw.Elapsed);
                    });

                var modelsList = models.Where(m => m != null).ToList();
                if (Args.ShowMetrics)
                    PrintMetrics(ch, modelsList);

                modelsList = _subModelSelector.Prune(modelsList).ToList();

                if (stackingTrainer != null)
                    stackingTrainer.Train(modelsList, _subsetSelector.GetTestData(null, batch), Host);

                foreach (var model in modelsList)
                    Utils.Add(ref Models, model);
                int modelSize = Utils.Size(Models);
                if (modelSize < Utils.Size(Trainers))
                    ch.Warning("{0} of {1} trainings failed.", Utils.Size(Trainers) - modelSize, Utils.Size(Trainers));
                ch.Check(modelSize > 0, "Ensemble training resulted in no valid models.");
            }
        }

        private bool EnsureMinimumFeaturesSelected(Subset subset)
        {
            if (subset.SelectedFeatures == null)
                return true;
            for (int i = 0; i < subset.SelectedFeatures.Count; i++)
            {
                if (subset.SelectedFeatures[i])
                    return true;
            }

            return false;
        }

        protected virtual void PrintMetrics(IChannel ch, List<FeatureSubsetModel<IPredictorProducing<TOutput>>> models)
        {
            // REVIEW tfinley: The formatting of this method is bizarre and seemingly not even self-consistent
            // w.r.t. its usage of |. Is this intentional?
            if (models.Count == 0 || models[0].Metrics == null)
                return;

            ch.Info("{0}| Name of Model |", string.Join("", models[0].Metrics.Select(m => string.Format("| {0} |", m.Key))));

            foreach (var model in models)
                ch.Info("{0}{1}", string.Join("", model.Metrics.Select(m => string.Format("| {0} |", m.Value))), model.Predictor.GetType().Name);
        }

        protected FeatureSubsetModel<T>[] CreateModels<T>() where T : IPredictor
        {
            var models = new FeatureSubsetModel<T>[Models.Count];
            for (int i = 0; i < Models.Count; i++)
            {
                models[i] = new FeatureSubsetModel<T>(
                    (T)Models[i].Predictor,
                    Models[i].SelectedFeatures,
                    Models[i].Metrics);
            }
            return models;
        }
    }
}
