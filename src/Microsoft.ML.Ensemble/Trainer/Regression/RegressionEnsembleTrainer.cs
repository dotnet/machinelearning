// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(typeof(RegressionEnsembleTrainer), typeof(RegressionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer) },
    RegressionEnsembleTrainer.UserNameValue,
    RegressionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TScalarPredictor = IPredictorProducing<float>;
    public sealed class RegressionEnsembleTrainer : EnsembleTrainerBase<RegressionEnsembleTrainer.Arguments, RegressionPredictionTransformer<TScalarPredictor>, TScalarPredictor,
       float, IRegressionSubModelSelector, IRegressionOutputCombiner>,
       IModelCombiner<TScalarPredictor, TScalarPredictor>
    {
        internal const string LoadNameValue = "EnsembleRegression";
        internal const string UserNameValue = "Regression Ensemble (bagging, stacking, etc)";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type", Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportRegressionSubModelSelectorFactory SubModelSelectorType = new AllSelectorFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportRegressionOutputCombinerFactory OutputCombiner = new MedianFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureRegressorTrainer))]
            public IComponentFactory<ITrainer<TScalarPredictor>>[] BasePredictors;

            internal override IComponentFactory<ITrainer<TScalarPredictor>>[] GetPredictorFactories() => BasePredictors;

            public Arguments()
            {
                BasePredictors = new[]
                {
                    ComponentFactoryUtils.CreateFromFunction(
                        env => new OnlineGradientDescentTrainer(env, DefaultColumnNames.Label, DefaultColumnNames.Features))
                };
            }
        }

        private readonly ISupportRegressionOutputCombinerFactory _outputCombiner;

        /// <summary>
        /// Initializes a new instance of <see cref="RegressionEnsembleTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public RegressionEnsembleTrainer(IHostEnvironment env, string featureColumn, string labelColumn,
            Action<Arguments> advancedSettings = null)
            : this(env, ArgsInit(featureColumn, labelColumn, advancedSettings))
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="RegressionEnsembleTrainer"/>
        /// </summary>
        public RegressionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue, TrainerUtils.MakeR4ScalarLabel(args.LabelColumn))
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        private static Arguments ArgsInit(string featureColumn, string labelColumn, Action<Arguments> advancedSettings)
        {
            Arguments args = new Arguments();
            advancedSettings?.Invoke(args);

            // Apply the advanced args, if the user supplied any.
            args.FeatureColumn = featureColumn;
            args.LabelColumn = labelColumn;

            return args;
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override RegressionPredictionTransformer<TScalarPredictor> MakeTransformer(TScalarPredictor model, ISchema trainSchema)
            => new RegressionPredictionTransformer<TScalarPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        private protected override TScalarPredictor CreatePredictor(List<FeatureSubsetModel<TScalarPredictor>> models)
        {
            return new EnsemblePredictor(Host, PredictionKind, CreateModels<TScalarPredictor>(models), Combiner);
        }

        public TScalarPredictor CombineModels(IEnumerable<TScalarPredictor> models)
        {
            var combiner = _outputCombiner.CreateComponent(Host);
            var p = models.First();

            var predictor = new EnsemblePredictor(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<TScalarPredictor>(k)).ToArray(), combiner);

            return predictor;
        }
    }
}
