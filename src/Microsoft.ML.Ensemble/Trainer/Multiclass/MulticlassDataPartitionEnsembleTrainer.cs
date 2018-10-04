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

[assembly: LoadableClass(MulticlassDataPartitionEnsembleTrainer.Summary, typeof(MulticlassDataPartitionEnsembleTrainer),
    typeof(MulticlassDataPartitionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MulticlassDataPartitionEnsembleTrainer.UserNameValue,
    MulticlassDataPartitionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    /// <summary>
    /// A generic ensemble classifier for multi-class classification
    /// </summary>
    public sealed class MulticlassDataPartitionEnsembleTrainer : EnsembleTrainerBase<MulticlassDataPartitionEnsembleTrainer.Arguments,
        MulticlassPredictionTransformer<EnsembleMultiClassPredictor>, EnsembleMultiClassPredictor,
        VBuffer<Single>, IMulticlassSubModelSelector, IMultiClassOutputCombiner>,
        IModelCombiner<TVectorPredictor, TVectorPredictor>
    {
        public const string LoadNameValue = "WeightedEnsembleMulticlass";
        public const string UserNameValue = "Multi-class Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for multi-class classification.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type", Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportMulticlassSubModelSelectorFactory SubModelSelectorType = new AllSelectorMultiClassFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportMulticlassOutputCombinerFactory OutputCombiner = new MultiMedian.Arguments();

            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureMultiClassClassifierTrainer))]
            public IComponentFactory<ITrainer<TVectorPredictor>>[] BasePredictors;

            internal override IComponentFactory<ITrainer<TVectorPredictor>>[] GetPredictorFactories() => BasePredictors;

            public Arguments()
            {
                BasePredictors = new[]
                {
                    ComponentFactoryUtils.CreateFromFunction(
                        env => new MulticlassLogisticRegression(env, FeatureColumn, LabelColumn))
                };
            }
        }

        private readonly ISupportMulticlassOutputCombinerFactory _outputCombiner;

        /// <summary>
        /// Initializes a new instance of <see cref="MulticlassDataPartitionEnsembleTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, string featureColumn, string labelColumn,
            Action<Arguments> advancedSettings = null)
            : this(env, ArgsInit(featureColumn, labelColumn, advancedSettings))
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="MulticlassDataPartitionEnsembleTrainer"/>
        /// </summary>
        public MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue, TrainerUtils.MakeU4ScalarLabel(args.LabelColumn))
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
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var scoreMetadata = new List<SchemaShape.Column>() { new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false) };
            scoreMetadata.AddRange(MetadataUtils.GetTrainerOutputMetadata());

            var predLabelMetadata = new SchemaShape(labelCol.Metadata.Columns.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                .Concat(MetadataUtils.GetTrainerOutputMetadata()));

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(scoreMetadata)),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, predLabelMetadata)
            };
        }

        protected override MulticlassPredictionTransformer<EnsembleMultiClassPredictor> MakeTransformer(EnsembleMultiClassPredictor model, ISchema trainSchema)
            => new MulticlassPredictionTransformer<EnsembleMultiClassPredictor>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        private protected override EnsembleMultiClassPredictor CreatePredictor(List<FeatureSubsetModel<TVectorPredictor>> models)
        {
            return new EnsembleMultiClassPredictor(Host, CreateModels<TVectorPredictor>(models), Combiner as IMultiClassOutputCombiner);
        }

        public TVectorPredictor CombineModels(IEnumerable<TVectorPredictor> models)
        {
            var predictor = new EnsembleMultiClassPredictor(Host,
                models.Select(k => new FeatureSubsetModel<TVectorPredictor>(k)).ToArray(),
                _outputCombiner.CreateComponent(Host));

            return predictor;
        }
    }
}
