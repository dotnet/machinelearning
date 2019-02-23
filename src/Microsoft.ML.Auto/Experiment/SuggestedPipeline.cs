// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    internal class SuggestedPipeline
    {
        private readonly MLContext _context;
        public readonly IList<SuggestedTransform> Transforms;
        public readonly SuggestedTrainer Trainer;

        public SuggestedPipeline(IEnumerable<SuggestedTransform> transforms,
            SuggestedTrainer trainer,
            MLContext context,
            bool autoNormalize = true)
        {
            Transforms = transforms.Select(t => t.Clone()).ToList();
            Trainer = trainer.Clone();
            _context = context;

            if(autoNormalize)
            {
                AddNormalizationTransforms();
            }
        }
        
        public override string ToString() => $"{string.Join(" xf=", this.Transforms)} tr={this.Trainer}";

        public override bool Equals(object obj)
        {
            var pipeline = obj as SuggestedPipeline;
            if(pipeline == null)
            {
                return false;
            }
            return pipeline.ToString() == this.ToString();
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public Pipeline ToPipeline()
        {
            var pipelineElements = new List<PipelineNode>();
            foreach(var transform in Transforms)
            {
                pipelineElements.Add(transform.PipelineNode);
            }
            pipelineElements.Add(Trainer.ToPipelineNode());
            return new Pipeline(pipelineElements.ToArray());
        }

        public static SuggestedPipeline FromPipeline(MLContext context, Pipeline pipeline)
        {
            var transforms = new List<SuggestedTransform>();
            SuggestedTrainer trainer = null;

            foreach(var pipelineNode in pipeline.Nodes)
            {
                if(pipelineNode.NodeType == PipelineNodeType.Trainer)
                {
                    var trainerName = (TrainerName)Enum.Parse(typeof(TrainerName), pipelineNode.Name);
                    var trainerExtension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                    var hyperParamSet = TrainerExtensionUtil.BuildParameterSet(trainerName, pipelineNode.Properties);
                    var columnInfo = TrainerExtensionUtil.BuildColumnInfo(pipelineNode.Properties);
                    trainer = new SuggestedTrainer(context, trainerExtension, columnInfo, hyperParamSet);
                }
                else if (pipelineNode.NodeType == PipelineNodeType.Transform)
                {
                    var estimatorName = (EstimatorName)Enum.Parse(typeof(EstimatorName), pipelineNode.Name);
                    var estimatorExtension = EstimatorExtensionCatalog.GetExtension(estimatorName);
                    var estimator = estimatorExtension.CreateInstance(context, pipelineNode);
                    var transform = new SuggestedTransform(pipelineNode, estimator);
                    transforms.Add(transform);
                }
            }

            return new SuggestedPipeline(transforms, trainer, context, false);
        }

        public IEstimator<ITransformer> ToEstimator()
        {
            IEstimator<ITransformer> pipeline = new EstimatorChain<ITransformer>();

            // append each transformer to the pipeline
            foreach (var transform in Transforms)
            {
                if(transform.Estimator != null)
                {
                    pipeline = pipeline.Append(transform.Estimator);
                }
            }

            // get learner
            var learner = Trainer.BuildTrainer();

            // append learner to pipeline
            pipeline = pipeline.Append(learner);

            return pipeline;
        }

        public ITransformer Fit(IDataView trainData)
        {
            var estimator = ToEstimator();
            return estimator.Fit(trainData);
        }

        private void AddNormalizationTransforms()
        {
            // get learner
            var learner = Trainer.BuildTrainer();

            // only add normalization if learner needs it
            if (!learner.Info.NeedNormalization)
            {
                return;
            }

            var transform = NormalizingExtension.CreateSuggestedTransform(_context, DefaultColumnNames.Features, DefaultColumnNames.Features);
            Transforms.Add(transform);
        }
    }
}