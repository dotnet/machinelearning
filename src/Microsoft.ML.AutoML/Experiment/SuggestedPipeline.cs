// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been executed.
    /// </summary>
    internal class SuggestedPipeline
    {
        public readonly IList<SuggestedTransform> Transforms;
        public readonly SuggestedTrainer Trainer;
        public readonly IList<SuggestedTransform> TransformsPostTrainer;

        private readonly MLContext _context;
        private readonly bool _cacheBeforeTrainer;

        public SuggestedPipeline(IEnumerable<SuggestedTransform> transforms,
            IEnumerable<SuggestedTransform> transformsPostTrainer,
            SuggestedTrainer trainer,
            MLContext context,
            bool cacheBeforeTrainer)
        {
            Transforms = transforms.Select(t => t.Clone()).ToList();
            TransformsPostTrainer = transformsPostTrainer.Select(t => t.Clone()).ToList();
            Trainer = trainer.Clone();
            _context = context;
            _cacheBeforeTrainer = cacheBeforeTrainer;
        }

        public override string ToString() => $"{string.Join(" ", Transforms.Select(t => $"xf={t}"))} tr={Trainer} {string.Join(" ", TransformsPostTrainer.Select(t => $"xf={t}"))} cache={(_cacheBeforeTrainer ? "+" : "-")}";

        public override bool Equals(object obj)
        {
            var pipeline = obj as SuggestedPipeline;
            if (pipeline == null)
            {
                return false;
            }
            return pipeline.ToString() == ToString();
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public MLContext GetContext()
        {
            return _context;
        }

        public Pipeline ToPipeline()
        {
            var pipelineElements = new List<PipelineNode>();
            foreach (var transform in Transforms)
            {
                pipelineElements.Add(transform.PipelineNode);
            }
            pipelineElements.Add(Trainer.ToPipelineNode());
            foreach (var transform in TransformsPostTrainer)
            {
                pipelineElements.Add(transform.PipelineNode);
            }
            return new Pipeline(pipelineElements.ToArray(), _cacheBeforeTrainer);
        }

        public static SuggestedPipeline FromPipeline(MLContext context, Pipeline pipeline)
        {
            var transforms = new List<SuggestedTransform>();
            var transformsPostTrainer = new List<SuggestedTransform>();
            SuggestedTrainer trainer = null;

            var trainerEncountered = false;
            foreach (var pipelineNode in pipeline.Nodes)
            {
                if (pipelineNode.NodeType == PipelineNodeType.Trainer)
                {
                    var trainerName = (TrainerName)Enum.Parse(typeof(TrainerName), pipelineNode.Name);
                    var trainerExtension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                    var hyperParamSet = TrainerExtensionUtil.BuildParameterSet(trainerName, pipelineNode.Properties);
                    var columnInfo = TrainerExtensionUtil.BuildColumnInfo(pipelineNode.Properties);
                    trainer = new SuggestedTrainer(context, trainerExtension, columnInfo, hyperParamSet);
                    trainerEncountered = true;
                }
                else if (pipelineNode.NodeType == PipelineNodeType.Transform)
                {
                    var estimatorName = (EstimatorName)Enum.Parse(typeof(EstimatorName), pipelineNode.Name);
                    var estimatorExtension = EstimatorExtensionCatalog.GetExtension(estimatorName);
                    var estimator = estimatorExtension.CreateInstance(context, pipelineNode);
                    var transform = new SuggestedTransform(pipelineNode, estimator);
                    if (!trainerEncountered)
                    {
                        transforms.Add(transform);
                    }
                    else
                    {
                        transformsPostTrainer.Add(transform);
                    }
                }
            }

            return new SuggestedPipeline(transforms, transformsPostTrainer, trainer, context, pipeline.CacheBeforeTrainer);
        }

        public IEstimator<ITransformer> ToEstimator(IDataView trainset = null,
            IDataView validationSet = null)
        {
            IEstimator<ITransformer> pipeline = new EstimatorChain<ITransformer>();

            // Append each transformer to the pipeline
            foreach (var transform in Transforms)
            {
                if (transform.Estimator != null)
                {
                    pipeline = pipeline.Append(transform.Estimator);
                }
            }

            if (validationSet != null)
            {
                validationSet = pipeline.Fit(validationSet).Transform(validationSet);
            }

            // Get learner
            var learner = Trainer.BuildTrainer(validationSet);

            if (_cacheBeforeTrainer)
            {
                pipeline = pipeline.AppendCacheCheckpoint(_context);
            }

            // Append learner to pipeline
            pipeline = pipeline.Append(learner);

            // Append each post-trainer transformer to the pipeline
            foreach (var transform in TransformsPostTrainer)
            {
                if (transform.Estimator != null)
                {
                    pipeline = pipeline.Append(transform.Estimator);
                }
            }

            return pipeline;
        }
    }
}
