// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace Microsoft.ML.AutoML
{
    [JsonConverter(typeof(MultiModelPipelineConverter))]
    public class MultiModelPipeline
    {
        private static readonly StringEntity _nilStringEntity = new StringEntity("Nil");
        private static readonly EstimatorEntity _nilSweepableEntity = new EstimatorEntity(null);
        private readonly Dictionary<string, SweepableEstimator> _estimators;
        private readonly Entity _schema;

        public MultiModelPipeline()
        {
            _estimators = new Dictionary<string, SweepableEstimator>();
            _schema = null;
        }

        internal MultiModelPipeline(Dictionary<string, SweepableEstimator> estimators, Entity schema)
        {
            _estimators = estimators;
            _schema = schema;
        }

        public Dictionary<string, SweepableEstimator> Estimators { get => _estimators; }

        internal Entity Schema { get => _schema; }

        /// <summary>
        /// Get the schema of all single model pipelines in the form of strings.
        /// the pipeline id can be used to create a single model pipeline through <see cref="MultiModelPipeline.BuildSweepableEstimatorPipeline(string)"/>.
        /// </summary>
        internal string[] PipelineIds { get => Schema.ToTerms().Select(t => t.ToString()).ToArray(); }

        public MultiModelPipeline Append(params SweepableEstimator[] estimators)
        {
            Entity entity = null;
            foreach (var estimator in estimators)
            {
                if (entity == null)
                {
                    entity = new EstimatorEntity(estimator);
                    continue;
                }

                entity += estimator;
            }

            return Append(entity);
        }

        public MultiModelPipeline AppendOrSkip(params SweepableEstimator[] estimators)
        {
            Entity entity = null;
            foreach (var estimator in estimators)
            {
                if (entity == null)
                {
                    entity = new EstimatorEntity(estimator);
                    continue;
                }

                entity += estimator;
            }

            return AppendOrSkip(entity);
        }

        public SweepableEstimatorPipeline BuildSweepableEstimatorPipeline(string schema)
        {
            var pipelineNodes = Entity.FromExpression(schema)
                                      .ValueEntities()
                                      .Where(e => e is StringEntity se && se.Value != "Nil")
                                      .Select((se) => _estimators[((StringEntity)se).Value]);

            return new SweepableEstimatorPipeline(pipelineNodes);
        }

        internal MultiModelPipeline Append(Entity entity)
        {
            return AppendEntity(false, entity);
        }

        internal MultiModelPipeline AppendOrSkip(Entity entity)
        {
            return AppendEntity(true, entity);
        }

        internal MultiModelPipeline AppendOrSkip(MultiModelPipeline pipeline)
        {
            return AppendPipeline(true, pipeline);
        }

        internal MultiModelPipeline Append(MultiModelPipeline pipeline)
        {
            return AppendPipeline(false, pipeline);
        }

        private MultiModelPipeline AppendPipeline(bool allowSkip, MultiModelPipeline pipeline)
        {
            var sweepableEntity = CreateSweepableEntityFromEntity(pipeline.Schema, pipeline.Estimators);
            return AppendEntity(allowSkip, sweepableEntity);
        }

        private MultiModelPipeline AppendEntity(bool allowSkip, Entity entity)
        {
            var estimators = _estimators.ToDictionary(x => x.Key, x => x.Value);
            var stringEntity = VisitAndReplaceSweepableEntityWithStringEntity(entity, ref estimators);
            if (allowSkip)
            {
                stringEntity += _nilStringEntity;
            }

            var schema = _schema;
            if (schema == null)
            {
                schema = stringEntity;
            }
            else
            {
                schema *= stringEntity;
            }

            return new MultiModelPipeline(estimators, schema);
        }

        private Entity CreateSweepableEntityFromEntity(Entity entity, Dictionary<string, SweepableEstimator> lookupTable)
        {
            if (entity is null)
            {
                return null;
            }

            if (entity is StringEntity stringEntity)
            {
                if (stringEntity == _nilStringEntity)
                {
                    return _nilSweepableEntity;
                }

                return new EstimatorEntity(lookupTable[stringEntity.Value]);
            }
            else if (entity is ConcatenateEntity concatenateEntity)
            {
                return new ConcatenateEntity()
                {
                    Left = CreateSweepableEntityFromEntity(concatenateEntity.Left, lookupTable),
                    Right = CreateSweepableEntityFromEntity(concatenateEntity.Right, lookupTable),
                };
            }
            else if (entity is OneOfEntity oneOfEntity)
            {
                return new OneOfEntity()
                {
                    Left = CreateSweepableEntityFromEntity(oneOfEntity.Left, lookupTable),
                    Right = CreateSweepableEntityFromEntity(oneOfEntity.Right, lookupTable),
                };
            }

            throw new ArgumentException();
        }

        private Entity VisitAndReplaceSweepableEntityWithStringEntity(Entity e, ref Dictionary<string, SweepableEstimator> estimators)
        {
            if (e is null)
            {
                return null;
            }

            if (e is EstimatorEntity sweepableEntity0)
            {
                if (sweepableEntity0 == _nilSweepableEntity)
                {
                    return _nilStringEntity;
                }

                var id = GetNextId(estimators);
                estimators[id] = (SweepableEstimator)sweepableEntity0.Estimator;
                return new StringEntity(id);
            }

            e.Left = VisitAndReplaceSweepableEntityWithStringEntity(e.Left, ref estimators);
            e.Right = VisitAndReplaceSweepableEntityWithStringEntity(e.Right, ref estimators);

            return e;
        }

        private string GetNextId(Dictionary<string, SweepableEstimator> estimators)
        {
            var count = estimators.Count();
            return "e" + count.ToString();
        }
    }
}
