// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.AutoML
{
    [JsonConverter(typeof(SweepablePipelineConverter))]
    internal class SweepablePipeline : ISweepable<EstimatorChain<ITransformer>>
    {
        private readonly Entity _schema;
        private const string SchemaOption = "_SCHEMA_";
        private readonly Dictionary<string, SweepableEstimator> _estimators = new Dictionary<string, SweepableEstimator>();
        private static readonly StringEntity _nilStringEntity = new StringEntity("Nil");
        private static readonly EstimatorEntity _nilSweepableEntity = new EstimatorEntity(null);
        private string _currentSchema;

        public SearchSpace.SearchSpace SearchSpace
        {
            get
            {
                var searchSpace = new SearchSpace.SearchSpace();
                var kvPairs = _estimators.Select((e, i) => new KeyValuePair<string, SearchSpace.SearchSpace>(i.ToString(), e.Value.SearchSpace));
                foreach (var kv in kvPairs)
                {
                    if (kv.Value != null)
                    {
                        searchSpace.Add(kv.Key, kv.Value);
                    }
                }

                var schemaOptions = _schema.ToTerms().Select(t => t.ToString()).ToArray();
                var choiceOption = new ChoiceOption(schemaOptions);
                searchSpace.Add(SchemaOption, choiceOption);

                return searchSpace;
            }
        }

        public Parameter CurrentParameter
        {
            get
            {
                var parameter = Parameter.CreateNestedParameter();
                var kvPairs = _estimators.Select((e, i) => new KeyValuePair<string, Parameter>(i.ToString(), e.Value.Parameter));
                foreach (var kv in kvPairs)
                {
                    if (kv.Value != null)
                    {
                        parameter[kv.Key] = kv.Value;
                    }
                }

                parameter[SchemaOption] = Parameter.FromString(_currentSchema);
                return parameter;
            }
        }

        internal SweepablePipeline()
        {
            _estimators = new Dictionary<string, SweepableEstimator>();
            _schema = null;
        }

        internal SweepablePipeline(Dictionary<string, SweepableEstimator> estimators, Entity schema, string currentSchema = null)
        {
            _estimators = estimators;
            _schema = schema;
            _currentSchema = currentSchema ?? schema.ToTerms().First().ToString();
        }

        public Dictionary<string, SweepableEstimator> Estimators { get => _estimators; }


        internal Entity Schema { get => _schema; }

        public EstimatorChain<ITransformer> BuildFromOption(MLContext context, Parameter parameter)
        {
            _currentSchema = parameter[SchemaOption].AsType<string>();
            var estimators = Entity.FromExpression(_currentSchema)
                                   .ValueEntities()
                                   .Where(e => e is StringEntity se && se.Value != "Nil")
                                   .Select((se) => _estimators[((StringEntity)se).Value]);

            var pipeline = new SweepableEstimatorPipeline(estimators);
            return pipeline.BuildTrainingPipeline(context, parameter);
        }

        public SweepablePipeline Append(params ISweepable<IEstimator<ITransformer>>[] sweepables)
        {
            Entity entity = null;
            foreach (var sweepable in sweepables)
            {
                if (sweepable is SweepableEstimator estimator)
                {
                    if (entity == null)
                    {
                        entity = new EstimatorEntity(estimator);
                        continue;
                    }
                    else
                    {
                        entity += estimator;
                    }
                }
                else if (sweepable is SweepablePipeline pipeline)
                {
                    if (entity == null)
                    {
                        entity = CreateSweepableEntityFromEntity(pipeline._schema, pipeline._estimators);
                        continue;
                    }
                    else
                    {
                        entity += CreateSweepableEntityFromEntity(pipeline._schema, pipeline._estimators);
                    }
                }
            }

            return AppendEntity(false, entity);
        }

        private SweepablePipeline AppendEntity(bool allowSkip, Entity entity)
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

            return new SweepablePipeline(estimators, schema);
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
