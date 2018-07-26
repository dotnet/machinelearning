using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public static class Ext
    {
        public static ISchema GetOutputSchema(this IDataTransform xf, IHostEnvironment env, ISchema inputSchema)
        {
            var dv = new EmptyDataView(env, inputSchema);
            var output = ApplyTransformUtils.ApplyTransformToData(env, xf, dv);
            return output.Schema;
        }

        public static IDataView Transform(this IDataTransform xf, IHostEnvironment env, IDataView input)
            => ApplyTransformUtils.ApplyTransformToData(env, xf, input);
    }

    public class AdHocTest
    {
        private static TextLoader.Arguments MakeTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                HasHeader = false,
                Column = new[] {
                            new TextLoader.Column()
                            {
                                Name = "Label",
                                Source = new [] { new TextLoader.Range() { Min = 0, Max = 0} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "SepalLength",
                                Source = new [] { new TextLoader.Range() { Min = 1, Max = 1} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "SepalWidth",
                                Source = new [] { new TextLoader.Range() { Min = 2, Max = 2} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "PetalLength",
                                Source = new [] { new TextLoader.Range() { Min = 3, Max = 3} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "PetalWidth",
                                Source = new [] { new TextLoader.Range() { Min = 4, Max = 4} },
                                Type = DataKind.R4
                            }
                        }
            };
        }

        public class MyTextLoader : IDataReaderEstimator<IMultiStreamSource>, IDataReader<IMultiStreamSource>
        {
            private readonly TextLoader.Arguments _args;
            private readonly IHostEnvironment _env;

            public MyTextLoader(IHostEnvironment env, TextLoader.Arguments args)
            {
                _env = env;
                _args = args;
            }

            public IDataReader<IMultiStreamSource> Fit(IMultiStreamSource input)
            {
                return this;
            }

            public SchemaShape GetOutputSchema()
            {
                var emptyData = new TextLoader(new TlcEnvironment(), _args, new MultiFileSource(null));
                return SchemaShape.Create(emptyData.Schema);
            }

            public IDataView Read(IMultiStreamSource input)
            {
                return new TextLoader(new TlcEnvironment(), _args, input);
            }

            ISchema IDataReader<IMultiStreamSource>.GetOutputSchema()
            {
                var emptyData = new TextLoader(new TlcEnvironment(), _args, new MultiFileSource(null));
                return emptyData.Schema;
            }
        }

        public class TransformerPipe<TIn> : IDataReader<TIn>
        {
            private readonly IDataReader<TIn> _start;
            private readonly ITransformer[] _chain;

            public TransformerPipe(IDataReader<TIn> start, ITransformer[] chain)
            {
                _start = start;
                _chain = chain;
            }

            public IDataView Read(TIn input)
            {
                var idv = _start.Read(input);
                foreach (var xf in _chain)
                    idv = xf.Transform(idv);
                return idv;
            }

            public (IDataReader<TIn> start, IEnumerable<ITransformer> chain) GetParts()
            {
                return (start: _start, chain: _chain);
            }

            public ISchema GetOutputSchema()
            {
                var s = _start.GetOutputSchema();
                foreach (var xf in _chain)
                    s = xf.GetOutputSchema(s);
                return s;
            }
        }

        public class EstimatorPipe<TIn, TTrans> : IDataReaderEstimator<TIn, TTrans>
            where TTrans : IDataReader<TIn>
        {
            private readonly IDataReaderEstimator<TIn> _start;
            private readonly IEstimator[] _chain;

            public EstimatorPipe(IDataReaderEstimator<TIn> start, IEstimator[] chain)
            {
                Contracts.CheckValue(start, nameof(start));
                Contracts.CheckNonEmpty(chain, nameof(chain));

                _start = start;
                _chain = chain;
            }

            public TTrans Fit(TIn input)
            {
                throw new System.NotImplementedException();
            }

            public SchemaShape GetOutputSchema()
            {
                throw new System.NotImplementedException();
            }

            IDataReader<TIn> IDataReaderEstimator<TIn>.Fit(TIn input)
            {
                return Fit(input);
            }
        }

        public class EstimatorPipe<TIn> : IDataReaderEstimator<TIn>
        {
            private readonly IDataReaderEstimator<TIn> _start;
            private readonly List<IEstimator> _estimatorChain = new List<IEstimator>();
            private readonly IHostEnvironment _env = new TlcEnvironment();


            public EstimatorPipe(IDataReaderEstimator<TIn> start)
            {
                _start = start;
            }

            public EstimatorPipe<TIn> Append(IEstimator est)
            {
                _estimatorChain.Add(est);
                return this;
            }

            public TransformerPipe<TIn> Fit(TIn input)
            {
                var start = _start.Fit(input);

                var idv = start.Read(input);
                var xfs = new List<ITransformer>();
                foreach (var est in _estimatorChain)
                {
                    var xf = est.Fit(idv);
                    xfs.Add(xf);
                    idv = xf.Transform(idv);
                }
                return new TransformerPipe<TIn>(start, xfs.ToArray());
            }

            public IDataReaderEstimator<TIn> GetEstimator()
            {
                return this;
            }

            public SchemaShape GetOutputSchema()
            {
                var shape = _start.GetOutputSchema();
                foreach (var xf in _estimatorChain)
                {
                    shape = xf.GetOutputSchema(shape);
                    if (shape == null)
                        return null;
                }
                return shape;
            }

            public (IDataReaderEstimator<TIn>, IEnumerable<IEstimator>) GetParts()
            {
                return (_start, _estimatorChain);
            }

            IDataReader<TIn> IDataReaderEstimator<TIn>.Fit(TIn input)
            {
                return Fit(input);
            }
        }

        public class MyConcat : IEstimator
        {
            private readonly ConcatTransform _xf;
            private readonly IHostEnvironment _env;
            private readonly string _name;
            private readonly string[] _source;

            public MyConcat(IHostEnvironment env, string name, params string[] source)
            {
                _env = env;
                _name = name;
                _source = source;
            }

            private MyConcat(IHostEnvironment env, ConcatTransform xf)
            {
                _env = env;
                _xf = xf;
            }

            public ITransformer Fit(IDataView input)
            {
                var xf = new ConcatTransform(_env, input, _name, _source);
                return new TransformWrapper(_env, xf);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                var cols = inputSchema.Columns.ToList();

                var selectedCols = cols.Where(x => _source.Contains(x.Name)).Cast<SchemaShape.Column>();
                var isFixed = selectedCols.All(x => x.Kind != SchemaShape.Column.VectorKind.VariableVector);
                var newCol = new SchemaShape.Column(_name,
                    isFixed ? SchemaShape.Column.VectorKind.Vector : SchemaShape.Column.VectorKind.VariableVector,
                    selectedCols.First().ItemKind, selectedCols.First().IsKey);

                cols.Add(newCol);
                return new SchemaShape(cols.ToArray());
            }
        }

        public class MyNormalizer : IEstimator
        {
            private readonly IHostEnvironment _env;
            private readonly string _col;

            public MyNormalizer(IHostEnvironment env, string col)
            {
                _env = env;
                _col = col;
            }

            public ITransformer Fit(IDataView input)
            {
                var xf = NormalizeTransform.CreateMinMaxNormalizer(_env, input, _col);
                return new TransformWrapper(_env, xf);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                return inputSchema;
            }
        }

        public class MySdca : IEstimator
        {

            private readonly IHostEnvironment _env;

            public MySdca(IHostEnvironment env)
            {
                _env = env;
            }

            public ITransformer Fit(IDataView input)
            {
                // Train
                var trainer = new SdcaMultiClassTrainer(_env, new SdcaMultiClassTrainer.Arguments() { NumThreads = 1 });

                // Explicity adding CacheDataView since caching is not working though trainer has 'Caching' On/Auto
                var cached = new CacheDataView(_env, input, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var pred = trainer.Train(trainRoles);

                var scoreRoles = new RoleMappedData(input, label: "Label", feature: "Features");
                IDataScorerTransform scorer = ScoreUtils.GetScorer(pred, scoreRoles, _env, trainRoles.Schema);
                return new Transformer(_env, pred, scorer);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                throw new System.NotImplementedException();
            }

            private sealed class Transformer : ITransformer
            {
                private IHostEnvironment _env;
                private IPredictor _pred;
                private IDataScorerTransform _xf;

                public Transformer(IHostEnvironment env, IPredictorProducing<VBuffer<float>> pred, IDataScorerTransform scorer)
                {
                    _env = env;
                    _pred = pred;
                    _xf = scorer;
                }

                public ISchema GetOutputSchema(ISchema inputSchema)
                {
                    var dv = new EmptyDataView(_env, inputSchema);
                    var output = ApplyTransformUtils.ApplyTransformToData(_env, _xf, dv);
                    return output.Schema;
                }

                public IDataView Transform(IDataView input)
                {
                    return ApplyTransformUtils.ApplyTransformToData(_env, _xf, input);
                }
            }
        }

        public class MyPredictionEngine<TSrc, TDst>
                    where TSrc : class
                    where TDst : class, new()
        {
            private readonly PredictionEngine<TSrc, TDst> _engine;

            public MyPredictionEngine(IHostEnvironment env, ISchema inputSchema, IEnumerable<ITransformer> steps)
            {
                IDataView dv = new EmptyDataView(env, inputSchema);
                foreach (var s in steps)
                    dv = s.Transform(dv);
                _engine = env.CreatePredictionEngine<TSrc, TDst>(dv);
            }

            public TDst Predict(TSrc example)
            {
                return _engine.Predict(example);
            }
        }

        public class MyLogisticRegression : IEstimator
        {
            private readonly IHostEnvironment _env;

            public MyLogisticRegression(IHostEnvironment env)
            {
                _env = env;
            }

            public ITransformer Fit(IDataView input)
            {
                // Train
                var trainer = new LogisticRegression(_env, new LogisticRegression.Arguments());

                // Explicity adding CacheDataView since caching is not working though trainer has 'Caching' On/Auto
                var cached = new CacheDataView(_env, input, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var predictor = trainer.Train(trainRoles);

                return new PredictorTransformer(_env, input.Schema, predictor, "Label", "Features");
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                var cols = inputSchema.Columns.ToList();
                cols.Add(new SchemaShape.Column("Score", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false));
                cols.Add(new SchemaShape.Column("Probability", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false));
                cols.Add(new SchemaShape.Column("PredictedLabel", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false));
                return new SchemaShape(cols.ToArray());
            }
        }

        public class PredictorTransformer : ITransformer
        {
            private readonly IHostEnvironment _env;
            public readonly IPredictor Predictor;
            private readonly IDataScorerTransform _scorer;

            public PredictorTransformer(IHostEnvironment env, ISchema inputSchema, IPredictor predictor, string labelColumn, string featureColumn)
            {
                _env = env;
                Predictor = predictor;

                var input = new EmptyDataView(env, inputSchema);
                var scoreRoles = new RoleMappedData(input, label: labelColumn, feature: featureColumn);

                _scorer = ScoreUtils.GetScorer(predictor, scoreRoles, _env, scoreRoles.Schema);
            }

            public ISchema GetOutputSchema(ISchema inputSchema)
            {
                return _scorer.GetOutputSchema(_env, inputSchema);
            }

            public IDataView Transform(IDataView input)
            {
                return _scorer.Transform(_env, input);
            }
        }

        public class TransformWrapper : ITransformer
        {
            private readonly IHostEnvironment _env;
            private readonly IDataTransform _xf;

            public TransformWrapper(IHostEnvironment env, IDataTransform xf)
            {
                _env = env;
                _xf = xf;
            }

            public ISchema GetOutputSchema(ISchema inputSchema)
            {
                return _xf.GetOutputSchema(_env, inputSchema);
            }

            public IDataView Transform(IDataView input)
            {
                return _xf.Transform(_env, input);
            }
        }

        public sealed class TransformerChain : ITransformer
        {
            private readonly ITransformer[] _transformers;

            public TransformerChain(params ITransformer[] transformers)
            {
                _transformers = transformers.ToArray();
            }

            public ISchema GetOutputSchema(ISchema inputSchema)
            {
                var s = inputSchema;
                foreach (var xf in _transformers)
                {
                    s = xf.GetOutputSchema(s);
                    if (s == null)
                        return null;
                }
                return s;
            }

            public IEnumerable<ITransformer> GetParts()
            {
                return _transformers;
            }

            public IDataView Transform(IDataView input)
            {
                var dv = input;
                foreach (var xf in _transformers)
                {
                    dv = xf.Transform(dv);
                }
                return dv;
            }
        }

        public class MyOva : IEstimator
        {
            private readonly IHostEnvironment _env;
            private readonly IEstimator _binaryEstimator;

            public MyOva(IHostEnvironment env, IEstimator binaryEstimator)
            {
                _env = env;
                _binaryEstimator = binaryEstimator;
            }

            public ITransformer Fit(IDataView input)
            {
                var cached = new CacheDataView(_env, input, prefetch: null);

                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                trainRoles.CheckMultiClassLabel(out var numClasses);

                var predictors = new ITransformer[numClasses];
                var names = Enumerable.Range(0, numClasses).Select(x => $"Score_{x}").ToArray();
                for (int iClass = 0; iClass < numClasses; iClass++)
                {
                    var data = new LabelIndicatorTransform(_env, cached, iClass, "Label");
                    var predictor = _binaryEstimator.Fit(data);
                    var outData = new EmptyDataView(_env, predictor.GetOutputSchema(data.Schema));
                    var copyTransform = new CopyColumnsTransform(_env, outData, names[iClass], "Score");

                    predictors[iClass] = new TransformerChain(predictor, new TransformWrapper(_env, copyTransform));
                }

                var allPredictors = new TransformerChain(predictors);

                var finalConcat = new MyConcat(_env, "Score", names).Fit(allPredictors.Transform(input));

                return new TransformerChain(allPredictors, finalConcat);
            }

            public ITransformer PredictorAwareFit(IDataView input)
            {
                var cached = new CacheDataView(_env, input, prefetch: null);

                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                trainRoles.CheckMultiClassLabel(out var numClasses);

                var predictors = new PredictorTransformer[numClasses];

                for (int iClass = 0; iClass < numClasses; iClass++)
                {
                    var data = new LabelIndicatorTransform(_env, cached, iClass, "Label");
                    predictors[iClass] = _binaryEstimator.Fit(data) as PredictorTransformer;
                }

                var prs = predictors.Select(x => x.Predictor as IPredictorProducing<float>);
                var finalPredictor = OvaPredictor.Create(_env.Register("ova"), prs.ToArray());

                return new PredictorTransformer(_env, input.Schema, finalPredictor, "Label", "Features");
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                var cols = inputSchema.Columns.ToList();
                cols.Add(new SchemaShape.Column("Score", SchemaShape.Column.VectorKind.Vector, DataKind.Num, false));
                cols.Add(new SchemaShape.Column("PredictedLabel", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false));
                return new SchemaShape(cols.ToArray());
            }
        }


        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        public class IrisData
        {
            public float SepalLength;
            public float SepalWidth;
            public float PetalLength;
            public float PetalWidth;
        }

        [Fact]
        public void TestEstimatorPipe()
        {
            var env = new TlcEnvironment();
            var sdca = new MySdca(env);
            var pipeline = new EstimatorPipe<IMultiStreamSource>(new MyTextLoader(env, MakeTextLoaderArgs()));
            pipeline.Append(new MyConcat(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .Append(new MyNormalizer(env, "Features"))
                    .Append(sdca);

            var model = pipeline.Fit(new MultiFileSource(@"e:\data\iris.txt"));

            var scoredTrainData = model.Read(new MultiFileSource(@"e:\data\iris.txt"))
                .AsEnumerable<IrisPrediction>(env, reuseRowObject: false)
                .ToArray();

            IDataReader<IMultiStreamSource> loader;
            IEnumerable<ITransformer> steps;
            (loader, steps) = model.GetParts();

            var engine = new MyPredictionEngine<IrisData, IrisPrediction>(env, loader.GetOutputSchema(), steps);
            IrisPrediction prediction = engine.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });
        }

        [Fact]
        public void TestOvaTraining()
        {
            var env = new TlcEnvironment();
            var pipeline = new EstimatorPipe<IMultiStreamSource>(new MyTextLoader(env, MakeTextLoaderArgs()));
            pipeline.Append(new MyConcat(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .Append(new MyNormalizer(env, "Features"));

            var lrEstimator = new MyLogisticRegression(env);

            pipeline.Append(new MyOva(env, lrEstimator));


            var model = pipeline.Fit(new MultiFileSource(@"e:\data\iris.txt"));
            var s = model.GetOutputSchema();

            IDataReader<IMultiStreamSource> loader;
            IEnumerable<ITransformer> steps;
            (loader, steps) = model.GetParts();

            var engine = new MyPredictionEngine<IrisData, IrisPrediction>(env, loader.GetOutputSchema(), steps);
            IrisPrediction prediction = engine.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });
        }
    }
}
