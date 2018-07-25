using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Learners;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Core.Tests.UnitTests
{
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

        public class MyTextLoader : IEstimator<IMultiStreamSource>, ITransformer<IMultiStreamSource>
        {
            private readonly TextLoader.Arguments _args;
            private readonly IHostEnvironment _env;

            public MyTextLoader(IHostEnvironment env, TextLoader.Arguments args)
            {
                _env = env;
                _args = args;
            }

            public ITransformer<IMultiStreamSource> Fit(IMultiStreamSource input)
            {
                return this;
            }

            public SchemaShape GetOutputSchema()
            {
                var emptyData = new TextLoader(new TlcEnvironment(), _args, new MultiFileSource(null));
                return SchemaShape.Create(emptyData.Schema);
            }

            public IDataView Transform(IMultiStreamSource input)
            {
                return new TextLoader(new TlcEnvironment(), _args, input);
            }

            ISchema ITransformer<IMultiStreamSource>.GetOutputSchema()
            {
                var emptyData = new TextLoader(new TlcEnvironment(), _args, new MultiFileSource(null));
                return emptyData.Schema;
            }
        }

        public class TransformerPipe<TIn> : ITransformer<TIn>
        {
            private readonly ITransformer<TIn> _start;
            private readonly IDataTransformer[] _chain;

            public TransformerPipe(ITransformer<TIn> start, IDataTransformer[] chain)
            {
                _start = start;
                _chain = chain;
            }

            public IDataView Transform(TIn input)
            {
                var idv = _start.Transform(input);
                foreach (var xf in _chain)
                    idv = xf.Transform(idv);
                return idv;
            }

            public (ITransformer<TIn>, IEnumerable<IDataTransformer>) GetParts()
            {
                return (_start, _chain);
            }

            public ISchema GetOutputSchema()
            {
                var s = _start.GetOutputSchema();
                foreach (var xf in _chain)
                    s = xf.GetOutputSchema(s);
                return s;
            }
        }

        public class EstimatorPipe<TIn> : IEstimator<TIn>
        {
            private readonly IEstimator<TIn> _start;
            private readonly List<IDataEstimator> _estimatorChain = new List<IDataEstimator>();
            private readonly IHostEnvironment _env = new TlcEnvironment();


            public EstimatorPipe(IEstimator<TIn> start)
            {
                _start = start;
            }

            public EstimatorPipe<TIn> Append(IDataEstimator est)
            {
                _estimatorChain.Add(est);
                return this;
            }

            public TransformerPipe<TIn> Fit(TIn input)
            {
                var start = _start.Fit(input);

                var idv = start.Transform(input);
                var xfs = new List<IDataTransformer>();
                foreach (var est in _estimatorChain)
                {
                    var xf = est.Fit(idv);
                    xfs.Add(xf);
                    idv = xf.Transform(idv);
                }
                return new TransformerPipe<TIn>(start, xfs.ToArray());
            }

            public IEstimator<TIn> GetEstimator()
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

            public (IEstimator<TIn>, IEnumerable<IDataEstimator>) GetParts()
            {
                return (_start, _estimatorChain);
            }

            ITransformer<TIn> IEstimator<TIn>.Fit(TIn input)
            {
                return Fit(input);
            }
        }

        public class MyConcatTransformer : IDataEstimator, IDataTransformer
        {
            private readonly ConcatTransform _xf;
            private readonly IHostEnvironment _env;
            private readonly string _name;
            private readonly string[] _source;

            public MyConcatTransformer(IHostEnvironment env, string name, params string[] source)
            {
                _env = env;
                _name = name;
                _source = source;
            }

            private MyConcatTransformer(IHostEnvironment env, ConcatTransform xf)
            {
                _env = env;
                _xf = xf;
            }

            public IDataTransformer Fit(IDataView input)
            {
                var xf = new ConcatTransform(_env, input, _name, _source);
                return new MyConcatTransformer(_env, xf);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                var cols = inputSchema.Columns.ToList();

                var selectedCols = cols.Where(x => _source.Contains(x.Name)).Cast<SchemaShape.RelaxedColumn>();
                var isFixed = selectedCols.All(x => x.Kind != SchemaShape.RelaxedColumn.VectorKind.VariableVector);
                var newCol = new SchemaShape.RelaxedColumn(_name,
                    isFixed ? SchemaShape.RelaxedColumn.VectorKind.Vector : SchemaShape.RelaxedColumn.VectorKind.VariableVector,
                    selectedCols.First().ItemKind, selectedCols.First().IsKey);

                cols.Add(newCol);
                return new SchemaShape(cols.ToArray());
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

        public class MyNormalizer : IDataEstimator
        {
            private readonly IHostEnvironment _env;
            private readonly string _col;

            public MyNormalizer(IHostEnvironment env, string col)
            {
                _env = env;
                _col = col;
            }

            public IDataTransformer Fit(IDataView input)
            {
                return new Transformer(_env, input, _col);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                return inputSchema;
            }

            private class Transformer : IDataTransformer
            {
                private IHostEnvironment _env;
                private IDataTransform _xf;

                public Transformer(IHostEnvironment env, IDataView input, string col)
                {
                    _env = env;
                    _xf = NormalizeTransform.CreateMinMaxNormalizer(env, input, col);
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

        public class MySdca : IDataEstimator
        {

            private readonly IHostEnvironment _env;

            public MySdca(IHostEnvironment env)
            {
                _env = env;
            }

            public IDataTransformer Fit(IDataView input)
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

            private sealed class Transformer : IDataTransformer
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

            public MyPredictionEngine(IHostEnvironment env, ISchema inputSchema, IEnumerable<IDataTransformer> steps)
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

            var pipeline = new EstimatorPipe<IMultiStreamSource>(new MyTextLoader(env, MakeTextLoaderArgs()));
            pipeline.Append(new MyConcatTransformer(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .Append(new MyNormalizer(env, "Features"))
                    .Append(new MySdca(env));

            var model = pipeline.Fit(new MultiFileSource(@"e:\data\iris.txt"));

            var scoredTrainData = model.Transform(new MultiFileSource(@"e:\data\iris.txt"))
                .AsEnumerable<IrisPrediction>(env, reuseRowObject: false)
                .ToArray();

            ITransformer<IMultiStreamSource> loader;
            IEnumerable<IDataTransformer> steps;
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
