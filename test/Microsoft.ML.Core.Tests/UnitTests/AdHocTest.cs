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
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Tests.Scenarios.Api;

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

        public class MyTextLoader : IDataReaderEstimator<IMultiStreamSource, IDataReader<IMultiStreamSource>>, IDataReader<IMultiStreamSource>
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

        public class CompositeReader<TIn> : IDataReader<TIn>
        {
            protected readonly IDataReader<TIn> _start;
            private readonly TransformerChain _chain;

            public CompositeReader(IDataReader<TIn> start, ITransformer[] chain)
            {
                _start = start;
                _chain = new TransformerChain(chain);
            }

            public IDataView Read(TIn input)
            {
                var idv = _start.Read(input);
                idv = _chain.Transform(idv);
                return idv;
            }

            public (IDataReader<TIn> start, TransformerChain chain) GetParts()
            {
                return (start: _start, chain: _chain);
            }

            public ISchema GetOutputSchema()
            {
                var s = _start.GetOutputSchema();
                s = _chain.GetOutputSchema(s);
                return s;
            }
        }

        public class CompositeReader<TIn, TLast> : CompositeReader<TIn>
            where TLast : class, ITransformer
        {
            private readonly TransformerChain<TLast> _typedChain;

            public CompositeReader(IDataReader<TIn> start, ITransformer[] transformers)
                : base(start, transformers)
            {
                _typedChain = new TransformerChain<TLast>(transformers);
            }

            public CompositeReader(IDataReader<TIn> start, TransformerChain<TLast> typedChain)
                : base(start, typedChain.GetParts().ToArray())
            {
                _typedChain = typedChain;
            }

            public new (IDataReader<TIn> start, TransformerChain<TLast> chain) GetParts()
            {
                return (start: _start, chain: _typedChain);
            }
        }

        public class EstimatorChain<TTrans> : IEstimator<TransformerChain<TTrans>>
            where TTrans : class, ITransformer
        {
            private readonly IEstimator<ITransformer>[] _chain;
            public readonly IEstimator<TTrans> Last;

            private EstimatorChain()
            {
                _chain = new IEstimator<ITransformer>[0];
                Last = null;
            }

            public static EstimatorChain<TTrans> CreateEmpty()
            {
                return new EstimatorChain<ITransformer>() as EstimatorChain<TTrans>;
            }

            public TransformerChain<TTrans> Fit(IDataView input)
            {
                var dv = input;
                var xfs = new ITransformer[_chain.Length];
                for (int i = 0; i < _chain.Length; i++)
                {
                    var est = _chain[i];
                    xfs[i] = est.Fit(dv);
                    dv = xfs[i].Transform(dv);
                }

                return new TransformerChain<TTrans>(xfs);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                var s = inputSchema;
                foreach (var est in _chain)
                {
                    s = est.GetOutputSchema(s);
                    if (s == null)
                        return null;
                }
                return s;
            }

            public EstimatorChain(IEstimator<ITransformer>[] chain)
            {
                Contracts.CheckNonEmpty(chain, nameof(chain));

                var last = chain.Last() as IEstimator<TTrans>;
                Contracts.Check(last != null);

                _chain = chain;
                Last = last;
            }

            public EstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator)
                where TNewTrans : class, ITransformer
            {
                Contracts.CheckValue(estimator, nameof(estimator));
                return new EstimatorChain<TNewTrans>(_chain.Append(estimator).ToArray());
            }
        }

        public class MyConcat : IEstimator<TransformWrapper>
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

            public TransformWrapper Fit(IDataView input)
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
                    selectedCols.First().ItemKind, selectedCols.First().IsKey, new string[0]);

                cols.Add(newCol);
                return new SchemaShape(cols.ToArray());
            }
        }

        public class MyNormalizer : IEstimator<TransformWrapper>
        {
            private readonly IHostEnvironment _env;
            private readonly string _col;

            public MyNormalizer(IHostEnvironment env, string col)
            {
                _env = env;
                _col = col;
            }

            public TransformWrapper Fit(IDataView input)
            {
                var xf = NormalizeTransform.CreateMinMaxNormalizer(_env, input, _col);
                return new TransformWrapper(_env, xf);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                return inputSchema;
            }
        }

        public class MySdca : IEstimator<MySdca.Transformer>
        {

            private readonly IHostEnvironment _env;

            public MySdca(IHostEnvironment env)
            {
                _env = env;
            }

            public Transformer Fit(IDataView input)
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

            public sealed class Transformer : ITransformer
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

            public MyPredictionEngine(IHostEnvironment env, ISchema inputSchema, ITransformer pipe)
            {
                IDataView dv = new EmptyDataView(env, inputSchema);
                _engine = env.CreatePredictionEngine<TSrc, TDst>(pipe.Transform(dv));
            }

            public TDst Predict(TSrc example)
            {
                return _engine.Predict(example);
            }
        }

        public class MyLogisticRegression : IEstimator<PredictorTransformer>
        {
            private readonly IHostEnvironment _env;

            public MyLogisticRegression(IHostEnvironment env)
            {
                _env = env;
            }

            public PredictorTransformer Fit(IDataView input)
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
                cols.Add(new SchemaShape.Column("Score", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false, new string[0]));
                cols.Add(new SchemaShape.Column("Probability", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false, new string[0]));
                cols.Add(new SchemaShape.Column("PredictedLabel", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false, new string[0]));
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

        public class TransformerChain : ITransformer
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

        public sealed class TransformerChain<TLast> : TransformerChain
            where TLast : class, ITransformer
        {
            public readonly TLast Last;

            public TransformerChain(params ITransformer[] transformers)
                : base(transformers)
            {
                if (Utils.Size(transformers) == 0)
                {
                    Last = null;
                }
                else
                {
                    Last = transformers.Last() as TLast;
                    Contracts.Check(Last != null);
                }
            }
        }

        public class MyOva : IEstimator<ITransformer>
        {
            private readonly IHostEnvironment _env;
            private readonly IEstimator<ITransformer> _binaryEstimator;

            public MyOva(IHostEnvironment env, IEstimator<ITransformer> binaryEstimator)
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
                cols.Add(new SchemaShape.Column("Score", SchemaShape.Column.VectorKind.Vector, DataKind.Num, false, new string[0]));
                cols.Add(new SchemaShape.Column("PredictedLabel", SchemaShape.Column.VectorKind.Scalar, DataKind.Num, false, new string[0]));
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
            var pipeline = new CompositeReaderEstimator<IMultiStreamSource, ITransformer>(new MyTextLoader(env, MakeTextLoaderArgs()));
            pipeline.Append(new MyConcat(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .Append(new MyNormalizer(env, "Features"))
                    .Append(sdca);

            var model = pipeline.Fit(new MultiFileSource(@"e:\data\iris.txt"));

            var scoredTrainData = model.Read(new MultiFileSource(@"e:\data\iris.txt"))
                .AsEnumerable<IrisPrediction>(env, reuseRowObject: false)
                .ToArray();

            (var loader, var steps) = model.GetParts();

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
            var pipeline = new CompositeReaderEstimator<IMultiStreamSource, ITransformer>(new MyTextLoader(env, MakeTextLoaderArgs()));
            pipeline.Append(new MyConcat(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .Append(new MyNormalizer(env, "Features"));

            var lrEstimator = new MyLogisticRegression(env);

            pipeline.Append(new MyOva(env, lrEstimator));


            var model = pipeline.Fit(new MultiFileSource(@"e:\data\iris.txt"));
            var s = model.GetOutputSchema();

            (var loader, var pipe) = model.GetParts();

            var x = new MySdca(env) as IEstimator<ITransformer>;

            var engine = new MyPredictionEngine<IrisData, IrisPrediction>(env, loader.GetOutputSchema(), pipe);
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
