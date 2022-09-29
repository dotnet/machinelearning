using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Trainers.XGBoost
{
    internal static class Defaults
    {
        public const int NumberOfIterations = 100;
    }

    public sealed class XGBoostBinaryClassificationTransformer : OneToOneTransformerBase
    {
	private Booster _booster;
	private int _numColumns;
	
        internal XGBoostBinaryClassificationTransformer(IHost host, Booster booster, params (string outputColumnName, string inputColumnName)[] columns) : base(host, columns)
        {
	  _booster = booster;
	  _numColumns = columns.Length;
        }

        internal XGBoostBinaryClassificationTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly XGBoostBinaryClassificationTransformer _parent;
            private readonly int _numColumns;
            public Mapper(XGBoostBinaryClassificationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
		_numColumns = _parent._numColumns;
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                throw new NotImplementedException();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
	        var result = new DataViewSchema.DetachedColumn[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                    result[i] = new DataViewSchema.DetachedColumn("PredictedLabel", NumberDataViewType.Int16, null);
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
		Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _numColumns);
                disposer = null;

                var srcGetter = input.GetGetter<VBuffer<float>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(VBuffer<float>);

                ValueGetter<VBuffer<float>> dstGetter = (ref VBuffer<float> dst) =>
                    {
                        srcGetter(ref src);
                        Predict(Host, in src, ref dst);
                    };

                return dstGetter;
            }

	    private void Predict(IExceptionContext ectx, in VBuffer<float> src, ref VBuffer<float> dst)
            {
	    #if false
	    	dst = _parent._booster.Predict(src);
	    #else
	        dst = new VBuffer<float>();
	    #endif
            }

        }
    }

    public sealed class XGBoostBinaryClassificationEstimator : IEstimator<XGBoostBinaryClassificationTransformer>
    {
        private readonly IHost _host;

       public sealed class Options : TrainerInputBase
       {
            /// <summary>
            /// Maximum tree depth for base learners
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum tree depth for base learners.", ShortName = "us")]
            public int MaxDepth = 3;
       }

        public XGBoostBinaryClassificationEstimator(IHost host, XGBoostBinaryClassificationTransformer transformer)         {
            _host = Contracts.CheckRef(host, nameof(host)).Register(nameof(XGBoostBinaryClassificationEstimator));
        }

        public XGBoostBinaryClassificationEstimator(IHostEnvironment env, string labelColumnName, string featureColumnName, int? numberOfLeaves, int? minimumExampleCountPerLeaf, double? learningRate, int? numberOfIterations)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(XGBoostBinaryClassificationEstimator));
        }

        public XGBoostBinaryClassificationTransformer Fit(IDataView input)
        {
	    	var featuresColumn = input.Schema["Features"];
		var labelColumn = input.Schema["Label"];
		int featureDimensionality = default(int);
		if (featuresColumn.Type is VectorDataViewType vt) {
		  featureDimensionality = vt.Size;
		} else {
		  _host.Except($"A vector input is expected");
		}
		int samples = 0;
		int maxSamples = 10000;

		float[] data = new float[ maxSamples * featureDimensionality];
		float[] dataLabels = new float[ maxSamples ];
		Span<float> dataSpan = new Span<float>(data);

		using (var cursor = input.GetRowCursor(new[] { featuresColumn, labelColumn })) {

		  float labelValue = default;
		  VBuffer<float> featureValues = default(VBuffer<float>);

		  var featureGetter = cursor.GetGetter< VBuffer<float> >(featuresColumn);
		  var labelGetter = cursor.GetGetter<float>(labelColumn);

		  while (cursor.MoveNext() && samples < maxSamples) {
		    featureGetter(ref featureValues);
		    labelGetter(ref labelValue);

		    int offset = samples * featureDimensionality;
		    Span<float> target = dataSpan.Slice(offset, featureDimensionality);
		    featureValues.GetValues().CopyTo(target);
		    dataLabels[samples] = labelValue;
		    samples++;
	    	  }

		  DMatrix trainMat = new DMatrix(data, (uint)maxSamples, (uint)featureDimensionality, dataLabels);
		  Booster booster = new Booster(trainMat);
  		  return new XGBoostBinaryClassificationTransformer(_host, booster, ("Features", "PredictedLabel"));
	  	}
        }

#if true
        // Used for schema propagation and verification in a pipeline (i.e., in an Estimator chain).
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
#else
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            return new SchemaShape(inputSchema);
        }
#endif
    }
}
