// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.OneDal;

#if false
using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
#endif

[assembly: LoadableClass(typeof(KnnClassificationTransformer), null, typeof(SignatureLoadModel),
    KnnClassificationTransformer.UserName, KnnClassificationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(KnnClassificationTransformer), null, typeof(SignatureLoadRowMapper),
    KnnClassificationTransformer.UserName, KnnClassificationTransformer.LoaderSignature)]

namespace Microsoft.ML.OneDal
{
    public class KnnClassificationTrainer : IEstimator<KnnClassificationTransformer>
    {
	private readonly IHost _host;
        private KnnAlgorithm _knnAlgorithm;

	internal class KnnClassificationOptions
        {
            private readonly int _numNeighbors = default(int);
            public int NumNeighbors { get;  }
            KnnClassificationOptions(int numNeighbors)
            {
                _numNeighbors = numNeighbors;
            }
        }

        KnnClassificationTrainer(IHostEnvironment env, int numClasses) // FIXME -- pass an Options instance rather than naked numClasses
        {
            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("KnnClassificationTrainer");
            Contracts.Assert(numClasses > 0);
            _knnAlgorithm = new KnnAlgorithm(numClasses);
        }

        public KnnClassificationTransformer Fit(IDataView input)
        {
            int featureDimensionality = default(int);
            if (featuresColumn.Type is VectorDataViewType vt) {
                featureDimensionality = vt.Size;
            } else {
                return null;
            }

            int samples = 0;

            using (var cursor = input.GetRowCursor(new[] { featuresColumn, labelColumn }))
            {
                float labelValue = default;
                VBuffer<float> featureValues = default(VBuffer<float>);

                var featureGetter = cursor.GetGetter<VBuffer<float>>(featuresColumn);
                var labelGetter = cursor.GetGetter<float>(labelColumn);

		List< VBuffer<float> > tempFeatures = new List <VBuffer<float>>();
		List< float > tempLabels = new List<float>();
                while (cursor.MoveNext())
                {
                    featureGetter(ref featureValues);
                    labelGetter(ref labelValue);
    		    tempFeatures.Add(featureValues);
       		    tempLabels.Add(labelValue);

                    samples++;
                }

		float[] data = new float[samples * featureDimensionality];
		float[] dataLabels = new float[samples];
                Span<float> dataSpan = new Span<float>(data);

		for( int i = 0; i < tempFeatures.Count(); i++) {
	    
                    int offset = i * featureDimensionality;
                    Span<float> target = dataSpan.Slice(offset, featureDimensionality);
                    tempFeatures.GetValues().CopyTo(target);
                    dataLabels[i] = tempLabels[i];
		 }
		 _knnAlgorithm.Train(data, dataLabels);

            }

            return new KnnClassificationTransformer(_host, _knnAlgorithm);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);

// FIXME
            // This loop checks if all input columns needed in the underlying transformer can be found
            // in inputSchema.
            for (var i = 0; i < Transformer.Inputs.Length; i++)
            {
                // Get the i-th IDataView input column's name in the underlying ONNX transformer.
                var input = Transformer.Inputs[i];

                // Make sure inputSchema contains the i-th input column.
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);

                // Make sure that the input columns in inputSchema are fixed shape tensors.
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());

                var inputsInfo = Transformer.Model.ModelInfo.InputsInfo;
                var idx = Transformer.Model.ModelInfo.InputNames.IndexOf(input);
                if (idx < 0)
                    throw Host.Except($"Column {input} doesn't match input node names of model.");

                var inputNodeInfo = inputsInfo[idx];
                var expectedType = ((VectorDataViewType)inputNodeInfo.DataViewType).ItemType;
                if (col.ItemType != expectedType)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }

            for (var i = 0; i < Transformer.Outputs.Length; i++)
            {
                resultDic[Transformer.Outputs[i]] = new SchemaShape.Column(Transformer.Outputs[i],
                    Transformer.OutputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, Transformer.OutputTypes[i].GetItemType(), false);
            }

            return new SchemaShape(resultDic.Values);
        }
    }

    public sealed class KnnClassificationTransformer : ITransformer, IDisposable
    {
        internal const string LoadName = "KNNClassificationTrainer";
        internal const string UserName = "KNN Classification Trainer";
        internal const string ShortName = "KNN";
        internal const string Summary = "k-Nearest Neighbors binary- and multi-class classification";

        internal const string LoaderSignature = "KNNCLASS";

	private KnnAlgorithm _knn; // Assumes the model is trained

        public KnnClassificationTransformer(IHost host, KnnAlgorithm knn)
	  : base(host)
        {
        }

        public IDataView Transform(IDataView input)
	{
	   _host.CheckValue(input, nameof(input));
	   return new KnnDataView(input, _knn);
	}

        public void Dispose()
        {
            _knn.Dispose();
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }
    }
 }