// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(KMeansModelParameters), null, typeof(SignatureLoadModel),
    "KMeans predictor", KMeansModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[KMeans](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Clustering/KMeans.cs)]
    /// ]]></format>
    /// </example>
    public sealed class KMeansModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveInTextFormat,
        ISingleCanSaveOnnx
    {
        internal const string LoaderSignature = "KMeansPredictor";

        /// <summary>
        /// Version information to be saved in binary format
        /// </summary>
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "KMEANS  ",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Allow sparse centroids
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(KMeansModelParameters).Assembly.FullName);
        }

        private protected override PredictionKind PredictionKind => PredictionKind.Clustering;

        private readonly DataViewType _inputType;
        private readonly DataViewType _outputType;
        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => _outputType;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        private readonly int _dimensionality;
        private readonly int _k;
        private readonly VBuffer<float>[] _centroids;
        private readonly float[] _centroidL2s; // L2 norms of the centroids

        /// <summary>
        /// Initialize predictor with a trained model.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="k">Number of centroids</param>
        /// <param name="centroids">Coordinates of the centroids</param>
        /// <param name="copyIn">If true then the <paramref name="centroids"/> vectors will be subject to
        /// a deep copy, if false then this constructor will take ownership of the passed in centroid vectors.
        /// If false then the caller must take care to not use or modify the input vectors once this object
        /// is constructed, and should probably remove all references.</param>
        internal KMeansModelParameters(IHostEnvironment env, int k, VBuffer<float>[] centroids, bool copyIn)
            : base(env, LoaderSignature)
        {
            Host.CheckParam(k > 0, nameof(k), "Need at least one cluster");
            Host.CheckParam(Utils.Size(centroids) >= k, nameof(centroids), "Not enough centroids for predictor initialization");
            Host.CheckParam(centroids[0].Length > 0, nameof(centroids), "Centroid vectors should have at least one length");

            _k = k;
            _dimensionality = centroids[0].Length;

            _centroidL2s = new float[_k];
            _centroids = new VBuffer<float>[_k];
            for (int i = 0; i < _k; i++)
            {
                Host.CheckParam(centroids[i].Length == _dimensionality,
                    nameof(centroids), "Inconsistent dimensions found among centroids");
                Host.CheckParam(FloatUtils.IsFinite(centroids[i].GetValues()),
                    nameof(centroids), "Cannot initialize K-means predictor with non-finite centroid coordinates");
                if (copyIn)
                    centroids[i].CopyTo(ref _centroids[i]);
                else
                    _centroids[i] = centroids[i];
            }

            InitPredictor();

            _inputType = new VectorDataViewType(NumberDataViewType.Single, _dimensionality);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, _k);
        }

        /// <summary>
        /// Initialize predictor from a binary file.
        /// </summary>
        /// <param name="ctx">The load context</param>
        /// <param name="env">The host environment</param>
        private KMeansModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: k, number of clusters
            // int: dimensionality, length of the centroid vectors
            // for each cluster, then:
            //     int: count of this centroid vector (sparse iff count < dimensionality)
            //     int[count]: only present if sparse, in order indices
            //     Float[count]: centroid vector values

            _k = ctx.Reader.ReadInt32();
            Host.CheckDecode(_k > 0);
            _dimensionality = ctx.Reader.ReadInt32();
            Host.CheckDecode(_dimensionality > 0);

            _centroidL2s = new float[_k];
            _centroids = new VBuffer<float>[_k];
            for (int i = 0; i < _k; i++)
            {
                // Prior to allowing sparse vectors, count was not written and was implicitly
                // always equal to dimensionality, and no indices were written either.
                int count = ctx.Header.ModelVerWritten >= 0x00010002 ? ctx.Reader.ReadInt32() : _dimensionality;
                Host.CheckDecode(0 <= count && count <= _dimensionality);
                var indices = count < _dimensionality ? ctx.Reader.ReadIntArray(count) : null;
                var values = ctx.Reader.ReadFloatArray(count);
                Host.CheckDecode(FloatUtils.IsFinite(values));
                _centroids[i] = new VBuffer<float>(_dimensionality, count, values, indices);
            }
            WarnOnOldNormalizer(ctx, GetType(), Host);

            InitPredictor();

            _inputType = new VectorDataViewType(NumberDataViewType.Single, _dimensionality);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, _k);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            ValueMapper<VBuffer<float>, VBuffer<float>> del =
                (in VBuffer<float> src, ref VBuffer<float> dst) =>
                {
                    if (src.Length != _dimensionality)
                        throw Host.Except($"Incorrect number of features: expected {_dimensionality}, got {src.Length}");
                    var editor = VBufferEditor.Create(ref dst, _k);
                    Map(in src, editor.Values);
                    dst = editor.Commit();
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private void Map(in VBuffer<float> src, Span<float> distances)
        {
            Host.Assert(distances.Length >= _k);

            float instanceL2 = VectorUtils.NormSquared(in src);
            for (int i = 0; i < _k; i++)
            {
                float distance = Math.Max(0,
                    -2 * VectorUtils.DotProduct(in _centroids[i], in src) + _centroidL2s[i] + instanceL2);
                distances[i] = distance;
            }
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("K: {0}", _k);
            writer.WriteLine("Dimensionality: {0}", _dimensionality);
            writer.WriteLine("Centroid coordinates");
            for (int i = 0; i < _k; i++)
            {
                Host.Assert(_dimensionality == _centroids[i].Length);
                var values = _centroids[i].GetValues();
                if (_centroids[i].IsDense)
                {
                    for (int j = 0; j < values.Length; j++)
                    {
                        if (j != 0)
                            writer.Write('\t');
                        writer.Write(values[j]);
                    }
                }
                else if (values.Length > 0)
                {
                    // Sparse and non-empty, write as key:value pairs.
                    bool isFirst = true;
                    foreach (var pair in _centroids[i].Items())
                    {
                        if (isFirst)
                            isFirst = false;
                        else
                            writer.Write('\t');
                        writer.Write(pair.Key);
                        writer.Write(':');
                        writer.Write(pair.Value);
                    }
                }
                else
                {
                    // Sparse but empty, write at least "0:0" since otherwise it will just look like a bug.
                    writer.Write("0:0");
                }
                writer.WriteLine();
            }
        }

        /// <summary>
        /// Save the predictor in binary format.
        /// </summary>
        /// <param name="ctx">The context to save to</param>
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            var writer = ctx.Writer;

            // *** Binary format ***
            // int: k, number of clusters
            // int: dimensionality, length of the centroid vectors
            // for each cluster, then:
            //     int: count of this centroid vector (sparse iff count < dimensionality)
            //     int[count]: only present if sparse, in order indices
            //     Float[count]: centroid vector values

            writer.Write(_k);
            writer.Write(_dimensionality);

            for (int i = 0; i < _k; i++)
            {
                Contracts.Assert(_centroids[i].Length == _dimensionality);
                var values = _centroids[i].GetValues();
                writer.Write(values.Length);
                if (!_centroids[i].IsDense)
                    writer.WriteIntsNoCount(_centroids[i].GetIndices());
                Contracts.Assert(FloatUtils.IsFinite(values));
                writer.WriteSinglesNoCount(values);
            }
        }

        /// <summary>
        /// This method is called by reflection to instantiate a predictor.
        /// </summary>
        private static KMeansModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new KMeansModelParameters(env, ctx);
        }

        /// <summary>
        /// Initialize internal parameters: L2 norms of the _centroids.
        /// </summary>
        private void InitPredictor()
        {
            for (int i = 0; i < _k; i++)
                _centroidL2s[i] = VectorUtils.NormSquared(_centroids[i]);
        }

        /// <summary>
        /// Copies the centroids to a set of provided buffers.
        /// </summary>
        /// <param name="centroids">The buffer to which to copy. Will be extended to
        /// an appropriate length, if necessary.</param>
        /// <param name="k">The number of clusters, corresponding to the logical size of
        /// <paramref name="centroids"/>.</param>
        public void GetClusterCentroids(ref VBuffer<float>[] centroids, out int k)
        {
            Contracts.Assert(_centroids.Length == _k);
            Utils.EnsureSize(ref centroids, _k, _k);
            for (int i = 0; i < _k; i++)
                _centroids[i].CopyTo(ref centroids[i]);
            k = _k;
        }

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
        {
            // Computation graph of distances to all centriods for a batch of examples. Note that a centriod is just
            // the center of a cluster. We use [] to denote the dimension of a variable; for example, X [3, 2] means
            // that X is a 3-by-2 tensor. In addition, for a matrix X, X^T denotes its transpose.
            //
            // Symbols:
            // l: # of examples.
            // n: # of features per input example.
            // X: input examples, l-by-n tensor.
            // C: centriods, k-by-n tensor.
            // C^2: 2-norm of all centriod vectors, its shape is [k].
            // Y: 2-norm of difference between examples and centriods, l-by-k tensor. The value at i-th row and k-th
            // column row, Y[i,k], is the distance from example i to centrioid k.
            // L: the id of the nearest centriod for each input example, its shape is [l].
            //
            // .------------------------------------------------------.
            // |                                                      |
            // |                                                      v
            // X [l, n] --> ReduceSumSquare --> X^2 [l]             Gemm (alpha=-2, transB=1) <-- C [k, n]
            //                                   |                    |
            //                                   |                    v
            //                                   `------> Add <---- -2XC^T [l, k]
            //                                             |
            //                                             v
            //                                             Z [l, k] ----------> Add <------------C^2 [k]
            //                                                                   |
            //                                                                   v
            //                                           L [l] <--- ArgMin <---  Y [l, k]

            // Allocate C, which is a constant tensor in prediction phase
            var shapeC = new long[] { _centroids.Length, _centroids[0].Length };
            var tensorC = new List<float>();
            foreach (var centriod in _centroids)
                tensorC.AddRange(centriod.DenseValues());
            var nameC = ctx.AddInitializer(tensorC, shapeC, "C");

            // Save C^2 as an initializer because it's a constant.
            var shapeC2 = new long[] { _centroidL2s.Length };
            var nameC2 = ctx.AddInitializer(_centroidL2s, shapeC2, "C2");

            // Retrieve the name of X
            var nameX = featureColumn;

            // Compute X^2 from X
            var nameX2 = ctx.AddIntermediateVariable(null, "X2", true);
            var reduceNodeX2 = ctx.CreateNode("ReduceSumSquare", nameX, nameX2, ctx.GetNodeName("ReduceSumSquare"), "");

            // Compute -2XC^T. Note that Gemm always takes three inputs. Since we only have two here,
            // a dummy one, named zero, is created.
            var zeroName = ctx.AddInitializer(new float[] { 0f }, null, "zero");
            var nameXC2 = ctx.AddIntermediateVariable(null, "XC2", true);
            var gemmNodeXC2 = ctx.CreateNode("Gemm", new[] { nameX, nameC, zeroName }, new[] { nameXC2 }, ctx.GetNodeName("Gemm"), "");
            gemmNodeXC2.AddAttribute("alpha", -2f);
            gemmNodeXC2.AddAttribute("transB", 1);

            // Compute Z = X^2 - 2XC^T
            var nameZ = ctx.AddIntermediateVariable(null, "Z", true);
            var addNodeZ = ctx.CreateNode("Add", new[] { nameX2, nameXC2 }, new[] { nameZ }, ctx.GetNodeName("Add"), "");

            // Compute Y = Z + C^2
            var nameY = outputNames[1];
            var addNodeY = ctx.CreateNode("Add", new[] { nameZ, nameC2 }, new[] { nameY }, ctx.GetNodeName("Add"), "");

            // Compute the most-matched cluster index, L
            var nameL = "ArgMinInt64";
            var predictNodeL = ctx.CreateNode("ArgMin", nameY, nameL, ctx.GetNodeName("ArgMin"), "");
            predictNodeL.AddAttribute("axis", 1);
            predictNodeL.AddAttribute("keepdims", 1);

            // ArgMin outputs an Int64. But ML.NET's KMeans trainer outputs a UINT32.
            // Cast the output here to UInt32 to make them compatible
            var predictedNode = ctx.CreateNode("Cast", nameL, outputNames[0], ctx.GetNodeName("Cast"), "");
            var t = InternalDataKindExtensions.ToInternalDataKind(DataKind.UInt32).ToType();
            predictedNode.AddAttribute("to", t);

            return true;
        }
    }
}