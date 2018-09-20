// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.IO;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Internal.Internallearn;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(KMeansPredictor), null, typeof(SignatureLoadModel),
    "KMeans predictor", KMeansPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.KMeans
{
    public sealed class KMeansPredictor :
        PredictorBase<VBuffer<Float>>,
        IValueMapper,
        ICanSaveInTextFormat,
        ISingleCanSaveOnnx
    {
        public const string LoaderSignature = "KMeansPredictor";

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
                loaderSignature: LoaderSignature);
        }

        public override PredictionKind PredictionKind => PredictionKind.Clustering;
        public ColumnType InputType { get; }
        public ColumnType OutputType { get; }

        bool ICanSaveOnnx.CanSaveOnnx => true;

        private readonly int _dimensionality;
        private readonly int _k;
        private readonly VBuffer<Float>[] _centroids;
        private readonly Float[] _centroidL2s; // L2 norms of the centroids

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
        public KMeansPredictor(IHostEnvironment env, int k, VBuffer<float>[] centroids, bool copyIn)
            : base(env, LoaderSignature)
        {
            Host.CheckParam(k > 0, nameof(k), "Need at least one cluster");
            Host.CheckParam(Utils.Size(centroids) >= k, nameof(centroids), "Not enough centroids for predictor initialization");
            Host.CheckParam(centroids[0].Length > 0, nameof(centroids), "Centroid vectors should have at least one length");

            _k = k;
            _dimensionality = centroids[0].Length;

            _centroidL2s = new Float[_k];
            _centroids = new VBuffer<Float>[_k];
            for (int i = 0; i < _k; i++)
            {
                Host.CheckParam(centroids[i].Length == _dimensionality,
                    nameof(centroids), "Inconsistent dimensions found among centroids");
                Host.CheckParam(FloatUtils.IsFinite(centroids[i].Values, centroids[i].Count),
                    nameof(centroids), "Cannot initialize K-means predictor with non-finite centroid coordinates");
                if (copyIn)
                    centroids[i].CopyTo(ref _centroids[i]);
                else
                    _centroids[i] = centroids[i];
            }

            InitPredictor();

            InputType = new VectorType(NumberType.Float, _dimensionality);
            OutputType = new VectorType(NumberType.Float, _k);
        }

        /// <summary>
        /// Initialize predictor from a binary file.
        /// </summary>
        /// <param name="ctx">The load context</param>
        /// <param name="env">The host environment</param>
        private KMeansPredictor(IHostEnvironment env, ModelLoadContext ctx)
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

            _centroidL2s = new Float[_k];
            _centroids = new VBuffer<Float>[_k];
            for (int i = 0; i < _k; i++)
            {
                // Prior to allowing sparse vectors, count was not written and was implicitly
                // always equal to dimensionality, and no indices were written either.
                int count = ctx.Header.ModelVerWritten >= 0x00010002 ? ctx.Reader.ReadInt32() : _dimensionality;
                Host.CheckDecode(0 <= count && count <= _dimensionality);
                var indices = count < _dimensionality ? ctx.Reader.ReadIntArray(count) : null;
                var values = ctx.Reader.ReadFloatArray(count);
                Host.CheckDecode(FloatUtils.IsFinite(values, count));
                _centroids[i] = new VBuffer<Float>(_dimensionality, count, values, indices);
            }
            WarnOnOldNormalizer(ctx, GetType(), Host);

            InitPredictor();

            InputType = new VectorType(NumberType.Float, _dimensionality);
            OutputType = new VectorType(NumberType.Float, _k);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<Float>));

            ValueMapper<VBuffer<Float>, VBuffer<Float>> del =
                (ref VBuffer<Float> src, ref VBuffer<Float> dst) =>
                {
                    if (src.Length != _dimensionality)
                        throw Host.Except($"Incorrect number of features: expected {_dimensionality}, got {src.Length}");
                    var values = dst.Values;
                    if (Utils.Size(values) < _k)
                        values = new Float[_k];
                    Map(ref src, values);
                    dst = new VBuffer<Float>(_k, values, dst.Indices);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private void Map(ref VBuffer<Float> src, Float[] distances)
        {
            Host.Assert(Utils.Size(distances) >= _k);

            Float instanceL2 = VectorUtils.NormSquared(src);
            for (int i = 0; i < _k; i++)
            {
                Float distance = Math.Max(0,
                    -2 * VectorUtils.DotProduct(ref _centroids[i], ref src) + _centroidL2s[i] + instanceL2);
                distances[i] = distance;
            }
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("K: {0}", _k);
            writer.WriteLine("Dimensionality: {0}", _dimensionality);
            writer.WriteLine("Centroid coordinates");
            for (int i = 0; i < _k; i++)
            {
                Host.Assert(_dimensionality == _centroids[i].Length);
                if (_centroids[i].IsDense)
                {
                    var values = _centroids[i].Values;
                    for (int j = 0; j < _dimensionality; j++)
                    {
                        if (j != 0)
                            writer.Write('\t');
                        writer.Write(values[j]);
                    }
                }
                else if (_centroids[i].Count > 0)
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
        protected override void SaveCore(ModelSaveContext ctx)
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
                writer.Write(_centroids[i].Count);
                if (!_centroids[i].IsDense)
                    writer.WriteIntsNoCount(_centroids[i].Indices, _centroids[i].Count);
                Contracts.Assert(FloatUtils.IsFinite(_centroids[i].Values, _centroids[i].Count));
                writer.WriteFloatsNoCount(_centroids[i].Values, _centroids[i].Count);
            }
        }

        /// <summary>
        /// This method is called by reflection to instantiate a predictor.
        /// </summary>
        public static KMeansPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new KMeansPredictor(env, ctx);
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
        public void GetClusterCentroids(ref VBuffer<Float>[] centroids, out int k)
        {
            Contracts.Assert(_centroids.Length == _k);
            Utils.EnsureSize(ref centroids, _k, _k);
            for (int i = 0; i < _k; i++)
                _centroids[i].CopyTo(ref centroids[i]);
            k = _k;
        }

        public bool SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
        {
            // Computation graph of distances to all centriods for a batch of exmples. Note that a centriod is just
            // the center of a cluster. We use [] to denote the dimension of a variable; for example, X [3, 2] means
            // that X is a 3-by-2 tensor. In addition, for a matrix X, X^T denotes its transpose.
            //
            // Symbols:
            // l: # of examples
            // n: # of features per input example
            // X: input examples, l-by-n tensor.
            // C: centriods, k-by-n tensor.
            // C^2: 2-norm of all centriod vectors, its shape is [k].
            // Y: 2-norm of difference between examples and centriods, l-by-k tensor. The value at i-th row and k-th
            // column row, Y[i,k], is the distance from example i to centrioid k.
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
            //                                                                   Y [l, k]

            // Allocate C, which is a constant tensor in prediction phase
            var shapeC = new long[] { _centroids.Length, _centroids[0].Length };
            var tensorC = new List<float>();
            foreach (var centriod in _centroids)
                tensorC.AddRange(centriod.DenseValues());
            var nameC = "C"; // ctx.AddInitializer(tensorC, shapeC, "C");

            // Save C^2 as an initializer because it's a constant.
            var shapeC2 = new[] { _centroidL2s.Length };
            var nameC2 = "C2"; // ctx.AddInitializer(_centroidL2s, shapeC2, "C2");

            // Retrieve the name of X
            var nameX = featureColumn;

            // Compute X^2 from X
            var nameX2 = ctx.AddIntermediateVariable(null , "X2", true);
            var reduceNodeX2 = ctx.CreateNode("ReduceSumSquare", nameX, nameX2, ctx.GetNodeName("ReduceSumSquare"));

            // Compute -2XC^T. Note that Gemm always takes three inputs. Since we only have two here,
            // a dummpy one is created.
            var zeroName = "zero"; // ctx.AddInitializer(0f, "zero");
            var nameXC2 = ctx.AddIntermediateVariable(null, "XC2", true);
            var gemmNodeXC2 = ctx.CreateNode("Gemm", new[] { nameX, nameC, zeroName}, new[] { nameXC2 }, ctx.GetNodeName("Gemm"));
            gemmNodeXC2.AddAttribute("alpha", -2f);
            gemmNodeXC2.AddAttribute("transB", 1);

            // Compute Z = X^2 - 2XC^T
            var nameZ = "Z"; // ctx.AddIntermediateVariable(null, "Z", true);
            var addNodeZ = ctx.CreateNode("Add", new[] { nameX2, nameXC2 }, new[] { nameZ }, ctx.GetNodeName("Add"));

            // Compute Y = Z + C^2
            var nameY = outputNames[1];
            var addNodeY = ctx.CreateNode("Add", new[] { nameZ, nameC2 }, new[] { nameY }, ctx.GetNodeName("Add"));

            // Compute the most-matched cluster index
            var nameL = outputNames[0];
            var predictNodeL = ctx.CreateNode("ArgMax", nameY, nameL, ctx.GetNodeName("ArgMax"));
            predictNodeL.AddAttribute("axis", 1);
            predictNodeL.AddAttribute("keepdims", 0);

            return true;
        }
    }
}