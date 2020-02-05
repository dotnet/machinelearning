// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(LdSvmModelParameters), null, typeof(SignatureLoadModel), "LDSVM binary predictor", LdSvmModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    public sealed class LdSvmModelParameters : ModelParametersBase<float>,
        IValueMapper,
        ICanSaveModel
    {
        internal const string LoaderSignature = "LDSVMBinaryPredictor";

        /// <summary>
        /// Version information to be saved in binary format
        /// </summary>
        /// <returns></returns>
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LDSVM BC",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LdSvmModelParameters).Assembly.FullName);
        }

        // Classifier Parameters
        private readonly int _numLeaf;
        private readonly float _sigma;
        private readonly VBuffer<float>[] _w;
        private readonly VBuffer<float>[] _thetaPrime;
        private readonly VBuffer<float>[] _theta;
        private readonly float[] _biasW;
        private readonly float[] _biasTheta;
        private readonly float[] _biasThetaPrime;

        /// <summary>
        /// Constructor. w, thetaPrime, theta must be dense <see cref="VBuffer{T}"/>s.
        /// Note that this takes over ownership of all such vectors.
        /// </summary>
        internal LdSvmModelParameters(IHostEnvironment env, VBuffer<float>[] w, VBuffer<float>[] thetaPrime, VBuffer<float>[] theta,
            float sigma, float[] biasW, float[] biasTheta, float[] biasThetaPrime, int treeDepth)
            : base(env, LoaderSignature)
        {
            // _numLeaf is 32-bit signed integer.
            Host.Assert(treeDepth > 0 && treeDepth < 31);
            int numLeaf = 1 << treeDepth;

            Host.Assert(w.Length == numLeaf * 2 - 1);
            Host.Assert(w.All(v => v.IsDense));
            Host.Assert(w.All(v => v.Length == w[0].Length));
            Host.Assert(thetaPrime.Length == numLeaf * 2 - 1);
            Host.Assert(thetaPrime.All(v => v.IsDense));
            Host.Assert(thetaPrime.All(v => v.Length == thetaPrime[0].Length));
            Host.Assert(theta.Length == numLeaf - 1);
            Host.Assert(theta.All(v => v.IsDense));
            Host.Assert(theta.All(v => v.Length == theta[0].Length));
            Host.Assert(biasW.Length == numLeaf * 2 - 1);
            Host.Assert(biasTheta.Length == numLeaf - 1);
            Host.Assert(biasThetaPrime.Length == numLeaf * 2 - 1);
            Host.Assert((w[0].Length > 0) && (w[0].Length == thetaPrime[0].Length) && (w[0].Length == theta[0].Length));

            _numLeaf = numLeaf;
            _sigma = sigma;
            _w = w;
            _thetaPrime = thetaPrime;
            _theta = theta;
            _biasW = biasW;
            _biasTheta = biasTheta;
            _biasThetaPrime = biasThetaPrime;

            InputType = new VectorDataViewType(NumberDataViewType.Single, _w[0].Length);

            AssertValid();
        }

        private LdSvmModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _numLeaf
            // int: numFeatures
            // float: _sigma
            // (_numLeaf * 2 - 1) times: a vector in _w
            //      float[numFeatures]
            // (_numLeaf * 2 - 1) times: a vector in _thetaPrime
            //      float[numFeatures]
            // (_numLeaf - 1) times: a vector in _theta
            //      float[numFeatures]
            // float[_numLeaf * 2 - 1]: _biasW
            // float[_numLeaf - 1]: _biasTheta
            // float[_numLeaf * 2 - 1]: _biasThetaPrime

            _numLeaf = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numLeaf > 1 && (_numLeaf & (_numLeaf - 1)) == 0);
            int numFeatures = ctx.Reader.ReadInt32();
            Host.CheckDecode(numFeatures > 0);

            _sigma = ctx.Reader.ReadFloat();

            _w = LoadVBufferArray(ctx, _numLeaf * 2 - 1, numFeatures);
            _thetaPrime = LoadVBufferArray(ctx, _numLeaf * 2 - 1, numFeatures);
            _theta = LoadVBufferArray(ctx, _numLeaf - 1, numFeatures);
            _biasW = ctx.Reader.ReadFloatArray(_numLeaf * 2 - 1);
            _biasTheta = ctx.Reader.ReadFloatArray(_numLeaf - 1);
            _biasThetaPrime = ctx.Reader.ReadFloatArray(_numLeaf * 2 - 1);
            WarnOnOldNormalizer(ctx, GetType(), Host);

            InputType = new VectorDataViewType(NumberDataViewType.Single, numFeatures);

            AssertValid();
        }

        private void AssertValid()
        {
            Host.Assert(_numLeaf > 1 && (_numLeaf & (_numLeaf - 1)) == 0); // Check if _numLeaf is power of 2
            Host.Assert(_w.Length == _numLeaf * 2 - 1);
            Host.Assert(_w.All(v => v.IsDense));
            Host.Assert(_w.All(v => v.Length == _w[0].Length));
            Host.Assert(_thetaPrime.Length == _numLeaf * 2 - 1);
            Host.Assert(_thetaPrime.All(v => v.IsDense));
            Host.Assert(_thetaPrime.All(v => v.Length == _thetaPrime[0].Length));
            Host.Assert(_theta.Length == _numLeaf - 1);
            Host.Assert(_theta.All(v => v.IsDense));
            Host.Assert(_theta.All(v => v.Length == _theta[0].Length));
            Host.Assert(_biasW.Length == _numLeaf * 2 - 1);
            Host.Assert(_biasTheta.Length == _numLeaf - 1);
            Host.Assert(_biasThetaPrime.Length == _numLeaf * 2 - 1);
            Host.Assert((_w[0].Length > 0) && (_w[0].Length == _thetaPrime[0].Length) && (_w[0].Length == _theta[0].Length)); // numFeatures
            Host.Assert(InputType != null && InputType.GetVectorSize() == _w[0].Length);
        }

        /// <summary>
        /// Create method to instantiate a predictor.
        /// </summary>
        private static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LdSvmModelParameters(env, ctx);
        }

        private protected override PredictionKind PredictionKind { get { return PredictionKind.BinaryClassification; } }

        /// <summary>
        /// Save the predictor in binary format.
        /// </summary>
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _numLeaf
            // int: numFeatures
            // float: _sigma
            // (_numLeaf * 2 - 1) times: a vector in _w
            //      float[numFeatures]
            // (_numLeaf * 2 - 1) times: a vector in _thetaPrime
            //      float[numFeatures]
            // (_numLeaf - 1) times: a vector in _theta
            //      float[numFeatures]
            // float[_numLeaf * 2 - 1]: _biasW
            // float[_numLeaf - 1]: _biasTheta
            // float[_numLeaf * 2 - 1]: _biasThetaPrime

            int numFeatures = _w[0].Length;

            ctx.Writer.Write(_numLeaf);
            ctx.Writer.Write(numFeatures);
            ctx.Writer.Write(_sigma);

            Host.Assert(_w.Length == _numLeaf * 2 - 1);
            SaveVBufferArray(ctx, _w);
            Host.Assert(_thetaPrime.Length == _numLeaf * 2 - 1);
            SaveVBufferArray(ctx, _thetaPrime);
            Host.Assert(_theta.Length == _numLeaf - 1);
            SaveVBufferArray(ctx, _theta);

            Host.Assert(_biasW.Length == _numLeaf * 2 - 1);
            ctx.Writer.WriteSinglesNoCount(_biasW.AsSpan());
            Host.Assert(_biasTheta.Length == _numLeaf - 1);
            ctx.Writer.WriteSinglesNoCount(_biasTheta.AsSpan());
            Host.Assert(_biasThetaPrime.Length == _numLeaf * 2 - 1);
            ctx.Writer.WriteSinglesNoCount(_biasThetaPrime.AsSpan());
        }

        /// <summary>
        /// Save an array of <see cref="VBuffer{T}"/> in binary format. The vectors must be dense.
        /// </summary>
        /// <param name="ctx">The context where we will save the vectors.</param>
        /// <param name="data">An array of vectors.</param>
        private void SaveVBufferArray(ModelSaveContext ctx, VBuffer<float>[] data)
        {
            if (data.Length == 0)
                return;

            int vectorLength = data[0].Length;
            for (int i = 0; i < data.Length; i++)
            {
                var vector = data[i];
                Host.Assert(vector.IsDense);
                Host.Assert(vector.Length == vectorLength);
                ctx.Writer.WriteSinglesNoCount(vector.GetValues());
            }
        }

        /// <summary>
        /// Load an array of <see cref="VBuffer{T}"/> from binary format.
        /// </summary>
        /// <param name="ctx">The context from which to read the vectors.</param>
        /// <param name="length">The length of the array of vectors.</param>
        /// <param name="vectorLength">The length of each vector.</param>
        /// <returns>An array of vectors.</returns>
        private VBuffer<float>[] LoadVBufferArray(ModelLoadContext ctx, int length, int vectorLength)
        {
            Host.Assert(length >= 0);
            Host.Assert(vectorLength >= 0);

            VBuffer<float>[] result = new VBuffer<float>[length];

            for (int i = 0; i < length; i++)
            {
                result[i] = new VBuffer<float>(vectorLength, ctx.Reader.ReadFloatArray(vectorLength));
                Host.Assert(result[i].IsDense);
                Host.Assert(result[i].Length == vectorLength);
            }
            return result;
        }

        /// <summary>
        /// Compute Margin.
        /// </summary>
        private float Margin(in VBuffer<float> src)
        {
            double score = 0;
            double childIndicator;
            int current = 0;
            while (current < _numLeaf - 1)
            {
                score += Math.Tanh(_sigma * (VectorUtils.DotProduct(in _thetaPrime[current], in src) + _biasThetaPrime[current])) *
                    (VectorUtils.DotProduct(in _w[current], in src) + _biasW[current]);
                childIndicator = VectorUtils.DotProduct(in _theta[current], in src) + _biasTheta[current];
                current = (childIndicator > 0) ? 2 * current + 1 : 2 * current + 2;
            }
            score += Math.Tanh(_sigma * (VectorUtils.DotProduct(in _thetaPrime[current], in src) + _biasThetaPrime[current])) *
                    (VectorUtils.DotProduct(in _w[current], in src) + _biasW[current]);
            return (float)score;
        }

        public DataViewType InputType { get; }

        public DataViewType OutputType => NumberDataViewType.Single;

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del =
                (in VBuffer<float> src, ref float dst) =>
                {
                    Host.Check(src.Length == InputType.GetVectorSize());
                    dst = Margin(in src);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}
