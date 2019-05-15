// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    public class SrCnnAnomalyDetectionBaseWrapper : IStatefulTransformer, ICanSaveModel
    {
        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)InternalTransform).IsRowToRowMapper;

        /// <summary>
        /// Create a clone of the transformer. Used for taking the snapshot of the state.
        /// </summary>
        /// <returns></returns>
        IStatefulTransformer IStatefulTransformer.Clone() => InternalTransform.Clone();

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => InternalTransform.GetOutputSchema(inputSchema);

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="ITransformer.IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception should be thrown. If the input schema is in any way
        /// unsuitable for constructing the mapper, an exception should likewise be thrown.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
            => ((ITransformer)InternalTransform).GetRowToRowMapper(inputSchema);

        /// <summary>
        /// Same as <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> but also supports mechanism to save the state.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        public IRowToRowMapper GetStatefulRowToRowMapper(DataViewSchema inputSchema)
            => ((IStatefulTransformer)InternalTransform).GetStatefulRowToRowMapper(inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input) => InternalTransform.Transform(input);

        /// <summary>
        /// For saving a model into a repository.
        /// </summary>
        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.SaveThis(ctx);
        }

        /// <summary>
        /// Creates a row mapper from Schema.
        /// </summary>
        internal IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => InternalTransform.MakeRowMapper(schema);

        /// <summary>
        /// Creates an IDataTransform from an IDataView.
        /// </summary>
        internal IDataTransform MakeDataTransform(IDataView input) => InternalTransform.MakeDataTransform(input);

        internal SrCnnAnomalyDetectionBase InternalTransform;

        internal SrCnnAnomalyDetectionBaseWrapper(SrCnnArgumentBase args, string name, IHostEnvironment env)
        {
            InternalTransform = new SrCnnAnomalyDetectionBase(args, name, env, this);
        }

        internal SrCnnAnomalyDetectionBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new SrCnnAnomalyDetectionBase(env, ctx, name, this);
        }

        internal sealed class SrCnnAnomalyDetectionBase : SrCnnTransformBase<Single, SrCnnAnomalyDetectionBase.State>
        {
            internal SrCnnAnomalyDetectionBaseWrapper Parent;

            public SrCnnAnomalyDetectionBase(SrCnnArgumentBase args, string name, IHostEnvironment env, SrCnnAnomalyDetectionBaseWrapper parent)
                : base(args, name, env)
            {
                InitialWindowSize = WindowSize;
                StateRef = new State();
                StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
                Parent = parent;
            }

            public SrCnnAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name, SrCnnAnomalyDetectionBaseWrapper parent)
                : base(env, ctx, name)
            {
                Host.CheckDecode(InitialWindowSize == 0);
                StateRef = new State(ctx.Reader);
                StateRef.InitState(this, Host);
                Parent = parent;
            }

            public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));

                if (!inputSchema.TryGetColumnIndex(InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName);

                var colType = inputSchema[col].Type;
                if (colType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", InputColumnName, NumberDataViewType.Single.ToString(), colType.ToString());

                return Transform(new EmptyDataView(Host, inputSchema)).Schema;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                ((ICanSaveModel)Parent).Save(ctx);
            }

            internal void SaveThis(ModelSaveContext ctx)
            {
                ctx.CheckAtModel();
                Host.Assert(InitialWindowSize == 0);
                base.SaveModel(ctx);

                // *** Binary format ***
                // <base>
                // State: StateRef
                StateRef.Save(ctx.Writer);
            }

            internal sealed class State : SrCnnStateBase
            {
                public State()
                {
                }

                internal State(BinaryReader reader)
                {
                    WindowedBuffer = TimeSeriesUtils.DeserializeFixedSizeQueueSingle(reader, Host);
                    InitialWindowedBuffer = TimeSeriesUtils.DeserializeFixedSizeQueueSingle(reader, Host);
                }

                internal override void Save(BinaryWriter writer)
                {
                    base.Save(writer);
                    TimeSeriesUtils.SerializeFixedSizeQueue(WindowedBuffer, writer);
                    TimeSeriesUtils.SerializeFixedSizeQueue(InitialWindowedBuffer, writer);
                }

                private protected override void CloneCore(State state)
                {
                    base.CloneCore(state);
                    Contracts.Assert(state is State);
                    var stateLocal = state as State;
                    stateLocal.WindowedBuffer = WindowedBuffer.Clone();
                    stateLocal.InitialWindowedBuffer = InitialWindowedBuffer.Clone();
                }

                private protected override void LearnStateFromDataCore(FixedSizeQueue<float> data)
                {
                }

                private protected override sealed void SpectralResidual(Single input, FixedSizeQueue<Single> data, ref VBufferEditor<double> result)
                {
                    // Step 1: Get backadd wave
                    List<Single> backAddList = BackAdd(input, data);

                    // Step 2: FFT transformation
                    int length = backAddList.Count;
                    float[] fftRe = new float[length];
                    float[] fftIm = new float[length];
                    FftUtils.ComputeForwardFft(backAddList.ToArray(), Enumerable.Repeat(0.0f, length).ToArray(), fftRe, fftIm, length);

                    // Step 3: Calculate mags of FFT
                    List<Single> magList = new List<Single>();
                    for (int i = 0; i < length; ++i)
                    {
                        magList.Add(MathUtils.Sqrt((fftRe[i] * fftRe[i] + fftIm[i] * fftIm[i])));
                    }

                    // Step 4: Calculate spectral
                    List<Single> magLogList = magList.Select(x => x != 0 ? MathUtils.Log(x) : 0).ToList();
                    List<Single> filteredLogList = AverageFilter(magLogList, Parent.AvergingWindowSize);
                    List<Single> spectralList = new List<Single>();
                    for (int i = 0; i < magLogList.Count; ++i)
                    {
                        spectralList.Add(magLogList[i] - filteredLogList[i]);
                    }

                    // Step 5: IFFT transformation
                    float[] transRe = new float[length];
                    float[] transIm = new float[length];
                    for (int i = 0; i < length; ++i)
                    {
                        if (magLogList[i] != 0)
                        {
                            transRe[i] = fftRe[i] * spectralList[i] / magList[i];
                            transIm[i] = fftIm[i] * spectralList[i] / magList[i];
                        }
                        else
                        {
                            transRe[i] = 0;
                            transIm[i] = 0;
                        }
                    }

                    float[] ifftRe = new float[length];
                    float[] ifftIm = new float[length];
                    FftUtils.ComputeBackwardFft(transRe, transIm, ifftRe, ifftIm, length);

                    // Step 6: Calculate mag and ave_mag of IFFT
                    List<Single> ifftMagList = new List<Single>();
                    for (int i = 0; i < length; ++i)
                    {
                        ifftMagList.Add(MathUtils.Sqrt((ifftRe[i] * ifftRe[i] + ifftIm[i] * ifftIm[i])));
                    }
                    List<Single> filteredIfftMagList = AverageFilter(ifftMagList, Parent.AvergingWindowSize);

                    // Step 7: Calculate score
                    var score = CalculateSocre(ifftMagList[data.Count-1], filteredIfftMagList[data.Count-1]);
                    var detres = score > Parent.AlertThreshold ? 1 : 0;
                    var mag = ifftMagList[data.Count-1];

                    //Step 8: Set result
                    result.Values[0] = detres;
                    result.Values[1] = score;
                    result.Values[2] = mag;
                }

                private List<Single> BackAdd(Single input, FixedSizeQueue<Single> data)
                {
                    List<Single> predictArray = new List<Single>();
                    for (int i = data.Count-Parent.LookaheadWindowSize-2; i < data.Count-1; ++i)
                    {
                        predictArray.Add(data[i]);
                    }
                    var predictedValue = PredictNext(input, predictArray);
                    List<Single> backAddArray = new List<Single>();
                    for (int i = 0; i < data.Count; ++i)
                    {
                        backAddArray.Add(data[i]);
                    }
                    backAddArray.AddRange(Enumerable.Repeat(predictedValue, Parent.BackAddWindowSize));
                    return backAddArray;
                }

                private Single PredictNext(Single input, List<Single> data)
                {
                    var n = data.Count;
                    Single slopeSum = 0.0f;
                    for (int i = 0; i < n-1; ++i)
                    {
                        slopeSum += (input - data[i]) / (n - 1 - i);
                    }
                    return (input + slopeSum);
                }

                private List<Single> AverageFilter(List<Single> data, int n)
                {
                    Single cumsum = 0.0f;
                    List<Single> cumSumList = data.Select(x => cumsum += x).ToList();
                    for (int i = n; i < cumSumList.Count; ++i)
                    {
                        cumSumList[i] -= cumSumList[i - n];
                    }
                    for (int i = 1; i < n; ++i)
                    {
                        cumSumList[i] /= (i + 1);
                    }
                    return cumSumList;
                }

                private Single CalculateSocre(Single mag, Single avgMag)
                {
                    double safeDivisor = avgMag;
                    if (safeDivisor < 1e-8)
                    {
                        safeDivisor = 1e-8;
                    }
                    return (float)(Math.Abs(mag - avgMag) / safeDivisor);
                }
            }
        }
    }
}
