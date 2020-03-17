// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    public class SrCnnAnomalyDetectionBase : IStatefulTransformer, ICanSaveModel
    {
        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)InternalTransform).IsRowToRowMapper;

        /// <summary>
        /// Create a clone of the transformer. Used for taking the snapshot of the state.
        /// </summary>
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
        /// Initialize a transformer which will do lambda transfrom on input data in prediction engine. No actual transformations happen here, just schema validation.
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

        internal SrCnnAnomalyDetectionBaseCore InternalTransform { get; }

        internal SrCnnAnomalyDetectionBase(SrCnnArgumentBase args, string name, IHostEnvironment env)
        {
            InternalTransform = new SrCnnAnomalyDetectionBaseCore(args, name, env, this);
        }

        internal SrCnnAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new SrCnnAnomalyDetectionBaseCore(env, ctx, name, this);
        }

        internal sealed class SrCnnAnomalyDetectionBaseCore : SrCnnTransformBase<Single, SrCnnAnomalyDetectionBaseCore.State>
        {
            internal SrCnnAnomalyDetectionBase Parent;

            public SrCnnAnomalyDetectionBaseCore(SrCnnArgumentBase args, string name, IHostEnvironment env, SrCnnAnomalyDetectionBase parent)
                : base(args, name, env)
            {
                InitialWindowSize = WindowSize;
                StateRef = new State();
                StateRef.InitState(WindowSize, InitialWindowSize, this, Host);
                Parent = parent;
            }

            public SrCnnAnomalyDetectionBaseCore(IHostEnvironment env, ModelLoadContext ctx, string name, SrCnnAnomalyDetectionBase parent)
                : base(env, ctx, name)
            {
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
                base.SaveModel(ctx);

                // *** Binary format ***
                // <base>
                // State: StateRef
                StateRef.Save(ctx.Writer);
            }

            internal sealed class State : SrCnnStateBase
            {
                internal class ForecastResult
                {
                    public float[] Forecast { get; set; }
                }

                internal class TimeSeriesData
                {
                    public float Value;

                    public TimeSeriesData(float value)
                    {
                        Value = value;
                    }
                }

                private static readonly Double[] _factors = new Double[]{
                    184331.62871148242, 141902.71648305038, 109324.12672037778, 84289.9974713784, 65038.57829581667, 50222.84038287002,
                    38812.08684920403, 30017.081863266845, 23233.035497884553, 17996.15452973242, 13950.50738738947, 10822.736530170265,
                    8402.745753237783, 6528.939979205737, 5076.93622022219, 3950.92312857758, 3077.042935029268, 2398.318733460069,
                    1870.7634426365591, 1460.393007522685, 1140.9320371270976, 892.0500681212648, 698.0047481387048, 546.5972968979678,
                    428.36778753759233, 335.97473532360186, 263.71643275007995, 207.16137686573444, 162.8627176617409, 128.13746472206208,
                    100.8956415134347, 79.50799173635517, 62.70346351447568, 49.48971074544253, 39.09139869308257, 30.90229145698227,
                    24.448015393182175, 19.35709849024717, 15.338429865489042, 12.163703303322, 9.653732780414286, 7.667778221139226,
                    6.095213212352326, 4.8490160798347866, 3.8606815922251485, 3.076240312529999, 2.4531421949999994, 1.9578149999999996,
                    1.5637499999999998, 1.25, 1.0, 0.8695652173913044, 0.7554867223208555, 0.655804446459076, 0.5687809596349316,
                    0.4928777813127657, 0.4267340097946024, 0.36914706729636887, 0.3190553736355825, 0.27552277516026125, 0.23772456873189068,
                    0.20493497304473338, 0.17651591132190647, 0.1519069804835684, 0.13061649224726435, 0.11221348131208278, 0.09632058481723846,
                    0.08260770567516164, 0.0707863801843716, 0.06060477755511267, 0.051843265658779024, 0.0443104834690419, 0.03783986632710667,
                    0.03228657536442549, 0.027524787181948417, 0.02344530424356765, 0.019953450420057577, 0.01696721974494692, 0.014415649740821513,
                    0.012237393667929978, 0.010379468759906684, 0.008796159966022614, 0.0074480609365136455, 0.006301235986898177,
                    0.00532648857725966, 0.004498723460523362, 0.0037963911059268884, 0.0032010043051660104, 0.002696718032995797,
                    0.0022699646742388863, 0.0019091376570554135, 0.0011570531254881296, 0.000697019955113331, 0.00041737721863073713,
                    0.000248438820613534, 0.00014700521929794912, 8.647365841055832e-05, 5.056939088336744e-05, 2.9400808653120604e-05,
                    1.6994687082728674e-05, 9.767061541798089e-06
                };

                public State()
                {
                }

                internal State(BinaryReader reader) : base(reader)
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
                    List<Single> backAddList = BackAdd(data);

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
                        spectralList.Add(MathUtils.ExpSlow(magLogList[i] - filteredLogList[i]));
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
                    List<Single> filteredIfftMagList = AverageFilter(ifftMagList, Parent.JudgementWindowSize);

                    // Step 7: Calculate score and set result
                    var score = CalculateSocre(ifftMagList[data.Count - 1], filteredIfftMagList[data.Count - 1]);
                    score /= 10.0f;
                    result.Values[1] = score;

                    score = Math.Min(score, 1);
                    score = Math.Max(score, 0);
                    var detres = score > Parent.AlertThreshold ? 1 : 0;
                    result.Values[0] = detres;

                    var mag = ifftMagList[data.Count - 1];
                    result.Values[2] = mag;

                    if (result.Values.Length == 3)
                        return;

                    //Optional Steps
                    //Step 8: Calculate Expected Value
                    List<Single> dataList = new List<Single>();
                    for (int i = 0; i < data.Count; ++i)
                    {
                        dataList.Add(data[i]);
                    }
                    //var exp = CalculateExpectedValueByFft(dataList);
                    var exp = CalculateExpectedValueBySsa(dataList);
                    result.Values[3] = exp;

                    //Step 9: Calculate Boundary Unit
                    var unit = CalculateBoundaryUnit(dataList.Select(x => (double)x).ToList());
                    result.Values[4] = unit;

                    //Step 10: Calculate UpperBound and LowerBound
                    var margin = CalculateMargin(unit, Parent.Sensitivity);
                    result.Values[5] = exp + margin;
                    result.Values[6] = exp - margin;

                    //Step 11: Update Anomaly Score
                    var anomalyScore = CalculateAnomalyScore((double)dataList[dataList.Count - 1], exp, unit, detres > 0);
                    result.Values[1] = anomalyScore;
                }

                private List<Single> BackAdd(FixedSizeQueue<Single> data)
                {
                    List<Single> predictArray = new List<Single>();
                    for (int i = data.Count - Parent.LookaheadWindowSize - 2; i < data.Count - 1; ++i)
                    {
                        predictArray.Add(data[i]);
                    }
                    var predictedValue = PredictNext(predictArray);
                    List<Single> backAddArray = new List<Single>();
                    for (int i = 0; i < data.Count; ++i)
                    {
                        backAddArray.Add(data[i]);
                    }
                    backAddArray.AddRange(Enumerable.Repeat(predictedValue, Parent.BackAddWindowSize));
                    return backAddArray;
                }

                private Single PredictNext(List<Single> data)
                {
                    var n = data.Count;
                    Single slopeSum = 0.0f;
                    for (int i = 0; i < n - 1; ++i)
                    {
                        slopeSum += (data[n-1] - data[i]) / (n - 1 - i);
                    }
                    return (data[1] + slopeSum);
                }

                private List<Single> AverageFilter(List<Single> data, int n)
                {
                    Single cumsum = 0.0f;
                    List<Single> cumSumList = data.Select(x => cumsum += x).ToList();
                    List<Single> cumSumShift = new List<Single>(cumSumList);
                    for (int i = n; i < cumSumList.Count; ++i)
                    {
                        cumSumList[i] = (cumSumList[i] - cumSumShift[i - n]) / n;
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

                private Single CalculateExpectedValueByFft(List<Single> data)
                {
                    int length = data.Count;
                    float[] fftRe = new float[length];
                    float[] fftIm = new float[length];
                    FftUtils.ComputeForwardFft(data.ToArray(), Enumerable.Repeat(0.0f, length).ToArray(), fftRe, fftIm, length);

                    for (int i = 0; i < length; ++i)
                    {
                        if (i > length*3/8 && i < length*5/8)
                        {
                            fftRe[i] = 0.0f;
                            fftIm[i] = 0.0f;
                        }
                    }

                    float[] ifftRe = new float[length];
                    float[] ifftIm = new float[length];
                    FftUtils.ComputeBackwardFft(fftRe, fftIm, ifftRe, ifftIm, length);

                    return ifftRe[length-1];
                }

                private Single CalculateExpectedValueBySsa(List<Single> data)
                {
                    var ml = new MLContext();

                    var tsData = data.GetRange(0, data.Count-1).Select(x => new TimeSeriesData(x)).ToList();
                    var dataView = ml.Data.LoadFromEnumerable(tsData);

                    var inputColumnName = nameof(TimeSeriesData.Value);
                    var outputColumnName = nameof(ForecastResult.Forecast);
                    var model = ml.Forecasting.ForecastBySsa(outputColumnName, inputColumnName, tsData.Count / 3, tsData.Count, tsData.Count, 1);
                    var transformer = model.Fit(dataView);
                    var forecastEngine = transformer.CreateTimeSeriesEngine<TimeSeriesData, ForecastResult>(ml);
                    var forecast = forecastEngine.Predict();

                    return forecast.Forecast[0];
                }

                private double CalculateBoundaryUnit(List<Double> data)
                {
                    if (data.Count == 0)
                    {
                        return 0.0f;
                    }

                    double unit = 0.0f;
                    int calculationSize = data.Count - 1;
                    int window = Math.Min(calculationSize / 3, 512);

                    List<Double> trend = MedianFilter(data.GetRange(0, calculationSize), window, true);
                    foreach (var val in trend)
                    {
                        unit += Math.Abs(val);
                    }

                    unit = Math.Max(unit / calculationSize, 1.0);
                    return unit;
                }

                private List<Double> MedianFilter(List<Double> data, int window, bool needTwoEnd)
                {
                    int wLen = window / 2 * 2 + 1;
                    int tLen = data.Count;
                    List<Double> val = new List<Double>(data);
                    List<Double> ans = new List<Double>(data);
                    List<Double> curWindow = new List<Double>(data).GetRange(0, wLen);
                    if (tLen < wLen)
                    {
                        return ans;
                    }

                    for (int i = 0; i < wLen; i++)
                    {
                        int index = i;
                        int addId = UpperBound(curWindow, 0, i, val[i]);
                        while (index > addId)
                        {
                            curWindow[index] = curWindow[index - 1];
                            index -= 1;
                        }
                        curWindow[addId] = data[i];
                        if (i >= wLen / 2 && needTwoEnd)
                            ans[i - wLen / 2] = SortedMedian(curWindow, 0, i + 1);
                    }

                    ans[window / 2] = SortedMedian(curWindow, 0, wLen);

                    for (int i = window / 2 + 1; i < tLen - window / 2; i++)
                    {
                        int deleteId = UpperBound(curWindow, 0, wLen, val[i - window / 2 - 1]) - 1;
                        int index = deleteId;
                        while (index < wLen - 1)
                        {
                            curWindow[index] = curWindow[index + 1];
                            index += 1;
                        }
                        int addId = UpperBound(curWindow, 0, wLen - 1, val[i + window / 2]);
                        index = wLen - 1;
                        while (index > addId)
                        {
                            curWindow[index] = curWindow[index - 1];
                            index -= 1;
                        }
                        curWindow[addId] = data[i + window / 2];
                        ans[i] = SortedMedian(curWindow, 0, wLen);
                    }

                    if (needTwoEnd)
                    {
                        for (int i = tLen - window / 2; i < tLen; i++)
                        {
                            int deleteId = UpperBound(curWindow, 0, wLen, data[i - window / 2 - 1]) - 1;
                            int index = deleteId;
                            while (index < wLen - 1)
                            {
                                curWindow[index] = curWindow[index + 1];
                                index += 1;
                            }
                            wLen -= 1;
                            ans[i] = SortedMedian(curWindow, 0, wLen);
                        }
                    }

                    return ans;
                }

                private int UpperBound(List<Double> arr, int begin, int end, double tar)
                {
                    while (begin < end)
                    {
                        int mid = begin + (end - begin) / 2;
                        if (arr[mid] <= tar)
                            begin = mid + 1;
                        else
                            end = mid;
                    }
                    return begin;
                }

                private double SortedMedian(List<Double> sortedValues, int begin, int end)
                {
                    int n = end - begin;
                    if (n % 2 == 1)
                        return sortedValues[begin + n / 2];
                    else
                    {
                        int mid = begin + n / 2;
                        return (sortedValues[mid - 1] + sortedValues[mid]) / 2;
                    }
                }

                private double CalculateMargin(double unit, double sensitivity)
                {
                    if (unit <= 0)
                    {
                        throw Host.Except("Get negative boundary unit");
                    }
                    if (Math.Floor(sensitivity) == sensitivity)
                    {
                        return unit * _factors[(int)sensitivity];
                    }
                    else
                    {
                        int lb = (int)sensitivity;
                        return (_factors[lb + 1] + (_factors[lb] - _factors[lb + 1]) * (1 - sensitivity + lb)) * unit;
                    }
                }

                private double CalculateAnomalyScore(double value, double exp, double unit, bool isAnomaly)
                {
                    double anomalyScore = 0.0f;

                    if (isAnomaly.Equals(false))
                    {
                        return anomalyScore;
                    }

                    double distance = Math.Abs(exp - value);
                    List<Double> margins = new List<Double>();
                    for (int i = 100; i >= 0; --i)
                    {
                        margins.Add(CalculateMargin(unit, i));
                    }

                    int lb = 0;
                    int ub = 100;
                    while(lb < ub)
                    {
                        int mid = (lb + ub) / 2;
                        if (margins[mid] < distance)
                        {
                            lb = mid + 1;
                        }
                        else
                        {
                            ub = mid;
                        }
                    }

                    if (Math.Abs(margins[lb] - distance) <1e-8 || lb == 0)
                    {
                        anomalyScore = lb;
                    }
                    else
                    {
                        double lowerMargin = margins[lb - 1];
                        double upperMargin = margins[lb];
                        anomalyScore = lb - 1 + (distance - lowerMargin) / (upperMargin - lowerMargin);
                    }

                    return anomalyScore / 100.0f;
                }
            }
        }
    }
}
