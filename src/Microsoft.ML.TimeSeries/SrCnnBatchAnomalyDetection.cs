// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataView;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// The detect modes of SrCnn models.
    /// </summary>
    public enum SrCnnDetectMode
    {
        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag).
        /// </summary>
        AnomalyOnly = 0,

        /// <summary>
        /// In this mode, output (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
        /// </summary>
        AnomalyAndMargin = 1,

        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag, ExpectedValue).
        /// </summary>
        AnomalyAndExpectedValue = 2
    }

    // TODO: SrCnn
    internal sealed class SrCnnBatchAnomalyDetector : BatchDataViewMapperBase<double, SrCnnBatchAnomalyDetector.Batch>
    {
        private const int MinBatchSize = 12;
        private const int AnomalyOnlyOutputLength = 3;
        private const int AnomalyAndExpectedValueOutputLength = 4;
        private const int AnomalyAndMarginOutputLength = 7;

        private readonly int _batchSize;
        private readonly string _inputColumnName;
        private readonly int _outputLength;
        private readonly SrCnnEntireModeler _modler;

        private class Bindings : ColumnBindingsBase
        {
            private readonly VectorDataViewType _outputColumnType;
            private readonly int _inputColumnIndex;

            public Bindings(DataViewSchema input, string inputColumnName, string outputColumnName, VectorDataViewType outputColumnType)
                : base(input, true, outputColumnName)
            {
                _outputColumnType = outputColumnType;
                _inputColumnIndex = Input[inputColumnName].Index;
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Check(iinfo == 0);
                return _outputColumnType;
            }

            // Get a predicate for the input columns.
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.Count];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                        active[_inputColumnIndex] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public SrCnnBatchAnomalyDetector(IHostEnvironment env, IDataView input, string inputColumnName, string outputColumnName, double threshold, int batchSize, double sensitivity, SrCnnDetectMode detectMode)
            : base(env, "SrCnnBatchAnomalyDetector", input)
        {
            Contracts.CheckValue(env, nameof(env));

            Contracts.CheckValue(inputColumnName, nameof(inputColumnName));
            _inputColumnName = inputColumnName;

            env.CheckUserArg(batchSize == -1 || batchSize >= MinBatchSize, nameof(batchSize), "BatchSize must be -1 or no less than 12.");
            _batchSize = batchSize;

            env.CheckUserArg(threshold >= 0 && threshold <= 1, nameof(threshold), "Must be in [0,1].");
            env.CheckUserArg(detectMode == SrCnnDetectMode.AnomalyOnly
                || detectMode == SrCnnDetectMode.AnomalyAndExpectedValue
                || detectMode == SrCnnDetectMode.AnomalyAndMargin, nameof(detectMode), "Invalid detectMode");

            if (detectMode.Equals(SrCnnDetectMode.AnomalyOnly))
            {
                _outputLength = AnomalyOnlyOutputLength;
                _modler = new SrCnnEntireModeler(threshold, sensitivity, detectMode, _outputLength);
            }
            else if (detectMode.Equals(SrCnnDetectMode.AnomalyAndMargin))
            {
                env.CheckUserArg(sensitivity >= 0 && sensitivity <= 100, nameof(sensitivity), "Must be in [0,100].");
                _outputLength = AnomalyAndMarginOutputLength;
                _modler = new SrCnnEntireModeler(threshold, sensitivity, detectMode, _outputLength);
            }
            else if (detectMode.Equals(SrCnnDetectMode.AnomalyAndExpectedValue))
            {
                _outputLength = AnomalyAndExpectedValueOutputLength;
                _modler = new SrCnnEntireModeler(threshold, sensitivity, detectMode, _outputLength);
            }

            SchemaBindings = new Bindings(input.Schema, inputColumnName, outputColumnName, new VectorDataViewType(NumberDataViewType.Double, _outputLength));
        }

        protected override ColumnBindingsBase SchemaBindings { get; }

        protected override Delegate[] CreateGetters(DataViewRowCursor input, Batch currentBatch, bool[] active)
        {
            if (!SchemaBindings.AnyNewColumnsActive(x => active[x]))
                return new Delegate[1];
            return new[] { currentBatch.CreateGetter(input, _inputColumnName) };
        }

        protected override Batch InitializeBatch(DataViewRowCursor input) => new Batch(_batchSize, _outputLength, _modler);

        protected override Func<bool> GetIsNewBatchDelegate(DataViewRowCursor input)
        {
            return () => input.Position % _batchSize == 0;
        }

        protected override Func<bool> GetLastInBatchDelegate(DataViewRowCursor input)
        {
            return () => (input.Position + 1) % _batchSize == 0;
        }

        protected override ValueGetter<double> GetLookAheadGetter(DataViewRowCursor input)
        {
            return input.GetGetter<double>(input.Schema[_inputColumnName]);
        }

        protected override Func<int, bool> GetSchemaBindingDependencies(Func<int, bool> predicate)
        {
            return (SchemaBindings as Bindings).GetDependencies(predicate);
        }

        protected override void ProcessExample(Batch currentBatch, double currentInput)
        {
            currentBatch.AddValue(currentInput);
        }

        protected override void ProcessBatch(Batch currentBatch)
        {
            currentBatch.Process();
            currentBatch.Reset();
        }

        public sealed class Batch
        {
            private List<double> _previousBatch;
            private List<double> _batch;
            private readonly int _batchSize;
            private readonly int _outputLength;
            private SrCnnEntireModeler _modler;
            private double[][] _results;

            public Batch(int batchSize, int outputLength, SrCnnEntireModeler modeler)
            {
                _batchSize = batchSize;
                _outputLength = outputLength;
                _previousBatch = new List<double>(batchSize);
                _batch = new List<double>(batchSize);
                _modler = modeler;
            }

            public void AddValue(double value)
            {
                _batch.Add(value);
            }

            public int Count => _batch.Count;

            public void Process()
            {
                if (_batch.Count < MinBatchSize)
                {
                    if (_previousBatch.Count + _batch.Count < MinBatchSize)
                        return;
                    var bLen = _previousBatch.Count - _batch.Count;
                    _previousBatch = _previousBatch.GetRange(_batch.Count, bLen);
                    _previousBatch.AddRange(_batch);
                    _results = _modler.Train(_previousBatch.ToArray()).Skip(bLen).ToArray();
                }
                else
                {
                    _results = _modler.Train(_batch.ToArray());
                }
            }

            public void Reset()
            {
                var tempBatch = _previousBatch;
                _previousBatch = _batch;
                _batch = tempBatch;
                _batch.Clear();
            }

            public ValueGetter<VBuffer<double>> CreateGetter(DataViewRowCursor input, string inputCol)
            {
                ValueGetter<double> srcGetter = input.GetGetter<double>(input.Schema[inputCol]);
                ValueGetter<VBuffer<double>> getter =
                    (ref VBuffer<double> dst) =>
                    {
                        double src = default;
                        srcGetter(ref src);
                        var result = VBufferEditor.Create(ref dst, _outputLength);
                        for (int i = 0; i < _outputLength; ++i)
                        {
                            result.Values[i] = _results[input.Position % _batchSize][i];
                        }
                        dst = result.Commit();
                    };
                return getter;
            }
        }

        public sealed class SrCnnEntireModeler
        {
            private static readonly int _lookaheadWindowSize = 5;
            private static readonly int _backAddWindowSize = 5;
            private static readonly int _avergingWindowSize = 3;
            private static readonly int _judgementWindowSize = 40;
            private static readonly double _eps = 1e-8;
            private static readonly double _deanomalyThreshold = 0.35;

            // A fixed lookup table which returns factor using sensitivity as index.
            // Since Margin = BoundaryUnit * factor, this factor is calculated to make sure Margin == Boundary when sensitivity is 50,
            // and increases/decreases exponentially as sensitivity increases/decreases.
            // The factor array is generated by formula:
            // f(x)=1, if x=50;
            // f(x)=f(x+1)*(1.25+0.001*x), if 0<=x<50;
            // f(x)=f(x+1)/(1.25+0.001*(x-50)), if 50<x<60;
            // f(x)=f(x+1)/(1.15+0.001*(x-50)),, if 60<=x<=100.
            private static readonly double[] _factors = new double[]{
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

            private readonly double _threshold;
            private readonly double _sensitivity;
            private readonly SrCnnDetectMode _detectMode;
            private readonly int _outputLength;

            public SrCnnEntireModeler(double threshold, double sensitivity, SrCnnDetectMode detectMode, int outputLength)
            {
                _threshold = threshold;
                _sensitivity = sensitivity;
                _detectMode = detectMode;
                _outputLength = outputLength;
            }

            public double[][] Train(double[] values)
            {
                double[][] results = new double[values.Length][];
                for (int i = 0; i < results.Length; ++i)
                {
                    results[i] = new double[_outputLength];
                }
                SpecturalResidual(values, results, _threshold);
                //Optional Steps
                if (_detectMode == SrCnnDetectMode.AnomalyAndMargin)
                {
                    GetMargin(values, results, _sensitivity);
                }
                else if (_detectMode == SrCnnDetectMode.AnomalyAndExpectedValue)
                {
                    GetExpectedValue(values, results);
                }
                return results;
            }

            private static void SpecturalResidual(double[] values, double[][] results, double threshold)
            {
                // Step 1: Get backadd wave
                double[] backAddList = BackAdd(values);

                // Step 2: FFT transformation
                int length = backAddList.Length;
                double[] fftRe = new double[length];
                double[] fftIm = new double[length];
                FftUtils.ComputeForwardFft(backAddList, Enumerable.Repeat((double)0.0f, length).ToArray(), fftRe, fftIm, length);

                // Step 3: Calculate mags of FFT
                double[] magList = new double[length];
                double[] magLogList = new double[length];
                for (int i = 0; i < length; ++i)
                {
                    magList[i] = Math.Sqrt((Math.Pow(fftRe[i], 2) + Math.Pow(fftIm[i], 2)));
                    if (magList[i] > _eps)
                    {
                        magLogList[i] = Math.Log(magList[i]);
                    }
                    else
                    {
                        magLogList[i] = 0;
                    }
                }

                // Step 4: Calculate spectral
                double[] filteredLogList = AverageFilter(magLogList, _avergingWindowSize);
                double[] spectralList = new double[length];
                for (int i = 0; i < length; ++i)
                {
                    spectralList[i] = Math.Exp(magLogList[i] - filteredLogList[i]);
                }

                // Step 5: IFFT transformation
                double[] transRe = new double[length];
                double[] transIm = new double[length];
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

                double[] ifftRe = new double[length];
                double[] ifftIm = new double[length];
                FftUtils.ComputeBackwardFft(transRe, transIm, ifftRe, ifftIm, length);

                // Step 6: Calculate mag and ave_mag of IFFT
                double[] ifftMagList = new double[length];
                for (int i = 0; i < length; ++i)
                {
                    ifftMagList[i] = Math.Sqrt((Math.Pow(ifftRe[i], 2) + Math.Pow(ifftIm[i], 2)));
                }
                double[] filteredIfftMagList = AverageFilter(ifftMagList, Math.Min(ifftMagList.Length, _judgementWindowSize));

                // Step 7: Calculate raw score and set result
                for (int i = 0; i < results.GetLength(0); ++i)
                {
                    var score = CalculateSocre(ifftMagList[i], filteredIfftMagList[i]);
                    score /= 10.0f;
                    score = Math.Min(score, 1);
                    score = Math.Max(score, 0);

                    var detres = score > threshold ? 1 : 0;

                    results[i][0] = detres;
                    results[i][1] = score;
                    results[i][2] = ifftMagList[i];
                }
            }

            private static double[] BackAdd(double[] data)
            {
                double[] predictArray = new double[_lookaheadWindowSize + 1];
                int j = 0;
                for (int i = data.Length - _lookaheadWindowSize - 2; i < data.Length - 1; ++i)
                {
                    predictArray[j] = data[i];
                }
                var predictedValue = PredictNext(predictArray);
                double[] backAddArray = new double[data.Length + _backAddWindowSize];
                for (int i = 0; i < data.Length; ++i)
                {
                    backAddArray[i] = data[i];
                }
                for (int i = 0; i < _backAddWindowSize; ++i)
                {
                    backAddArray[data.Length + i] = predictedValue;
                }
                return backAddArray;
            }

            private static double PredictNext(double[] data)
            {
                var n = data.Length;
                double slopeSum = 0.0f;
                for (int i = 0; i < n - 1; ++i)
                {
                    slopeSum += (data[n - 1] - data[i]) / (n - 1 - i);
                }
                return (data[1] + slopeSum);
            }

            private static double[] AverageFilter(double[] data, int n)
            {
                double cumsum = 0.0f;
                int length = data.Length;
                double[] cumSumList = new double[length];
                double[] cumSumShift = new double[length];

                for (int i = 0; i < length; ++i)
                {
                    cumsum += cumSumList[i];
                    cumSumList[i] = cumsum;
                    cumSumShift[i] = cumsum;
                }
                for (int i = n; i < length; ++i)
                {
                    cumSumList[i] = (cumSumList[i] - cumSumShift[i - n]) / n;
                }
                for (int i = 1; i < n; ++i)
                {
                    cumSumList[i] /= (i + 1);
                }
                return cumSumList;
            }

            private static double CalculateSocre(double mag, double avgMag)
            {
                double safeDivisor = avgMag;
                if (Math.Abs(safeDivisor) < _eps)
                {
                    safeDivisor = _eps;
                }
                return (Math.Abs(mag - avgMag) / safeDivisor);
            }

            private static void GetExpectedValue(double[] values, double[][] results)
            {
                //Step 8: Calculate Expected Value
                var exps = CalculateExpectedValueByFft(GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToArray())));

                for (int i = 0; i < results.Length; ++i)
                {
                    results[i][3] = exps[i];
                }
            }

            private static void GetMargin(double[] values, double[][] results, double sensitivity)
            {
                //Step 8: Calculate Expected Value
                var exps = CalculateExpectedValueByFft(GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToArray())));

                //Step 9: Calculate Boundary Unit
                var units = CalculateBoundaryUnit(values, results.Select(x => x[0] > 0 ? true : false).ToArray());

                //Step 10: Calculate UpperBound and LowerBound
                var margins = units.Select(x => CalculateMargin(x, sensitivity)).ToList();

                for (int i = 0; i < results.Length; ++i)
                {
                    results[i][3] = exps[i];
                    results[i][4] = units[i];
                    results[i][5] = exps[i] + margins[i];
                    results[i][6] = exps[i] - margins[i];
                    //Step 11: Update Anomaly Score
                    results[i][1] = CalculateAnomalyScore(values[i], exps[i], units[i], results[i][0] > 0);
                }
            }

            private static int[] GetAnomalyIndex(double[] scores)
            {
                List<int> anomalyIdxList = new List<int>();
                for (int i = 0; i < scores.Length; ++i)
                    if (scores[i] > _deanomalyThreshold)
                    {
                        anomalyIdxList.Add(i);
                    }

                return anomalyIdxList.ToArray();
            }

            private static double[] GetDeanomalyData(double[] data, int[] anomalyIdxList)
            {
                double[] deAnomalyData = (double[])data.Clone();
                int minPointsToFit = 4;
                foreach (var idx in anomalyIdxList)
                {
                    int step = 1;
                    int start = Math.Max(idx - step, 0);
                    int end = Math.Min(data.Length - 1, idx + step);

                    List<Tuple<int, double>> fitValues = new List<Tuple<int, double>>();
                    for (int i = start; i <= end; ++i)
                    {
                        if (!anomalyIdxList.Contains(i))
                        {
                            fitValues.Add(new Tuple<int, double>(i, data[i]));
                        }
                    }

                    while (fitValues.Count < minPointsToFit && (start > 0 || end < data.Length - 1))
                    {
                        step += 2;
                        start = Math.Max(idx - step, 0);
                        end = Math.Min(data.Length - 1, idx + step);
                        fitValues.Clear();
                        for (int i = start; i <= end; ++i)
                        {
                            if (!anomalyIdxList.Contains(i))
                            {
                                fitValues.Add(new Tuple<int, double>(i, data[i]));
                            }
                        }
                    }

                    if (fitValues.Count > 1)
                    {
                        deAnomalyData[idx] = CalculateInterplate(fitValues, idx);
                    }
                }

                return deAnomalyData;
            }

            private static double CalculateInterplate(List<Tuple<int, double>> values, int idx)
            {
                var n = values.Count;
                double sumX = values.Sum(item => item.Item1);
                double sumY = values.Sum(item => item.Item2);
                double sumXX = values.Sum(item => Math.Pow(item.Item1, 2));
                double sumXY = values.Sum(item => item.Item1 * item.Item2);

                var a = ((double)n * sumXY - sumX * sumY) / ((double)n * sumXX - sumX * sumX);
                var b = (sumXX * sumY - sumX * sumXY) / ((double)n * sumXX - sumX * sumX);

                return a * (double)idx + b;
            }

            private static double[] CalculateExpectedValueByFft(double[] data)
            {
                int length = data.Length;
                double[] fftRe = new double[length];
                double[] fftIm = new double[length];
                FftUtils.ComputeForwardFft(data, Enumerable.Repeat((double)0.0f, length).ToArray(), fftRe, fftIm, length);

                for (int i = 0; i < length; ++i)
                {
                    if (i > (double)length * 3 / 8 && i < (double)length * 5 / 8)
                    {
                        fftRe[i] = 0.0f;
                        fftIm[i] = 0.0f;
                    }
                }

                double[] ifftRe = new double[length];
                double[] ifftIm = new double[length];
                FftUtils.ComputeBackwardFft(fftRe, fftIm, ifftRe, ifftIm, length);

                return ifftRe.Take(length).ToArray();
            }

            private static double[] CalculateBoundaryUnit(double[] data, bool[] isAnomalys)
            {
                int window = Math.Min(data.Length / 3, 512);
                double trendFraction = 0.5;    // mix trend and average of trend
                double trendSum = 0;
                int calculationSize = 0;

                double[] trends = MedianFilter(data, window, true);
                for (int i = 0; i < trends.Length; ++i)
                {
                    if (!isAnomalys[i])
                    {
                        trendSum += Math.Abs(trends[i]);
                        ++calculationSize;
                    }
                }

                double averageTrendPart = 0;
                if (calculationSize > 0)
                {
                    averageTrendPart = trendSum / calculationSize * (1 - trendFraction);
                }
                else
                {
                    trendFraction = 1.0;
                }

                double[] units = new double[trends.Length];
                for (int i = 0; i < units.Length; ++i)
                {
                    units[i] = Math.Max(1, averageTrendPart + Math.Abs(trends[i]) * trendFraction);
                    if (double.IsInfinity(units[i]))
                    {
                        throw new ArithmeticException("Not finite unit value");
                    }
                }

                return units;
            }

            private static double[] MedianFilter(double[] data, int window, bool needTwoEnd = false)
            {
                int wLen = window / 2 * 2 + 1;
                int tLen = data.Length;
                double[] val = (double[]) data.Clone();
                double[] ans = (double[])data.Clone();
                double[] curWindow = new double[wLen];
                if (tLen < wLen)
                {
                    return ans;
                }

                for (int i = 0; i < wLen; i++)
                {
                    int index = i;
                    int addId = BisectRight(curWindow, 0, i, val[i]);
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
                    int deleteId = BisectRight(curWindow, 0, wLen, val[i - window / 2 - 1]) - 1;
                    int index = deleteId;
                    while (index < wLen - 1)
                    {
                        curWindow[index] = curWindow[index + 1];
                        index += 1;
                    }
                    int addId = BisectRight(curWindow, 0, wLen - 1, val[i + window / 2]);
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
                        int deleteId = BisectRight(curWindow, 0, wLen, data[i - window / 2 - 1]) - 1;
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

            private static int BisectRight(double[] arr, int begin, int end, double tar)
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

            private static double SortedMedian(double[] sortedValues, int begin, int end)
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

            private static double CalculateMargin(double unit, double sensitivity)
            {
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

            private static double CalculateAnomalyScore(double value, double exp, double unit, bool isAnomaly)
            {
                double anomalyScore = 0.0f;

                if (isAnomaly.Equals(false))
                {
                    return anomalyScore;
                }

                double distance = Math.Abs(exp - value);
                List<double> margins = new List<double>();
                for (int i = 100; i >= 0; --i)
                {
                    margins.Add(CalculateMargin(unit, i));
                }

                int lb = 0;
                int ub = 100;
                while (lb < ub)
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

                if (Math.Abs(margins[lb] - distance) < _eps || lb == 0)
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
