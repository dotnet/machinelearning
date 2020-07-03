// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataView;
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

    /// <summary>
    /// The Deseasonality modes of SrCnn models. The de-seasonality mode is envoked when the period of the series is greater than 0.
    /// </summary>
    public enum SrCnnDeseasonalityMode
    {
        /// <summary>
        /// In this mode, the stl decompose algorithm is used to de-seasonality.
        /// </summary>
        Stl = 0,

        /// <summary>
        /// In this mode, the mean value of points in the same position in a period is substracted to de-seasonality.
        /// </summary>
        Mean = 1,

        /// <summary>
        /// In this mode, the median value of points in the same position in a period is substracted to de-seasonality.
        /// </summary>
        Median = 2
    }
    public sealed class SrCnnEntireAnomalyDetectorOptions
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
            SortOrder = 3, ShortName = "thr")]
        public double Threshold = Defaults.Threshold;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of data points to be detected in each batch. It should be at least 12. Set this parameter to -1 to detect anomaly on the entire series.",
            SortOrder = 4, ShortName = "bsz")]
        public int BatchSize = Defaults.BatchSize;

        [Argument(ArgumentType.AtMostOnce, HelpText = "This parameter is used in AnomalyAndMargin mode the determine the range of the boundaries.",
            SortOrder = 4, ShortName = "sen")]
        public double Sensitivity = Defaults.Sensitivity;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Specify the detect mode as one of AnomalyOnly, AnomalyAndExpectedValue and AnomalyAndMargin.",
            SortOrder = 5, ShortName = "dtmd")]
        public SrCnnDetectMode DetectMode = Defaults.DetectMode;

        [Argument(ArgumentType.AtMostOnce, HelpText = "If there is circular pattern in the series, set this value to the number of points in one cycle.",
            SortOrder = 5, ShortName = "prd")]
        public int Period = Defaults.Period;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Specify the deseasonality mode as one of stl, mean and median.",
            SortOrder = 6, ShortName = "dsmd")]
        public SrCnnDeseasonalityMode DeseasonalityMode = Defaults.DeseasonalityMode;

        internal static class Defaults
        {
            public const double Threshold = 0.3;
            public const int BatchSize = 2000;
            public const double Sensitivity = 55;
            public const SrCnnDetectMode DetectMode = SrCnnDetectMode.AnomalyOnly;
            public const int Period = 0;
            public const SrCnnDeseasonalityMode DeseasonalityMode = SrCnnDeseasonalityMode.Stl;
        }
    }

    /// <summary>
    /// Detect timeseries anomalies for entire input using Spectral Residual(SR) algorithm.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this detector, use
    /// [DetectEntireAnomalyBySrCnn](xref:Microsoft.ML.TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(Microsoft.ML.AnomalyDetectionCatalog,Microsoft.ML.IDataView,System.String,System.String,System.Double,System.Int32,System.Double,SrCnnDetectMode))
    ///
    /// ### Background
    /// At Microsoft, we developed a time-series anomaly detection service which helps customers to monitor the time-series continuously
    /// and alert for potential incidents on time. To tackle the problem of time-series anomaly detection,
    /// we proposed a novel algorithm based on Spectral Residual (SR) and Convolutional Neural Network
    /// (CNN). The SR model is borrowed from visual saliency detection domain to time-series anomaly detection.
    /// And here we onboarded this SR algorithm firstly.
    ///
    /// The Spectral Residual (SR) algorithm is unsupervised, which means a training step is not needed when using SR. It consists of three major steps:
    /// (1) Fourier Transform to get the log amplitude spectrum;
    /// (2) calculation of spectral residual;
    /// (3) Inverse Fourier Transform that transforms the sequence back to spatial domain.
    /// Mathematically, given a sequence $\mathbf{x}$, we have
    /// $$A(f) = Amplitude(\mathfrak{F}(\mathbf{x}))\\P(f) = Phrase(\mathfrak{F}(\mathbf{x}))\\L(f) = log(A(f))\\AL(f) = h_n(f) \cdot L(f)\\R(f) = L(f) - AL(f)\\S(\mathbf{x}) = \mathfrak{F}^{-1}(exp(R(f) + P(f))^{2})$$
    /// where $\mathfrak{F}$ and $\mathfrak{F}^{-1}$ denote Fourier Transform and Inverse Fourier Transform respectively.
    /// $\mathbf{x}$ is the input sequence with shape $n × 1$; $A(f)$ is the amplitude spectrum of sequence $\mathbf{x}$;
    /// $P(f)$ is the corresponding phase spectrum of sequence $\mathbf{x}$; $L(f)$ is the log representation of $A(f)$;
    /// and $AL(f)$ is the average spectrum of $L(f)$ which can be approximated by convoluting the input sequence by $h_n(f)$,
    /// where $h_n(f)$ is an $n × n$ matrix defined as:
    /// $$n_f(f) = \begin{bmatrix}1&1&1&\cdots&1\\1&1&1&\cdots&1\\\vdots&\vdots&\vdots&\ddots&\vdots\\1&1&1&\cdots&1\end{bmatrix}$$
    /// $R(f)$ is the spectral residual, i.e., the log spectrum $L(f)$ subtracting the averaged log spectrum $AL(f)$.
    /// The spectral residual serves as a compressed representation of the sequence while the innovation part of the original sequence becomes more significant.
    /// At last, we transfer the sequence back to spatial domain via Inverse Fourier Transform. The result sequence $S(\mathbf{x})$ is called the saliency map.
    /// Given the saliency map $S(\mathbf{x})$, the output sequence $O(\mathbf{x})$ is computed by:
    /// $$O(x_i) = \begin{cases}1, if \frac{S(x_i)-\overline{S(x_i)}}{S(x_i)} > \tau\\0,otherwise,\end{cases}$$
    /// where $x_i$ represents an arbitrary point in sequence $\mathbf{x}$; $S(x_i)$is the corresponding point in the saliency map;
    /// and $\overline{S(x_i)}$ is the local average of the preceding points of $S(x_i)$.
    ///
    /// * [Link to the KDD 2019 paper](https://dl.acm.org/doi/10.1145/3292500.3330680)
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(AnomalyDetectionCatalog, IDataView, string, string, double, int, double, SrCnnDetectMode)"/>
    /// <seealso cref="TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(AnomalyDetectionCatalog, IDataView, string, string, SrCnnEntireAnomalyDetectorOptions)"/>
    internal sealed class SrCnnEntireAnomalyDetector : BatchDataViewMapperBase<double, SrCnnEntireAnomalyDetector.Batch>
    {
        private const int MinBatchSize = 12;

        private static readonly int[] _outputLengthArray = {3, 7, 4};
        private readonly SrCnnEntireAnomalyDetectorOptions _options;
        private readonly string _inputColumnName;
        private readonly int _outputLength;
        private readonly Bindings _bindings;

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

        public SrCnnEntireAnomalyDetector(IHostEnvironment env, IDataView input, string outputColumnName, string inputColumnName, SrCnnEntireAnomalyDetectorOptions options)
            : base(env, nameof(SrCnnEntireAnomalyDetector), input)
        {
            Host.CheckValue(outputColumnName, nameof(outputColumnName));

            Host.CheckValue(inputColumnName, nameof(inputColumnName));
            _inputColumnName = inputColumnName;

            Host.CheckValue(options, nameof(options));
            CheckOptionArguments(options);

            _options = options;
            _outputLength = _outputLengthArray[(int)options.DetectMode];

            _bindings = new Bindings(input.Schema, inputColumnName, outputColumnName, new VectorDataViewType(NumberDataViewType.Double, _outputLength));
        }

        private void CheckOptionArguments(SrCnnEntireAnomalyDetectorOptions options)
        {
            Host.CheckUserArg(options.Period >= 0, nameof(options.Period), "Must be an integer equal to or greater than 0.");

            Host.CheckUserArg(options.BatchSize == -1 || options.BatchSize >= MinBatchSize, nameof(options.BatchSize), "Must be -1 or no less than 12.");
            Host.CheckUserArg(options.BatchSize >= 4 * options.Period || options.BatchSize == -1 || options.Period == 0, nameof(options.BatchSize), "Must be at least four times the length of one period.");

            Host.CheckUserArg(options.Threshold >= 0 && options.Threshold <= 1, nameof(options.Threshold), "Must be in [0,1].");
            Host.CheckUserArg(options.DetectMode == SrCnnDetectMode.AnomalyOnly
                || options.DetectMode == SrCnnDetectMode.AnomalyAndExpectedValue
                || options.DetectMode == SrCnnDetectMode.AnomalyAndMargin, nameof(options.DetectMode), "Invalid detectMode");

            Host.CheckUserArg(options.DeseasonalityMode == SrCnnDeseasonalityMode.Stl
                || options.DeseasonalityMode == SrCnnDeseasonalityMode.Mean
                || options.DeseasonalityMode == SrCnnDeseasonalityMode.Median, nameof(options.DeseasonalityMode), "Invalid detectMode");

            Host.CheckUserArg(options.Sensitivity >= 0 && options.Sensitivity <= 100, nameof(options.Sensitivity), "Must be in [0,100].");
        }

        protected override ColumnBindingsBase SchemaBindings => _bindings;

        protected override Delegate[] CreateGetters(DataViewRowCursor input, Batch currentBatch, bool[] active)
        {
            if (!SchemaBindings.AnyNewColumnsActive(x => active[x]))
                return new Delegate[1];
            return new[] { currentBatch.CreateGetter(input, _inputColumnName) };
        }

        protected override Batch CreateBatch(DataViewRowCursor input)
            => new Batch(_options.BatchSize, _outputLength, _options.Threshold, _options.Sensitivity, _options.DetectMode, _options.Period, _options.DeseasonalityMode);

        protected override Func<bool> GetIsNewBatchDelegate(DataViewRowCursor input)
        {
            return () => _options.BatchSize == -1 ? input.Position == 0 : input.Position % _options.BatchSize == 0;
        }

        protected override Func<bool> GetLastInBatchDelegate(DataViewRowCursor input)
        {
            return () => _options.BatchSize == -1 ? input.Position == -1 : (input.Position + 1) % _options.BatchSize == 0;
        }

        protected override ValueGetter<double> GetLookAheadGetter(DataViewRowCursor input)
        {
            return input.GetGetter<double>(input.Schema[_inputColumnName]);
        }

        protected override Func<int, bool> GetSchemaBindingDependencies(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
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

        internal sealed class Batch
        {
            private List<double> _previousBatch;
            private List<double> _batch;
            private readonly int _outputLength;
            private SrCnnEntireModeler _modeler;
            private int _batchSize;
            private double[][] _results;
            private int _bLen;

            public Batch(int batchSize, int outputLength, double threshold, double sensitivity, SrCnnDetectMode detectMode, int period, SrCnnDeseasonalityMode deseasonalityMode)
            {
                _batchSize = batchSize;
                _outputLength = outputLength;
                if (batchSize == -1)
                {
                    _previousBatch = new List<double>();
                    _batch = new List<double>();
                }
                else
                {
                    _previousBatch = new List<double>(batchSize);
                    _batch = new List<double>(batchSize);
                }
                _modeler = new SrCnnEntireModeler(threshold, sensitivity, detectMode, period, deseasonalityMode);
            }

            public void AddValue(double value)
            {
                _batch.Add(value);
            }

            public int Count => _batch.Count;

            public void Process()
            {
                _batchSize = _batch.Count;
                if (_batch.Count < MinBatchSize)
                {
                    if (_previousBatch.Count == 0)
                    {
                        throw Contracts.Except("The input must contain no less than 12 points.");
                    }
                    _bLen = _previousBatch.Count - _batch.Count;
                    _previousBatch = _previousBatch.GetRange(_batch.Count, _bLen);
                    _previousBatch.AddRange(_batch);
                    _modeler.Train(_previousBatch.ToArray(), ref _results);
                }
                else
                {
                    _modeler.Train(_batch.ToArray(), ref _results);
                }
            }

            public void Reset()
            {
                var tempBatch = _previousBatch;
                _previousBatch = _batch;
                _batch = tempBatch;
                _batch.Clear();
                _bLen = 0;
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
                        _results[input.Position % _batchSize + _bLen].CopyTo(result.Values);
                        dst = result.Commit();
                    };
                return getter;
            }
        }

        internal sealed class SrCnnEntireModeler
        {
            private static readonly int _lookaheadWindowSize = 5;
            private static readonly int _backAddWindowSize = 5;
            private static readonly int _averagingWindowSize = 3;
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
            private readonly int _period;
            private readonly IDeseasonality _deseasonalityFunction;

            //used in all modes
            private readonly double[] _predictArray;
            private double[] _backAddArray;
            private double[] _fftRe;
            private double[] _fftIm;
            private double[] _magList;
            private double[] _magLogList;
            private double[] _spectralList;
            private double[] _transRe;
            private double[] _transIm;
            private double[] _ifftRe;
            private double[] _ifftIm;
            private double[] _ifftMagList;
            private double[] _cumSumList;
            private double[] _cumSumShift;
            private double[] _zeroArray;
            private double[] _seriesToDetect;
            //used in AnomalyAndExpectedValue and AnomalyAndMargin
            private double[] _deAnomalyData;
            //used in AnomalyAndMargin mode
            private double[] _units;
            private double[] _val;
            private double[] _trends;
            private double[] _curWindow;

            public SrCnnEntireModeler(double threshold, double sensitivity, SrCnnDetectMode detectMode, int period, SrCnnDeseasonalityMode deseasonalityMode)
            {
                _threshold = threshold;
                _sensitivity = sensitivity;
                _detectMode = detectMode;
                _period = period;
                _predictArray = new double[_lookaheadWindowSize + 1];

                switch (deseasonalityMode)
                {
                    case SrCnnDeseasonalityMode.Stl:
                        _deseasonalityFunction = new StlDeseasonality();
                        break;
                    case SrCnnDeseasonalityMode.Mean:
                        _deseasonalityFunction = new MeanDeseasonality();
                        break;
                    default:
                        Contracts.Assert(deseasonalityMode == SrCnnDeseasonalityMode.Median);
                        _deseasonalityFunction = new MedianDeseasonality();
                        break;
                }
            }

            public void Train(double[] values, ref double[][] results)
            {
                if (results == null)
                {
                    results = new double[values.Length][];
                    for (int i = 0; i < results.Length; ++i)
                    {
                        results[i] = new double[_outputLengthArray[(int)_detectMode]];
                    }
                }
                else if (results.Length > values.Length)
                {
                    Array.Resize<double[]>(ref results, values.Length);
                }

                Array.Resize(ref _seriesToDetect, values.Length);
                for (int i = 0; i < values.Length; ++i)
                {
                    _seriesToDetect[i] = values[i];
                }

                if (_period > 0)
                {
                    _deseasonalityFunction.Deseasonality(ref values, _period, ref _seriesToDetect);
                }

                SpectralResidual(_seriesToDetect, results, _threshold);

                //Optional Steps
                if (_detectMode == SrCnnDetectMode.AnomalyAndMargin)
                {
                    if (_period > 0)
                    {
                        GetMarginPeriod(values, results, _seriesToDetect, _sensitivity);
                    }
                    else
                    {
                        GetMargin(values, results, _sensitivity);
                    }
                }
                else if (_detectMode == SrCnnDetectMode.AnomalyAndExpectedValue)
                {
                    if (_period > 0)
                    {
                        GetExpectedValuePeriod(values, results, _seriesToDetect);
                    }
                    else
                    {
                        GetExpectedValue(values, results);
                    }
                }
            }

            private void SpectralResidual(double[] values, double[][] results, double threshold)
            {
                // Step 1: Get backadd wave
                BackAdd(values);

                // Step 2: FFT transformation
                int length = _backAddArray.Length;
                Array.Resize(ref _fftRe, length);
                Array.Resize(ref _fftIm, length);

                Array.Resize(ref _zeroArray, length);
                FftUtils.ComputeForwardFft(_backAddArray, _zeroArray, _fftRe, _fftIm, length);

                // Step 3: Calculate mags of FFT
                Array.Resize(ref _magList, length);
                Array.Resize(ref _magLogList, length);
                for (int i = 0; i < length; ++i)
                {
                    _magList[i] = Math.Sqrt(_fftRe[i] * _fftRe[i] + _fftIm[i] * _fftIm[i]);
                    if (_magList[i] > _eps)
                    {
                        _magLogList[i] = Math.Log(_magList[i]);
                    }
                    else
                    {
                        _magLogList[i] = 0;
                    }
                }

                // Step 4: Calculate spectral
                AverageFilter(_magLogList, _averagingWindowSize);
                Array.Resize(ref _spectralList, length);
                for (int i = 0; i < length; ++i)
                {
                    _spectralList[i] = Math.Exp(_magLogList[i] - _cumSumList[i]);
                }

                // Step 5: IFFT transformation
                Array.Resize(ref _transRe, length);
                Array.Resize(ref _transIm, length);
                for (int i = 0; i < length; ++i)
                {
                    if (_magLogList[i] != 0)
                    {
                        _transRe[i] = _fftRe[i] * _spectralList[i] / _magList[i];
                        _transIm[i] = _fftIm[i] * _spectralList[i] / _magList[i];
                    }
                    else
                    {
                        _transRe[i] = 0;
                        _transIm[i] = 0;
                    }
                }

                Array.Resize(ref _ifftRe, length);
                Array.Resize(ref _ifftIm, length);
                FftUtils.ComputeBackwardFft(_transRe, _transIm, _ifftRe, _ifftIm, length);

                // Step 6: Calculate mag and ave_mag of IFFT
                Array.Resize(ref _ifftMagList, length);
                for (int i = 0; i < length; ++i)
                {
                    _ifftMagList[i] = Math.Sqrt(_ifftRe[i] * _ifftRe[i] + _ifftIm[i] * _ifftIm[i]);
                }
                AverageFilter(_ifftMagList, Math.Min(_ifftMagList.Length, _judgementWindowSize));

                // Step 7: Calculate raw score and set result
                for (int i = 0; i < results.GetLength(0); ++i)
                {
                    var score = CalculateScore(_ifftMagList[i], _cumSumList[i]);
                    score /= 10.0f;
                    score = Math.Min(score, 1);
                    score = Math.Max(score, 0);

                    var detres = score > threshold ? 1 : 0;

                    results[i][0] = detres;
                    results[i][1] = score;
                    results[i][2] = _ifftMagList[i];
                }
            }

            private void BackAdd(double[] data)
            {
                int j = 0;
                for (int i = data.Length - _lookaheadWindowSize - 2; i < data.Length - 1; ++i)
                {
                    _predictArray[j++] = data[i];
                }
                var predictedValue = PredictNext(_predictArray);
                Array.Resize(ref _backAddArray, data.Length + _backAddWindowSize);
                for (int i = 0; i < data.Length; ++i)
                {
                    _backAddArray[i] = data[i];
                }
                for (int i = 0; i < _backAddWindowSize; ++i)
                {
                    _backAddArray[data.Length + i] = predictedValue;
                }
            }

            private double PredictNext(double[] data)
            {
                var n = data.Length;
                double slopeSum = 0.0f;
                for (int i = 0; i < n - 1; ++i)
                {
                    slopeSum += (data[n - 1] - data[i]) / (n - 1 - i);
                }
                return (data[1] + slopeSum);
            }

            private void AverageFilter(double[] data, int n)
            {
                double cumsum = 0.0f;
                int length = data.Length;

                Array.Resize(ref _cumSumList, length);
                Array.Resize(ref _cumSumShift, length);

                for (int i = 0; i < length; ++i)
                {
                    cumsum += data[i];
                    _cumSumList[i] = cumsum;
                    _cumSumShift[i] = cumsum;
                }
                for (int i = n; i < length; ++i)
                {
                    _cumSumList[i] = (_cumSumList[i] - _cumSumShift[i - n]) / n;
                }
                for (int i = 1; i < n; ++i)
                {
                    _cumSumList[i] /= (i + 1);
                }
            }

            private double CalculateScore(double mag, double avgMag)
            {
                double safeDivisor = avgMag;
                if (Math.Abs(safeDivisor) < _eps)
                {
                    safeDivisor = _eps;
                }
                return (Math.Abs(mag - avgMag) / safeDivisor);
            }

            private void GetExpectedValue(double[] values, double[][] results)
            {
                //Step 8: Calculate Expected Value
                GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToArray()));
                CalculateExpectedValueByFft(_deAnomalyData);

                for (int i = 0; i < results.Length; ++i)
                {
                    results[i][3] = _ifftRe[i];
                }
            }

            private void GetExpectedValuePeriod(double[] values, double[][] results, IReadOnlyList<double> residual)
            {
                //Step 8: Calculate Expected Value
                for (int i = 0; i < values.Length; ++i)
                {
                    results[i][3] = values[i] - residual[i];
                }
            }

            private void GetMarginPeriod(double[] values, double[][] results, IReadOnlyList<double> residual, double sensitivity)
            {
                //Step 8: Calculated Expected Value
                GetExpectedValuePeriod(values, results, residual);

                //Step 9: Calculate Boundary Unit
                CalculateBoundaryUnit(values, results.Select(x => x[0] > 0).ToArray());

                for (int i = 0; i < results.Length; ++i)
                {
                    //Step 10: Calculate UpperBound and LowerBound
                    var margin = CalculateMargin(_units[i], sensitivity);
                    results[i][4] = _units[i];
                    results[i][5] = results[i][3] + margin;
                    results[i][6] = results[i][3] - margin;

                    // update anomaly result according to the boundary
                    results[i][0] = results[i][0] > 0 && (values[i] < results[i][6] || results[i][5] < values[i]) ? 1 : 0;
                }

                List<Tuple<int, int>> segments = new List<Tuple<int, int>>();
                int start = -1;
                int cursor = -1;
                for(int i = 0; i < values.Length; ++i)
                {
                    // this is a outlier
                    if (results[i][6] > values[i] || values[i] > results[i][5])
                    {
                        if (cursor + 1 == i)
                        {
                            cursor = i;
                        }
                        else
                        {
                            if (start > -1)
                            {
                                segments.Add(new Tuple<int, int>(start, cursor));
                            }
                            start = i;
                            cursor = i;
                        }
                    }
                }

                if (start > -1)
                {
                    segments.Add(new Tuple<int, int>(start, Math.Max(start, cursor)));
                }

                List<int> anomalyIndex = new List<int>();
                for (int i = 0; i < values.Length; ++i)
                {
                    if(results[i][0] > 0)
                    {
                        anomalyIndex.Add(i);
                    }
                }

                // more than one anomaly, update anomaly results
                if (anomalyIndex.Count > 1)
                {
                    cursor = 0;
                    for(int i = 0; i < anomalyIndex.Count - 1; ++i)
                    {
                        while (cursor < segments.Count && anomalyIndex[i] >= segments[cursor].Item2)
                        {
                            ++cursor;
                        }

                        if (cursor < segments.Count && segments[cursor].Item1 <= anomalyIndex[i] && anomalyIndex[i+1] <= segments[cursor].Item2)
                        {
                            for (int j = anomalyIndex[i]; j < anomalyIndex[i+1]; ++j)
                            {
                                results[j][0] = 1;
                            }
                        }
                    }
                }

                //Step 11: Update Anomaly Score
                for (int i = 0; i < results.Length; ++i)
                {
                    results[i][1] = CalculateAnomalyScore(values[i], _ifftRe[i], _units[i], results[i][0] > 0);
                }

            }

            private void GetMargin(double[] values, double[][] results, double sensitivity)
            {
                //Step 8: Calculate Expected Value
                GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToArray()));
                CalculateExpectedValueByFft(_deAnomalyData);

                //Step 9: Calculate Boundary Unit
                CalculateBoundaryUnit(values, results.Select(x => x[0] > 0).ToArray());

                for (int i = 0; i < results.Length; ++i)
                {
                    //Step 10: Calculate UpperBound and LowerBound
                    var margin = CalculateMargin(_units[i], sensitivity);
                    results[i][3] = _ifftRe[i];
                    results[i][4] = _units[i];
                    results[i][5] = _ifftRe[i] + margin;
                    results[i][6] = _ifftRe[i] - margin;

                    //Step 11: Update Anomaly Score
                    results[i][1] = CalculateAnomalyScore(values[i], _ifftRe[i], _units[i], results[i][0] > 0);

                    //Step 12: Update IsAnomaly
                    results[i][0] = results[i][0] > 0 && (values[i] < results[i][6] || values[i] > results[i][5]) ? 1 : 0;
                }
            }

            private int[] GetAnomalyIndex(double[] scores)
            {
                List<int> anomalyIdxList = new List<int>();
                for (int i = 0; i < scores.Length; ++i)
                    if (scores[i] > _deanomalyThreshold)
                    {
                        anomalyIdxList.Add(i);
                    }

                return anomalyIdxList.ToArray();
            }

            private void GetDeanomalyData(double[] data, int[] anomalyIdxList)
            {
                Array.Resize(ref _deAnomalyData, data.Length);
                Array.Copy(data, _deAnomalyData, data.Length);
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
                        _deAnomalyData[idx] = CalculateInterpolate(fitValues, idx);
                    }
                }
            }

            private double CalculateInterpolate(List<Tuple<int, double>> values, int idx)
            {
                var n = values.Count;
                double sumX = values.Sum(item => item.Item1);
                double sumY = values.Sum(item => item.Item2);
                double sumXX = values.Sum(item => item.Item1 * item.Item1);
                double sumXY = values.Sum(item => item.Item1 * item.Item2);

                var a = ((n * sumXY) - (sumX * sumY)) / ((n * sumXX) - (sumX * sumX));
                var b = ((sumXX * sumY) - (sumX * sumXY)) / ((n * sumXX) - (sumX * sumX));

                return a * (double)idx + b;
            }

            private void CalculateExpectedValueByFft(double[] data)
            {
                int length = data.Length;
                Array.Resize(ref _fftRe, length);
                Array.Resize(ref _fftIm, length);
                Array.Resize(ref _zeroArray, length);
                FftUtils.ComputeForwardFft(data, _zeroArray, _fftRe, _fftIm, length);

                for (int i = 0; i < length; ++i)
                {
                    if (i > (double)length * 3 / 8 && i < (double)length * 5 / 8)
                    {
                        _fftRe[i] = 0.0f;
                        _fftIm[i] = 0.0f;
                    }
                }

                Array.Resize(ref _ifftRe, length);
                Array.Resize(ref _ifftIm, length);
                FftUtils.ComputeBackwardFft(_fftRe, _fftIm, _ifftRe, _ifftIm, length);
            }

            private void CalculateBoundaryUnit(double[] data, bool[] isAnomalys)
            {
                int window = Math.Min(data.Length / 3, 512);
                double trendFraction = 0.5;    // mix trend and average of trend
                double trendSum = 0;
                int calculationSize = 0;

                MedianFilter(data, window, true);
                for (int i = 0; i < _trends.Length; ++i)
                {
                    if (!isAnomalys[i])
                    {
                        trendSum += Math.Abs(_trends[i]);
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

                Array.Resize(ref _units, _trends.Length);
                for (int i = 0; i < _units.Length; ++i)
                {
                    _units[i] = Math.Max(1, averageTrendPart + Math.Abs(_trends[i]) * trendFraction);
                    if (double.IsInfinity(_units[i]))
                    {
                        throw new ArithmeticException("Not finite unit value");
                    }
                }
            }

            private void MedianFilter(double[] data, int window, bool needTwoEnd = false)
            {
                int wLen = window / 2 * 2 + 1;
                int tLen = data.Length;
                Array.Resize(ref _val, tLen);
                Array.Copy(data, _val, tLen);
                Array.Resize(ref _trends, tLen);
                Array.Copy(data, _trends, tLen);
                Array.Resize(ref _curWindow, wLen);

                if (tLen < wLen)
                    return;

                for (int i = 0; i < wLen; i++)
                {
                    int index = i;
                    int addId = BisectRight(_curWindow, 0, i, _val[i]);
                    while (index > addId)
                    {
                        _curWindow[index] = _curWindow[index - 1];
                        index -= 1;
                    }
                    _curWindow[addId] = data[i];
                    if (i >= wLen / 2 && needTwoEnd)
                        _trends[i - wLen / 2] = SortedMedian(_curWindow, 0, i + 1);
                }

                _trends[window / 2] = SortedMedian(_curWindow, 0, wLen);

                for (int i = window / 2 + 1; i < tLen - window / 2; i++)
                {
                    int deleteId = BisectRight(_curWindow, 0, wLen, _val[i - window / 2 - 1]) - 1;
                    int index = deleteId;
                    while (index < wLen - 1)
                    {
                        _curWindow[index] = _curWindow[index + 1];
                        index += 1;
                    }
                    int addId = BisectRight(_curWindow, 0, wLen - 1, _val[i + window / 2]);
                    index = wLen - 1;
                    while (index > addId)
                    {
                        _curWindow[index] = _curWindow[index - 1];
                        index -= 1;
                    }
                    _curWindow[addId] = data[i + window / 2];
                    _trends[i] = SortedMedian(_curWindow, 0, wLen);
                }

                if (needTwoEnd)
                {
                    for (int i = tLen - window / 2; i < tLen; i++)
                    {
                        int deleteId = BisectRight(_curWindow, 0, wLen, data[i - window / 2 - 1]) - 1;
                        int index = deleteId;
                        while (index < wLen - 1)
                        {
                            _curWindow[index] = _curWindow[index + 1];
                            index += 1;
                        }
                        wLen -= 1;
                        _trends[i] = SortedMedian(_curWindow, 0, wLen);
                    }
                }
            }

            private int BisectRight(double[] arr, int begin, int end, double tar)
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

            private double SortedMedian(double[] sortedValues, int begin, int end)
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
