using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{

    public sealed class SrCnnEntireTransformer : OneToOneTransformerBase
    {
        internal const string Summary = "This transformer detect anomalies for input timeseries using SRCNN";
        internal const string LoaderSignature = "SrCnnEntireTransformer";
        internal const string UserName = "SrCnn Entire Anomaly Detection";
        internal const string ShortName = "srcnn entire";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.", ShortName = "tgt", SortOrder = 2)]
            public string Target;

            [Argument(ArgumentType.Required, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
                ShortName = "thre", SortOrder = 101)]
            public double Threshold = 0.3;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The batch size of computing.",
                ShortName = "batch", SortOrder = 202)]
            public int BatchSize = 1024;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The detection mode, affects output vector length.",
                ShortName = "mode", SortOrder = 303)]
            public SrCnnDetectMode SrCnnDetectMode = SrCnnDetectMode.AnomalyOnly;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The sensitivity of boundary.",
                ShortName = "sens", SortOrder = 404)]
            public double Sensitivity = 99;
        }

        internal readonly string InputColumnName;

        internal readonly string OutputColumnName;

        internal double Threshold { get; }

        internal int BatchSize { get; }

        internal SrCnnDetectMode SrCnnDetectMode { get; }

        internal double Sensitivity { get; }

        internal int OutputLength { get; }

        internal SrCnnEntireModeler AnomalyModel;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SRENTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SrCnnEntireTransformer).Assembly.FullName);
        }

        internal SrCnnEntireTransformer(IHostEnvironment env, Options options, IDataView input)
            :base(Contracts.CheckRef(env, nameof(env)).Register(LoaderSignature), new[] { (options.Target, options.Source) })
        {
            InputColumnName = options.Source;
            OutputColumnName = options.Target;

            Host.CheckUserArg(options.Threshold >= 0 && options.Threshold <= 1, nameof(Options.Threshold), "Must be in [0,1]");
            Threshold = options.Threshold;

            Host.CheckUserArg(options.BatchSize == -1 || options.BatchSize > 12, nameof(Options.BatchSize), "BatchSize must be -1 or larger than 12");
            BatchSize = options.BatchSize;

            SrCnnDetectMode = options.SrCnnDetectMode;
            if (SrCnnDetectMode.Equals(SrCnnDetectMode.AnomalyOnly))
            {
                Sensitivity = options.Sensitivity;
                OutputLength = 3;
            }
            else if (SrCnnDetectMode.Equals(SrCnnDetectMode.AnomalyAndMargin))
            {
                Host.CheckUserArg(options.Sensitivity >= 0 && options.Sensitivity <= 100, nameof(Options.Sensitivity), "Must be in [0,100]");
                Sensitivity = options.Sensitivity;
                OutputLength = 7;
            }
            else if (SrCnnDetectMode.Equals(SrCnnDetectMode.AnomalyAndExpectedValue))
            {
                Sensitivity = options.Sensitivity;
                OutputLength = 4;
            }

            AnomalyModel = new SrCnnEntireModeler(Host, this, input);
        }

        private SrCnnEntireTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            //TODO
        }

        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SrCnnEntireTransformer(env, options, input).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnEntireTransformer(host, ctx).MakeDataTransform(input);
        }

        internal static SrCnnEntireTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnEntireTransformer(host, ctx);
        }

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            //TODO
            throw new NotImplementedException();
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly SrCnnEntireTransformer _parent;
            private readonly VBuffer<ReadOnlyMemory<Char>> _slotNames;
            private readonly int _inputColumnIndex;

            public Mapper(SrCnnEntireTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                if (!inputSchema.TryGetColumnIndex(parent.InputColumnName, out _inputColumnIndex))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.InputColumnName);

                var colType = inputSchema[_inputColumnIndex].Type;
                if (!(colType is SrCnnTsPointDataViewType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.InputColumnName, "SrCnnTsPoint", colType.ToString());

                _parent = parent;
                if (_parent.OutputLength == 3)
                {
                    _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Alert".AsMemory(), "Raw Score".AsMemory(), "Mag".AsMemory() });
                }
                else if (_parent.OutputLength == 7)
                {
                    _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Is Anomaly".AsMemory(), "Anomaly Score".AsMemory(), "Mag".AsMemory(),
                        "Expected Value".AsMemory(), "Boundary Unit".AsMemory(), "Upper Boundary".AsMemory(), "Lower Boundary".AsMemory() });
                }
                else if (_parent.OutputLength == 4)
                {
                    _slotNames = new VBuffer<ReadOnlyMemory<char>>(_parent.OutputLength, new[] { "Is Anomaly".AsMemory(), "Anomaly Score".AsMemory(), "Mag".AsMemory(),
                        "Expected Value".AsMemory() });
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var meta = new DataViewSchema.Annotations.Builder();
                meta.AddSlotNames(_parent.OutputLength, GetSlotNames);
                var info = new DataViewSchema.DetachedColumn[1];
                info[0] = new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Double, _parent.OutputLength), meta.ToAnnotations());
                return info;
            }

            public void GetSlotNames(ref VBuffer<ReadOnlyMemory<char>> dst) => _slotNames.CopyTo(ref dst, 0, _parent.OutputLength);

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(SrCnnTsPoint);
                var getSrc = input.GetGetter<SrCnnTsPoint>(input.Schema[ColMapNewToOld[iinfo]]);

                disposer = null;

                ValueGetter<VBuffer<double>> del =
                    (ref VBuffer<double> dst) =>
                    {
                        getSrc(ref src);
                        if (src == null)
                            return;
                        var result = VBufferEditor.Create(ref dst, _parent.OutputLength);
                        result.Values.Fill(Double.NaN);
                        Double[] values = new Double[_parent.OutputLength];
                        if (_parent.AnomalyModel.Inference(src, out values))
                        {
                            for (int i = 0; i < result.Values.Length; ++i)
                            {
                                result.Values[i] = values[i];
                            }
                        }
                        dst = result.Commit();
                    };

                return del;
            }
        }

        internal class SrCnnEntireModeler : ICanSaveModel
        {
            private IHost _host;
            private SrCnnEntireTransformer _parent;
            private Dictionary<DateTime, Double[]> _anomalyDict;
            private static readonly int _minTrainingPoint = 12;
            private static readonly int _lookaheadWindowSize = 5;
            private static readonly int _backAddWindowSize = 5;
            private static readonly int _avergingWindowSize = 3;
            private static readonly int _judgementWindowSize = 21;
            private static readonly double _eps = 1e-8;
            private static readonly double _deanomalyThreshold = 0.35;

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

            public SrCnnEntireModeler(IHostEnvironment env, SrCnnEntireTransformer parent, IDataView data)
                : this(env, parent)
            {
                Train(data);
            }

            public SrCnnEntireModeler(IHostEnvironment env, SrCnnEntireTransformer parent)
            {
                Contracts.CheckValue(env, nameof(env));
                Contracts.CheckValue(parent, nameof(parent));
                _host = env.Register(nameof(SrCnnEntireModeler));
                _parent = parent;
                _anomalyDict = new Dictionary<DateTime, Double[]>();
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

            private void SaveModel(ModelSaveContext ctx)
            {
                //TODO
                throw new NotImplementedException();
            }

            public void Train(IDataView data) => Train(new RoleMappedData(data, null, _parent.InputColumnName));

            public void Train(RoleMappedData data)
            {
                _host.CheckValue(data, nameof(data));
                _host.CheckParam(data.Schema.Feature.HasValue, nameof(data), "Must have features column.");
                var featureCol = data.Schema.Feature.Value;
                if (!(featureCol.Type is SrCnnTsPointDataViewType))
                    throw _host.ExceptSchemaMismatch(nameof(data), "feature", featureCol.Name, "SrCnnTsPointDataViewType", featureCol.Type.ToString());

                List<Double> trainingData = new List<Double>();
                List<DateTime> trainingTimestamp = new List<DateTime>();
                DateTime prevTimestamp = DateTime.MinValue;
                using (var cursor = data.Data.GetRowCursor(featureCol))
                {
                    var getVal = cursor.GetGetter<SrCnnTsPoint>(featureCol);
                    SrCnnTsPoint val = null;
                    while (cursor.MoveNext())
                    {
                        getVal(ref val);
                        if (val != null && val.Timestamp > prevTimestamp && !Double.IsNaN(val.Value))
                        {
                            trainingTimestamp.Add(val.Timestamp);
                            trainingData.Add(val.Value);
                            prevTimestamp = val.Timestamp;
                        }
                    }
                }
                if (trainingTimestamp.Count >= _minTrainingPoint)
                {
                    var batchSize = (_parent.BatchSize == -1) ? trainingData.Count : _parent.BatchSize;
                    List<Double[]> batchResult = new List<double[]>();
                    for (int i = 0; i * batchSize < trainingData.Count; ++i)
                    {
                        batchResult.AddRange(TrainCore(trainingData.GetRange(i * batchSize, Math.Min(batchSize, trainingData.Count - i * batchSize))));
                    }

                    _anomalyDict = trainingTimestamp.Zip(batchResult, (k, v) => new { k, v }).ToDictionary(x => x.k, x => x.v);
                }
            }

            public bool Inference(SrCnnTsPoint key, out Double[] result)
            {
                return _anomalyDict.TryGetValue(key.Timestamp, out result);
            }

            private List<Double[]> TrainCore(List<Double> values)
            {
                List<Double[]> results = values.Select(x => new Double[_parent.OutputLength]).ToList();
                SpecturalResidual(values, results);
                //Optional Steps
                if (_parent.SrCnnDetectMode == SrCnnDetectMode.AnomalyAndMargin)
                {
                    GetMargin(values, results);
                }
                else if (_parent.SrCnnDetectMode == SrCnnDetectMode.AnomalyAndExpectedValue)
                {
                    GetExpectedValue(values, results);
                }
                return results;
            }

            private void SpecturalResidual(List<Double> values, List<Double[]> results)
            {
                // Step 1: Get backadd wave
                List<Double> backAddList = BackAdd(values);

                // Step 2: FFT transformation
                int length = backAddList.Count;
                double[] fftRe = new double[length];
                double[] fftIm = new double[length];
                FftUtils.ComputeForwardFft(backAddList.ToArray(), Enumerable.Repeat((double)0.0f, length).ToArray(), fftRe, fftIm, length);

                // Step 3: Calculate mags of FFT
                List<Double> magList = new List<Double>();
                for (int i = 0; i < length; ++i)
                {
                    magList.Add(Math.Sqrt((Math.Pow(fftRe[i], 2) + Math.Pow(fftIm[i], 2))));
                }

                // Step 4: Calculate spectral
                List<Double> magLogList = magList.Select(x => x > _eps ? Math.Log(x) : 0).ToList();
                List<Double> filteredLogList = AverageFilter(magLogList, _avergingWindowSize);
                List<Double> spectralList = new List<Double>();
                for (int i = 0; i < magLogList.Count; ++i)
                {
                    spectralList.Add(Math.Exp(magLogList[i] - filteredLogList[i]));
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
                List<Double> ifftMagList = new List<Double>();
                for (int i = 0; i < length; ++i)
                {
                    ifftMagList.Add(Math.Sqrt((Math.Pow(ifftRe[i], 2) + Math.Pow(ifftIm[i], 2))));
                }
                List<Double> filteredIfftMagList = AverageFilter(ifftMagList, Math.Min(ifftMagList.Count, _judgementWindowSize));

                // Step 7: Calculate score and set result
                for (int i = 0; i < results.Count; ++i)
                {
                    var score = CalculateSocre(ifftMagList[i], filteredIfftMagList[i]);
                    score /= 10.0f;
                    score = Math.Min(score, 1);
                    score = Math.Max(score, 0);

                    var detres = score > _parent.Threshold ? 1 : 0;

                    results[i][0] = detres;
                    results[i][1] = score;
                    results[i][2] = ifftMagList[i];
                }
            }

            private List<Double> BackAdd(List<Double> data)
            {
                List<Double> predictArray = new List<Double>();
                for (int i = data.Count - _lookaheadWindowSize - 2; i < data.Count - 1; ++i)
                {
                    predictArray.Add(data[i]);
                }
                var predictedValue = PredictNext(predictArray);
                List<Double> backAddArray = new List<Double>();
                for (int i = 0; i < data.Count; ++i)
                {
                    backAddArray.Add(data[i]);
                }
                backAddArray.AddRange(Enumerable.Repeat(predictedValue, _backAddWindowSize));
                return backAddArray;
            }

            private Double PredictNext(List<Double> data)
            {
                var n = data.Count;
                Double slopeSum = 0.0f;
                for (int i = 0; i < n - 1; ++i)
                {
                    slopeSum += (data[n - 1] - data[i]) / (n - 1 - i);
                }
                return (data[1] + slopeSum);
            }

            private List<Double> AverageFilter(List<Double> data, int n)
            {
                Double cumsum = 0.0f;
                List<Double> cumSumList = data.Select(x => cumsum += x).ToList();
                List<Double> cumSumShift = new List<Double>(cumSumList);
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

            private Double CalculateSocre(Double mag, Double avgMag)
            {
                double safeDivisor = avgMag;
                if (Math.Abs(safeDivisor) < _eps)
                {
                    safeDivisor = _eps;
                }
                return (Math.Abs(mag - avgMag) / safeDivisor);
            }

            private void GetExpectedValue(List<Double> values, List<Double[]> results)
            {
                //Step 8: Calculate Expected Value
                var exps = CalculateExpectedValueByFft(GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToList())));

                for (int i = 0; i < results.Count; ++i)
                {
                    results[i][3] = exps[i];
                }
            }

            private void GetMargin(List<Double> values, List<Double[]> results)
            {
                //Step 8: Calculate Expected Value
                var exps = CalculateExpectedValueByFft(GetDeanomalyData(values, GetAnomalyIndex(results.Select(x => x[1]).ToList())));

                //Step 9: Calculate Boundary Unit
                var units = CalculateBoundaryUnit(values, results.Select(x => x[0] > 0 ? true : false).ToList());

                //Step 10: Calculate UpperBound and LowerBound
                var margins = units.Select(x => CalculateMargin(x, _parent.Sensitivity)).ToList();

                for (int i = 0; i < results.Count; ++i)
                {
                    results[i][3] = exps[i];
                    results[i][4] = units[i];
                    results[i][5] = exps[i] + margins[i];
                    results[i][6] = exps[i] - margins[i];
                    //Step 11: Update Anomaly Score
                    results[i][1] = CalculateAnomalyScore(values[i], exps[i], units[i], results[i][0] > 0);
                }
            }

            private List<int> GetAnomalyIndex(List<Double> scores)
            {
                List<int> anomalyIdxList = new List<int>();
                for (int i = 0; i < scores.Count; ++i)
                if (scores[i] > _deanomalyThreshold)
                {
                    anomalyIdxList.Add(i);
                }

                return anomalyIdxList;
            }

            private List<Double> GetDeanomalyData(List<Double> data, List<int> anomalyIdxList)
            {
                List<Double> deAnomalyData = new List<Double>(data);
                int minPointsToFit = 4;
                foreach (var idx in anomalyIdxList)
                {
                    int step = 1;
                    int start = Math.Max(idx - step, 0);
                    int end = Math.Min(data.Count - 1, idx + step);

                    List<Tuple<int, double>> fitValues = new List<Tuple<int, double>>();
                    for (int i = start; i <= end; ++i)
                    {
                        if (!anomalyIdxList.Contains(i))
                        {
                            fitValues.Add(new Tuple<int, double>(i, data[i]));
                        }
                    }

                    while (fitValues.Count < minPointsToFit && (start > 0 || end < data.Count - 1))
                    {
                        step += 2;
                        start = Math.Max(idx - step, 0);
                        end = Math.Min(data.Count - 1, idx + step);
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

            private double CalculateInterplate(List<Tuple<int, double>> values, int idx)
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

            private List<Double> CalculateExpectedValueByFft(List<double> data)
            {
                int length = data.Count;
                double[] fftRe = new double[length];
                double[] fftIm = new double[length];
                FftUtils.ComputeForwardFft(data.ToArray(), Enumerable.Repeat((double)0.0f, length).ToArray(), fftRe, fftIm, length);

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

                return ifftRe.ToList().GetRange(0, length);
            }

            private List<Double> CalculateBoundaryUnit(List<Double> data, List<Boolean> isAnomalys)
            {
                if (data.Count == 0)
                {
                    return new List<Double>();
                }

                int window = Math.Min(data.Count / 3, 512);
                double trendFraction = 0.5;    // mix trend and average of trend
                double trendSum = 0;
                int calculationSize = 0;

                List<Double> trends = MedianFilter(data, window, true);
                for (int i = 0; i < trends.Count; ++i)
                {
                    if(!isAnomalys[i])
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

                List<Double> units = new List<Double>();
                foreach (var t in trends)
                {
                    units.Add(Math.Max(1, averageTrendPart + Math.Abs(t) * trendFraction));
                }

                foreach (var v in units)
                {
                    if (Double.IsInfinity(v))
                    {
                        throw new ArithmeticException("Not finite unit value");
                    }
                }

                return units;
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
                    throw _host.Except("Get negative boundary unit");
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
