// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    // REVIEW: This base class and its children classes generate one output column of type VBuffer<Double> to output 3 different anomaly scores as well as
    // the alert flag. Ideally these 4 output information should be put in four seaparate columns instead of one VBuffer<> column. However, this is not currently
    // possible due to our design restriction. This must be fixed in the next version and will potentially affect the children classes.

    /// <summary>
    /// The base class for sequential anomaly detection transforms that supports the p-value as well as the martingales scores computation from the sequence of
    /// raw anomaly scores whose calculation is specified by the children classes. This class also provides mechanism for the threshold-based alerting on
    /// the raw anomaly score, the p-value score or the martingale score. Currently, this class supports Power and Mixture martingales.
    /// For more details, please refer to http://arxiv.org/pdf/1204.3251.pdf
    /// </summary>
    /// <typeparam name="TInput">The type of the input sequence</typeparam>
    /// <typeparam name="TState">The type of the state object for sequential anomaly detection. Must be a class inherited from AnomalyDetectionStateBase</typeparam>
    public abstract class SequentialAnomalyDetectionTransformBase<TInput, TState> : SequentialTransformBase<TInput, VBuffer<Double>, TState>
        where TState : SequentialAnomalyDetectionTransformBase<TInput, TState>.AnomalyDetectionStateBase, new()
    {
        /// <summary>
        /// The type of the martingale.
        /// </summary>
        public enum MartingaleType : byte
        {
            /// <summary>
            /// (None) No martingale is used.
            /// </summary>
            None,
            /// <summary>
            /// (Power) The Power martingale is used.
            /// </summary>
            Power,
            /// <summary>
            /// (Mixture) The Mixture martingale is used.
            /// </summary>
            Mixture
        }

        /// <summary>
        /// The side of anomaly detection.
        /// </summary>
        public enum AnomalySide : byte
        {
            /// <summary>
            /// (Positive) Only positive anomalies are detected.
            /// </summary>
            Positive,
            /// <summary>
            /// (Negative) Only negative anomalies are detected.
            /// </summary>
            Negative,
            /// <summary>
            /// (TwoSided) Both positive and negative anomalies are detected.
            /// </summary>
            TwoSided
        }

        /// <summary>
        /// The score that should be thresholded to generate alerts.
        /// </summary>
        public enum AlertingScore : byte
        {
            /// <summary>
            /// (RawScore) The raw anomaly score is thresholded.
            /// </summary>
            RawScore,
            /// <summary>
            /// (PValueScore) The p-value score is thresholded.
            /// </summary>
            PValueScore,
            /// <summary>
            /// (MartingaleScore) The martingale score is thresholded.
            /// </summary>
            MartingaleScore
        }

        /// <summary>
        /// The base class that can be inherited by the 'Argument' classes in the derived classes containing the shared input parameters.
        /// </summary>
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The argument that determines whether to detect positive or negative anomalies, or both", ShortName = "side",
                SortOrder = 3)]
            public AnomalySide Side = AnomalySide.TwoSided;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the p-value.", ShortName = "wnd",
                SortOrder = 4)]
            public int WindowSize = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the initial window for computing the p-value as well as training if needed. The default value is set to 0, which means there is no initial window considered.",
                ShortName = "initwnd", SortOrder = 5)]
            public int InitialWindowSize = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The martingale used for scoring",
                ShortName = "martingale", SortOrder = 6)]
            public MartingaleType Martingale = MartingaleType.Power;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The argument that determines whether anomalies should be detected based on the raw anomaly score, the p-value or the martingale score",
                ShortName = "alert", SortOrder = 7)]
            public AlertingScore AlertOn = AlertingScore.MartingaleScore;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The epsilon parameter for the Power martingale",
                ShortName = "eps", SortOrder = 8)]
            public Double PowerMartingaleEpsilon = 0.1;

            [Argument(ArgumentType.Required, HelpText = "The threshold for alerting",
                ShortName = "thr", SortOrder = 9)]
            public Double AlertThreshold;
        }

        // Determines the side of anomaly detection for this transform.
        protected AnomalySide Side;

        // Determines the type of martingale used by this transform.
        protected MartingaleType Martingale;

        // The epsilon parameter used by the Power martingale.
        protected Double PowerMartingaleEpsilon;

        // Determines the score that should be thresholded to generate alerts by this transform.
        protected AlertingScore ThresholdScore;

        // Determines the threshold for generating alerts.
        protected Double AlertThreshold;

        // The size of the VBuffer in the dst column.
        private int _outputLength;

        private readonly SchemaImpl _wrappedSchema;

        public override ISchema Schema => _wrappedSchema;

        private static int GetOutputLength(AlertingScore alertingScore, IHostEnvironment host)
        {
            switch (alertingScore)
            {
                case AlertingScore.RawScore:
                    return 2;
                case AlertingScore.PValueScore:
                    return 3;
                case AlertingScore.MartingaleScore:
                    return 4;
                default:
                    throw host.Except("The alerting score can be only (0) RawScore, (1) PValueScore or (2) MartingaleScore.");
            }
        }

        private static SchemaImpl CreateSchema(ISchema parentSchema, string colName, int length)
        {
            Contracts.AssertValue(parentSchema);
            Contracts.Assert(2 <= length && length <= 4);

            string[] names = { "Alert", "Raw Score", "P-Value Score", "Martingale Score" };
            int col;
            bool result = parentSchema.TryGetColumnIndex(colName, out col);
            Contracts.Assert(result);

            return new SchemaImpl(parentSchema, col, names, length);
        }

        protected SequentialAnomalyDetectionTransformBase(int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName, string name, IHostEnvironment env, IDataView input,
            AnomalySide anomalySide, MartingaleType martingale, AlertingScore alertingScore, Double powerMartingaleEpsilon,
            Double alertThreshold)
            : base(windowSize, initialWindowSize, inputColumnName, outputColumnName, name, env, input, new VectorType(NumberType.R8, GetOutputLength(alertingScore, env)))
        {
            Host.CheckUserArg(Enum.IsDefined(typeof(MartingaleType), martingale), nameof(ArgumentsBase.Martingale), "Value is undefined.");
            Host.CheckUserArg(Enum.IsDefined(typeof(AnomalySide), anomalySide), nameof(ArgumentsBase.Side), "Value is undefined.");
            Host.CheckUserArg(Enum.IsDefined(typeof(AlertingScore), alertingScore), nameof(ArgumentsBase.AlertOn), "Value is undefined.");
            Host.CheckUserArg(martingale != MartingaleType.None || alertingScore != AlertingScore.MartingaleScore, nameof(ArgumentsBase.Martingale), "A martingale type should be specified if alerting is based on the martingale score.");
            Host.CheckUserArg(windowSize > 0 || alertingScore == AlertingScore.RawScore, nameof(ArgumentsBase.AlertOn),
                "When there is no windowed buffering (i.e., " + nameof(ArgumentsBase.WindowSize) + " = 0), the alert can be generated only based on the raw score (i.e., "
                + nameof(ArgumentsBase.AlertOn) + " = " + nameof(AlertingScore.RawScore) + ")");
            Host.CheckUserArg(0 < powerMartingaleEpsilon && powerMartingaleEpsilon < 1, nameof(ArgumentsBase.PowerMartingaleEpsilon), "Should be in (0,1).");
            Host.CheckUserArg(alertThreshold >= 0, nameof(ArgumentsBase.AlertThreshold), "Must be non-negative.");
            Host.CheckUserArg(alertingScore != AlertingScore.PValueScore || (0 <= alertThreshold && alertThreshold <= 1), nameof(ArgumentsBase.AlertThreshold), "Must be in [0,1].");

            ThresholdScore = alertingScore;
            Side = anomalySide;
            Martingale = martingale;
            PowerMartingaleEpsilon = powerMartingaleEpsilon;
            AlertThreshold = alertThreshold;
            _outputLength = GetOutputLength(ThresholdScore, Host);
            _wrappedSchema = CreateSchema(base.Schema, outputColumnName, _outputLength);
        }

        protected SequentialAnomalyDetectionTransformBase(ArgumentsBase args, string name, IHostEnvironment env, IDataView input)
            : this(args.WindowSize, args.InitialWindowSize, args.Source, args.Name, name, env, input, args.Side, args.Martingale,
                args.AlertOn, args.PowerMartingaleEpsilon, args.AlertThreshold)
        {
        }

        protected SequentialAnomalyDetectionTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name, IDataView input)
            : base(env, ctx, name, input)
        {
            // *** Binary format ***
            // <base>
            // byte: _martingale
            // byte: _alertingScore
            // byte: _anomalySide
            // Double: _powerMartingaleEpsilon
            // Double: _alertThreshold

            byte temp;
            temp = ctx.Reader.ReadByte();
            Host.CheckDecode(Enum.IsDefined(typeof(MartingaleType), temp));
            Martingale = (MartingaleType)temp;

            temp = ctx.Reader.ReadByte();
            Host.CheckDecode(Enum.IsDefined(typeof(AlertingScore), temp));
            ThresholdScore = (AlertingScore)temp;

            Host.CheckDecode(Martingale != MartingaleType.None || ThresholdScore != AlertingScore.MartingaleScore);
            Host.CheckDecode(WindowSize > 0 || ThresholdScore == AlertingScore.RawScore);

            temp = ctx.Reader.ReadByte();
            Host.CheckDecode(Enum.IsDefined(typeof(AnomalySide), temp));
            Side = (AnomalySide)temp;

            PowerMartingaleEpsilon = ctx.Reader.ReadDouble();
            Host.CheckDecode(0 < PowerMartingaleEpsilon && PowerMartingaleEpsilon < 1);

            AlertThreshold = ctx.Reader.ReadDouble();
            Host.CheckDecode(AlertThreshold >= 0);
            Host.CheckDecode(ThresholdScore != AlertingScore.PValueScore || (0 <= AlertThreshold && AlertThreshold <= 1));

            _outputLength = GetOutputLength(ThresholdScore, Host);
            _wrappedSchema = CreateSchema(base.Schema, OutputColumnName, _outputLength);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();

            Host.Assert(Enum.IsDefined(typeof(MartingaleType), Martingale));
            Host.Assert(Enum.IsDefined(typeof(AlertingScore), ThresholdScore));
            Host.Assert(Martingale != MartingaleType.None || ThresholdScore != AlertingScore.MartingaleScore);
            Host.Assert(WindowSize > 0 || ThresholdScore == AlertingScore.RawScore);
            Host.Assert(Enum.IsDefined(typeof(AnomalySide), Side));
            Host.Assert(0 < PowerMartingaleEpsilon && PowerMartingaleEpsilon < 1);
            Host.Assert(AlertThreshold >= 0);
            Host.Assert(ThresholdScore != AlertingScore.PValueScore || (0 <= AlertThreshold && AlertThreshold <= 1));

            // *** Binary format ***
            // <base>
            // byte: _martingale
            // byte: _alertingScore
            // byte: _anomalySide
            // Double: _powerMartingaleEpsilon
            // Double: _alertThreshold

            base.Save(ctx);
            ctx.Writer.Write((byte)Martingale);
            ctx.Writer.Write((byte)ThresholdScore);
            ctx.Writer.Write((byte)Side);
            ctx.Writer.Write(PowerMartingaleEpsilon);
            ctx.Writer.Write(AlertThreshold);
        }

        // The minimum value for p-values. The smaller p-values are ceiled to this value.
        private const Double MinPValue = 1e-8;

        // The maximun value for p-values. The larger p-values are floored to this value.
        private const Double MaxPValue = 1 - MinPValue;

        /// <summary>
        /// Calculates the betting function for the Power martingale in the log scale.
        /// For more details, please refer to http://arxiv.org/pdf/1204.3251.pdf.
        /// </summary>
        /// <param name="p">The p-value</param>
        /// <param name="epsilon">The epsilon</param>
        /// <returns>The Power martingale betting function value in the natural logarithmic scale.</returns>
        protected Double LogPowerMartigaleBettingFunc(Double p, Double epsilon)
        {
            Host.Assert(MinPValue > 0);
            Host.Assert(MaxPValue < 1);
            Host.Assert(MinPValue <= p && p <= MaxPValue);
            Host.Assert(0 < epsilon && epsilon < 1);

            return Math.Log(epsilon) + (epsilon - 1) * Math.Log(p);
        }

        /// <summary>
        /// Calculates the betting function for the Mixture martingale in the log scale.
        /// For more details, please refer to http://arxiv.org/pdf/1204.3251.pdf.
        /// </summary>
        /// <param name="p">The p-value</param>
        /// <returns>The Mixure (marginalized over epsilon) martingale betting function value in the natural logarithmic scale.</returns>
        protected Double LogMixtureMartigaleBettingFunc(Double p)
        {
            Host.Assert(MinPValue > 0);
            Host.Assert(MaxPValue < 1);
            Host.Assert(MinPValue <= p && p <= MaxPValue);

            Double logP = Math.Log(p);
            return Math.Log(p * logP + 1 - p) - 2 * Math.Log(-logP) - logP;
        }

        /// <summary>
        /// The base state class for sequential anomaly detection: this class implements the p-values and martinagle calculations for anomaly detection
        /// given that the raw anomaly score calculation is specified by the derived classes.
        /// </summary>
        public abstract class AnomalyDetectionStateBase : StateBase
        {
            // A reference to the parent transform.
            protected SequentialAnomalyDetectionTransformBase<TInput, TState> Parent;

            // A windowed buffer to cache the update values to the martingale score in the log scale.
            private FixedSizeQueue<Double> _logMartingaleUpdateBuffer;

            // A windowed buffer to cache the raw anomaly scores for p-value calculation.
            private FixedSizeQueue<Single> _rawScoreBuffer;

            // The current martingale score in the log scale.
            private Double _logMartingaleValue;

            // Sum of the squared Euclidean distances among the raw socres in the buffer.
            // Used for computing the optimal bandwidth for Kernel Density Estimation of p-values.
            private Double _sumSquaredDist;

            private int _martingaleAlertCounter;

            protected Double LatestMartingaleScore {
                get { return Math.Exp(_logMartingaleValue); }
            }

            private Double ComputeKernelPValue(Double rawScore)
            {
                int i;
                int n = _rawScoreBuffer.Count;

                if (n == 0)
                    return 0.5;

                Double pValue = 0;
                Double bandWidth = Math.Sqrt(2) * ((n == 1) ? 1 : Math.Sqrt(_sumSquaredDist) / n);
                bandWidth = Math.Max(bandWidth, 1e-6);

                Double diff;
                for (i = 0; i < n; ++i)
                {
                    diff = rawScore - _rawScoreBuffer[i];
                    pValue -= ProbabilityFunctions.Erf(diff / bandWidth);
                    _sumSquaredDist += diff * diff;
                }

                pValue = 0.5 + pValue / (2 * n);
                if (_rawScoreBuffer.IsFull)
                {
                    for (i = 1; i < n; ++i)
                    {
                        diff = _rawScoreBuffer[0] - _rawScoreBuffer[i];
                        _sumSquaredDist -= diff * diff;
                    }

                    diff = _rawScoreBuffer[0] - rawScore;
                    _sumSquaredDist -= diff * diff;
                }

                return pValue;
            }

            protected override void SetNaOutput(ref VBuffer<Double> dst)
            {
                var values = dst.Values;
                var outputLength = Parent._outputLength;
                if (Utils.Size(values) < outputLength)
                    values = new Double[outputLength];

                for (int i = 0; i < outputLength; ++i)
                    values[i] = Double.NaN;

                dst = new VBuffer<Double>(Utils.Size(values), values, dst.Indices);
            }

            protected override sealed void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref VBuffer<Double> dst)
            {
                var outputLength = Parent._outputLength;
                Host.Assert(outputLength >= 2);

                var result = dst.Values;
                if (Utils.Size(result) < outputLength)
                    result = new Double[outputLength];

                float rawScore = 0;

                for (int i = 0; i < outputLength; ++i)
                    result[i] = Double.NaN;

                // Step 1: Computing the raw anomaly score
                result[1] = ComputeRawAnomalyScore(ref input, windowedBuffer, iteration);

                if (Double.IsNaN(result[1]))
                    result[0] = 0;
                else
                {
                    if (WindowSize > 0)
                    {
                        // Step 2: Computing the p-value score
                        rawScore = (float)result[1];
                        if (Parent.ThresholdScore == AlertingScore.RawScore)
                        {
                            switch (Parent.Side)
                            {
                                case AnomalySide.Negative:
                                    rawScore = (float)(-result[1]);
                                    break;

                                case AnomalySide.Positive:
                                    break;

                                default:
                                    rawScore = (float)Math.Abs(result[1]);
                                    break;
                            }
                        }
                        else
                        {
                            result[2] = ComputeKernelPValue(rawScore);

                            switch (Parent.Side)
                            {
                                case AnomalySide.Negative:
                                    result[2] = 1 - result[2];
                                    break;

                                case AnomalySide.Positive:
                                    break;

                                default:
                                    result[2] = Math.Min(result[2], 1 - result[2]);
                                    break;
                            }

                            // Keeping the p-value in the safe range
                            if (result[2] < MinPValue)
                                result[2] = MinPValue;
                            else if (result[2] > MaxPValue)
                                result[2] = MaxPValue;

                            _rawScoreBuffer.AddLast(rawScore);

                            // Step 3: Computing the martingale value
                            if (Parent.Martingale != MartingaleType.None && Parent.ThresholdScore == AlertingScore.MartingaleScore)
                            {
                                Double martingaleUpdate = 0;
                                switch (Parent.Martingale)
                                {
                                    case MartingaleType.Power:
                                        martingaleUpdate = Parent.LogPowerMartigaleBettingFunc(result[2], Parent.PowerMartingaleEpsilon);
                                        break;

                                    case MartingaleType.Mixture:
                                        martingaleUpdate = Parent.LogMixtureMartigaleBettingFunc(result[2]);
                                        break;
                                }

                                if (_logMartingaleUpdateBuffer.Count == 0)
                                {
                                    for (int i = 0; i < _logMartingaleUpdateBuffer.Capacity; ++i)
                                        _logMartingaleUpdateBuffer.AddLast(martingaleUpdate);
                                    _logMartingaleValue += _logMartingaleUpdateBuffer.Capacity * martingaleUpdate;
                                }
                                else
                                {
                                    _logMartingaleValue += martingaleUpdate;
                                    _logMartingaleValue -= _logMartingaleUpdateBuffer.PeekFirst();
                                    _logMartingaleUpdateBuffer.AddLast(martingaleUpdate);
                                }

                                result[3] = Math.Exp(_logMartingaleValue);
                            }
                        }
                    }

                    // Generating alert
                    bool alert = false;

                    if (_rawScoreBuffer.IsFull) // No alert until the buffer is completely full.
                    {
                        switch (Parent.ThresholdScore)
                        {
                            case AlertingScore.RawScore:
                                alert = rawScore >= Parent.AlertThreshold;
                                break;
                            case AlertingScore.PValueScore:
                                alert = result[2] <= Parent.AlertThreshold;
                                break;
                            case AlertingScore.MartingaleScore:
                                alert = (Parent.Martingale != MartingaleType.None) && (result[3] >= Parent.AlertThreshold);

                                if (alert)
                                {
                                    if (_martingaleAlertCounter > 0)
                                        alert = false;
                                    else
                                        _martingaleAlertCounter = Parent.WindowSize;
                                }

                                _martingaleAlertCounter--;
                                _martingaleAlertCounter = _martingaleAlertCounter < 0 ? 0 : _martingaleAlertCounter;
                                break;
                        }
                    }

                    result[0] = Convert.ToDouble(alert);
                }

                dst = new VBuffer<Double>(outputLength, result, dst.Indices);
            }

            protected override sealed void InitializeStateCore()
            {
                Parent = (SequentialAnomalyDetectionTransformBase<TInput, TState>)ParentTransform;
                Host.Assert(WindowSize >= 0);

                if (Parent.Martingale != MartingaleType.None)
                    _logMartingaleUpdateBuffer = new FixedSizeQueue<Double>(WindowSize == 0 ? 1 : WindowSize);

                _rawScoreBuffer = new FixedSizeQueue<float>(WindowSize == 0 ? 1 : WindowSize);

                _logMartingaleValue = 0;
                InitializeAnomalyDetector();
            }

            /// <summary>
            /// The abstract method that realizes the initialization functionality for the anomaly detector.
            /// </summary>
            protected abstract void InitializeAnomalyDetector();

            /// <summary>
            /// The abstract method that realizes the main logic for calculating the raw anomaly score bfor the current input given a windowed buffer
            /// </summary>
            /// <param name="input">A reference to the input object.</param>
            /// <param name="windowedBuffer">A reference to the windowed buffer.</param>
            /// <param name="iteration">A long number that indicates the number of times ComputeRawAnomalyScore has been called so far (starting value = 0).</param>
            /// <returns>The raw anomaly score for the input. The Assumption is the higher absolute value of the raw score, the more anomalous the input is.
            /// The sign of the score determines whether it's a positive anomaly or a negative one.</returns>
            protected abstract Double ComputeRawAnomalyScore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration);
        }

        /// <summary>
        /// Schema implementation to add slot name metadata to the produced output column.
        /// </summary>
        private sealed class SchemaImpl : ISchema
        {
            private readonly ISchema _parent;
            private readonly int _col;
            private readonly ColumnType _type;
            private readonly string[] _names;
            private readonly int _namesLength;
            private readonly MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> _getter;

            public int ColumnCount { get { return _parent.ColumnCount; } }

            /// <summary>
            /// Constructs the schema.
            /// </summary>
            /// <param name="schema">The schema we will wrap.
            /// Aside from presenting that additional piece of metadata, the constructed schema
            /// will appear identical to this input schema.</param>
            /// <param name="col">The column in <paramref name="schema"/> that has the metadata.</param>
            /// <param name="names"></param>
            /// <param name="length"></param>
            public SchemaImpl(ISchema schema, int col, string[] names, int length)
            {
                Contracts.Assert(length > 0);
                Contracts.Assert(Utils.Size(names) >= length);
                Contracts.AssertValue(schema);
                Contracts.Assert(0 <= col && col < schema.ColumnCount);
                _parent = schema;
                _col = col;

                _names = names;
                _namesLength = length;

                _type = new VectorType(TextType.Instance, _namesLength);
                Contracts.AssertValue(_type);
                _getter = GetSlotNames;
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                return _parent.TryGetColumnIndex(name, out col);
            }

            public string GetColumnName(int col)
            {
                return _parent.GetColumnName(col);
            }

            public ColumnType GetColumnType(int col)
            {
                return _parent.GetColumnType(col);
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                var result = _parent.GetMetadataTypes(col);
                if (col == _col)
                    return result.Prepend(_type.GetPair(MetadataUtils.Kinds.SlotNames));
                return result;
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                if (col == _col && kind == MetadataUtils.Kinds.SlotNames)
                    return _type;
                return _parent.GetMetadataTypeOrNull(kind, col);
            }

            public void GetSlotNames(int col, ref VBuffer<ReadOnlyMemory<char>> slotNames)
            {
                Contracts.Assert(col == _col);

                var result = slotNames.Values;
                if (Utils.Size(result) < _namesLength)
                    result = new ReadOnlyMemory<char>[_namesLength];

                for (int i = 0; i < _namesLength; ++i)
                    result[i] = _names[i].AsMemory();

                slotNames = new VBuffer<ReadOnlyMemory<char>>(_namesLength, result, slotNames.Indices);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                if (col == _col && kind == MetadataUtils.Kinds.SlotNames)
                {
                    _getter.Marshal(col, ref value);
                    return;
                }
                _parent.GetMetadata(kind, col, ref value);
            }
        }
    }
}
