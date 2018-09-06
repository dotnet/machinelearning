// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

[assembly: LoadableClass(typeof(NormalizerTransformer), null, typeof(SignatureLoadModel),
    "", NormalizerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NormalizerTransformer), null, typeof(SignatureLoadRowMapper),
    "", NormalizerTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class Normalizer : IEstimator<NormalizerTransformer>
    {
        internal static class Defaults
        {
            public const bool FixZero = true;
            public const bool MeanVarCdf = false;
            public const bool LogMeanVarCdf = true;
            public const int NumBins = 1024;
            public const int MinBinSize = 10;
            public const long MaxTrainingExamples = 1000000000;
        }

        public enum NormalizerMode
        {
            MinMax = 0,
            MeanVariance = 1,
            LogMeanVariance = 2,
            Binning = 3
        }

        public abstract class ColumnBase
        {
            public readonly string Input;
            public readonly string Output;
            public readonly long MaxTrainingExamples;

            private protected ColumnBase(string input, string output, long maxTrainingExamples)
            {
                Contracts.CheckNonEmpty(input, nameof(input));
                Contracts.CheckNonEmpty(output, nameof(output));
                Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");

                Input = input;
                Output = output;
                MaxTrainingExamples = maxTrainingExamples;
            }

            internal abstract IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, IRowCursor cursor);

            internal static ColumnBase Create(string input, string output, NormalizerMode mode)
            {
                switch (mode)
                {
                    case NormalizerMode.MinMax:
                        return new MinMaxColumn(input, output);
                    case NormalizerMode.MeanVariance:
                        return new MeanVarColumn(input, output);
                    case NormalizerMode.LogMeanVariance:
                        return new LogMeanVarColumn(input, output);
                    case NormalizerMode.Binning:
                        return new BinningColumn(input, output);
                    default:
                        throw Contracts.ExceptParam(nameof(mode), "Unknown normalizer mode");
                }
            }
        }

        public abstract class FixZeroColumnBase : ColumnBase
        {
            public readonly bool FixZero;

            private protected FixZeroColumnBase(string input, string output, long maxTrainingExamples, bool fixZero)
                : base(input, output, maxTrainingExamples)
            {
                FixZero = fixZero;
            }
        }

        public sealed class MinMaxColumn : FixZeroColumnBase
        {
            public MinMaxColumn(string input, string output = null, long maxTrainingExamples = Defaults.MaxTrainingExamples, bool fixZero = Defaults.FixZero)
                : base(input, output ?? input, maxTrainingExamples, fixZero)
            {
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, IRowCursor cursor)
                => NormalizeTransform.MinMaxUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        public sealed class MeanVarColumn : FixZeroColumnBase
        {
            public readonly bool UseCdf;

            public MeanVarColumn(string input, string output = null,
                long maxTrainingExamples = Defaults.MaxTrainingExamples, bool fixZero = Defaults.FixZero, bool useCdf = Defaults.MeanVarCdf)
                : base(input, output ?? input, maxTrainingExamples, fixZero)
            {
                UseCdf = useCdf;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, IRowCursor cursor)
                => NormalizeTransform.MeanVarUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        public sealed class LogMeanVarColumn : ColumnBase
        {
            public readonly bool UseCdf;

            public LogMeanVarColumn(string input, string output = null,
                long maxTrainingExamples = Defaults.MaxTrainingExamples, bool useCdf = Defaults.LogMeanVarCdf)
                : base(input, output ?? input, maxTrainingExamples)
            {
                UseCdf = useCdf;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, IRowCursor cursor)
                => NormalizeTransform.LogMeanVarUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        public sealed class BinningColumn : FixZeroColumnBase
        {
            public readonly int NumBins;

            public BinningColumn(string input, string output = null,
                long maxTrainingExamples = Defaults.MaxTrainingExamples, bool fixZero = true, int numBins = Defaults.NumBins)
                : base(input, output ?? input, maxTrainingExamples, fixZero)
            {
                NumBins = numBins;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, IRowCursor cursor)
                => NormalizeTransform.BinUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        private readonly IHost _host;
        private readonly ColumnBase[] _columns;

        public Normalizer(IHostEnvironment env, string columnName, NormalizerMode mode = NormalizerMode.MinMax)
            : this(env, mode, (columnName, columnName))
        {
        }

        public Normalizer(IHostEnvironment env, NormalizerMode mode, params (string inputColumn, string outputColumn)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(Normalizer));
            _host.CheckValue(columns, nameof(columns));
            _columns = columns.Select(x => ColumnBase.Create(x.inputColumn, x.outputColumn, mode)).ToArray();
        }

        public Normalizer(IHostEnvironment env, params ColumnBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(Normalizer));
            _host.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
        }

        public NormalizerTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return NormalizerTransformer.Train(_host, input, _columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);

            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, "fixed-size vector or scalar", col.GetTypeString());

                if (!col.ItemType.Equals(NumberType.R4) && !col.ItemType.Equals(NumberType.R8))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, "vector or scalar of R4 or R8", col.GetTypeString());

                var isNormalizedMeta = new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar,
                    BoolType.Instance, false);
                var newMetadataKinds = new List<SchemaShape.Column> { isNormalizedMeta };
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    newMetadataKinds.Add(slotMeta);
                var meta = new SchemaShape(newMetadataKinds);
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, col.ItemType, col.IsKey, meta);
            }

            return new SchemaShape(result.Values);
        }
    }

    public sealed partial class NormalizerTransformer : OneToOneTransformerBase
    {
        public const string LoaderSignature = "Normalizer";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NORMALZR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly ColumnType InputType;
            public readonly IColumnFunction ColumnFunction;

            public ColumnInfo(string input, string output, ColumnType inputType, IColumnFunction columnFunction)
            {
                Input = input;
                Output = output;
                InputType = inputType;
                ColumnFunction = columnFunction;
            }

            internal static ColumnType LoadType(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                //   - bool: is vector
                //   - int: vector size
                //   - byte: ItemKind of input column (only R4 and R8 are valid)
                bool isVector = ctx.Reader.ReadBoolean();
                int vectorSize = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(vectorSize >= 0);
                Contracts.CheckDecode(vectorSize > 0 || !isVector);

                DataKind itemKind = (DataKind)ctx.Reader.ReadByte();
                Contracts.CheckDecode(itemKind == DataKind.R4 || itemKind == DataKind.R8);

                var itemType = PrimitiveType.FromKind(itemKind);
                return isVector ? (ColumnType)(new VectorType(itemType, vectorSize)) : itemType;
            }

            internal static void SaveType(ModelSaveContext ctx, ColumnType type)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                //   - bool: is vector
                //   - int: vector size
                //   - byte: ItemKind of input column (only R4 and R8 are valid)
                ctx.Writer.Write(type.IsVector);

                Contracts.Assert(!type.IsVector || type.VectorSize > 0);
                ctx.Writer.Write(type.VectorSize);

                var itemKind = type.ItemType.RawKind;
                Contracts.Assert(itemKind == DataKind.R4 || itemKind == DataKind.R8);
                ctx.Writer.Write((byte)itemKind);
            }
        }

        private sealed class ColumnFunctionAccessor : IReadOnlyList<IColumnFunction>
        {
            private readonly ColumnInfo[] _infos;

            public ColumnFunctionAccessor(ColumnInfo[] infos)
            {
                _infos = infos;
            }

            public IColumnFunction this[int index] => _infos[index].ColumnFunction;
            public int Count => _infos.Length;
            public IEnumerator<IColumnFunction> GetEnumerator() => _infos.Select(info => info.ColumnFunction).GetEnumerator();
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }

        /// <summary>An accessor of the column functions within <see cref="_columns"/>.</summary>
        internal readonly IReadOnlyList<IColumnFunction> ColumnFunctions;

        private readonly ColumnInfo[] _columns;

        public (string input, string output)[] Columns => ColumnPairs;

        private NormalizerTransformer(IHostEnvironment env, ColumnInfo[] columns)
            : base(env.Register(nameof(NormalizerTransformer)), columns.Select(x => (x.Input, x.Output)).ToArray())
        {
            _columns = columns;
            ColumnFunctions = new ColumnFunctionAccessor(_columns);
        }

        public static NormalizerTransformer Train(IHostEnvironment env, IDataView data, Normalizer.ColumnBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(columns, nameof(columns));

            bool[] activeInput = new bool[data.Schema.ColumnCount];

            var srcCols = new int[columns.Length];
            var srcTypes = new ColumnType[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var info = columns[i];
                bool success = data.Schema.TryGetColumnIndex(info.Input, out srcCols[i]);
                if (!success)
                    throw env.ExceptSchemaMismatch(nameof(data), "input", info.Input);
                srcTypes[i] = data.Schema.GetColumnType(srcCols[i]);
                activeInput[srcCols[i]] = true;
            }

            var functionBuilders = new IColumnFunctionBuilder[columns.Length];
            var needMoreData = new bool[columns.Length];

            // Go through the input data and pass it to the column function builders.
            using (var pch = env.StartProgressChannel("Normalize"))
            {
                long numRows = 0;

                pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, numRows));
                using (var cursor = data.GetRowCursor(col => activeInput[col]))
                {
                    for (int i = 0; i < columns.Length; i++)
                    {
                        needMoreData[i] = true;
                        var info = columns[i];
                        var host = env.Register($"Column_{i:000}");

                        functionBuilders[i] = info.MakeBuilder(host, srcCols[i], srcTypes[i], cursor);
                    }

                    while (cursor.MoveNext())
                    {
                        // If the row has bad values, the good values are still being used for training.
                        // The comparisons in the code below are arranged so that NaNs in the input are not recorded.
                        // REVIEW: Should infinities and/or NaNs be filtered before the normalization? Should we not record infinities for min/max?
                        // Currently, infinities are recorded and will result in zero scale which in turn will result in NaN output for infinity input.
                        bool any = false;
                        for (int i = 0; i < columns.Length; i++)
                        {
                            if (!needMoreData[i])
                                continue;
                            var info = columns[i];
                            env.Assert(!srcTypes[i].IsVector || srcTypes[i].IsVector && srcTypes[i].IsKnownSizeVector);
                            env.Assert(functionBuilders[i] != null);
                            any |= needMoreData[i] = functionBuilders[i].ProcessValue();
                        }
                        numRows++;

                        if (!any)
                            break;
                    }
                }

                pch.Checkpoint(numRows);

                var result = new ColumnInfo[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    var func = functionBuilders[i].CreateColumnFunction();
                    result[i] = new ColumnInfo(columns[i].Input, columns[i].Output, srcTypes[i], func);
                }

                return new NormalizerTransformer(env, result);
            }
        }

        private NormalizerTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // for each added column:
            //   - source type
            //   - separate model for column function

            _columns = new ColumnInfo[ColumnPairs.Length];
            ColumnFunctions = new ColumnFunctionAccessor(_columns);
            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                var typeSrc = ColumnInfo.LoadType(ctx);
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out var function, dir, Host, typeSrc);
                _columns[iinfo] = new ColumnInfo(ColumnPairs[iinfo].input, ColumnPairs[iinfo].output, typeSrc, function);
            }
        }

        public static NormalizerTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new NormalizerTransformer(env.Register(nameof(NormalizerTransformer)), ctx);
        }

        // Factory method for SignatureLoadRowMapper.
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each added column:
            //   - source type
            //   - separate model for column function
            base.SaveColumns(ctx);

            // Individual normalization models.
            for (int iinfo = 0; iinfo < _columns.Length; iinfo++)
            {
                ColumnInfo.SaveType(ctx, _columns[iinfo].InputType);
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                ctx.SaveSubModel(dir, _columns[iinfo].ColumnFunction.Save);
            }
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            const string expectedType = "scalar or known-size vector of R4";

            var colType = inputSchema.GetColumnType(srcCol);
            if (colType.IsVector && !colType.IsKnownSizeVector)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, expectedType, "variable-size vector");
            if (!colType.ItemType.Equals(NumberType.R4) && !colType.ItemType.Equals(NumberType.R8))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, expectedType, colType.ToString());
        }

        // Temporary: enables SignatureDataTransform factory methods.
        public new IDataTransform MakeDataTransform(IDataView input)
            => base.MakeDataTransform(input);

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : MapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private NormalizerTransformer _parent;

            public bool CanSaveOnnx => true;
            public bool CanSavePfa => true;

            public Mapper(NormalizerTransformer parent, ISchema schema)
                : base(parent.Host.Register(nameof(Mapper)), parent, schema)
            {
                _parent = parent;
            }

            public override RowMapperColumnInfo[] GetOutputColumns()
            {
                var result = new RowMapperColumnInfo[_parent._columns.Length];
                for (int i = 0; i < _parent.Columns.Length; i++)
                    result[i] = new RowMapperColumnInfo(_parent._columns[i].Output, _parent._columns[i].InputType, MakeMetadata(i));
                return result;
            }

            private ColumnMetadataInfo MakeMetadata(int iinfo)
            {
                var colInfo = _parent._columns[iinfo];
                var result = new ColumnMetadataInfo(colInfo.Output);
                result.Add(MetadataUtils.Kinds.IsNormalized, new MetadataInfo<DvBool>(BoolType.Instance, IsNormalizedGetter));
                if (InputSchema.HasSlotNames(ColMapNewToOld[iinfo], colInfo.InputType.VectorSize))
                {
                    MetadataUtils.MetadataGetter<VBuffer<DvText>> getter = (int col, ref VBuffer<DvText> slotNames) =>
                        InputSchema.GetMetadata(MetadataUtils.Kinds.SlotNames, ColMapNewToOld[iinfo], ref slotNames);
                    var metaType = InputSchema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, ColMapNewToOld[iinfo]);
                    Contracts.AssertValue(metaType);
                    result.Add(MetadataUtils.Kinds.SlotNames, new MetadataInfo<VBuffer<DvText>>(metaType, getter));
                }
                return result;
            }

            private void IsNormalizedGetter(int col, ref DvBool dst)
            {
                dst = true;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                disposer = null;
                return _parent._columns[iinfo].ColumnFunction.GetGetter(input, ColMapNewToOld[iinfo]);
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    var info = _parent._columns[iinfo];
                    string sourceColumnName = info.Input;
                    if (!ctx.ContainsColumn(sourceColumnName))
                    {
                        ctx.RemoveColumn(info.Output, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(sourceColumnName),
                        ctx.AddIntermediateVariable(info.InputType, info.Output)))
                    {
                        ctx.RemoveColumn(info.Output, true);
                    }
                }
            }

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    var info = _parent._columns[iinfo];
                    var srcName = info.Input;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Output);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, info, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Output);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Output, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColumnInfo info, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                Contracts.Assert(_parent._columns[iinfo] == info);
                Contracts.AssertValue(srcToken);
                Contracts.Assert(CanSavePfa);
                return info.ColumnFunction.PfaInfo(ctx, srcToken);
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColumnInfo info, string srcVariableName, string dstVariableName)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                Contracts.Assert(_parent._columns[iinfo] == info);
                Contracts.Assert(CanSaveOnnx);

                if (info.InputType.ValueCount == 0)
                    return false;

                if (info.ColumnFunction.CanSaveOnnx)
                {
                    string opType = "Scaler";
                    var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                    info.ColumnFunction.OnnxInfo(ctx, node, info.InputType.ValueCount);
                    return true;
                }

                return false;
            }
        }

        /// <summary>
        /// An interface implemented by items of <see cref="ColumnFunctions"/> corresponding to the
        /// <see cref="NormalizeTransform.AffineColumnFunction"/> items.
        /// </summary>
        internal interface IAffineData<TData>
        {
            /// <summary>
            /// The scales. In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots. Function is <c>(input - offset) * scale</c>.
            /// </summary>
            TData Scale { get; }

            /// <summary>
            /// The offsets. In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots, or of length zero if all the offsets are zero.
            /// </summary>
            TData Offset { get; }
        }

        /// <summary>
        /// An interface implemented by items of <see cref="ColumnFunctions"/> corresponding to the
        /// <see cref="NormalizeTransform.CdfColumnFunction"/> items. The function is the value of the
        /// cumulative density function of the normal distribution parameterized with mean <see cref="Mean"/>
        /// and standard deviation <see cref="Stddev"/>.
        /// </summary>
        internal interface ICdfData<TData>
        {
            /// <summary>
            /// The mean(s). In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots.
            /// </summary>
            TData Mean { get; }

            /// <summary>
            /// The standard deviation(s). In the scalar case, this is a single value. In the vector case this is of
            /// length equal to the number of slots.
            /// </summary>
            TData Stddev { get; }

            /// <summary>
            /// Whether the input should first be considered to be logged, that is, whether this was configured
            /// over a log-normal as opposed to regular normal distribution.
            /// </summary>
            bool UseLog { get; }
        }

        /// <summary>
        /// An interface implemented by items of <see cref="ColumnFunctions"/> corresponding to the
        /// <see cref="NormalizeTransform.BinColumnFunction"/> items.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        internal interface IBinData<TData>
        {
            /// <summary>
            /// The standard deviation(s). In the scalar case, these are the bin upper bounds for that single value.
            /// In the vector case it is a jagged array of the bin upper bounds for all slots.
            /// </summary>
            ImmutableArray<TData> UpperBounds { get; }
        }
    }
}

public static class NormalizerStaticExtensions
{
    private const long MaxTrain = Normalizer.Defaults.MaxTrainingExamples;
    private const bool FZ = Normalizer.Defaults.FixZero;

    /// <summary>
    /// Learns an affine function based on the minimum and maximum, so that all values between the minimum and
    /// maximum observed during fitting fall into the range of -1 to 1. Note that if values are later transformed
    /// that are lower than the minimum, or higher than the maximum, observed during fitting, that the output
    /// values may be outside the range of -1 to 1.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>false</c>, then the observed minimum and maximum during fitting
    /// will map to -1 and 1 respectively, exactly. If however set to <c>true</c>, then 0 will always map to 0.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate that can be called whenever the function is fit, with the learned slopes
    /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<float> NormalizeByMinMax(
        this Vector<float> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
        OnFitAffine<ImmutableArray<float>> onFit = null)
    {
        return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
    }

    /// <summary>
    /// Learns an affine function based on the minimum and maximum, so that all values between the minimum and
    /// maximum observed during fitting fall into the range of -1 to 1. Note that if values are later transformed
    /// that are lower than the minimum, or higher than the maximum, observed during fitting, that the output
    /// values may be outside the range of -1 to 1.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>false</c>, then the observed minimum and maximum during fitting
    /// will map to -1 and 1 respectively, exactly. If however set to <c>true</c>, then 0 will always map to 0.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
    /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<double> NormalizeByMinMax(
        this Vector<double> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
        OnFitAffine<ImmutableArray<double>> onFit = null)
    {
        return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
    }

    private static NormVector<T> NormalizeByMinMaxCore<T>(Vector<T> input, bool fixZero, long maxTrainingExamples,
        OnFitAffine<ImmutableArray<T>> onFit)
    {
        Contracts.CheckValue(input, nameof(input));
        Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
        return new Impl<T>(input, (src, name) => new Normalizer.MinMaxColumn(src, name, maxTrainingExamples, fixZero), AffineMapper(onFit));
    }

    // We have a slightly different breaking up of categories of normalizers versus the dynamic API. Both the mean-var and
    // CDF normalizers are initialized in the same way because they gather exactly the same statistics, but from the point of
    // view of the static API what is more important is the type of mapping that winds up being computed.

    /// <summary>
    /// Learns an affine function based on the observed mean and standard deviation. This is less susceptible
    /// to outliers as compared to <see cref="NormalizeByMinMax(Vector{float}, bool, long, OnFitAffine{ImmutableArray{float}})"/>.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>true</c> then the offset will always be considered zero.</param>
    /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
    /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
    /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<float> NormalizeByMeanVar(
        this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
        OnFitAffine<ImmutableArray<float>> onFit = null)
    {
        return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
    }

    /// <summary>
    /// Learns an affine function based on the observed mean and standard deviation. This is less susceptible
    /// to outliers as compared to <see cref="NormalizeByMinMax(Vector{double}, bool, long, OnFitAffine{ImmutableArray{double}})"/>.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>true</c> then the offset will always be considered zero.</param>
    /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
    /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned slopes
    /// and, if <paramref name="fixZero"/> is <c>false</c>, the offsets as well.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<double> NormalizeByMeanVar(
        this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
        OnFitAffine<ImmutableArray<double>> onFit = null)
    {
        return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
    }

    /// <summary>
    /// Learns a function based on the cumulative density function of a normal distribution parameterized by
    /// a mean and variance as observed during fitting.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>false</c>, then the learned distributional parameters will be
    /// adjusted in such a way as to ensure that the input 0 maps to the output 0.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
    /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned mean and standard
    /// deviation for all slots.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<float> NormalizeByCumulativeDistribution(
        this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
        OnFitCumulativeDistribution<ImmutableArray<float>> onFit = null)
    {
        return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
    }

    /// <summary>
    /// Learns a function based on the cumulative density function of a normal distribution parameterized by
    /// a mean and variance as observed during fitting.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="fixZero">If set to <c>false</c>, then the learned distributional parameters will be
    /// adjusted in such a way as to ensure that the input 0 maps to the output 0.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="useLog">If set to true then we transform over the logarithm of the values, rather
    /// than just the raw values. If this is set to <c>true</c> then <paramref name="fixZero"/> is ignored.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the learned mean and standard
    /// deviation for all slots.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<double> NormalizeByCumulativeDistribution(
        this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
        OnFitCumulativeDistribution<ImmutableArray<double>> onFit = null)
    {
        return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
    }

    private static NormVector<T> NormalizeByMVCdfCore<T>(Vector<T> input, bool fixZero, bool useLog, bool useCdf, long maxTrainingExamples, Action<IColumnFunction> onFit)
    {
        Contracts.CheckValue(input, nameof(input));
        Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
        return new Impl<T>(input, (src, name) =>
        {
            if (useLog)
                return new Normalizer.LogMeanVarColumn(src, name, maxTrainingExamples, useCdf);
            return new Normalizer.MeanVarColumn(src, name, maxTrainingExamples, fixZero, useCdf);
        }, onFit);
    }

    /// <summary>
    /// Learns a function based on a discretization of the input values. The observed values for each slot are
    /// analyzed, and the range of numbers is partitioned into monotonically increasing bins. An attempt is made
    /// to make these bins equal in population, but under some circumstances this may be impossible (e.g., a slot
    /// with a very dominant mode). The way the mapping works is, if there are <c>N</c> bins in a slot, and a value
    /// falls in the range of bin <c>n</c> (indexed from 0), the output value is <c>n / (N - 1)</c>, and then possibly
    /// subtracting off the binned value for what 0 would have been if <paramref name="fixZero"/> is true.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="maxBins">The maximum number of discretization points to learn per slot.</param>
    /// <param name="fixZero">Normally the output is in the range of 0 to 1, but if set to <c>true</c>, then what
    /// would have been the output for a zero input is subtracted off the value.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the bin upper bounds for each slot.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<float> NormalizeByBinning(
        this Vector<float> input, int maxBins = Normalizer.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
        OnFitBinned<ImmutableArray<float>> onFit = null)
    {
        return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
    }

    /// <summary>
    /// Learns a function based on a discretization of the input values. The observed values for each slot are
    /// analyzed, and the range of numbers is partitioned into monotonically increasing bins. An attempt is made
    /// to make these bins equal in population, but under some circumstances this may be impossible (e.g., a slot
    /// with a very dominant mode). The way the mapping works is, if there are <c>N</c> bins in a slot, and a value
    /// falls in the range of bin <c>n</c> (indexed from 0), the output value is <c>n / (N - 1)</c>, and then possibly
    /// subtracting off the binned value for what 0 would have been if <paramref name="fixZero"/> is true.
    /// </summary>
    /// <param name="input">The input column.</param>
    /// <param name="maxBins">The maximum number of discretization points to learn per slot.</param>
    /// <param name="fixZero">Normally the output is in the range of 0 to 1, but if set to <c>true</c>, then what
    /// would have been the output for a zero input is subtracted off the value.
    /// This is valuable for the sake of sparsity preservation, if normalizing sparse vectors.</param>
    /// <param name="maxTrainingExamples">When gathering statistics only look at most this many examples.</param>
    /// <param name="onFit">A delegate called whenever the estimator is fit, with the bin upper bounds for each slot.</param>
    /// <remarks>Note that the statistics gathering and normalization is done independently per slot of the
    /// vector values.</remarks>
    /// <returns>The normalized column.</returns>
    public static NormVector<double> NormalizeByBinning(
        this Vector<double> input, int maxBins = Normalizer.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
        OnFitBinned<ImmutableArray<double>> onFit = null)
    {
        return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
    }

    private static NormVector<T> NormalizeByBinningCore<T>(Vector<T> input, int numBins, bool fixZero, long maxTrainingExamples,
        OnFitBinned<ImmutableArray<T>> onFit)
    {
        Contracts.CheckValue(input, nameof(input));
        Contracts.CheckParam(numBins > 1, nameof(maxTrainingExamples), "Must be greater than 1");
        Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
        return new Impl<T>(input, (src, name) => new Normalizer.BinningColumn(src, name, maxTrainingExamples, fixZero, numBins), BinMapper(onFit));
    }

    /// <summary>
    /// For user provided delegates to receive information when an affine normalizer is fitted.
    /// The function of the normalizer transformer is <c>(input - offset) * scale</c>.
    /// </summary>
    /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
    /// <param name="scale">The scales. In the scalar case, this is a single value. In the vector case this is of length equal
    /// to the number of slots.</param>
    /// <param name="offset">The offsets. In the scalar case, this is a single value. In the vector case this is of length equal
    /// to the number of slots, or of length zero if all the offsets are zero.</param>
    public delegate void OnFitAffine<TData>(TData scale, TData offset);

    /// <summary>
    /// For user provided delegates to receive information when a cumulative distribution function normalizer is fitted.
    /// </summary>
    /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
    /// <param name="mean">The mean value. In the scalar case, this is a single value. In the vector case this is of length equal
    /// to the number of slots.</param>
    /// <param name="standardDeviation">The standard deviation. In the scalar case, this is a single value. In the vector case
    /// this is of length equal to the number of slots.</param>
    public delegate void OnFitCumulativeDistribution<TData>(TData mean, TData standardDeviation);

    /// <summary>
    /// For user provided delegates to receive information when a binning normalizer is fitted.
    /// The function fo the normalizer transformer is, given a value, find its index in the upper bounds, then divide that value
    /// by the number of upper bounds minus 1, so as to scale the index between 0 and 1. Then, if zero had been fixed, subtract
    /// off the value that would have been computed by the above procedure for the value zero.
    /// </summary>
    /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
    /// <param name="upperBounds">For a scalar column a single sequence of the bin upper bounds. For a vector, the same, but
    /// for all slots.</param>
    public delegate void OnFitBinned<TData>(ImmutableArray<TData> upperBounds);

    #region Implementation support
    private delegate Normalizer.ColumnBase CreateNormCol(string input, string name);

    private sealed class Rec : EstimatorReconciler
    {
        // All settings are self contained in the columns.
        public static readonly Rec Inst = new Rec();

        public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
            IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
        {
            var cols = new Normalizer.ColumnBase[toOutput.Length];
            List<(int idx, Action<IColumnFunction> onFit)> onFits = null;

            for (int i = 0; i < toOutput.Length; ++i)
            {
                var col = (INormColCreator)toOutput[i];
                cols[i] = col.CreateNormCol(inputNames[col.Input], outputNames[toOutput[i]]);
                if (col.OnFit != null)
                    Utils.Add(ref onFits, (i, col.OnFit));
            }
            var norm = new Normalizer(env, cols);
            if (Utils.Size(onFits) == 0)
                return norm;
            return norm.WithOnFitDelegate(normTrans =>
            {
                Contracts.Assert(normTrans.ColumnFunctions.Count == toOutput.Length);
                foreach ((int idx, Action<IColumnFunction> onFit) in onFits)
                    onFit(normTrans.ColumnFunctions[idx]);
            });
        }
    }

    private static Action<IColumnFunction> AffineMapper<TData>(OnFitAffine<TData> onFit)
    {
        Contracts.AssertValueOrNull(onFit);
        if (onFit == null)
            return null;
        return col =>
        {
            var aCol = (NormalizerTransformer.IAffineData<TData>)col;
            onFit(aCol.Scale, aCol.Offset);
        };
    }

    private static Action<IColumnFunction> CdfMapper<TData>(OnFitCumulativeDistribution<TData> onFit)
    {
        Contracts.AssertValueOrNull(onFit);
        if (onFit == null)
            return null;
        return col =>
        {
            var aCol = (NormalizerTransformer.ICdfData<TData>)col;
            onFit(aCol.Mean, aCol.Stddev);
        };
    }

    private static Action<IColumnFunction> BinMapper<TData>(OnFitBinned<TData> onFit)
    {
        Contracts.AssertValueOrNull(onFit);
        if (onFit == null)
            return null;
        return col =>
        {
            var aCol = (NormalizerTransformer.IBinData<TData>)col;
            onFit(aCol.UpperBounds);
        };
    }

    private interface INormColCreator
    {
        CreateNormCol CreateNormCol { get; }
        PipelineColumn Input { get; }
        Action<IColumnFunction> OnFit { get; }
    }

    private sealed class Impl<T> : NormVector<T>, INormColCreator
    {
        public PipelineColumn Input { get; }
        public CreateNormCol CreateNormCol { get; }
        public Action<IColumnFunction> OnFit { get; }

        public Impl(Vector<T> input, CreateNormCol del, Action<IColumnFunction> onFitDel)
            : base(Rec.Inst, input)
        {
            Contracts.AssertValue(input);
            Contracts.AssertValue(del);
            Contracts.AssertValueOrNull(onFitDel);
            Input = input;
            CreateNormCol = del;
            OnFit = onFitDel;
        }
    }
    #endregion
}
