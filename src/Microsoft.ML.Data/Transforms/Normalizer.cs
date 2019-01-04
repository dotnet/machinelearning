// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Transforms.Normalizers;
using Newtonsoft.Json.Linq;
using static Microsoft.ML.Transforms.Normalizers.NormalizeTransform;

[assembly: LoadableClass(typeof(NormalizingTransformer), null, typeof(SignatureLoadModel),
    "", NormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NormalizingTransformer), null, typeof(SignatureLoadRowMapper),
    "", NormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(NormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    "", NormalizingTransformer.LoaderSignature, "NormalizeTransform")]

namespace Microsoft.ML.Transforms.Normalizers
{
    public sealed class NormalizingEstimator : IEstimator<NormalizingTransformer>
    {
        [BestFriend]
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
            /// <summary>
            /// Linear rescale such that minimum and maximum values are mapped between -1 and 1.
            /// </summary>
            MinMax = 0,
            /// <summary>
            /// Rescale to unit variance and, optionally, zero mean.
            /// </summary>
            MeanVariance = 1,
            /// <summary>
            /// Rescale to unit variance on the log scale.
            /// </summary>
            LogMeanVariance = 2,
            /// <summary>
            /// Bucketize and then rescale to between -1 and 1.
            /// </summary>
            Binning = 3,
            /// <summary>
            /// Bucketize and then rescale to between -1 and 1. Calculates bins based on correlation with the Label column.
            /// </summary>
            SupervisedBinning = 4
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

            internal abstract IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor);

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
                    case NormalizerMode.SupervisedBinning:
                        return new SupervisedBinningColumn(input, output);
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

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor)
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

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor)
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

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor)
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

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor)
                => NormalizeTransform.BinUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        public sealed class SupervisedBinningColumn : FixZeroColumnBase
        {
            public readonly int NumBins;
            public readonly string LabelColumn;
            public readonly int MinBinSize;

            public SupervisedBinningColumn(string input, string output = null,
                string labelColumn = DefaultColumnNames.Label,
                long maxTrainingExamples = Defaults.MaxTrainingExamples,
                bool fixZero = true,
                int numBins = Defaults.NumBins,
                int minBinSize = Defaults.MinBinSize)
                : base(input, output ?? input, maxTrainingExamples, fixZero)
            {
                NumBins = numBins;
                LabelColumn = labelColumn;
                MinBinSize = minBinSize;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, ColumnType srcType, RowCursor cursor)
                => NormalizeTransform.SupervisedBinUtils.CreateBuilder(this, host, LabelColumn, srcIndex, srcType, cursor);
        }

        private readonly IHost _host;
        private readonly ColumnBase[] _columns;

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the output column.</param>
        /// <param name="outputColumn">Name of the column to be transformed. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="mode">The <see cref="NormalizerMode"/> indicating how to the old values are mapped to the new values.</param>
        public NormalizingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, NormalizerMode mode = NormalizerMode.MinMax)
            : this(env, mode, (inputColumn, outputColumn ?? inputColumn))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="mode">The <see cref="NormalizerMode"/> indicating how to the old values are mapped to the new values.</param>
        /// <param name="columns">An array of (inputColumn, outputColumn) tuples.</param>
        public NormalizingEstimator(IHostEnvironment env, NormalizerMode mode, params (string inputColumn, string outputColumn)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NormalizingEstimator));
            _host.CheckValue(columns, nameof(columns));
            _columns = columns.Select(x => ColumnBase.Create(x.inputColumn, x.outputColumn, mode)).ToArray();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">The private instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">An array of <see cref="ColumnBase"/> defining the inputs to the Normalizer, and their settings.</param>
        public NormalizingEstimator(IHostEnvironment env, params ColumnBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NormalizingEstimator));
            _host.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
        }

        public NormalizingTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return NormalizingTransformer.Train(_host, input, _columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);

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

    public sealed partial class NormalizingTransformer : OneToOneTransformerBase
    {
        public const string LoaderSignature = "Normalizer";

        internal const string LoaderSignatureOld = "NormalizeFunction";

        private static VersionInfo GetOldVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NORMFUNC",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Changed to OneToOneColumn
                verWrittenCur: 0x00010003,    // Support generic column functions
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                 loaderAssemblyName: typeof(NormalizingTransformer).Assembly.FullName);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NORMALZR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NormalizingTransformer).Assembly.FullName);
        }

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly NormalizerModelParametersBase ModelParameters;
            internal readonly ColumnType InputType;
            internal readonly IColumnFunction ColumnFunction;

            internal ColumnInfo(string input, string output, ColumnType inputType, IColumnFunction columnFunction)
            {
                Input = input;
                Output = output;
                InputType = inputType;
                ColumnFunction = columnFunction;
                ModelParameters = columnFunction.GetNormalizerModelParams();
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
            private readonly ImmutableArray<ColumnInfo> _infos;

            public ColumnFunctionAccessor(ImmutableArray<ColumnInfo> infos)
            {
                _infos = infos;
            }

            public IColumnFunction this[int index] => _infos[index].ColumnFunction;
            public int Count => _infos.Length;
            public IEnumerator<IColumnFunction> GetEnumerator() => _infos.Select(info => info.ColumnFunction).GetEnumerator();
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }

        /// <summary>An accessor of the column functions within <see cref="Columns"/>.</summary>
        [BestFriend]
        internal readonly IReadOnlyList<IColumnFunction> ColumnFunctions;

        public readonly ImmutableArray<ColumnInfo> Columns;
        private NormalizingTransformer(IHostEnvironment env, ColumnInfo[] columns)
            : base(env.Register(nameof(NormalizingTransformer)), columns.Select(x => (x.Input, x.Output)).ToArray())
        {
            Columns = ImmutableArray.Create(columns);
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
        }

        public static NormalizingTransformer Train(IHostEnvironment env, IDataView data, NormalizingEstimator.ColumnBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(columns, nameof(columns));

            bool[] activeInput = new bool[data.Schema.Count];

            var srcCols = new int[columns.Length];
            var srcTypes = new ColumnType[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var info = columns[i];
                bool success = data.Schema.TryGetColumnIndex(info.Input, out srcCols[i]);
                if (!success)
                    throw env.ExceptSchemaMismatch(nameof(data), "input", info.Input);
                srcTypes[i] = data.Schema[srcCols[i]].Type;
                activeInput[srcCols[i]] = true;

                var supervisedBinColumn = info as NormalizingEstimator.SupervisedBinningColumn;
                if(supervisedBinColumn != null)
                {
                    var labelColumnId = SupervisedBinUtils.GetLabelColumnId(env, data.Schema, supervisedBinColumn.LabelColumn);
                    activeInput[labelColumnId] = true;
                }
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

                return new NormalizingTransformer(env, result);
            }
        }

        private NormalizingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // for each added column:
            //   - source type
            //   - separate model for column function

            var cols = new ColumnInfo[ColumnPairs.Length];
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                var typeSrc = ColumnInfo.LoadType(ctx);
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out var function, dir, Host, typeSrc);
                cols[iinfo] = new ColumnInfo(ColumnPairs[iinfo].input, ColumnPairs[iinfo].output, typeSrc, function);
            }

            Columns = ImmutableArray.Create(cols);
        }

        // This constructor for models in old format.
        private NormalizingTransformer(IHost host, ModelLoadContext ctx, IDataView input)
          : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // for each added column:
            //   - separate model for column function
            var cols = new ColumnInfo[ColumnPairs.Length];
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                var typeSrc = input.Schema[ColumnPairs[iinfo].input].Type;
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out var function, dir, Host, typeSrc);
                cols[iinfo] = new ColumnInfo(ColumnPairs[iinfo].input, ColumnPairs[iinfo].output, typeSrc, function);
            }

            Columns = ImmutableArray.Create(cols);
        }

        public static NormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new NormalizingTransformer(env.Register(nameof(NormalizingTransformer)), ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetOldVersionInfo());
            int cbFloat = ctx.Reader.ReadInt32();
            env.CheckDecode(cbFloat == sizeof(float));
            var transformer = new NormalizingTransformer(env.Register(nameof(NormalizingTransformer)), ctx, input);
            return transformer.MakeDataTransform(input);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
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
            for (int iinfo = 0; iinfo < Columns.Length; iinfo++)
            {
                ColumnInfo.SaveType(ctx, Columns[iinfo].InputType);
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                ctx.SaveSubModel(dir, Columns[iinfo].ColumnFunction.Save);
            }
        }

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            const string expectedType = "scalar or known-size vector of R4";

            var colType = inputSchema[srcCol].Type;
            if (colType.IsVector && !colType.IsKnownSizeVector)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, expectedType, "variable-size vector");
            if (!colType.ItemType.Equals(NumberType.R4) && !colType.ItemType.Equals(NumberType.R8))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, expectedType, colType.ToString());
        }

        // Temporary: enables SignatureDataTransform factory methods.
        public new IDataTransform MakeDataTransform(IDataView input)
            => base.MakeDataTransform(input);

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private NormalizingTransformer _parent;

            public bool CanSaveOnnx(OnnxContext ctx) => true;
            public bool CanSavePfa => true;

            public Mapper(NormalizingTransformer parent, Schema schema)
                : base(parent.Host.Register(nameof(Mapper)), parent, schema)
            {
                _parent = parent;
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.Columns.Length];
                for (int i = 0; i < _parent.Columns.Length; i++)
                    result[i] = new Schema.DetachedColumn(_parent.Columns[i].Output, _parent.Columns[i].InputType, MakeMetadata(i));
                return result;
            }

            private Schema.Metadata MakeMetadata(int iinfo)
            {
                var colInfo = _parent.Columns[iinfo];
                var builder = new MetadataBuilder();

                builder.Add(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, (ValueGetter<bool>)IsNormalizedGetter);
                builder.Add(InputSchema[ColMapNewToOld[iinfo]].Metadata, name => name == MetadataUtils.Kinds.SlotNames);
                return builder.GetMetadata();
            }

            private void IsNormalizedGetter(ref bool dst)
            {
                dst = true;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                return _parent.Columns[iinfo].ColumnFunction.GetGetter(input, ColMapNewToOld[iinfo]);
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _parent.Columns.Length; ++iinfo)
                {
                    var info = _parent.Columns[iinfo];
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

                for (int iinfo = 0; iinfo < _parent.Columns.Length; ++iinfo)
                {
                    var info = _parent.Columns[iinfo];
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
                Contracts.Assert(0 <= iinfo && iinfo < _parent.Columns.Length);
                Contracts.Assert(_parent.Columns[iinfo] == info);
                Contracts.AssertValue(srcToken);
                Contracts.Assert(CanSavePfa);
                return info.ColumnFunction.PfaInfo(ctx, srcToken);
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColumnInfo info, string srcVariableName, string dstVariableName)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.Columns.Length);
                Contracts.Assert(_parent.Columns[iinfo] == info);
                Contracts.Assert(CanSaveOnnx(ctx));

                if (info.InputType.ValueCount == 0)
                    return false;

                if (info.ColumnFunction.CanSaveOnnx(ctx))
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
        /// Base class for all the NormalizerData classes: <see cref="AffineNormalizerModelParameters{TData}"/>,
        /// <see cref="BinNormalizerModelParameters{TData}"/>, <see cref="CdfNormalizerModelParameters{TData}"/>.
        /// </summary>
        public abstract class NormalizerModelParametersBase
        {
            private protected NormalizerModelParametersBase() { }
        }

        /// <summary>
        /// The model parameters generated by affine normalization transformations.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public sealed class AffineNormalizerModelParameters<TData> : NormalizerModelParametersBase
        {
            /// <summary>
            /// The scales. In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots. Function is <c>(input - offset) * scale</c>.
            /// </summary>
            public TData Scale { get; }

            /// <summary>
            /// The offsets. In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots, or of length zero if all the offsets are zero.
            /// </summary>
            public TData Offset { get; }

            /// <summary>
            /// Initializes a new instance of <see cref="AffineNormalizerModelParameters{TData}"/>
            /// </summary>
            internal AffineNormalizerModelParameters(TData scale, TData offset)
            {
                Scale = scale;
                Offset = offset;
            }
        }

        /// <summary>
        /// The model parameters generated by cumulative distribution normalization transformations.
        /// The cumulative density function is parameterized by <see cref="CdfNormalizerModelParameters{TData}.Mean"/> and
        /// the <see cref="CdfNormalizerModelParameters{TData}.Stddev"/> as observed during fitting.
        /// </summary>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Normalize](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Normalizer.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public sealed class CdfNormalizerModelParameters<TData> : NormalizerModelParametersBase
        {
            /// <summary>
            /// The mean(s). In the scalar case, this is a single value. In the vector case this is of length equal
            /// to the number of slots.
            /// </summary>
            public TData Mean { get; }

            /// <summary>
            /// The standard deviation(s). In the scalar case, this is a single value. In the vector case this is of
            /// length equal to the number of slots.
            /// </summary>
            public TData Stddev { get; }

            /// <summary>
            /// Whether the we ought to apply a logarithm to the input first.
            /// </summary>
            public bool UseLog { get; }

            /// <summary>
            /// Initializes a new instance of <see cref="CdfNormalizerModelParameters{TData}"/>
            /// </summary>
            internal CdfNormalizerModelParameters(TData mean, TData stddev, bool useLog)
            {
                Mean = mean;
                Stddev = stddev;
                UseLog = useLog;
            }
        }

        /// <summary>
        /// The model parameters generated by buckettizing the data into bins with monotonically
        /// increasing <see cref="BinNormalizerModelParameters{TData}.UpperBounds"/>.
        /// The <see cref="BinNormalizerModelParameters{TData}.Density"/> value is constant from bin to bin, for most cases.
        /// /// </summary>
        public sealed class BinNormalizerModelParameters<TData> : NormalizerModelParametersBase
        {
            /// <summary>
            /// The standard deviation(s). In the scalar case, these are the bin upper bounds for that single value.
            /// In the vector case it is a jagged array of the bin upper bounds for all slots.
            /// </summary>
            public ImmutableArray<TData> UpperBounds { get; }

            /// <summary>
            /// The frequency of the datapoints per each bin.
            /// </summary>
            public TData Density { get; }

            /// <summary>
            /// If normalization is performed with <see cref="NormalizeTransform.FixZeroArgumentsBase.FixZero"/> set to <value>true</value>,
            /// the offset indicates the displacement of zero, if any.
            /// </summary>
            public TData Offset { get; }

            /// <summary>
            /// Initializes a new instance of <see cref="BinNormalizerModelParameters{TData}"/>
            /// </summary>
            internal BinNormalizerModelParameters(ImmutableArray<TData> upperBounds, TData density, TData offset)
            {
                UpperBounds = upperBounds;
                Density = density;
                Offset = offset;
            }
        }
    }
}
