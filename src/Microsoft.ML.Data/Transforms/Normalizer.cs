// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Linq;

[assembly: LoadableClass(typeof(NormalizerTransformer), null, typeof(SignatureLoadModel),
    "", NormalizerTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class Normalizer : IEstimator<NormalizerTransformer>
    {
        private static class Defaults
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

            protected ColumnBase(string input, string output, long maxTrainingExamples)
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

            protected FixZeroColumnBase(string input, string output, long maxTrainingExamples, bool fixZero)
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
                var col = inputSchema.FindColumn(colInfo.Input);

                if (col == null)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, "fixed-size vector or scalar", col.GetTypeString());

                if (!col.ItemType.Equals(NumberType.R4) && !col.ItemType.Equals(NumberType.R8))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, "vector or scalar of R4 or R8", col.GetTypeString());

                var newMetadataKinds = new List<string> { MetadataUtils.Kinds.IsNormalized };
                if (col.MetadataKinds.Contains(MetadataUtils.Kinds.SlotNames))
                    newMetadataKinds.Add(MetadataUtils.Kinds.SlotNames);
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, col.ItemType, col.IsKey, newMetadataKinds.ToArray());
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

        private readonly ColumnInfo[] _columns;

        public (string input, string output)[] Columns => ColumnPairs;

        private NormalizerTransformer(IHostEnvironment env, ColumnInfo[] columns)
            : base(env.Register(nameof(NormalizerTransformer)), columns.Select(x => (x.Input, x.Output)).ToArray())
        {
            _columns = columns;
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

        // Factory method for SignatureRowMapper.
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

        private sealed class Mapper : MapperBase
        {
            private NormalizerTransformer _parent;

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
        }
    }
}
