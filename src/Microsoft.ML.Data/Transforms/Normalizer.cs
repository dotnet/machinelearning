// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(NormalizingTransformer), null, typeof(SignatureLoadModel),
    "", NormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NormalizingTransformer), null, typeof(SignatureLoadRowMapper),
    "", NormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(NormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    "", NormalizingTransformer.LoaderSignature, "NormalizeTransform")]

namespace Microsoft.ML.Transforms
{
    public sealed class NormalizingEstimator : IEstimator<NormalizingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const bool EnsureZeroUntouched = true;
            public const bool MeanVarCdf = false;
            public const bool LogMeanVarCdf = true;
            public const int MaximumBinCount = 1024;
            public const int MininimumBinSize = 10;
            public const long MaximumExampleCount = 1000000000;
        }

        public enum NormalizationMode
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

        [BestFriend]
        internal abstract class ColumnOptionsBase
        {
            public readonly string Name;
            public readonly string InputColumnName;
            public readonly long MaximumExampleCount;

            private protected ColumnOptionsBase(string name, string inputColumnName, long maximumExampleCount)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckNonEmpty(inputColumnName, nameof(inputColumnName));
                Contracts.CheckParam(maximumExampleCount > 1, nameof(maximumExampleCount), "Must be greater than 1");

                Name = name;
                InputColumnName = inputColumnName;
                MaximumExampleCount = maximumExampleCount;
            }

            internal abstract IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor);

            internal static ColumnOptionsBase Create(string outputColumnName, string inputColumnName, NormalizationMode mode)
            {
                switch (mode)
                {
                    case NormalizationMode.MinMax:
                        return new MinMaxColumnOptions(outputColumnName, inputColumnName);
                    case NormalizationMode.MeanVariance:
                        return new MeanVarianceColumnOptions(outputColumnName, inputColumnName);
                    case NormalizationMode.LogMeanVariance:
                        return new LogMeanVarianceColumnOptions(outputColumnName, inputColumnName);
                    case NormalizationMode.Binning:
                        return new BinningColumnOptions(outputColumnName, inputColumnName);
                    case NormalizationMode.SupervisedBinning:
                        return new SupervisedBinningColumOptions(outputColumnName, inputColumnName);
                    default:
                        throw Contracts.ExceptParam(nameof(mode), "Unknown normalizer mode");
                }
            }
        }

        internal abstract class ControlZeroColumnOptionsBase : ColumnOptionsBase
        {
            public readonly bool EnsureZeroUntouched;

            private protected ControlZeroColumnOptionsBase(string outputColumnName, string inputColumnName, long maximumExampleCount, bool ensureZeroUntouched)
                : base(outputColumnName, inputColumnName, maximumExampleCount)
            {
                EnsureZeroUntouched = ensureZeroUntouched;
            }
        }

        [BestFriend]
        internal sealed class MinMaxColumnOptions : ControlZeroColumnOptionsBase
        {
            public MinMaxColumnOptions(string outputColumnName, string inputColumnName = null, long maximumExampleCount = Defaults.MaximumExampleCount, bool ensureZeroUntouched = Defaults.EnsureZeroUntouched)
                : base(outputColumnName, inputColumnName ?? outputColumnName, maximumExampleCount, ensureZeroUntouched)
            {
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
                => NormalizeTransform.MinMaxUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        [BestFriend]
        internal sealed class MeanVarianceColumnOptions : ControlZeroColumnOptionsBase
        {
            public readonly bool UseCdf;

            public MeanVarianceColumnOptions(string outputColumnName, string inputColumnName = null,
                long maximumExampleCount = Defaults.MaximumExampleCount, bool fixZero = Defaults.EnsureZeroUntouched, bool useCdf = Defaults.MeanVarCdf)
                : base(outputColumnName, inputColumnName ?? outputColumnName, maximumExampleCount, fixZero)
            {
                UseCdf = useCdf;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
                => NormalizeTransform.MeanVarUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        [BestFriend]
        internal sealed class LogMeanVarianceColumnOptions : ColumnOptionsBase
        {
            public readonly bool UseCdf;

            public LogMeanVarianceColumnOptions(string outputColumnName, string inputColumnName = null,
                long maximumExampleCount = Defaults.MaximumExampleCount, bool useCdf = Defaults.LogMeanVarCdf)
                : base(outputColumnName, inputColumnName ?? outputColumnName, maximumExampleCount)
            {
                UseCdf = useCdf;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
                => NormalizeTransform.LogMeanVarUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        [BestFriend]
        internal sealed class BinningColumnOptions : ControlZeroColumnOptionsBase
        {
            public readonly int MaximumBinCount;

            public BinningColumnOptions(string outputColumnName, string inputColumnName = null,
                long maximumExampleCount = Defaults.MaximumExampleCount, bool fixZero = true, int maximumBinCount = Defaults.MaximumBinCount)
                : base(outputColumnName, inputColumnName ?? outputColumnName, maximumExampleCount, fixZero)
            {
                MaximumBinCount = maximumBinCount;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
                => NormalizeTransform.BinUtils.CreateBuilder(this, host, srcIndex, srcType, cursor);
        }

        [BestFriend]
        internal sealed class SupervisedBinningColumOptions : ControlZeroColumnOptionsBase
        {
            public readonly int MaximumBinCount;
            public readonly string LabelColumnName;
            public readonly int MininimumBinSize;

            public SupervisedBinningColumOptions(string outputColumnName, string inputColumnName = null,
                string labelColumnName = DefaultColumnNames.Label,
                long maximumExampleCount = Defaults.MaximumExampleCount,
                bool fixZero = true,
                int maximumBinCount = Defaults.MaximumBinCount,
                int mininimumBinSize = Defaults.MininimumBinSize)
                : base(outputColumnName, inputColumnName ?? outputColumnName, maximumExampleCount, fixZero)
            {
                MaximumBinCount = maximumBinCount;
                LabelColumnName = labelColumnName;
                MininimumBinSize = mininimumBinSize;
            }

            internal override IColumnFunctionBuilder MakeBuilder(IHost host, int srcIndex, DataViewType srcType, DataViewRowCursor cursor)
                => NormalizeTransform.SupervisedBinUtils.CreateBuilder(this, host, LabelColumnName, srcIndex, srcType, cursor);
        }

        private readonly IHost _host;
        private readonly ColumnOptionsBase[] _columns;

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="mode">The <see cref="NormalizationMode"/> indicating how to the old values are mapped to the new values.</param>
        internal NormalizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, NormalizationMode mode = NormalizationMode.MinMax)
            : this(env, mode, (outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="mode">The <see cref="NormalizationMode"/> indicating how to the old values are mapped to the new values.</param>
        /// <param name="columns">An array of (outputColumnName, inputColumnName) tuples.</param>
        internal NormalizingEstimator(IHostEnvironment env, NormalizationMode mode, params (string outputColumnName, string inputColumnName)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NormalizingEstimator));
            _host.CheckValue(columns, nameof(columns));
            _columns = columns.Select(x => ColumnOptionsBase.Create(x.outputColumnName, x.inputColumnName, mode)).ToArray();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NormalizingEstimator"/>.
        /// </summary>
        /// <param name="env">The private instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">An array of <see cref="ColumnOptionsBase"/> defining the inputs to the Normalizer, and their settings.</param>
        internal NormalizingEstimator(IHostEnvironment env, params ColumnOptionsBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NormalizingEstimator));
            _host.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
        }

        /// <summary>
        /// Trains and returns a <see cref="NormalizingTransformer"/>.
        /// </summary>
        public NormalizingTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return NormalizingTransformer.Train(_host, input, _columns);
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);

            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "known-size vector or scalar", col.GetTypeString());

                if (!col.ItemType.Equals(NumberDataViewType.Single) && !col.ItemType.Equals(NumberDataViewType.Double))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "vector or scalar of float or double", col.GetTypeString());

                var isNormalizedMeta = new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar,
                    BooleanDataViewType.Instance, false);
                var newMetadataKinds = new List<SchemaShape.Column> { isNormalizedMeta };
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    newMetadataKinds.Add(slotMeta);
                var meta = new SchemaShape(newMetadataKinds);
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, col.Kind, col.ItemType, col.IsKey, meta);
            }

            return new SchemaShape(result.Values);
        }
    }

    public sealed partial class NormalizingTransformer : OneToOneTransformerBase
    {
        internal const string LoaderSignature = "Normalizer";

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

        [BestFriend]
        internal sealed class ColumnOptions
        {
            public readonly string Name;
            public readonly string InputColumnName;
            public readonly NormalizerModelParametersBase ModelParameters;
            internal readonly DataViewType InputType;
            internal readonly IColumnFunction ColumnFunction;

            internal ColumnOptions(string name, string inputColumnName, DataViewType inputType, IColumnFunction columnFunction)
            {
                Name = name;
                InputColumnName = inputColumnName;
                InputType = inputType;
                ColumnFunction = columnFunction;
                ModelParameters = columnFunction.GetNormalizerModelParams();
            }

            internal static DataViewType LoadType(ModelLoadContext ctx)
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

                InternalDataKind itemKind = (InternalDataKind)ctx.Reader.ReadByte();
                Contracts.CheckDecode(itemKind == InternalDataKind.R4 || itemKind == InternalDataKind.R8);

                var itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(itemKind);
                return isVector ? (DataViewType)(new VectorType(itemType, vectorSize)) : itemType;
            }

            internal static void SaveType(ModelSaveContext ctx, DataViewType type)
            {
                Contracts.AssertValue(ctx);
                // *** Binary format ***
                //   - bool: is vector
                //   - int: vector size
                //   - byte: ItemKind of input column (only R4 and R8 are valid)
                VectorType vectorType = type as VectorType;
                ctx.Writer.Write(vectorType != null);

                Contracts.Assert(vectorType == null || vectorType.IsKnownSize);
                ctx.Writer.Write(vectorType?.Size ?? 0);

                DataViewType itemType = vectorType?.ItemType ?? type;
                itemType.RawType.TryGetDataKind(out InternalDataKind itemKind);
                Contracts.Assert(itemKind == InternalDataKind.R4 || itemKind == InternalDataKind.R8);
                ctx.Writer.Write((byte)itemKind);
            }
        }

        private sealed class ColumnFunctionAccessor : IReadOnlyList<IColumnFunction>
        {
            private readonly ImmutableArray<ColumnOptions> _infos;

            public ColumnFunctionAccessor(ImmutableArray<ColumnOptions> infos)
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

        /// <summary>
        /// The configuration of the normalizer. The i-th element describes the i-th input-output column pair.
        /// </summary>
        [BestFriend]
        internal readonly ImmutableArray<ColumnOptions> Columns;

        /// <summary>
        /// The normalization configurations of input columns. It returns the normalization parameters applied to the <paramref name="index"/>-th input column.
        /// </summary>
        /// <param name="index">column index.</param>
        /// <returns>the normalization parameters applied to the <paramref name="index"/>-th input column.</returns>
        public NormalizerModelParametersBase GetNormalizerModelParameters(int index)
        {
            string errMsg = "Not valid. Valid range is from 0 (inclusive) to " + Columns.Length + " (exclusive) but got " + index + ".";
            Contracts.CheckUserArg(index >= 0 && index < Columns.Length, nameof(index), errMsg);

            return Columns[index].ModelParameters;
        }

        private NormalizingTransformer(IHostEnvironment env, ColumnOptions[] columns)
            : base(env.Register(nameof(NormalizingTransformer)), columns.Select(x => (x.Name, x.InputColumnName)).ToArray())
        {
            Columns = ImmutableArray.Create(columns);
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
        }

        internal static NormalizingTransformer Train(IHostEnvironment env, IDataView data, NormalizingEstimator.ColumnOptionsBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckValue(columns, nameof(columns));

            var activeCols = new List<DataViewSchema.Column>();

            var srcCols = new int[columns.Length];
            var srcTypes = new DataViewType[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var info = columns[i];
                bool success = data.Schema.TryGetColumnIndex(info.InputColumnName, out srcCols[i]);
                if (!success)
                    throw env.ExceptSchemaMismatch(nameof(data), "input", info.InputColumnName);
                srcTypes[i] = data.Schema[srcCols[i]].Type;
                activeCols.Add(data.Schema[srcCols[i]]);

                var supervisedBinColumn = info as NormalizingEstimator.SupervisedBinningColumOptions;
                if(supervisedBinColumn != null)
                    activeCols.Add(data.Schema[supervisedBinColumn.LabelColumnName]);
            }

            var functionBuilders = new IColumnFunctionBuilder[columns.Length];
            var needMoreData = new bool[columns.Length];

            // Go through the input data and pass it to the column function builders.
            using (var pch = env.StartProgressChannel("Normalize"))
            {
                long numRows = 0;

                pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, numRows));
                using (var cursor = data.GetRowCursor(activeCols))
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
                            env.Assert(!(srcTypes[i] is VectorType vectorType) || vectorType.IsKnownSize);
                            env.Assert(functionBuilders[i] != null);
                            any |= needMoreData[i] = functionBuilders[i].ProcessValue();
                        }
                        numRows++;

                        if (!any)
                            break;
                    }
                }

                pch.Checkpoint(numRows);

                var result = new ColumnOptions[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    var func = functionBuilders[i].CreateColumnFunction();
                    result[i] = new ColumnOptions(columns[i].Name, columns[i].InputColumnName, srcTypes[i], func);
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

            var cols = new ColumnOptions[ColumnPairs.Length];
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                var typeSrc = ColumnOptions.LoadType(ctx);
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out var function, dir, Host, typeSrc);
                cols[iinfo] = new ColumnOptions(ColumnPairs[iinfo].outputColumnName, ColumnPairs[iinfo].inputColumnName, typeSrc, function);
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
            var cols = new ColumnOptions[ColumnPairs.Length];
            ColumnFunctions = new ColumnFunctionAccessor(Columns);
            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                var typeSrc = input.Schema[ColumnPairs[iinfo].inputColumnName].Type;
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out var function, dir, Host, typeSrc);
                cols[iinfo] = new ColumnOptions(ColumnPairs[iinfo].outputColumnName, ColumnPairs[iinfo].inputColumnName, typeSrc, function);
            }

            Columns = ImmutableArray.Create(cols);
        }

        private static NormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
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
                ColumnOptions.SaveType(ctx, Columns[iinfo].InputType);
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                ctx.SaveSubModel(dir, Columns[iinfo].ColumnFunction.Save);
            }
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            const string expectedType = "scalar or known-size vector of R4";

            var colType = inputSchema[srcCol].Type;
            VectorType vectorType = colType as VectorType;
            if (vectorType != null && !vectorType.IsKnownSize)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, expectedType, "variable-size vector");
            DataViewType itemType = vectorType?.ItemType ?? colType;
            if (!itemType.Equals(NumberDataViewType.Single) && !itemType.Equals(NumberDataViewType.Double))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, expectedType, colType.ToString());
        }

        // Temporary: enables SignatureDataTransform factory methods.
        internal new IDataTransform MakeDataTransform(IDataView input)
            => base.MakeDataTransform(input);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private NormalizingTransformer _parent;

            public bool CanSaveOnnx(OnnxContext ctx) => true;
            public bool CanSavePfa => true;

            public Mapper(NormalizingTransformer parent, DataViewSchema schema)
                : base(parent.Host.Register(nameof(Mapper)), parent, schema)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.Columns.Length];
                for (int i = 0; i < _parent.Columns.Length; i++)
                    result[i] = new DataViewSchema.DetachedColumn(_parent.Columns[i].Name, _parent.Columns[i].InputType, MakeMetadata(i));
                return result;
            }

            private DataViewSchema.Annotations MakeMetadata(int iinfo)
            {
                var colInfo = _parent.Columns[iinfo];
                var builder = new DataViewSchema.Annotations.Builder();

                builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ValueGetter<bool>)IsNormalizedGetter);
                builder.Add(InputSchema[ColMapNewToOld[iinfo]].Annotations, name => name == AnnotationUtils.Kinds.SlotNames);
                return builder.ToAnnotations();
            }

            private void IsNormalizedGetter(ref bool dst)
            {
                dst = true;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
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
                    string inputColumnName = info.InputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(inputColumnName),
                        ctx.AddIntermediateVariable(info.InputType, info.Name)))
                    {
                        ctx.RemoveColumn(info.Name, true);
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
                    var srcName = info.InputColumnName;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, info, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Name, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColumnOptions info, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.Columns.Length);
                Contracts.Assert(_parent.Columns[iinfo] == info);
                Contracts.AssertValue(srcToken);
                Contracts.Assert(CanSavePfa);
                return info.ColumnFunction.PfaInfo(ctx, srcToken);
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColumnOptions info, string srcVariableName, string dstVariableName)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.Columns.Length);
                Contracts.Assert(_parent.Columns[iinfo] == info);
                Contracts.Assert(CanSaveOnnx(ctx));

                int valueCount = info.InputType.GetValueCount();
                if (valueCount == 0)
                    return false;

                if (info.ColumnFunction.CanSaveOnnx(ctx))
                {
                    string opType = "Scaler";
                    var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                    info.ColumnFunction.OnnxInfo(ctx, node, valueCount);
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
        /// the <see cref="CdfNormalizerModelParameters{TData}.StandardDeviation"/> as observed during fitting.
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
            public TData StandardDeviation { get; }

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
                StandardDeviation = stddev;
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
            /// If normalization is performed with <see cref="NormalizeTransform.ControlZeroArgumentsBase.EnsureZeroUntouched"/> set to <value>true</value>,
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
