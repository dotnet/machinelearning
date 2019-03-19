// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(MissingValueReplacingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueReplacingTransformer), typeof(MissingValueReplacingTransformer.Options), typeof(SignatureDataTransform),
    MissingValueReplacingTransformer.FriendlyName, MissingValueReplacingTransformer.LoadName, "NAReplace", MissingValueReplacingTransformer.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(MissingValueReplacingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueReplacingTransformer), null, typeof(SignatureLoadDataTransform),
    MissingValueReplacingTransformer.FriendlyName, MissingValueReplacingTransformer.LoadName)]

[assembly: LoadableClass(MissingValueReplacingTransformer.Summary, typeof(MissingValueReplacingTransformer), null, typeof(SignatureLoadModel),
    MissingValueReplacingTransformer.FriendlyName, MissingValueReplacingTransformer.LoadName)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(MissingValueReplacingTransformer), null, typeof(SignatureLoadRowMapper),
   MissingValueReplacingTransformer.FriendlyName, MissingValueReplacingTransformer.LoadName)]

namespace Microsoft.ML.Transforms
{
    // This transform can transform either scalars or vectors (both fixed and variable size),
    // creating output columns that are identical to the input columns except for replacing NA values
    // with either the default value, user input, or imputed values (min/max/mean are currently supported).
    // Imputation modes are supported for vectors both by slot and across all slots.
    // REVIEW: May make sense to implement the transform template interface.
    /// <include file='doc.xml' path='doc/members/member[@name="NAReplace"]/*' />
    public sealed partial class MissingValueReplacingTransformer : OneToOneTransformerBase
    {
        internal enum ReplacementKind : byte
        {
            // REVIEW: What should the full list of options for this transform be?
            DefaultValue = 0,
            Mean = 1,
            Minimum = 2,
            Maximum = 3,
            SpecifiedValue = 4,

            [HideEnumValue]
            Def = DefaultValue,
            [HideEnumValue]
            Default = DefaultValue,
            [HideEnumValue]
            Min = Minimum,
            [HideEnumValue]
            Max = Maximum,

            [HideEnumValue]
            Val = SpecifiedValue,
            [HideEnumValue]
            Value = SpecifiedValue,
        }

        // REVIEW: Need to add support for imputation modes for replacement values:
        // *default: use default value
        // *custom: use replacementValue string
        // *mean: use domain value closest to the mean
        // Potentially also min/max; probably will not include median due to its relatively low value and high computational cost.
        // Note: Will need to support different replacement values for different slots to implement this.
        internal sealed class Column : OneToOneColumn
        {
            // REVIEW: Should flexibility for different replacement values for slots be introduced?
            [Argument(ArgumentType.AtMostOnce, HelpText = "Replacement value for NAs (uses default value if not given)", ShortName = "rep")]
            public string ReplacementString;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The replacement method to utilize")]
            public ReplacementKind? Kind;

            // REVIEW: The default is to perform imputation by slot. If the input column is an unknown size vector type, then imputation
            // will be performed across columns. Should the default be changed/an imputation method required?
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to impute values by slot")]
            public bool? Slot;

            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                // We accept N:R:S where N is the new column name, R is the replacement string,
                // and S is source column names.
                return base.TryParse(str, out ReplacementString);
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Kind != null || Slot != null)
                    return false;
                if (ReplacementString == null)
                    return TryUnparseCore(sb);

                return TryUnparseCore(sb, ReplacementString);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:rep:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The replacement method to utilize", ShortName = "kind")]
            public ReplacementKind ReplacementKind = (ReplacementKind)MissingValueReplacingEstimator.Defaults.Mode;

            // Specifying by-slot imputation for vectors of unknown size will cause a warning, and the imputation will be global.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to impute values by slot", ShortName = "slot")]
            public bool ImputeBySlot = MissingValueReplacingEstimator.Defaults.ImputeBySlot;
        }

        internal const string LoadName = "NAReplaceTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                // REVIEW: temporary name
                modelSignature: "NAREP TF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x0010002, // Added imputation methods.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName,
                loaderAssemblyName: typeof(MissingValueReplacingTransformer).Assembly.FullName);
        }

        internal const string Summary = "Create an output column of the same type and size of the input column, where missing values "
         + "are replaced with either the default value or the mean/min/max value (for non-text columns only).";

        internal const string FriendlyName = "NA Replace Transform";
        internal const string ShortName = "NARep";

        internal static string TestType(DataViewType type)
        {
            // Item type must have an NA value that exists and is not equal to its default value.
            Func<DataViewType, string> func = TestType<int>;
            var itemType = type.GetItemType();
            return Utils.MarshalInvoke(func, itemType.RawType, itemType);
        }

        private static string TestType<T>(DataViewType type)
        {
            Contracts.Assert(type.GetItemType().RawType == typeof(T));
            if (!Data.Conversion.Conversions.Instance.TryGetIsNAPredicate(type.GetItemType(), out InPredicate<T> isNA))
            {
                return string.Format("Type '{0}' is not supported by {1} since it doesn't have an NA value",
                    type, LoadName);
            }
            var t = default(T);
            if (isNA(in t))
            {
                // REVIEW: Key values will be handled in a "new key value" transform.
                return string.Format("Type '{0}' is not supported by {1} since its NA value is equivalent to its default value",
                    type, LoadName);
            }
            return null;
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(MissingValueReplacingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        // The output column types, parallel to Infos.
        private readonly DataViewType[] _replaceTypes;

        // The replacementValues for the columns, parallel to Infos.
        // The elements of this array can be either primitive values or arrays of primitive values. When replacing a scalar valued column in Infos,
        // this array will hold a primitive value. When replacing a vector valued column in Infos, this array will either hold a primitive
        // value, indicating that NAs in all slots will be replaced with this value, or an array of primitives holding the value that each slot
        // will have its NA values replaced with respectively. The case with an array of primitives can only occur when dealing with a
        // vector of known size.
        private readonly object[] _repValues;

        // Marks if the replacement values in given slots of _repValues are the default value.
        // REVIEW: Currently these arrays are constructed on load but could be changed to being constructed lazily.
        private readonly BitArray[] _repIsDefault;

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            string reason = TestType(type);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        internal MissingValueReplacingTransformer(IHostEnvironment env, IDataView input, params MissingValueReplacingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MissingValueReplacingTransformer)), GetColumnPairs(columns))
        {
            // Check that all the input columns are present and correct.
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].inputColumnName, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].inputColumnName);
                CheckInputColumn(input.Schema, i, srcCol);
            }
            GetReplacementValues(input, columns, out _repValues, out _repIsDefault, out _replaceTypes);
        }

        private MissingValueReplacingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            _repValues = new object[columnsLength];
            _repIsDefault = new BitArray[columnsLength];
            _replaceTypes = new DataViewType[columnsLength];
            var saver = new BinarySaver(Host, new BinarySaver.Arguments());
            for (int i = 0; i < columnsLength; i++)
            {
                if (!saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out DataViewType savedType, out object repValue))
                    throw Host.ExceptDecode();
                _replaceTypes[i] = savedType;
                if (savedType is VectorType savedVectorType)
                {
                    // REVIEW: The current implementation takes the serialized VBuffer, densifies it, and stores the values array.
                    // It might be of value to consider storing the VBuffer in order to possibly benefit from sparsity. However, this would
                    // necessitate a reimplementation of the FillValues code to accomodate sparse VBuffers.
                    object[] args = new object[] { repValue, savedVectorType, i };
                    Func<VBuffer<int>, VectorType, int, int[]> func = GetValuesArray<int>;
                    var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(savedVectorType.ItemType.RawType);
                    _repValues[i] = meth.Invoke(this, args);
                }
                else
                    _repValues[i] = repValue;

                Host.Assert(repValue.GetType() == _replaceTypes[i].RawType || repValue.GetType() == _replaceTypes[i].GetItemType().RawType);
            }
        }

        private T[] GetValuesArray<T>(VBuffer<T> src, VectorType srcType, int iinfo)
        {
            Host.Assert(srcType != null);
            Host.Assert(srcType.Size == src.Length);
            VBufferUtils.Densify<T>(ref src);
            InPredicate<T> defaultPred = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(srcType.ItemType);
            _repIsDefault[iinfo] = new BitArray(srcType.Size);
            var srcValues = src.GetValues();
            for (int slot = 0; slot < srcValues.Length; slot++)
            {
                if (defaultPred(in srcValues[slot]))
                    _repIsDefault[iinfo][slot] = true;
            }
            // copy the result array out. Copying is OK because this method is only called on model load.
            T[] valReturn = srcValues.ToArray();
            Host.Assert(valReturn.Length == src.Length);
            return valReturn;
        }

        /// <summary>
        /// Fill the repValues array with the correct replacement values based on the user-given replacement kinds.
        /// Vectors default to by-slot imputation unless otherwise specified, except for unknown sized vectors
        /// which force across-slot imputation.
        /// </summary>
        private void GetReplacementValues(IDataView input, MissingValueReplacingEstimator.ColumnOptions[] columns, out object[] repValues, out BitArray[] slotIsDefault, out DataViewType[] types)
        {
            repValues = new object[columns.Length];
            slotIsDefault = new BitArray[columns.Length];
            types = new DataViewType[columns.Length];
            var sources = new int[columns.Length];
            ReplacementKind[] imputationModes = new ReplacementKind[columns.Length];

            List<int> columnsToImpute = null;
            // REVIEW: Would like to get rid of the sourceColumns list but seems to be the best way to provide
            // the cursor with what columns to cursor through.
            var sourceColumns = new List<DataViewSchema.Column>();
            for (int iinfo = 0; iinfo < columns.Length; iinfo++)
            {
                input.Schema.TryGetColumnIndex(columns[iinfo].InputColumnName, out int colSrc);
                sources[iinfo] = colSrc;
                var type = input.Schema[colSrc].Type;
                if (type is VectorType vectorType)
                    type = new VectorType(vectorType.ItemType, vectorType);
                Delegate isNa = GetIsNADelegate(type);
                types[iinfo] = type;
                var kind = (ReplacementKind)columns[iinfo].Replacement;
                switch (kind)
                {
                    case ReplacementKind.SpecifiedValue:
                        repValues[iinfo] = GetSpecifiedValue(columns[iinfo].ReplacementString, _replaceTypes[iinfo], isNa);
                        break;
                    case ReplacementKind.DefaultValue:
                        repValues[iinfo] = GetDefault(type);
                        break;
                    case ReplacementKind.Mean:
                    case ReplacementKind.Minimum:
                    case ReplacementKind.Maximum:
                        if (!(type.GetItemType() is NumberDataViewType))
                            throw Host.Except("Cannot perform mean imputations on non-numeric '{0}'", type.GetItemType());
                        imputationModes[iinfo] = kind;
                        Utils.Add(ref columnsToImpute, iinfo);
                        sourceColumns.Add(input.Schema[colSrc]);
                        break;
                    default:
                        Host.Assert(false);
                        throw Host.Except("Internal error, undefined ReplacementKind '{0}' assigned in NAReplaceTransform.", columns[iinfo].Replacement);
                }
            }

            // Exit if there are no columns needing a replacement value imputed.
            if (Utils.Size(columnsToImpute) == 0)
                return;

            // Impute values.
            using (var ch = Host.Start("Computing Statistics"))
            using (var cursor = input.GetRowCursor(sourceColumns))
            {
                StatAggregator[] statAggregators = new StatAggregator[columnsToImpute.Count];
                for (int ii = 0; ii < columnsToImpute.Count; ii++)
                {
                    int iinfo = columnsToImpute[ii];
                    bool bySlot = columns[ii].ImputeBySlot;
                    if (types[iinfo] is VectorType vectorType && !vectorType.IsKnownSize && bySlot)
                    {
                        ch.Warning("By-slot imputation can not be done on variable-length column");
                        bySlot = false;
                    }

                    statAggregators[ii] = CreateStatAggregator(ch, types[iinfo], imputationModes[iinfo], bySlot,
                        cursor, sources[iinfo]);
                }

                while (cursor.MoveNext())
                {
                    for (int ii = 0; ii < statAggregators.Length; ii++)
                        statAggregators[ii].ProcessRow();
                }

                for (int ii = 0; ii < statAggregators.Length; ii++)
                    repValues[columnsToImpute[ii]] = statAggregators[ii].GetStat();
            }

            // Construct the slotIsDefault bit arrays.
            for (int ii = 0; ii < columnsToImpute.Count; ii++)
            {
                int slot = columnsToImpute[ii];
                if (repValues[slot] is Array)
                {
                    Func<DataViewType, int[], BitArray> func = ComputeDefaultSlots<int>;
                    var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(types[slot].GetItemType().RawType);
                    slotIsDefault[slot] = (BitArray)meth.Invoke(this, new object[] { types[slot], repValues[slot] });
                }
            }
        }

        private BitArray ComputeDefaultSlots<T>(DataViewType type, T[] values)
        {
            Host.Assert(values.Length == type.GetVectorSize());
            BitArray defaultSlots = new BitArray(values.Length);
            InPredicate<T> defaultPred = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type.GetItemType());
            for (int slot = 0; slot < values.Length; slot++)
            {
                if (defaultPred(in values[slot]))
                    defaultSlots[slot] = true;
            }
            return defaultSlots;
        }

        private object GetDefault(DataViewType type)
        {
            Func<object> func = GetDefault<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.GetItemType().RawType);
            return meth.Invoke(this, null);
        }

        private object GetDefault<T>()
        {
            return default(T);
        }

        /// <summary>
        /// Returns the isNA predicate for the respective type.
        /// </summary>
        private Delegate GetIsNADelegate(DataViewType type)
        {
            Func<DataViewType, Delegate> func = GetIsNADelegate<int>;
            return Utils.MarshalInvoke(func, type.GetItemType().RawType, type);
        }

        private Delegate GetIsNADelegate<T>(DataViewType type)
            => Data.Conversion.Conversions.Instance.GetIsNAPredicate<T>(type.GetItemType());

        /// <summary>
        /// Converts a string to its respective value in the corresponding type.
        /// </summary>
        private object GetSpecifiedValue(string srcStr, DataViewType dstType, Delegate isNA)
        {
            Func<string, DataViewType, InPredicate<int>, object> func = GetSpecifiedValue<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(dstType.GetItemType().RawType);
            return meth.Invoke(this, new object[] { srcStr, dstType, isNA });
        }

        private object GetSpecifiedValue<T>(string srcStr, DataViewType dstType, InPredicate<T> isNA)
        {
            var val = default(T);
            if (!string.IsNullOrEmpty(srcStr))
            {
                // Handles converting input strings to correct types.
                var srcTxt = srcStr.AsMemory();
                var strToT = Data.Conversion.Conversions.Instance.GetStandardConversion<ReadOnlyMemory<char>, T>(TextDataViewType.Instance, dstType.GetItemType(), out bool identity);
                strToT(in srcTxt, ref val);
                // Make sure that the srcTxt can legitimately be converted to dstType, throw error otherwise.
                if (isNA(in val))
                    throw Contracts.Except("No conversion of '{0}' to '{1}'", srcStr, dstType.GetItemType());
            }

            return val;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new MissingValueReplacingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                var kind = item.Kind ?? options.ReplacementKind;
                if (!Enum.IsDefined(typeof(ReplacementKind), kind))
                    throw env.ExceptUserArg(nameof(options.ReplacementKind), "Undefined sorting criteria '{0}' detected for column '{1}'", kind, item.Name);

                cols[i] = new MissingValueReplacingEstimator.ColumnOptions(
                    item.Name,
                    item.Source,
                    (MissingValueReplacingEstimator.ReplacementMode)(item.Kind ?? options.ReplacementKind),
                    item.Slot ?? options.ImputeBySlot,
                    item.ReplacementString);
            };
            return new MissingValueReplacingTransformer(env, input, cols).MakeDataTransform(input);
        }

        internal static IDataTransform Create(IHostEnvironment env, IDataView input, params MissingValueReplacingEstimator.ColumnOptions[] columns)
        {
            return new MissingValueReplacingTransformer(env, input, columns).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static MissingValueReplacingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoadName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new MissingValueReplacingTransformer(host, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private VBuffer<T> CreateVBuffer<T>(T[] array)
        {
            Host.AssertValue(array);
            return new VBuffer<T>(array.Length, array);
        }

        private void WriteTypeAndValue<T>(Stream stream, BinarySaver saver, DataViewType type, T rep)
        {
            Host.AssertValue(stream);
            Host.AssertValue(saver);
            Host.Assert(type.RawType == typeof(T) || type.GetItemType().RawType == typeof(T));

            if (!saver.TryWriteTypeAndValue<T>(stream, type, ref rep, out int bytesWritten))
                throw Host.Except("We do not know how to serialize terms of type '{0}'", type);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);
            var saver = new BinarySaver(Host, new BinarySaver.Arguments());
            for (int iinfo = 0; iinfo < _replaceTypes.Length; iinfo++)
            {
                var repValue = _repValues[iinfo];
                var repType = _replaceTypes[iinfo].GetItemType();
                if (_repIsDefault[iinfo] != null)
                {
                    Host.Assert(repValue is Array);
                    Func<int[], VBuffer<int>> function = CreateVBuffer<int>;
                    var method = function.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(repType.RawType);
                    repValue = method.Invoke(this, new object[] { _repValues[iinfo] });
                    repType = _replaceTypes[iinfo];
                }
                Host.Assert(!(repValue is Array));
                object[] args = new object[] { ctx.Writer.BaseStream, saver, repType, repValue };
                Action<Stream, BinarySaver, DataViewType, int> func = WriteTypeAndValue<int>;
                Host.Assert(repValue.GetType() == _replaceTypes[iinfo].RawType || repValue.GetType() == _replaceTypes[iinfo].GetItemType().RawType);
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(repValue.GetType());
                meth.Invoke(this, args);
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string InputColumnName;
                public readonly DataViewType TypeSrc;

                public ColInfo(string outputColumnName, string inputColumnName, DataViewType type)
                {
                    Name = outputColumnName;
                    InputColumnName = inputColumnName;
                    TypeSrc = type;
                }
            }

            private readonly MissingValueReplacingTransformer _parent;
            private readonly ColInfo[] _infos;
            private readonly DataViewType[] _types;
            // The isNA delegates, parallel to Infos.
            private readonly Delegate[] _isNAs;
            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public Mapper(MissingValueReplacingTransformer parent, DataViewSchema inputSchema)
             : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _isNAs = new Delegate[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var type = _infos[i].TypeSrc;
                    VectorType vectorType = type as VectorType;
                    if (vectorType != null)
                    {
                        vectorType = new VectorType(vectorType.ItemType, vectorType);
                        type = vectorType;
                    }
                    var repType = _parent._repIsDefault[i] != null ? _parent._replaceTypes[i] : _parent._replaceTypes[i].GetItemType();
                    if (!type.GetItemType().Equals(repType.GetItemType()))
                        throw Host.ExceptParam(nameof(InputSchema), "Column '{0}' item type '{1}' does not match expected ColumnType of '{2}'",
                            _infos[i].InputColumnName, _parent._replaceTypes[i].GetItemType().ToString(), _infos[i].TypeSrc);
                    // If type is a vector and the value is not either a scalar or a vector of the same size, throw an error.
                    if (repType is VectorType repVectorType)
                    {
                        if (vectorType == null)
                            throw Host.ExceptParam(nameof(inputSchema), "Column '{0}' item type '{1}' cannot be a vector when Columntype is a scalar of type '{2}'",
                                _infos[i].InputColumnName, repType, type);
                        if (!vectorType.IsKnownSize)
                            throw Host.ExceptParam(nameof(inputSchema), "Column '{0}' is unknown size vector '{1}' must be a scalar instead of type '{2}'", _infos[i].InputColumnName, type, parent._replaceTypes[i]);
                        if (vectorType.Size != repVectorType.Size)
                            throw Host.ExceptParam(nameof(inputSchema), "Column '{0}' item type '{1}' must be a scalar or a vector of the same size as Columntype '{2}'",
                                 _infos[i].InputColumnName, repType, type);
                    }
                    _types[i] = type;
                    _isNAs[i] = _parent.GetIsNADelegate(type);
                }
            }

            private ColInfo[] CreateInfos(DataViewSchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    var type = inputSchema[colSrc].Type;
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].outputColumnName, _parent.ColumnPairs[i].inputColumnName, type);
                }
                return infos;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new DataViewSchema.Annotations.Builder();
                    builder.Add(InputSchema[colIndex].Annotations, x => x == AnnotationUtils.Kinds.SlotNames || x == AnnotationUtils.Kinds.IsNormalized);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], builder.ToAnnotations());
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                if (!(_infos[iinfo].TypeSrc is VectorType))
                    return ComposeGetterOne(input, iinfo);
                return ComposeGetterVec(input, iinfo);
            }

            /// <summary>
            /// Getter generator for single valued inputs.
            /// </summary>
            private Delegate ComposeGetterOne(DataViewRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterOne<int>, _infos[iinfo].TypeSrc.RawType, input, iinfo);

            /// <summary>
            ///  Replaces NA values for scalars.
            /// </summary>
            private Delegate ComposeGetterOne<T>(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<T>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(T);
                var isNA = (InPredicate<T>)_isNAs[iinfo];
                Host.Assert(_parent._repValues[iinfo] is T);
                T rep = (T)_parent._repValues[iinfo];
                ValueGetter<T> getter;

                return getter =
                    (ref T dst) =>
                    {
                        getSrc(ref src);
                        dst = isNA(in src) ? rep : src;
                    };
            }

            /// <summary>
            /// Getter generator for vector valued inputs.
            /// </summary>
            private Delegate ComposeGetterVec(DataViewRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterVec<int>, _infos[iinfo].TypeSrc.GetItemType().RawType, input, iinfo);

            /// <summary>
            ///  Replaces NA values for vectors.
            /// </summary>
            private Delegate ComposeGetterVec<T>(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<T>>(input.Schema[ColMapNewToOld[iinfo]]);
                var isNA = (InPredicate<T>)_isNAs[iinfo];
                var isDefault = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(_infos[iinfo].TypeSrc.GetItemType());

                var src = default(VBuffer<T>);
                ValueGetter<VBuffer<T>> getter;

                if (_parent._repIsDefault[iinfo] == null)
                {
                    // One replacement value for all slots.
                    Host.Assert(_parent._repValues[iinfo] is T);
                    T rep = (T)_parent._repValues[iinfo];
                    bool repIsDefault = isDefault(in rep);
                    return getter =
                        (ref VBuffer<T> dst) =>
                        {
                            getSrc(ref src);
                            FillValues(in src, ref dst, isNA, rep, repIsDefault);
                        };
                }

                // Replacement values by slot.
                Host.Assert(_parent._repValues[iinfo] is T[]);
                // The replacement array.
                T[] repArray = (T[])_parent._repValues[iinfo];

                return getter =
                    (ref VBuffer<T> dst) =>
                    {
                        getSrc(ref src);
                        Host.Check(src.Length == repArray.Length);
                        FillValues(in src, ref dst, isNA, repArray, _parent._repIsDefault[iinfo]);
                    };
            }

            /// <summary>
            ///  Fills values for vectors where there is one replacement value.
            /// </summary>
            private void FillValues<T>(in VBuffer<T> src, ref VBuffer<T> dst, InPredicate<T> isNA, T rep, bool repIsDefault)
            {
                Host.AssertValue(isNA);

                int srcSize = src.Length;
                var srcValues = src.GetValues();
                int srcCount = srcValues.Length;

                // REVIEW: One thing that changing the code to simply ensure that there are srcCount indices in the arrays
                // does is over-allocate space if the replacement value is the default value in a dataset with a
                // signficiant amount of NA values -- is it worth handling allocation of memory for this case?
                var dstEditor = VBufferEditor.Create(ref dst, srcSize, srcCount);

                int iivDst = 0;
                if (src.IsDense)
                {
                    // The source vector is dense.
                    Host.Assert(srcSize == srcCount);

                    for (int ivSrc = 0; ivSrc < srcCount; ivSrc++)
                    {
                        var srcVal = srcValues[ivSrc];

                        // The output for dense inputs is always dense.
                        // Note: Theoretically, one could imagine a dataset with NA values that one wished to replace with
                        // the default value, resulting in more than half of the indices being the default value.
                        // In this case, changing the dst vector to be sparse would be more memory efficient -- the current decision
                        // is it is not worth handling this case at the expense of running checks that will almost always not be triggered.
                        dstEditor.Values[ivSrc] = isNA(in srcVal) ? rep : srcVal;
                    }
                    iivDst = srcCount;
                }
                else
                {
                    // The source vector is sparse.
                    Host.Assert(srcCount < srcSize);
                    var srcIndices = src.GetIndices();

                    // Note: ivPrev is only used for asserts.
                    int ivPrev = -1;
                    for (int iivSrc = 0; iivSrc < srcCount; iivSrc++)
                    {
                        Host.Assert(iivDst <= iivSrc);
                        var srcVal = srcValues[iivSrc];
                        int iv = srcIndices[iivSrc];
                        Host.Assert(ivPrev < iv & iv < srcSize);
                        ivPrev = iv;

                        if (!isNA(in srcVal))
                        {
                            dstEditor.Values[iivDst] = srcVal;
                            dstEditor.Indices[iivDst++] = iv;
                        }
                        else if (!repIsDefault)
                        {
                            // Allow for further sparsification.
                            dstEditor.Values[iivDst] = rep;
                            dstEditor.Indices[iivDst++] = iv;
                        }
                    }
                    Host.Assert(iivDst <= srcCount);
                }
                Host.Assert(0 <= iivDst);
                Host.Assert(repIsDefault || iivDst == srcCount);
                dst = dstEditor.CommitTruncated(iivDst);
            }

            /// <summary>
            ///  Fills values for vectors where there is slot-wise replacement values.
            /// </summary>
            private void FillValues<T>(in VBuffer<T> src, ref VBuffer<T> dst, InPredicate<T> isNA, T[] rep, BitArray repIsDefault)
            {
                Host.AssertValue(rep);
                Host.Assert(rep.Length == src.Length);
                Host.AssertValue(repIsDefault);
                Host.Assert(repIsDefault.Length == src.Length);
                Host.AssertValue(isNA);

                int srcSize = src.Length;
                var srcValues = src.GetValues();
                int srcCount = srcValues.Length;

                // REVIEW: One thing that changing the code to simply ensure that there are srcCount indices in the arrays
                // does is over-allocate space if the replacement value is the default value in a dataset with a
                // signficiant amount of NA values -- is it worth handling allocation of memory for this case?
                var dstEditor = VBufferEditor.Create(ref dst, srcSize, srcCount);

                int iivDst = 0;
                if (src.IsDense)
                {
                    // The source vector is dense.
                    Host.Assert(srcSize == srcCount);

                    for (int ivSrc = 0; ivSrc < srcCount; ivSrc++)
                    {
                        var srcVal = srcValues[ivSrc];

                        // The output for dense inputs is always dense.
                        // Note: Theoretically, one could imagine a dataset with NA values that one wished to replace with
                        // the default value, resulting in more than half of the indices being the default value.
                        // In this case, changing the dst vector to be sparse would be more memory efficient -- the current decision
                        // is it is not worth handling this case at the expense of running checks that will almost always not be triggered.
                        dstEditor.Values[ivSrc] = isNA(in srcVal) ? rep[ivSrc] : srcVal;
                    }
                    iivDst = srcCount;
                }
                else
                {
                    // The source vector is sparse.
                    Host.Assert(srcCount < srcSize);
                    var srcIndices = src.GetIndices();

                    // Note: ivPrev is only used for asserts.
                    int ivPrev = -1;
                    for (int iivSrc = 0; iivSrc < srcCount; iivSrc++)
                    {
                        Host.Assert(iivDst <= iivSrc);
                        var srcVal = srcValues[iivSrc];
                        int iv = srcIndices[iivSrc];
                        Host.Assert(ivPrev < iv & iv < srcSize);
                        ivPrev = iv;

                        if (!isNA(in srcVal))
                        {
                            dstEditor.Values[iivDst] = srcVal;
                            dstEditor.Indices[iivDst++] = iv;
                        }
                        else if (!repIsDefault[iv])
                        {
                            // Allow for further sparsification.
                            dstEditor.Values[iivDst] = rep[iv];
                            dstEditor.Indices[iivDst++] = iv;
                        }
                    }
                    Host.Assert(iivDst <= srcCount);
                }
                Host.Assert(0 <= iivDst);
                dst = dstEditor.CommitTruncated(iivDst);
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    ColInfo info = _infos[iinfo];
                    string inputColumnName = info.InputColumnName;
                    if (!ctx.ContainsColumn(inputColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(inputColumnName),
                        ctx.AddIntermediateVariable(_parent._replaceTypes[iinfo], info.Name)))
                    {
                        ctx.RemoveColumn(info.Name, true);
                    }
                }
            }

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
            {
                Type rawType;
                var type = _infos[iinfo].TypeSrc;
                if (type is VectorType vectorType)
                    rawType = vectorType.ItemType.RawType;
                else
                    rawType = type.RawType;

                if (rawType != typeof(float))
                    return false;

                string opType = "Imputer";
                var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                node.AddAttribute("replaced_value_float", Single.NaN);

                if (!(_infos[iinfo].TypeSrc is VectorType))
                    node.AddAttribute("imputed_value_floats", Enumerable.Repeat((float)_parent._repValues[iinfo], 1));
                else
                {
                    if (_parent._repIsDefault[iinfo] != null)
                        node.AddAttribute("imputed_value_floats", (float[])_parent._repValues[iinfo]);
                    else
                        node.AddAttribute("imputed_value_floats", Enumerable.Repeat((float)_parent._repValues[iinfo], 1));
                }
                return true;
            }
        }
    }

    public sealed class MissingValueReplacingEstimator : IEstimator<MissingValueReplacingTransformer>
    {
        /// <summary>
        /// The possible ways to replace missing values.
        /// </summary>
        public enum ReplacementMode : byte
        {
            /// <summary>
            /// Replace with the default value of the column based on its type. For example, 'zero' for numeric and 'empty' for string/text columns.
            /// </summary>
            DefaultValue = 0,
            /// <summary>
            /// Replace with the mean value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Mean = 1,
            /// <summary>
            /// Replace with the minimum value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Minimum = 2,
            /// <summary>
            /// Replace with the maximum value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Maximum = 3,
        }

        [BestFriend]
        internal static class Defaults
        {
            public const ReplacementMode Mode = ReplacementMode.DefaultValue;
            public const bool ImputeBySlot = true;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary> Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;
            /// <summary> Name of column to transform. </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// If true, per-slot imputation of replacement is performed.
            /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
            /// where imputation is always for the entire column.
            /// </summary>
            public readonly bool ImputeBySlot;
            /// <summary> How to replace the missing values.</summary>
            public readonly ReplacementMode Replacement;
            /// <summary> Replacement value for missing values (only used in entrypoing and command line API).</summary>
            internal readonly string ReplacementString;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="replacementMode">How to replace the missing values.</param>
            /// <param name="imputeBySlot">If true, per-slot imputation of replacement is performed.
            /// Otherwise, replacement value is imputed for the entire vector column. This setting is ignored for scalars and variable vectors,
            /// where imputation is always for the entire column.</param>
            public ColumnOptions(string name, string inputColumnName = null, ReplacementMode replacementMode = Defaults.Mode,
                bool imputeBySlot = Defaults.ImputeBySlot)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));
                Name = name;
                InputColumnName = inputColumnName ?? name;
                ImputeBySlot = imputeBySlot;
                Replacement = replacementMode;
            }

            /// <summary>
            /// This constructor is used internally to convert from <see cref="MissingValueReplacingTransformer.Options"/> to <see cref="ColumnOptions"/>
            /// as we support <paramref name="replacementString"/> in command line and entrypoint API only.
            /// </summary>
            internal ColumnOptions(string name, string inputColumnName, ReplacementMode replacementMode, bool imputeBySlot, string replacementString)
                : this(name, inputColumnName, replacementMode, imputeBySlot)
            {
                ReplacementString = replacementString;
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        internal MissingValueReplacingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, ReplacementMode replacementKind = Defaults.Mode)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, replacementKind))
        {

        }

        [BestFriend]
        internal MissingValueReplacingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(MissingValueReplacingEstimator));
            _columns = columns;
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
                string reason = MissingValueReplacingTransformer.TestType(col.ItemType);
                if (reason != null)
                    throw _host.ExceptParam(nameof(inputSchema), reason);
                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.IsNormalized, out var normalized))
                    metadata.Add(normalized);
                var type = !(col.ItemType is VectorType vectorType) ?
                    col.ItemType :
                    new VectorType(vectorType.ItemType, vectorType);
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, col.Kind, type, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="MissingValueReplacingTransformer"/>.
        /// </summary>
        public MissingValueReplacingTransformer Fit(IDataView input) => new MissingValueReplacingTransformer(_host, input, _columns);
    }
}
