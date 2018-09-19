// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;

[assembly: LoadableClass(typeof(NAReplaceTransform), typeof(NAReplaceTransform.Arguments), typeof(SignatureDataTransform),
   NAReplaceTransform.FriendlyName, NAReplaceTransform.LoadName, "NAReplace", NAReplaceTransform.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(typeof(NAReplaceTransform), null, typeof(SignatureLoadDataTransform),
    NAReplaceTransform.FriendlyName, NAReplaceTransform.LoadName)]

namespace Microsoft.ML.Runtime.Data
{
    // This transform can transform either scalars or vectors (both fixed and variable size),
    // creating output columns that are identical to the input columns except for replacing NA values
    // with either the default value, user input, or imputed values (min/max/mean are currently supported).
    // Imputation modes are supported for vectors both by slot and across all slots.
    // REVIEW: May make sense to implement the transform template interface.
    /// <include file='doc.xml' path='doc/members/member[@name="NAReplace"]/*' />
    public sealed partial class NAReplaceTransform : OneToOneTransformBase
    {
        public enum ReplacementKind
        {
            // REVIEW: What should the full list of options for this transform be?
            DefaultValue,
            Mean,
            Minimum,
            Maximum,
            SpecifiedValue,

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
        public sealed class Column : OneToOneColumn
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

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                // We accept N:R:S where N is the new column name, R is the replacement string,
                // and S is source column names.
                return base.TryParse(str, out ReplacementString);
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Kind != null || Slot != null)
                    return false;
                if (ReplacementString == null)
                    return TryUnparseCore(sb);

                return TryUnparseCore(sb, ReplacementString);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:rep:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The replacement method to utilize", ShortName = "kind")]
            public ReplacementKind ReplacementKind = ReplacementKind.DefaultValue;

            // Specifying by-slot imputation for vectors of unknown size will cause a warning, and the imputation will be global.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to impute values by slot", ShortName = "slot")]
            public bool ImputeBySlot = true;
        }

        public const string LoadName = "NAReplaceTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                // REVIEW: temporary name
                modelSignature: "NAREP TF",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x0010002, // Added imputation methods.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName);
        }

        internal const string Summary = "Create an output column of the same type and size of the input column, where missing values "
            + "are replaced with either the default value or the mean/min/max value (for non-text columns only).";

        internal const string FriendlyName = "NA Replace Transform";
        internal const string ShortName = "NARep";

        private static string TestType(ColumnType type)
        {
            // Item type must have an NA value that exists and is not equal to its default value.
            Func<ColumnType, string> func = TestType<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
            return (string)meth.Invoke(null, new object[] { type.ItemType });
        }

        private static string TestType<T>(ColumnType type)
        {
            Contracts.Assert(type.ItemType.RawType == typeof(T));
            RefPredicate<T> isNA;
            if (!Conversions.Instance.TryGetIsNAPredicate(type.ItemType, out isNA))
            {
                return string.Format("Type '{0}' is not supported by {1} since it doesn't have an NA value",
                    type, LoadName);
            }
            var t = default(T);
            if (isNA(ref t))
            {
                // REVIEW: Key values will be handled in a "new key value" transform.
                return string.Format("Type '{0}' is not supported by {1} since its NA value is equivalent to its default value",
                    type, LoadName);
            }
            return null;
        }

        // The output column types, parallel to Infos.
        private readonly ColumnType[] _types;

        // The replacementValues for the columns, parallel to Infos.
        // The elements of this array can be either primitive values or arrays of primitive values. When replacing a scalar valued column in Infos,
        // this array will hold a primitive value. When replacing a vector valued column in Infos, this array will either hold a primitive
        // value, indicating that NAs in all slots will be replaced with this value, or an array of primitives holding the value that each slot
        // will have its NA values replaced with respectively. The case with an array of primitives can only occur when dealing with a
        // vector of known size.
        private readonly object[] _repValues;

        // Marks if the replacement values in given slots of _repValues are the default value.
        // REVIEW: Currently these arrays are constructed on load but could be changed to being constructed lazily.
        private BitArray[] _repIsDefault;

        // The isNA delegates, parallel to Infos.
        private readonly Delegate[] _isNAs;

        public override bool CanSaveOnnx(OnnxContext ctx) => true;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="replacementKind">The replacement method to utilize.</param>
        public NAReplaceTransform(IHostEnvironment env, IDataView input, string name, string source = null, ReplacementKind replacementKind = ReplacementKind.DefaultValue)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, ReplacementKind = replacementKind }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public NAReplaceTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, LoadName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestType)
        {
            Host.CheckValue(args, nameof(args));
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            GetInfoAndMetadata(out _types, out _isNAs);
            GetReplacementValues(args, out _repValues, out _repIsDefault);
        }

        private NAReplaceTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestType)
        {
            Host.AssertValue(ctx);
            Host.AssertNonEmpty(Infos);

            GetInfoAndMetadata(out _types, out _isNAs);

            // *** Binary format ***
            // <base>
            // for each column:
            //   type and value
            _repValues = new object[Infos.Length];
            _repIsDefault = new BitArray[Infos.Length];
            var saver = new BinarySaver(Host, new BinarySaver.Arguments());
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                object repValue;
                ColumnType repType;
                if (!saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out repType, out repValue))
                    throw Host.ExceptDecode();
                if (!_types[iinfo].ItemType.Equals(repType.ItemType))
                    throw Host.ExceptParam(nameof(input), "Decoded serialization of type '{0}' does not match expected ColumnType of '{1}'", repType.ItemType, _types[iinfo].ItemType);
                // If type is a vector and the value is not either a scalar or a vector of the same size, throw an error.
                if (repType.IsVector)
                {
                    if (!_types[iinfo].IsVector)
                        throw Host.ExceptParam(nameof(input), "Decoded serialization of type '{0}' cannot be a vector when Columntype is a scalar of type '{1}'", repType, _types[iinfo]);
                    if (!_types[iinfo].IsKnownSizeVector)
                        throw Host.ExceptParam(nameof(input), "Decoded serialization for unknown size vector '{0}' must be a scalar instead of type '{1}'", _types[iinfo], repType);
                    if (_types[iinfo].VectorSize != repType.VectorSize)
                    {
                        throw Host.ExceptParam(nameof(input), "Decoded serialization of type '{0}' must be a scalar or a vector of the same size as Columntype '{1}'",
                            repType, _types[iinfo]);
                    }

                    // REVIEW: The current implementation takes the serialized VBuffer, densifies it, and stores the values array.
                    // It might be of value to consider storing the VBUffer in order to possibly benefit from sparsity. However, this would
                    // necessitate a reimplementation of the FillValues code to accomodate sparse VBuffers.
                    object[] args = new object[] { repValue, _types[iinfo], iinfo };
                    Func<VBuffer<int>, ColumnType, int, int[]> func = GetValuesArray<int>;
                    var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(repType.ItemType.RawType);
                    _repValues[iinfo] = meth.Invoke(this, args);
                }
                else
                    _repValues[iinfo] = repValue;

                Host.Assert(repValue.GetType() == _types[iinfo].RawType || repValue.GetType() == _types[iinfo].ItemType.RawType);
            }
        }

        private T[] GetValuesArray<T>(VBuffer<T> src, ColumnType srcType, int iinfo)
        {
            Host.Assert(srcType.IsVector);
            Host.Assert(srcType.VectorSize == src.Length);
            VBufferUtils.Densify<T>(ref src);
            RefPredicate<T> defaultPred = Conversions.Instance.GetIsDefaultPredicate<T>(srcType.ItemType);
            _repIsDefault[iinfo] = new BitArray(srcType.VectorSize);
            for (int slot = 0; slot < src.Length; slot++)
            {
                if (defaultPred(ref src.Values[slot]))
                    _repIsDefault[iinfo][slot] = true;
            }
            T[] valReturn = src.Values;
            Array.Resize<T>(ref valReturn, srcType.VectorSize);
            Host.Assert(valReturn.Length == src.Length);
            return valReturn;
        }

        public static NAReplaceTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoadName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NAReplaceTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each column:
            //   type and value
            SaveBase(ctx);
            var saver = new BinarySaver(Host, new BinarySaver.Arguments());
            for (int iinfo = 0; iinfo < _types.Length; iinfo++)
            {
                var repValue = _repValues[iinfo];
                var repType = _types[iinfo].ItemType;
                if (_repIsDefault[iinfo] != null)
                {
                    Host.Assert(repValue is Array);
                    Func<int[], VBuffer<int>> function = CreateVBuffer<int>;
                    var method = function.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_types[iinfo].ItemType.RawType);
                    repValue = method.Invoke(this, new object[] { _repValues[iinfo] });
                    repType = _types[iinfo];
                }
                Host.Assert(!(repValue is Array));
                object[] args = new object[] { ctx.Writer.BaseStream, saver, repType, repValue };
                Action<Stream, BinarySaver, ColumnType, int> func = WriteTypeAndValue<int>;
                Host.Assert(repValue.GetType() == _types[iinfo].RawType || repValue.GetType() == _types[iinfo].ItemType.RawType);
                var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(repValue.GetType());
                meth.Invoke(this, args);
            }
        }

        private VBuffer<T> CreateVBuffer<T>(T[] array)
        {
            Host.AssertValue(array);
            return new VBuffer<T>(array.Length, array);
        }

        private void WriteTypeAndValue<T>(Stream stream, BinarySaver saver, ColumnType type, T rep)
        {
            Host.AssertValue(stream);
            Host.AssertValue(saver);
            Host.Assert(type.RawType == typeof(T) || type.ItemType.RawType == typeof(T));

            int bytesWritten;
            if (!saver.TryWriteTypeAndValue<T>(stream, type, ref rep, out bytesWritten))
                throw Host.Except("We do not know how to serialize terms of type '{0}'", type);
        }

        /// <summary>
        /// Fill the repValues array with the correct replacement values based on the user-given replacement kinds.
        /// Vectors default to by-slot imputation unless otherwise specified, except for unknown sized vectors
        /// which force across-slot imputation.
        /// </summary>
        private void GetReplacementValues(Arguments args, out object[] repValues, out BitArray[] slotIsDefault)
        {
            repValues = new object[Infos.Length];
            slotIsDefault = new BitArray[Infos.Length];

            ReplacementKind?[] imputationModes = new ReplacementKind?[Infos.Length];

            List<int> columnsToImpute = null;
            // REVIEW: Would like to get rid of the sourceColumns list but seems to be the best way to provide
            // the cursor with what columns to cursor through.
            HashSet<int> sourceColumns = null;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                ReplacementKind kind = args.Column[iinfo].Kind ?? args.ReplacementKind;
                switch (kind)
                {
                case ReplacementKind.SpecifiedValue:
                    repValues[iinfo] = GetSpecifiedValue(args.Column[iinfo].ReplacementString, _types[iinfo], _isNAs[iinfo]);
                    break;
                case ReplacementKind.DefaultValue:
                    repValues[iinfo] = GetDefault(_types[iinfo]);
                    break;
                case ReplacementKind.Mean:
                case ReplacementKind.Min:
                case ReplacementKind.Max:
                    if (!_types[iinfo].ItemType.IsNumber && !_types[iinfo].ItemType.IsTimeSpan && !_types[iinfo].ItemType.IsDateTime)
                        throw Host.Except("Cannot perform mean imputations on non-numeric '{0}'", _types[iinfo].ItemType);
                    imputationModes[iinfo] = kind;
                    Utils.Add(ref columnsToImpute, iinfo);
                    Utils.Add(ref sourceColumns, Infos[iinfo].Source);
                    break;
                default:
                    Host.Assert(false);
                    throw Host.Except("Internal error, undefined ReplacementKind '{0}' assigned in NAReplaceTransform.", kind);
                }
            }

            // Exit if there are no columns needing a replacement value imputed.
            if (Utils.Size(columnsToImpute) == 0)
                return;

            // Impute values.
            using (var ch = Host.Start("Computing Statistics"))
            using (var cursor = Source.GetRowCursor(sourceColumns.Contains))
            {
                StatAggregator[] statAggregators = new StatAggregator[columnsToImpute.Count];
                for (int ii = 0; ii < columnsToImpute.Count; ii++)
                {
                    int iinfo = columnsToImpute[ii];
                    bool bySlot = args.Column[ii].Slot ?? args.ImputeBySlot;
                    if (_types[iinfo].IsVector && !_types[iinfo].IsKnownSizeVector && bySlot)
                    {
                        ch.Warning("By-slot imputation can not be done on variable-length column");
                        bySlot = false;
                    }
                    statAggregators[ii] = CreateStatAggregator(ch, _types[iinfo], imputationModes[iinfo], bySlot,
                        cursor, Infos[iinfo].Source);
                }

                while (cursor.MoveNext())
                {
                    for (int ii = 0; ii < statAggregators.Length; ii++)
                        statAggregators[ii].ProcessRow();
                }

                for (int ii = 0; ii < statAggregators.Length; ii++)
                    repValues[columnsToImpute[ii]] = statAggregators[ii].GetStat();

                ch.Done();
            }

            // Construct the slotIsDefault bit arrays.
            for (int ii = 0; ii < columnsToImpute.Count; ii++)
            {
                int slot = columnsToImpute[ii];
                if (repValues[slot] is Array)
                {
                    Func<ColumnType, int[], BitArray> func = ComputeDefaultSlots<int>;
                    var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_types[slot].ItemType.RawType);
                    slotIsDefault[slot] = (BitArray)meth.Invoke(this, new object[] { _types[slot], repValues[slot] });
                }
            }
        }

        private BitArray ComputeDefaultSlots<T>(ColumnType type, T[] values)
        {
            Host.Assert(values.Length == type.VectorSize);
            BitArray defaultSlots = new BitArray(values.Length);
            RefPredicate<T> defaultPred = Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);
            for (int slot = 0; slot < values.Length; slot++)
            {
                if (defaultPred(ref values[slot]))
                    defaultSlots[slot] = true;
            }
            return defaultSlots;
        }

        private void GetInfoAndMetadata(out ColumnType[] types, out Delegate[] isNAs)
        {
            var md = Metadata;
            types = new ColumnType[Infos.Length];
            isNAs = new Delegate[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var type = Infos[iinfo].TypeSrc;

                if (!type.IsVector)
                    types[iinfo] = type;
                else
                    types[iinfo] = new VectorType(type.ItemType.AsPrimitive, type.AsVector);

                isNAs[iinfo] = GetIsNADelegate(type);

                // Pass through slot name metadata and normalization data.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source,
                    MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized))
                {
                }
            }
            md.Seal();
        }

        private object GetDefault(ColumnType type)
        {
            Func<object> func = GetDefault<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
            return meth.Invoke(this, null);
        }

        private object GetDefault<T>()
        {
            return default(T);
        }

        /// <summary>
        /// Returns the isNA predicate for the respective type.
        /// </summary>
        private Delegate GetIsNADelegate(ColumnType type)
        {
            Func<ColumnType, Delegate> func = GetIsNADelegate<int>;
            return Utils.MarshalInvoke(func, type.ItemType.RawType, type);
        }

        private Delegate GetIsNADelegate<T>(ColumnType type)
        {
            return Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        /// <summary>
        /// Converts a string to its respective value in the corresponding type.
        /// </summary>
        private object GetSpecifiedValue(string srcStr, ColumnType dstType, Delegate isNA)
        {
            Func<string, ColumnType, RefPredicate<int>, object> func = GetSpecifiedValue<int>;
            var meth = func.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(dstType.ItemType.RawType);
            return meth.Invoke(this, new object[] { srcStr, dstType, isNA });
        }

        private object GetSpecifiedValue<T>(string srcStr, ColumnType dstType, RefPredicate<T> isNA)
        {
            var val = default(T);
            if (!string.IsNullOrEmpty(srcStr))
            {
                // Handles converting input strings to correct types.
                DvText srcTxt = new DvText(srcStr);
                bool identity;
                var strToT = Conversions.Instance.GetStandardConversion<DvText, T>(TextType.Instance, dstType.ItemType, out identity);
                strToT(ref srcTxt, ref val);
                // Make sure that the srcTxt can legitimately be converted to dstType, throw error otherwise.
                if (isNA(ref val))
                    throw Contracts.Except("No conversion of '{0}' to '{1}'", srcStr, dstType.ItemType);
            }

            return val;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            if (!Infos[iinfo].TypeSrc.IsVector)
                return ComposeGetterOne(input, iinfo);
            return ComposeGetterVec(input, iinfo);
        }

        /// <summary>
        /// Getter generator for single valued inputs.
        /// </summary>
        private Delegate ComposeGetterOne(IRow input, int iinfo)
            => Utils.MarshalInvoke(ComposeGetterOne<int>, Infos[iinfo].TypeSrc.RawType, input, iinfo);

        /// <summary>
        ///  Replaces NA values for scalars.
        /// </summary>
        private Delegate ComposeGetterOne<T>(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<T>(input, iinfo);
            var src = default(T);
            var isNA = (RefPredicate<T>)_isNAs[iinfo];
            Host.Assert(_repValues[iinfo] is T);
            T rep = (T)_repValues[iinfo];
            ValueGetter<T> getter;

            return getter =
                (ref T dst) =>
                {
                    getSrc(ref src);
                    dst = isNA(ref src) ? rep : src;
                };
        }

        /// <summary>
        /// Getter generator for vector valued inputs.
        /// </summary>
        private Delegate ComposeGetterVec(IRow input, int iinfo)
            => Utils.MarshalInvoke(ComposeGetterVec<int>, Infos[iinfo].TypeSrc.ItemType.RawType, input, iinfo);

        /// <summary>
        ///  Replaces NA values for vectors.
        /// </summary>
        private Delegate ComposeGetterVec<T>(IRow input, int iinfo)
        {
            var getSrc = GetSrcGetter<VBuffer<T>>(input, iinfo);
            var isNA = (RefPredicate<T>)_isNAs[iinfo];
            var isDefault = Conversions.Instance.GetIsDefaultPredicate<T>(input.Schema.GetColumnType(Infos[iinfo].Source).ItemType);

            var src = default(VBuffer<T>);
            ValueGetter<VBuffer<T>> getter;

            if (_repIsDefault[iinfo] == null)
            {
                // One replacement value for all slots.
                Host.Assert(_repValues[iinfo] is T);
                T rep = (T)_repValues[iinfo];
                bool repIsDefault = isDefault(ref rep);
                return getter =
                    (ref VBuffer<T> dst) =>
                    {
                        getSrc(ref src);
                        FillValues(ref src, ref dst, isNA, rep, repIsDefault);
                    };
            }

            // Replacement values by slot.
            Host.Assert(_repValues[iinfo] is T[]);
            // The replacement array.
            T[] repArray = (T[])_repValues[iinfo];

            return getter =
                (ref VBuffer<T> dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == repArray.Length);
                    FillValues(ref src, ref dst, isNA, repArray, _repIsDefault[iinfo]);
                };
        }

        protected override bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            DataKind rawKind;
            var type = Infos[iinfo].TypeSrc;
            if (type.IsVector)
                rawKind = type.AsVector.ItemType.RawKind;
            else if (type.IsKey)
                rawKind = type.AsKey.RawKind;
            else
                rawKind = type.RawKind;

            if (rawKind != DataKind.R4)
                return false;

            string opType = "Imputer";
            var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
            node.AddAttribute("replaced_value_float", Single.NaN);

            if (!Infos[iinfo].TypeSrc.IsVector)
                node.AddAttribute("imputed_value_floats", Enumerable.Repeat((float)_repValues[iinfo], 1));
            else
            {
                if (_repIsDefault[iinfo] != null)
                    node.AddAttribute("imputed_value_floats", (float[])_repValues[iinfo]);
                else
                    node.AddAttribute("imputed_value_floats", Enumerable.Repeat((float)_repValues[iinfo], 1));
            }

            return true;
        }

        protected override VectorType GetSlotTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return Infos[iinfo].SlotTypeSrc;
        }

        protected override ISlotCursor GetSlotCursorCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.AssertValue(Infos[iinfo].SlotTypeSrc);

            ISlotCursor cursor = InputTranspose.GetSlotCursor(Infos[iinfo].Source);
            var type = GetSlotTypeCore(iinfo);
            Host.AssertValue(type);
            return Utils.MarshalInvoke(GetSlotCursorCore<int>, type.ItemType.RawType, this, iinfo, cursor, type);
        }

        private ISlotCursor GetSlotCursorCore<T>(NAReplaceTransform parent, int iinfo, ISlotCursor cursor, VectorType type)
            => new SlotCursor<T>(parent, iinfo, cursor, type);

        private sealed class SlotCursor<T> : SynchronizedCursorBase<ISlotCursor>, ISlotCursor
        {
            private readonly ValueGetter<VBuffer<T>> _getter;
            private readonly VectorType _type;

            public SlotCursor(NAReplaceTransform parent, int iinfo, ISlotCursor cursor, VectorType type)
                : base(parent.Host, cursor)
            {
                Ch.Assert(0 <= iinfo && iinfo < parent.Infos.Length);
                Ch.AssertValue(cursor);
                Ch.AssertValue(type);
                var srcGetter = cursor.GetGetter<T>();
                _type = type;
                _getter = CreateGetter(parent, iinfo, cursor, type);
            }

            private ValueGetter<VBuffer<T>> CreateGetter(NAReplaceTransform parent, int iinfo, ISlotCursor cursor, VectorType type)
            {
                var src = default(VBuffer<T>);
                ValueGetter<VBuffer<T>> getter;

                var getSrc = cursor.GetGetter<T>();
                var isNA = (RefPredicate<T>)parent._isNAs[iinfo];
                var isDefault = Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);

                if (parent._repIsDefault[iinfo] == null)
                {
                    // One replacement value for all slots.
                    Ch.Assert(parent._repValues[iinfo] is T);
                    T rep = (T)parent._repValues[iinfo];
                    bool repIsDefault = isDefault(ref rep);

                    return (ref VBuffer<T> dst) =>
                    {
                        getSrc(ref src);
                        parent.FillValues(ref src, ref dst, isNA, rep, repIsDefault);
                    };
                }

                // Replacement values by slot.
                Ch.Assert(parent._repValues[iinfo] is T[]);
                // The replacement array.
                T[] repArray = (T[])parent._repValues[iinfo];

                return getter =
                    (ref VBuffer<T> dst) =>
                    {
                        getSrc(ref src);
                        Ch.Check(0 <= Position && Position < repArray.Length);
                        T rep = repArray[(int)Position];
                        parent.FillValues(ref src, ref dst, isNA, rep, isDefault(ref rep));
                    };
            }

            public VectorType GetSlotType() => _type;

            public ValueGetter<VBuffer<TValue>> GetGetter<TValue>()
            {
                ValueGetter<VBuffer<TValue>> getter = _getter as ValueGetter<VBuffer<TValue>>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }
        }

        /// <summary>
        ///  Fills values for vectors where there is one replacement value.
        /// </summary>
        private void FillValues<T>(ref VBuffer<T> src, ref VBuffer<T> dst, RefPredicate<T> isNA, T rep, bool repIsDefault)
        {
            Host.AssertValue(isNA);

            int srcSize = src.Length;
            int srcCount = src.Count;
            var srcValues = src.Values;
            Host.Assert(Utils.Size(srcValues) >= srcCount);
            var srcIndices = src.Indices;

            var dstValues = dst.Values;
            var dstIndices = dst.Indices;

            // If the values array is not large enough, allocate sufficient space.
            // Note: We can't set the max to srcSize as vectors can be of variable lengths.
            Utils.EnsureSize(ref dstValues, srcCount, keepOld: false);

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
                    dstValues[ivSrc] = isNA(ref srcVal) ? rep : srcVal;
                }
                iivDst = srcCount;
            }
            else
            {
                // The source vector is sparse.
                Host.Assert(Utils.Size(srcIndices) >= srcCount);
                Host.Assert(srcCount < srcSize);

                // Allocate more space if necessary.
                // REVIEW: One thing that changing the code to simply ensure that there are srcCount indices in the arrays
                // does is over-allocate space if the replacement value is the default value in a dataset with a
                // signficiant amount of NA values -- is it worth handling allocation of memory for this case?
                Utils.EnsureSize(ref dstIndices, srcCount, keepOld: false);

                // Note: ivPrev is only used for asserts.
                int ivPrev = -1;
                for (int iivSrc = 0; iivSrc < srcCount; iivSrc++)
                {
                    Host.Assert(iivDst <= iivSrc);
                    var srcVal = srcValues[iivSrc];
                    int iv = srcIndices[iivSrc];
                    Host.Assert(ivPrev < iv & iv < srcSize);
                    ivPrev = iv;

                    if (!isNA(ref srcVal))
                    {
                        dstValues[iivDst] = srcVal;
                        dstIndices[iivDst++] = iv;
                    }
                    else if (!repIsDefault)
                    {
                        // Allow for further sparsification.
                        dstValues[iivDst] = rep;
                        dstIndices[iivDst++] = iv;
                    }
                }
                Host.Assert(iivDst <= srcCount);
            }
            Host.Assert(0 <= iivDst);
            Host.Assert(repIsDefault || iivDst == srcCount);
            dst = new VBuffer<T>(srcSize, iivDst, dstValues, dstIndices);
        }

        /// <summary>
        ///  Fills values for vectors where there is slot-wise replacement values.
        /// </summary>
        private void FillValues<T>(ref VBuffer<T> src, ref VBuffer<T> dst, RefPredicate<T> isNA, T[] rep, BitArray repIsDefault)
        {
            Host.AssertValue(rep);
            Host.Assert(rep.Length == src.Length);
            Host.AssertValue(repIsDefault);
            Host.Assert(repIsDefault.Length == src.Length);
            Host.AssertValue(isNA);

            int srcSize = src.Length;
            int srcCount = src.Count;
            var srcValues = src.Values;
            Host.Assert(Utils.Size(srcValues) >= srcCount);
            var srcIndices = src.Indices;

            var dstValues = dst.Values;
            var dstIndices = dst.Indices;

            // If the values array is not large enough, allocate sufficient space.
            Utils.EnsureSize(ref dstValues, srcCount, srcSize, keepOld: false);

            int iivDst = 0;
            Host.Assert(Utils.Size(srcValues) >= srcCount);
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
                    dstValues[ivSrc] = isNA(ref srcVal) ? rep[ivSrc] : srcVal;
                }
                iivDst = srcCount;
            }
            else
            {
                // The source vector is sparse.
                Host.Assert(Utils.Size(srcIndices) >= srcCount);
                Host.Assert(srcCount < srcSize);

                // Allocate more space if necessary.
                // REVIEW: One thing that changing the code to simply ensure that there are srcCount indices in the arrays
                // does is over-allocate space if the replacement value is the default value in a dataset with a
                // signficiant amount of NA values -- is it worth handling allocation of memory for this case?
                Utils.EnsureSize(ref dstIndices, srcCount, srcSize, keepOld: false);

                // Note: ivPrev is only used for asserts.
                int ivPrev = -1;
                for (int iivSrc = 0; iivSrc < srcCount; iivSrc++)
                {
                    Host.Assert(iivDst <= iivSrc);
                    var srcVal = srcValues[iivSrc];
                    int iv = srcIndices[iivSrc];
                    Host.Assert(ivPrev < iv & iv < srcSize);
                    ivPrev = iv;

                    if (!isNA(ref srcVal))
                    {
                        dstValues[iivDst] = srcVal;
                        dstIndices[iivDst++] = iv;
                    }
                    else if (!repIsDefault[iv])
                    {
                        // Allow for further sparsification.
                        dstValues[iivDst] = rep[iv];
                        dstIndices[iivDst++] = iv;
                    }
                }
                Host.Assert(iivDst <= srcCount);
            }
            Host.Assert(0 <= iivDst);
            dst = new VBuffer<T>(srcSize, iivDst, dstValues, dstIndices);
        }
    }
}
