// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(TermLookupTransform.Summary, typeof(TermLookupTransform), typeof(TermLookupTransform.Arguments), typeof(SignatureDataTransform),
    "Term Lookup Transform", "TermLookup", "Lookup", "LookupTransform", "TermLookupTransform")]

[assembly: LoadableClass(TermLookupTransform.Summary, typeof(TermLookupTransform), null, typeof(SignatureLoadDataTransform),
    "Term Lookup Transform", TermLookupTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// This transform maps text values columns to new columns using a map dataset provided through its arguments.
    /// </summary>
    public sealed class TermLookupTransform : OneToOneTransformBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file containing the terms", ShortName = "data", SortOrder = 2)]
            public string DataFile;

            [Argument(ArgumentType.Multiple, HelpText = "The data loader", NullName = "<Auto>", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, IDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the text column containing the terms", ShortName = "term")]
            public string TermColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the column containing the values", ShortName = "value")]
            public string ValueColumn;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "If term and value columns are unspecified, specifies whether the values are key values or numeric.", ShortName = "key")]
            public bool KeyValues = true;
        }

        /// <summary>
        /// Holds the values that the terms map to.
        /// </summary>
        private abstract class ValueMap
        {
            public readonly ColumnType Type;

            protected ValueMap(ColumnType type)
            {
                Type = type;
            }

            public static ValueMap Create(ColumnType type)
            {
                Contracts.AssertValue(type);

                if (!type.IsVector)
                {
                    Func<PrimitiveType, OneValueMap<int>> del = CreatePrimitive<int>;
                    var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
                    return (ValueMap)meth.Invoke(null, new object[] { type });
                }
                else
                {
                    Func<VectorType, VecValueMap<int>> del = CreateVector<int>;
                    var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
                    return (ValueMap)meth.Invoke(null, new object[] { type });
                }
            }

            public static OneValueMap<TVal> CreatePrimitive<TVal>(PrimitiveType type)
            {
                Contracts.AssertValue(type);
                Contracts.Assert(type.RawType == typeof(TVal));
                return new OneValueMap<TVal>(type);
            }

            public static VecValueMap<TVal> CreateVector<TVal>(VectorType type)
            {
                Contracts.AssertValue(type);
                Contracts.Assert(type.ItemType.RawType == typeof(TVal));
                return new VecValueMap<TVal>(type);
            }

            public abstract void Train(IExceptionContext ectx, IRowCursor cursor, int colTerm, int colValue);

            public abstract Delegate GetGetter(ValueGetter<DvText> getSrc);
        }

        /// <summary>
        /// Holds the values that the terms map to - where the destination type is TRes.
        /// </summary>
        private abstract class ValueMap<TRes> : ValueMap
        {
            private NormStr.Pool _terms;
            private TRes[] _values;

            protected ValueMap(ColumnType type)
                : base(type)
            {
                Contracts.Assert(type.RawType == typeof(TRes));
            }

            /// <summary>
            /// Bind this value map to the given cursor for "training".
            /// </summary>
            public override void Train(IExceptionContext ectx, IRowCursor cursor, int colTerm, int colValue)
            {
                Contracts.AssertValue(ectx);
                ectx.Assert(_terms == null);
                ectx.Assert(_values == null);
                ectx.AssertValue(cursor);
                ectx.Assert(0 <= colTerm && colTerm < cursor.Schema.ColumnCount);
                ectx.Assert(cursor.Schema.GetColumnType(colTerm).IsText);
                ectx.Assert(0 <= colValue && colValue < cursor.Schema.ColumnCount);
                ectx.Assert(cursor.Schema.GetColumnType(colValue).Equals(Type));

                var getTerm = cursor.GetGetter<DvText>(colTerm);
                var getValue = cursor.GetGetter<TRes>(colValue);
                var terms = new NormStr.Pool();
                var values = new List<TRes>();

                DvText term = default(DvText);
                while (cursor.MoveNext())
                {
                    getTerm(ref term);
                    // REVIEW: Should we trim?
                    term = term.Trim();
                    // REVIEW: Should we handle mapping "missing" to something?
                    if (term.IsNA)
                        throw ectx.Except("Missing term in lookup data around row: {0}", values.Count);

                    var nstr = term.AddToPool(terms);
                    if (nstr.Id != values.Count)
                        throw ectx.Except("Duplicate term in lookup data: '{0}'", nstr);

                    TRes res = default(TRes);
                    getValue(ref res);
                    values.Add(res);
                    ectx.Assert(terms.Count == values.Count);
                }

                _terms = terms;
                _values = values.ToArray();
                ectx.Assert(_terms.Count == _values.Length);
            }

            /// <summary>
            /// Given the term getter, produce a value getter from this value map.
            /// </summary>
            public override Delegate GetGetter(ValueGetter<DvText> getTerm)
            {
                Contracts.Assert(_terms != null);
                Contracts.Assert(_values != null);
                Contracts.Assert(_terms.Count == _values.Length);

                return GetGetterCore(getTerm);
            }

            private ValueGetter<TRes> GetGetterCore(ValueGetter<DvText> getTerm)
            {
                var src = default(DvText);
                return
                    (ref TRes dst) =>
                    {
                        getTerm(ref src);
                        src = src.Trim();
                        var nstr = src.FindInPool(_terms);
                        if (nstr == null)
                            GetMissing(ref dst);
                        else
                        {
                            Contracts.Assert(0 <= nstr.Id && nstr.Id < _values.Length);
                            CopyValue(ref _values[nstr.Id], ref dst);
                        }
                    };
            }

            protected abstract void GetMissing(ref TRes dst);

            protected abstract void CopyValue(ref TRes src, ref TRes dst);
        }

        /// <summary>
        /// Holds the values that the terms map to when the destination type is a PrimitiveType (non-vector).
        /// </summary>
        private sealed class OneValueMap<TRes> : ValueMap<TRes>
        {
            private readonly TRes _badValue;

            public OneValueMap(PrimitiveType type)
                : base(type)
            {
                // REVIEW: This uses the fact that standard conversions map NA to NA to get the NA for TRes.
                // We should probably have a mapping from type to its bad value somewhere, perhaps in Conversions.
                bool identity;
                ValueMapper<DvText, TRes> conv;
                if (Conversions.Instance.TryGetStandardConversion<DvText, TRes>(TextType.Instance, type,
                    out conv, out identity))
                {
                    var bad = DvText.NA;
                    conv(ref bad, ref _badValue);
                }
            }

            protected override void GetMissing(ref TRes dst)
            {
                dst = _badValue;
            }

            protected override void CopyValue(ref TRes src, ref TRes dst)
            {
                dst = src;
            }
        }

        /// <summary>
        /// Holds the values that the terms map to when the destination type is a VectorType.
        /// TItem is the represtation type for the vector's ItemType.
        /// </summary>
        private sealed class VecValueMap<TItem> : ValueMap<VBuffer<TItem>>
        {
            public VecValueMap(VectorType type)
                : base(type)
            {
            }

            protected override void GetMissing(ref VBuffer<TItem> dst)
            {
                dst = new VBuffer<TItem>(Type.VectorSize, 0, dst.Values, dst.Indices);
            }

            protected override void CopyValue(ref VBuffer<TItem> src, ref VBuffer<TItem> dst)
            {
                src.CopyTo(ref dst);
            }
        }

        public const string LoaderSignature = "TermLookupTransform";

        internal const string Summary = "Maps text values columns to new columns using a map dataset.";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TXTLOOKT",
                // verWrittenCur: 0x00010001, // Initial.
                verWrittenCur: 0x00010002, // Dropped sizeof(Float).
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        // This is the byte array containing the binary .idv file contents for the lookup data.
        // This is persisted; the _termMap and _valueMap are constructed from it.
        private readonly byte[] _bytes;

        // The BinaryLoader over the byte array above. We keep this
        // active simply for metadata requests.
        private readonly BinaryLoader _ldr;

        // The value map.
        private readonly ValueMap _valueMap;

        // Stream names for the binary idv streams.
        private const string DefaultMapName = "DefaultMap.idv";

        private const string RegistrationName = "TextLookup";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public TermLookupTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column,
                input, TestIsText)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            Host.CheckUserArg(!string.IsNullOrWhiteSpace(args.DataFile), nameof(args.DataFile), "must specify dataFile");
            Host.CheckUserArg(string.IsNullOrEmpty(args.TermColumn) == string.IsNullOrEmpty(args.ValueColumn), nameof(args.TermColumn),
                "Either both term and value column should be specified, or neither.");

            using (var ch = Host.Start("Training"))
            {
                _bytes = GetBytes(Host, Infos, args);
                _ldr = GetLoader(Host, _bytes);
                _valueMap = Train(ch, _ldr);
                SetMetadata();
                ch.Done();
            }
        }

        public TermLookupTransform(IHostEnvironment env, IDataView input, IDataView lookup, string sourceTerm, string sourceValue, string targetTerm, string targetValue)
            : base(env, RegistrationName, new[] { new Column { Name = sourceValue, Source = sourceTerm } }, input, TestIsText)
        {
            Host.AssertNonEmpty(Infos);
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(lookup, nameof(lookup));
            Host.Assert(Infos.Length == 1);
            Host.CheckNonEmpty(targetTerm, nameof(targetTerm), "Term column must be specified when passing in a data view as lookup table.");
            Host.CheckNonEmpty(targetValue, nameof(targetValue), "Value column must be specified when passing in a data view as lookup table.");

            using (var ch = Host.Start("Training"))
            {
                _bytes = GetBytesFromDataView(Host, lookup, targetTerm, targetValue);
                _ldr = GetLoader(Host, _bytes);
                _valueMap = Train(ch, _ldr);
                SetMetadata();
                ch.Done();
            }
        }

        // This method is called if only a datafile is specified, without a loader/term and value columns.
        // It determines the type of the Value column and returns the appropriate TextLoader component factory.
        private static IComponentFactory<IMultiStreamSource, IDataLoader> GetLoaderFactory(string filename, bool keyValues, IHost host)
        {
            Contracts.AssertValue(host);

            // If the user specified non-key values, we define the value column to be numeric.
            if (!keyValues)
                return ComponentFactoryUtils.CreateFromFunction<IMultiStreamSource, IDataLoader>(
                    (env, files) => new TextLoader(
                        env,
                        new TextLoader.Arguments()
                        {
                            Column = new[]
                            {
                                new TextLoader.Column("Term", DataKind.TX, 0),
                                new TextLoader.Column("Value", DataKind.Num, 1)
                            }
                        },
                        files));

            // If the user specified key values, we scan the values to determine the range of the key type.
            ulong min = ulong.MaxValue;
            ulong max = ulong.MinValue;
            try
            {
                var txtArgs = new TextLoader.Arguments();
                bool parsed = CmdParser.ParseArguments(host, "col=Term:TX:0 col=Value:TX:1", txtArgs);
                host.Assert(parsed);
                var data = TextLoader.ReadFile(host, txtArgs, new MultiFileSource(filename));
                using (var cursor = data.GetRowCursor(c => true))
                {
                    var getTerm = cursor.GetGetter<DvText>(0);
                    var getVal = cursor.GetGetter<DvText>(1);
                    DvText txt = default(DvText);

                    using (var ch = host.Start("Creating Text Lookup Loader"))
                    {
                        long countNonKeys = 0;
                        while (cursor.MoveNext())
                        {
                            getVal(ref txt);
                            ulong res;
                            // Try to parse the text as a key value between 1 and ulong.MaxValue. If this succeeds and res>0,
                            // we update max and min accordingly. If res==0 it means the value is missing, in which case we ignore it for
                            // computing max and min.
                            if (Conversions.Instance.TryParseKey(ref txt, 1, ulong.MaxValue, out res))
                            {
                                if (res < min && res != 0)
                                    min = res;
                                if (res > max)
                                    max = res;
                            }
                            // If parsing as key did not succeed, the value can still be 0, so we try parsing it as a ulong. If it succeeds,
                            // then the value is 0, and we update min accordingly.
                            else if (Conversions.Instance.TryParse(ref txt, out res))
                            {
                                ch.Assert(res == 0);
                                min = 0;
                            }
                            //If parsing as a ulong fails, we increment the counter for the non-key values.
                            else
                            {
                                var term = default(DvText);
                                getTerm(ref term);
                                if (countNonKeys < 5)
                                    ch.Warning("Term '{0}' in mapping file is mapped to non key value '{1}'", term, txt);
                                countNonKeys++;
                            }
                        }
                        if (countNonKeys > 0)
                            ch.Warning("Found {0} non key values in the file '{1}'", countNonKeys, filename);
                        if (min > max)
                        {
                            min = 0;
                            max = uint.MaxValue - 1;
                            ch.Warning("did not find any valid key values in the file '{0}'", filename);
                        }
                        else
                            ch.Info("Found key values in the range {0} to {1} in the file '{2}'", min, max, filename);
                        ch.Done();
                    }
                }
            }
            catch (Exception e)
            {
                throw host.Except(e, "Failed to parse the lookup file '{0}' in TermLookupTransform", filename);
            }

            TextLoader.Column valueColumn = new TextLoader.Column("Value", DataKind.U4, 1);
            if (max - min < (ulong)int.MaxValue)
            {
                valueColumn.KeyRange = new KeyRange(min, max);
            }
            else if (max - min < (ulong)uint.MaxValue)
            {
                valueColumn.KeyRange = new KeyRange(min);
            }
            else
            {
                valueColumn.Type = DataKind.U8;
                valueColumn.KeyRange = new KeyRange(min);
            }

            return ComponentFactoryUtils.CreateFromFunction<IMultiStreamSource, IDataLoader>(
                   (env, files) => new TextLoader(
                       env,
                       new TextLoader.Arguments()
                       {
                           Column = new[]
                           {
                                new TextLoader.Column("Term", DataKind.TX, 0),
                                valueColumn
                           }
                       },
                       files));
        }

        // This saves the lookup data as a byte array encoded as a binary .idv file.
        private static byte[] GetBytes(IHost host, ColInfo[] infos, Arguments args)
        {
            Contracts.AssertValue(host);
            host.AssertNonEmpty(infos);
            host.AssertValue(args);

            string dataFile = args.DataFile;
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = args.Loader;
            string termColumn;
            string valueColumn;
            if (!string.IsNullOrEmpty(args.TermColumn))
            {
                host.Assert(!string.IsNullOrEmpty(args.ValueColumn));
                termColumn = args.TermColumn;
                valueColumn = args.ValueColumn;
            }
            else
            {
                var ext = Path.GetExtension(dataFile);
                if (loaderFactory != null || string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase))
                    throw host.ExceptUserArg(nameof(args.TermColumn), "Term and value columns needed.");
                loaderFactory = GetLoaderFactory(args.DataFile, args.KeyValues, host);
                termColumn = "Term";
                valueColumn = "Value";
            }
            return GetBytesOne(host, dataFile, loaderFactory, termColumn, valueColumn);
        }

        private static byte[] GetBytesFromDataView(IHost host, IDataView lookup, string termColumn, string valueColumn)
        {
            Contracts.AssertValue(host);
            host.AssertValue(lookup);
            host.AssertNonEmpty(termColumn);
            host.AssertNonEmpty(valueColumn);

            int colTerm;
            int colValue;
            var schema = lookup.Schema;

            if (!schema.TryGetColumnIndex(termColumn, out colTerm))
                throw host.ExceptUserArg(nameof(Arguments.TermColumn), "column not found: '{0}'", termColumn);
            if (!schema.TryGetColumnIndex(valueColumn, out colValue))
                throw host.ExceptUserArg(nameof(Arguments.ValueColumn), "column not found: '{0}'", valueColumn);

            // REVIEW: Should we allow term to be a vector of text (each term in the vector
            // would map to the same value)?
            var typeTerm = schema.GetColumnType(colTerm);
            host.CheckUserArg(typeTerm.IsText, nameof(Arguments.TermColumn), "term column must contain text");
            var typeValue = schema.GetColumnType(colValue);

            var args = new ChooseColumnsTransform.Arguments();
            args.Column = new[] {
                new ChooseColumnsTransform.Column {Name = "Term", Source = termColumn},
                new ChooseColumnsTransform.Column {Name = "Value", Source = valueColumn},
            };
            var view = new ChooseColumnsTransform(host, args, lookup);

            var saver = new BinarySaver(host, new BinarySaver.Arguments());
            using (var strm = new MemoryStream())
            {
                saver.SaveData(strm, view, 0, 1);
                return strm.ToArray();
            }
        }

        private static byte[] GetBytesOne(IHost host, string dataFile, IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory,
            string termColumn, string valueColumn)
        {
            Contracts.AssertValue(host);
            host.Assert(!string.IsNullOrWhiteSpace(dataFile));
            host.AssertNonEmpty(termColumn);
            host.AssertNonEmpty(valueColumn);

            IMultiStreamSource fileSource = new MultiFileSource(dataFile);
            IDataLoader loader;
            if (loaderFactory == null)
            {
                // REVIEW: Should there be defaults for loading from text?
                var ext = Path.GetExtension(dataFile);
                bool isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                bool isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);
                if (!isBinary && !isTranspose)
                    throw host.ExceptUserArg(nameof(Arguments.Loader), "must specify the loader");
                host.Assert(isBinary != isTranspose); // One or the other must be true.
                if (isBinary)
                {
                    loader = new BinaryLoader(host, new BinaryLoader.Arguments(), fileSource);
                }
                else
                {
                    loader = new TransposeLoader(host, new TransposeLoader.Arguments(), fileSource);
                }
            }
            else
            {
                loader = loaderFactory.CreateComponent(host, fileSource);
            }

            return GetBytesFromDataView(host, loader, termColumn, valueColumn);
        }

        private static BinaryLoader GetLoader(IHostEnvironment env, byte[] bytes)
        {
            env.AssertValue(env);
            env.AssertValue(bytes);

            var strm = new MemoryStream(bytes, writable: false);
            return new BinaryLoader(env, new BinaryLoader.Arguments(), strm);
        }

        private static ValueMap Train(IExceptionContext ectx, BinaryLoader ldr)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(ldr);
            ectx.Assert(ldr.Schema.ColumnCount == 2);

            // REVIEW: Should we allow term to be a vector of text (each term in the vector
            // would map to the same value)?
            ectx.Assert(ldr.Schema.GetColumnType(0).IsText);

            var schema = ldr.Schema;
            var typeValue = schema.GetColumnType(1);

            // REVIEW: We should know the number of rows - use that info to set initial capacity.
            var values = ValueMap.Create(typeValue);
            using (var cursor = ldr.GetRowCursor(c => true))
                values.Train(ectx, cursor, 0, 1);
            return values;
        }

        private TermLookupTransform(IChannel ch, ModelLoadContext ctx, IHost host, IDataView input)
            : base(host, ctx, input, TestIsText)
        {
            Host.AssertValue(ch);

            // *** Binary format ***
            // <base>
            ch.AssertNonEmpty(Infos);

            // Extra streams:
            // DefaultMap.idv
            byte[] rgb = null;
            Action<BinaryReader> fn = r => rgb = ReadAllBytes(ch, r);

            if (!ctx.TryLoadBinaryStream(DefaultMapName, fn))
                throw ch.ExceptDecode();
            _bytes = rgb;

            // Process the bytes into the loader and map.
            _ldr = GetLoader(Host, _bytes);
            ValidateLoader(ch, _ldr);
            _valueMap = Train(ch, _ldr);
            SetMetadata();
        }

        private static byte[] ReadAllBytes(IExceptionContext ectx, BinaryReader rdr)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(rdr);
            ectx.Assert(rdr.BaseStream.CanSeek);

            long size = rdr.BaseStream.Length;
            ectx.CheckDecode(size <= int.MaxValue);

            var rgb = new byte[(int)size];
            int cb = rdr.Read(rgb, 0, rgb.Length);
            ectx.CheckDecode(cb == rgb.Length);

            return rgb;
        }

        public static TermLookupTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new TermLookupTransform(ch, ctx, h, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            SaveBase(ctx);

            // Extra streams:
            // DefaultMap.idv
            Host.Assert(_ldr != null);
            Host.AssertValue(_bytes);
            DebugValidateLoader(_ldr);
            ctx.SaveBinaryStream(DefaultMapName, w => w.Write(_bytes));
        }

        [Conditional("DEBUG")]
        private static void DebugValidateLoader(BinaryLoader ldr)
        {
            Contracts.Assert(ldr != null);
            Contracts.Assert(ldr.Schema.ColumnCount == 2);
            Contracts.Assert(ldr.Schema.GetColumnType(0).IsText);
        }

        private static void ValidateLoader(IExceptionContext ectx, BinaryLoader ldr)
        {
            if (ldr == null)
                return;
            ectx.CheckDecode(ldr.Schema.ColumnCount == 2);
            ectx.CheckDecode(ldr.Schema.GetColumnType(0).IsText);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _valueMap.Type;
        }

        private void SetMetadata()
        {
            // Metadata is passed through from the Value column of the map data view.
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                using (var bldr = md.BuildMetadata(iinfo, _ldr.Schema, 1))
                {
                    // No additional metadata.
                }
            }
            md.Seal();
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var getSrc = GetSrcGetter<DvText>(input, iinfo);
            return _valueMap.GetGetter(getSrc);
        }
    }
}
