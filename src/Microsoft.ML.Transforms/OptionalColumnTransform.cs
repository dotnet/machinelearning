// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.DataPipe;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(OptionalColumnTransform.Summary, typeof(OptionalColumnTransform),
    typeof(OptionalColumnTransform.Arguments), typeof(SignatureDataTransform),
    OptionalColumnTransform.UserName, OptionalColumnTransform.LoaderSignature, OptionalColumnTransform.ShortName)]

[assembly: LoadableClass(typeof(OptionalColumnTransform), null, typeof(SignatureLoadDataTransform),
    OptionalColumnTransform.UserName, OptionalColumnTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(OptionalColumnTransform))]

namespace Microsoft.ML.Runtime.DataPipe
{
    /// <include file='doc.xml' path='doc/members/member[@name="OptionalColumnTransform"]/*' />
    public class OptionalColumnTransform : RowToRowMapperTransformBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public string[] Column;
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            public readonly ColumnType[] ColumnTypes;
            public readonly int[] SrcCols;

            private readonly MetadataDispatcher _metadata;
            private readonly OptionalColumnTransform _parent;
            // The input schema of the original data view that contains the source columns. We need this
            // so that we can have the metadata even when we load this transform with new data that does not have
            // these columns.
            private readonly ISchema _inputWithOptionalColumn;
            private readonly int[] _srcColsWithOptionalColumn;

            private Bindings(OptionalColumnTransform parent, ColumnType[] columnTypes, int[] srcCols,
                int[] srcColsWithOptionalColumn, ISchema input, ISchema inputWithOptionalColumn, bool user, string[] names)
                : base(input, user, names)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(Utils.Size(columnTypes) == InfoCount);
                Contracts.Assert(Utils.Size(srcCols) == InfoCount);
                Contracts.AssertValue(inputWithOptionalColumn);
                ColumnTypes = columnTypes;
                SrcCols = srcCols;
                _parent = parent;
                _metadata = new MetadataDispatcher(InfoCount);
                _inputWithOptionalColumn = inputWithOptionalColumn;
                _srcColsWithOptionalColumn = srcColsWithOptionalColumn;
                SetMetadata();
            }

            public static Bindings Create(Arguments args, ISchema input, OptionalColumnTransform parent)
            {
                var names = new string[args.Column.Length];
                var columnTypes = new ColumnType[args.Column.Length];
                var srcCols = new int[args.Column.Length];
                for (int i = 0; i < args.Column.Length; i++)
                {
                    names[i] = args.Column[i];
                    int col;
                    bool success = input.TryGetColumnIndex(names[i], out col);
                    Contracts.CheckUserArg(success, nameof(args.Column));
                    columnTypes[i] = input.GetColumnType(col);
                    srcCols[i] = col;
                }

                return new Bindings(parent, columnTypes, srcCols, srcCols, input, input, true, names);
            }

            public static Bindings Create(IHostEnvironment env, ModelLoadContext ctx, ISchema input, OptionalColumnTransform parent)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(input);

                // *** Binary format ***
                // Schema of the data view containing the optional columns
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   ColumnType: the type of the column

                byte[] buffer = null;
                if (!ctx.TryLoadBinaryStream("Schema.idv", r => buffer = r.ReadByteArray()))
                    throw env.ExceptDecode();
                BinaryLoader loader = null;
                var strm = new MemoryStream(buffer, writable: false);
                loader = new BinaryLoader(env, new BinaryLoader.Arguments(), strm);

                int size = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(size > 0);

                var saver = new BinarySaver(env, new BinarySaver.Arguments());
                var names = new string[size];
                var columnTypes = new ColumnType[size];
                var srcCols = new int[size];
                var srcColsWithOptionalColumn = new int[size];
                for (int i = 0; i < size; i++)
                {
                    names[i] = ctx.LoadNonEmptyString();
                    columnTypes[i] = saver.LoadTypeDescriptionOrNull(ctx.Reader.BaseStream);
                    int col;
                    bool success = input.TryGetColumnIndex(names[i], out col);
                    srcCols[i] = success ? col : -1;

                    success = loader.Schema.TryGetColumnIndex(names[i], out var colWithOptionalColumn);
                    env.CheckDecode(success);
                    srcColsWithOptionalColumn[i] = colWithOptionalColumn;
                }

                return new Bindings(parent, columnTypes, srcCols, srcColsWithOptionalColumn, input, loader.Schema, false, names);
            }

            public void Save(IHostEnvironment env, ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // Schema of the data view containing the optional columns
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   ColumnType: the type of the column

                var noRows = new EmptyDataView(env, _inputWithOptionalColumn);
                var saverArgs = new BinarySaver.Arguments();
                saverArgs.Silent = true;
                var saver = new BinarySaver(env, saverArgs);
                using (var strm = new MemoryStream())
                {
                    saver.SaveData(strm, noRows, _srcColsWithOptionalColumn);
                    ctx.SaveBinaryStream("Schema.idv", w => w.WriteByteArray(strm.ToArray()));
                }

                int size = InfoCount;
                ctx.Writer.Write(size);

                saver = new BinarySaver(env, new BinarySaver.Arguments());
                for (int i = 0; i < size; i++)
                {
                    ctx.SaveNonEmptyString(GetColumnNameCore(i));
                    var columnType = ColumnTypes[i];
                    int written;
                    saver.TryWriteTypeDescription(ctx.Writer.BaseStream, columnType, out written);
                }
            }

            private void SetMetadata()
            {
                var md = _metadata;
                for (int iinfo = 0; iinfo < InfoCount; iinfo++)
                {
                    // Pass through metadata from source columns.
                    using (var bldr = md.BuildMetadata(iinfo, _inputWithOptionalColumn, _srcColsWithOptionalColumn[iinfo]))
                    {
                        // No metadata to add.
                    }
                }
                md.Seal();
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                return ColumnTypes[iinfo];
            }

            protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
            {
                return _metadata.GetMetadataTypes(iinfo);
            }

            protected override ColumnType GetMetadataTypeCore(string kind, int iinfo)
            {
                return _metadata.GetMetadataTypeOrNull(kind, iinfo);
            }

            protected override void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                _metadata.GetMetadata(_parent.Host, kind, iinfo, ref value);
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = GetActiveInput(predicate);
                Contracts.Assert(active.Length == Input.ColumnCount);

                foreach (int srcCol in SrcCols)
                {
                    if (srcCol >= 0)
                        active[srcCol] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        internal const string Summary = "If the source column does not exist after deserialization," +
            " create a column with the right type and default values.";
        internal const string UserName = "Optional Column Transform";
        public const string LoaderSignature = "OptColTransform";
        internal const string ShortName = "optional";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OPTCOL T",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Save the input schema, for metadata
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "OptionalColumn";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columns">Columns to transform.</param>
        public OptionalColumnTransform(IHostEnvironment env, IDataView input, params string[] columns)
            : this(env, new Arguments() { Column = columns }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public OptionalColumnTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            _bindings = Bindings.Create(args, Source.Schema, this);
        }

        private OptionalColumnTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // bindings
            _bindings = Bindings.Create(host, ctx, Source.Schema, this);
        }

        public static OptionalColumnTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new OptionalColumnTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(Host, ctx);
        }

        public override ISchema Schema { get { return _bindings; } }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred);
            return new RowCursor(Host, _bindings, input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            IRowCursor input;

            if (n > 1 && ShouldUseParallelCursors(predicate) != false)
            {
                var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n);
                Host.AssertNonEmpty(inputs);

                if (inputs.Length != 1)
                {
                    var cursors = new IRowCursor[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                        cursors[i] = new RowCursor(Host, _bindings, inputs[i], active);
                    return cursors;
                }
                input = inputs[0];
            }
            else
                input = Source.GetRowCursor(inputPred);

            consolidator = null;
            return new IRowCursor[] { new RowCursor(Host, _bindings, input, active) };
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disposer)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            var getters = new Delegate[_bindings.InfoCount];
            disposer = null;
            using (var ch = Host.Start("CreateGetters"))
            {
                for (int iinfo = 0; iinfo < _bindings.InfoCount; iinfo++)
                {
                    if (!activeInfos(iinfo))
                        continue;
                    if (_bindings.SrcCols[iinfo] < 0)
                        getters[iinfo] = MakeGetter(iinfo);
                    else
                    {
                        Func<IRow, int, ValueGetter<int>> srcDel = GetSrcGetter<int>;
                        var meth = srcDel.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_bindings.ColumnTypes[iinfo].ItemType.RawType);
                        getters[iinfo] = (Delegate)meth.Invoke(this, new object[] { input, iinfo });
                    }
                }
                ch.Done();
                return getters;
            }
        }

        private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo)
        {
            return input.GetGetter<T>(_bindings.SrcCols[iinfo]);
        }

        private Delegate MakeGetter(int iinfo)
        {
            var columnType = _bindings.ColumnTypes[iinfo];
            if (columnType.IsVector)
                return Utils.MarshalInvoke(MakeGetterVec<int>, columnType.ItemType.RawType, columnType.VectorSize);
            return Utils.MarshalInvoke(MakeGetterOne<int>, columnType.RawType);
        }

        private Delegate MakeGetterOne<T>()
        {
            return (ValueGetter<T>)((ref T value) => value = default(T));
        }

        private Delegate MakeGetterVec<T>(int length)
        {
            return (ValueGetter<VBuffer<T>>)((ref VBuffer<T> value) => value = new VBuffer<T>(length, 0, value.Values, value.Indices));
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            public RowCursor(IChannelProvider provider, Bindings bindings, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.CheckValue(bindings, nameof(bindings));
                Ch.CheckValue(input, nameof(input));
                Ch.CheckParam(active == null || active.Length == bindings.ColumnCount, nameof(active));

                _bindings = bindings;
                _active = active;
                var length = _bindings.InfoCount;
                _getters = new Delegate[length];
                for (int iinfo = 0; iinfo < length; iinfo++)
                {
                    if (_bindings.SrcCols[iinfo] < 0)
                        _getters[iinfo] = MakeGetter(iinfo);
                }
            }

            public ISchema Schema { get { return _bindings; } }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                if (_getters[index] == null)
                    return Input.GetGetter<TValue>(_bindings.SrcCols[index]);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private Delegate MakeGetter(int iinfo)
            {
                var columnType = _bindings.ColumnTypes[iinfo];
                if (columnType.IsVector)
                    return Utils.MarshalInvoke(MakeGetterVec<int>, columnType.ItemType.RawType, columnType.VectorSize);
                return Utils.MarshalInvoke(MakeGetterOne<int>, columnType.RawType);
            }

            private Delegate MakeGetterOne<T>()
            {
                return (ValueGetter<T>)((ref T value) => value = default(T));
            }

            private Delegate MakeGetterVec<T>(int length)
            {
                return (ValueGetter<VBuffer<T>>)((ref VBuffer<T> value) => value = new VBuffer<T>(length, 0, value.Values, value.Indices));
            }
        }

        [TlcModule.EntryPoint(Desc = Summary,
            Name = "Transforms.OptionalColumnCreator",
            UserName = UserName,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""OptionalColumnTransform""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""OptionalColumnTransform""]/*' />"})]

        public static CommonOutputs.TransformOutput MakeOptional(IHostEnvironment env, Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "OptionalColumn", input);
            var xf = new OptionalColumnTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
