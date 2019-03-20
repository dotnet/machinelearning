// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(OptionalColumnTransform.Summary, typeof(OptionalColumnTransform),
    typeof(OptionalColumnTransform.Arguments), typeof(SignatureDataTransform),
    OptionalColumnTransform.UserName, OptionalColumnTransform.LoaderSignature, OptionalColumnTransform.ShortName)]

[assembly: LoadableClass(typeof(OptionalColumnTransform), null, typeof(SignatureLoadDataTransform),
    OptionalColumnTransform.UserName, OptionalColumnTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(OptionalColumnTransform))]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="OptionalColumnTransform"]/*' />
    [BestFriend]
    internal sealed class OptionalColumnTransform : RowToRowMapperTransformBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public string[] Columns;
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            public readonly DataViewType[] ColumnTypes;
            public readonly int[] SrcCols;

            private readonly MetadataDispatcher _metadata;
            private readonly OptionalColumnTransform _parent;
            // The input schema of the original data view that contains the source columns. We need this
            // so that we can have the metadata even when we load this transform with new data that does not have
            // these columns.
            private readonly DataViewSchema _inputWithOptionalColumn;
            private readonly int[] _srcColsWithOptionalColumn;

            private Bindings(OptionalColumnTransform parent, DataViewType[] columnTypes, int[] srcCols,
                int[] srcColsWithOptionalColumn, DataViewSchema input, DataViewSchema inputWithOptionalColumn, bool user, string[] names)
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

            public static Bindings Create(Arguments args, DataViewSchema input, OptionalColumnTransform parent)
            {
                var names = new string[args.Columns.Length];
                var columnTypes = new DataViewType[args.Columns.Length];
                var srcCols = new int[args.Columns.Length];
                for (int i = 0; i < args.Columns.Length; i++)
                {
                    names[i] = args.Columns[i];
                    int col;
                    bool success = input.TryGetColumnIndex(names[i], out col);
                    Contracts.CheckUserArg(success, nameof(args.Columns));
                    columnTypes[i] = input[col].Type;
                    srcCols[i] = col;
                }

                return new Bindings(parent, columnTypes, srcCols, srcCols, input, input, true, names);
            }

            public static Bindings Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema input, OptionalColumnTransform parent)
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
                var columnTypes = new DataViewType[size];
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

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < InfoCount);
                return ColumnTypes[iinfo];
            }

            protected override IEnumerable<KeyValuePair<string, DataViewType>> GetAnnotationTypesCore(int iinfo)
            {
                return _metadata.GetMetadataTypes(iinfo);
            }

            protected override DataViewType GetAnnotationTypeCore(string kind, int iinfo)
            {
                return _metadata.GetMetadataTypeOrNull(kind, iinfo);
            }

            protected override void GetAnnotationCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                _metadata.GetMetadata(_parent.Host, kind, iinfo, ref value);
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = GetActiveInput(predicate);
                Contracts.Assert(active.Length == Input.Count);

                foreach (int srcCol in SrcCols)
                {
                    if (srcCol >= 0)
                        active[srcCol] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            public IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                Contracts.AssertValue(dependingColumns);
                var predicate = RowCursorUtils.FromColumnsToPredicate(dependingColumns, AsSchema);
                Func<int, bool> dependencies = GetDependencies(predicate);

                return Input.Where(c => dependencies(c.Index));
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OptionalColumnTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "OptionalColumn";

        /// <summary>
        /// Initializes a new instance of <see cref="OptionalColumnTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columns">Columns to transform.</param>
        public OptionalColumnTransform(IHostEnvironment env, IDataView input, params string[] columns)
            : this(env, new Arguments() { Columns = columns }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public OptionalColumnTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Columns) > 0, nameof(args.Columns));

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

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(Host, ctx);
        }

        public override DataViewSchema OutputSchema => _bindings.AsSchema;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return null;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var input = Source.GetRowCursor(inputCols);
            return new Cursor(Host, _bindings, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var inputPred = _bindings.GetDependencies(predicate);
            var inputCols = Source.Schema.Where(x => inputPred(x.Index));

            var active = _bindings.GetActive(predicate);
            DataViewRowCursor input;

            if (n > 1 && ShouldUseParallelCursors(predicate) != false)
            {
                var inputs = Source.GetRowCursorSet(inputCols, n);
                Host.AssertNonEmpty(inputs);

                if (inputs.Length != 1)
                {
                    var cursors = new DataViewRowCursor[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                        cursors[i] = new Cursor(Host, _bindings, inputs[i], active);
                    return cursors;
                }
                input = inputs[0];
            }
            else
                input = Source.GetRowCursor(inputCols);

            return new DataViewRowCursor[] { new Cursor(Host, _bindings, input, active) };
        }

        protected override IEnumerable<DataViewSchema.Column> GetDependenciesCore(IEnumerable<DataViewSchema.Column> dependingColumns)
            => _bindings.GetDependencies(dependingColumns);

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        protected override Delegate[] CreateGetters(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns, out Action disposer)
        {
            var activeIndices = new HashSet<int>(activeColumns.Select(c => c.Index));
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return activeIndices.Contains(col);
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
                        Func<DataViewRow, int, ValueGetter<int>> srcDel = GetSrcGetter<int>;
                        var meth = srcDel.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(_bindings.ColumnTypes[iinfo].GetItemType().RawType);
                        getters[iinfo] = (Delegate)meth.Invoke(this, new object[] { input, iinfo });
                    }
                }
                return getters;
            }
        }

        private ValueGetter<T> GetSrcGetter<T>(DataViewRow input, int iinfo)
        {
            return input.GetGetter<T>(input.Schema[_bindings.SrcCols[iinfo]]);
        }

        private Delegate MakeGetter(int iinfo)
        {
            var columnType = _bindings.ColumnTypes[iinfo];
            if (columnType is VectorType vectorType)
                return Utils.MarshalInvoke(MakeGetterVec<int>, vectorType.ItemType.RawType, vectorType.Size);
            return Utils.MarshalInvoke(MakeGetterOne<int>, columnType.RawType);
        }

        private Delegate MakeGetterOne<T>()
        {
            return (ValueGetter<T>)((ref T value) => value = default(T));
        }

        private Delegate MakeGetterVec<T>(int length)
        {
            return (ValueGetter<VBuffer<T>>)((ref VBuffer<T> value) =>
                VBufferUtils.Resize(ref value, length, 0));
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            public Cursor(IChannelProvider provider, Bindings bindings, DataViewRowCursor input, bool[] active)
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

            public override DataViewSchema Schema => _bindings.AsSchema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _bindings.ColumnCount);
                return _active == null || _active[column.Index];
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.Check(IsColumnActive(column));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.GetGetter<TValue>(Input.Schema[index]);

                if (_getters[index] == null)
                    return Input.GetGetter<TValue>(_bindings.AsSchema[_bindings.SrcCols[index]]);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private Delegate MakeGetter(int iinfo)
            {
                var columnType = _bindings.ColumnTypes[iinfo];
                if (columnType is VectorType vectorType)
                    return Utils.MarshalInvoke(MakeGetterVec<int>, vectorType.ItemType.RawType, vectorType.Size);
                return Utils.MarshalInvoke(MakeGetterOne<int>, columnType.RawType);
            }

            private Delegate MakeGetterOne<T>()
            {
                return (ValueGetter<T>)((ref T value) => value = default(T));
            }

            private Delegate MakeGetterVec<T>(int length)
            {
                return (ValueGetter<VBuffer<T>>)((ref VBuffer<T> value) =>
                    VBufferUtils.Resize(ref value, length, 0));
            }
        }

        [TlcModule.EntryPoint(Desc = Summary,
            Name = "Transforms.OptionalColumnCreator",
            UserName = UserName,
            ShortName = ShortName)]

        public static CommonOutputs.TransformOutput MakeOptional(IHostEnvironment env, Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "OptionalColumn", input);
            var xf = new OptionalColumnTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
