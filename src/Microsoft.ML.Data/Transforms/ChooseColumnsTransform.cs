// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Float = System.Single;

[assembly: LoadableClass(typeof(ChooseColumnsTransform), typeof(ChooseColumnsTransform.Arguments), typeof(SignatureDataTransform),
    "Choose Columns Transform", "ChooseColumnsTransform", "ChooseColumns", "Choose", DocName = "transform/DropKeepChooseTransforms.md")]

[assembly: LoadableClass(typeof(ChooseColumnsTransform), null, typeof(SignatureLoadDataTransform),
    "Choose Columns Transform", ChooseColumnsTransform.LoaderSignature, ChooseColumnsTransform.LoaderSignatureOld)]

namespace Microsoft.ML.Transforms
{
    public sealed class ChooseColumnsTransform : RowToRowTransformBase
    {
        // These values are serialized so should not be changed.
        public enum HiddenColumnOption : byte
        {
            Drop = 1,
            Keep = 2,
            Rename = 3
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "What to do with hidden columns")]
            public HiddenColumnOption? Hidden;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Hidden != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments
        {
            public Arguments()
            {

            }

            internal Arguments(params string[] columns)
            {
                Column = new Column[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    Column[i] = new Column() { Source = columns[i], Name = columns[i] };
                }
            }

            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Multiple, HelpText = "What to do with hidden columns")]
            public HiddenColumnOption Hidden = HiddenColumnOption.Drop;
        }

        private sealed class Bindings : ISchema
        {
            /// <summary>
            /// This encodes the information specified in the Arguments object and is what is persisted.
            /// </summary>
            public sealed class RawColInfo
            {
                public readonly string Name;
                public readonly string Source;
                public readonly HiddenColumnOption Hid;

                public RawColInfo(string name, string source, HiddenColumnOption hid)
                {
                    Contracts.AssertNonEmpty(name);
                    Contracts.AssertNonEmpty(source);
                    Contracts.Assert(Enum.IsDefined(typeof(HiddenColumnOption), hid));

                    Name = name;
                    Source = source;
                    Hid = hid;
                }
            }

            public sealed class ColInfo
            {
                public readonly string Name;
                public readonly int Source;
                public readonly ColumnType TypeSrc;

                public ColInfo(string name, int src, ColumnType typeSrc)
                {
                    Contracts.AssertNonEmpty(name);
                    Contracts.Assert(src >= 0);
                    Contracts.AssertValue(typeSrc);

                    Name = name;
                    Source = src;
                    TypeSrc = typeSrc;
                }
            }

            public readonly ISchema Input;
            public readonly RawColInfo[] RawInfos;
            public readonly HiddenColumnOption HidDefault;
            public readonly ColInfo[] Infos;
            public readonly Dictionary<string, int> NameToInfoIndex;

            public Schema AsSchema { get; }

            public Bindings(Arguments args, ISchema schemaInput)
            {
                Contracts.AssertValue(args);
                Contracts.AssertValue(schemaInput);

                Input = schemaInput;

                Contracts.Check(Enum.IsDefined(typeof(HiddenColumnOption), args.Hidden), "hidden");
                HidDefault = args.Hidden;

                RawInfos = new RawColInfo[Utils.Size(args.Column)];
                if (RawInfos.Length > 0)
                {
                    var names = new HashSet<string>();
                    for (int i = 0; i < RawInfos.Length; i++)
                    {
                        var item = args.Column[i];
                        string dst = item.Name;
                        string src = item.Source;

                        if (string.IsNullOrWhiteSpace(src))
                            src = dst;
                        else if (string.IsNullOrWhiteSpace(dst))
                            dst = src;
                        Contracts.CheckUserArg(!string.IsNullOrWhiteSpace(dst), nameof(Column.Name));

                        if (!names.Add(dst))
                            throw Contracts.ExceptUserArg(nameof(args.Column), "New column '{0}' specified multiple times", dst);

                        var hid = item.Hidden ?? args.Hidden;
                        Contracts.CheckUserArg(Enum.IsDefined(typeof(HiddenColumnOption), hid), nameof(args.Hidden));

                        RawInfos[i] = new RawColInfo(dst, src, hid);
                    }
                }

                BuildInfos(out Infos, out NameToInfoIndex, user: true);
                AsSchema = Schema.Create(this);
            }

            private void BuildInfos(out ColInfo[] infosArray, out Dictionary<string, int> nameToCol, bool user)
            {
                var raws = RawInfos;
                var tops = new List<int>();

                bool rename = false;
                Dictionary<string, List<int>> dups = null;
                if (raws.Length == 0)
                {
                    // Empty raws means take all with default HiddenColumnOption.
                    var rawList = new List<RawColInfo>();
                    for (int col = 0; col < Input.ColumnCount; col++)
                    {
                        string src = Input.GetColumnName(col);
                        int tmp;
                        if (!Input.TryGetColumnIndex(src, out tmp))
                        {
                            Contracts.Assert(false, "Why couldn't the schema find the name?");
                            continue;
                        }

                        if (tmp == col)
                        {
                            var raw = new RawColInfo(src, src, HidDefault);
                            rawList.Add(raw);
                            tops.Add(col);
                        }
                        else if (HidDefault != HiddenColumnOption.Drop)
                        {
                            if (dups == null)
                                dups = new Dictionary<string, List<int>>();
                            List<int> list;
                            if (!dups.TryGetValue(src, out list))
                                dups[src] = list = new List<int>();
                            list.Add(col);
                        }
                    }
                    if (dups != null && HidDefault == HiddenColumnOption.Rename)
                        rename = true;

                    raws = rawList.ToArray();
                }
                else
                {
                    for (int i = 0; i < raws.Length; i++)
                    {
                        var raw = raws[i];

                        int col;
                        if (!Input.TryGetColumnIndex(raw.Source, out col))
                        {
                            throw user ?
                                Contracts.ExceptUserArg(nameof(Arguments.Column), "source column '{0}' not found", raw.Source) :
                                Contracts.ExceptDecode("source column '{0}' not found", raw.Source);
                        }
                        tops.Add(col);

                        if (raw.Hid != HiddenColumnOption.Drop)
                        {
                            if (dups == null)
                                dups = new Dictionary<string, List<int>>();
                            dups[raw.Source] = null;
                            if (raw.Hid == HiddenColumnOption.Rename)
                                rename = true;
                        }
                    }

                    if (dups != null)
                    {
                        for (int col = 0; col < Input.ColumnCount; col++)
                        {
                            string src = Input.GetColumnName(col);
                            List<int> list;
                            if (!dups.TryGetValue(src, out list))
                                continue;
                            int tmp;
                            if (!Input.TryGetColumnIndex(src, out tmp))
                            {
                                Contracts.Assert(false, "Why couldn't the schema find the name?");
                                continue;
                            }
                            if (tmp == col)
                                continue;
                            if (list == null)
                                dups[src] = list = new List<int>();
                            list.Add(col);
                        }
                    }
                }
                Contracts.Assert(tops.Count == raws.Length);

                HashSet<string> names = null;
                if (rename)
                    names = new HashSet<string>(raws.Select(r => r.Name));

                var infos = new List<ColInfo>();
                for (int i = 0; i < raws.Length; i++)
                {
                    var raw = raws[i];

                    int colSrc;
                    ColumnType type;
                    ColInfo info;

                    // Handle dups.
                    List<int> list;
                    if (raw.Hid != HiddenColumnOption.Drop &&
                        dups != null && dups.TryGetValue(raw.Source, out list) && list != null)
                    {
                        int iinfo = infos.Count;
                        int inc = 0;
                        for (int iv = list.Count; --iv >= 0; )
                        {
                            colSrc = list[iv];
                            type = Input.GetColumnType(colSrc);
                            string name = raw.Name;
                            if (raw.Hid == HiddenColumnOption.Rename)
                                name = GetUniqueName(names, name, ref inc);
                            info = new ColInfo(name, colSrc, type);
                            infos.Insert(iinfo, info);
                        }
                    }

                    colSrc = tops[i];
                    type = Input.GetColumnType(colSrc);
                    info = new ColInfo(raw.Name, colSrc, type);
                    infos.Add(info);
                }

                infosArray = infos.ToArray();
                nameToCol = new Dictionary<string, int>(Infos.Length);
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    nameToCol[Infos[iinfo].Name] = iinfo;
            }

            private static string GetUniqueName(HashSet<string> names, string name, ref int inc)
            {
                for (; ; )
                {
                    string tmp = string.Format("{0}_{1:000}", name, ++inc);
                    if (names.Add(tmp))
                        return tmp;
                }
            }

            public Bindings(ModelLoadContext ctx, ISchema schemaInput)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(schemaInput);

                Input = schemaInput;

                // *** Binary format ***
                // byte: default HiddenColumnOption value
                // int: number of raw column infos
                // for each raw column info
                //   int: id of output column name
                //   int: id of input column name
                //   byte: HiddenColumnOption
                HidDefault = (HiddenColumnOption)ctx.Reader.ReadByte();
                Contracts.CheckDecode(Enum.IsDefined(typeof(HiddenColumnOption), HidDefault));

                int count = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(count >= 0);

                RawInfos = new RawColInfo[count];
                if (count > 0)
                {
                    var names = new HashSet<string>();
                    for (int i = 0; i < count; i++)
                    {
                        string dst = ctx.LoadNonEmptyString();
                        Contracts.CheckDecode(names.Add(dst));
                        string src = ctx.LoadNonEmptyString();

                        var hid = (HiddenColumnOption)ctx.Reader.ReadByte();
                        Contracts.CheckDecode(Enum.IsDefined(typeof(HiddenColumnOption), hid));
                        RawInfos[i] = new RawColInfo(dst, src, hid);
                    }
                }

                BuildInfos(out Infos, out NameToInfoIndex, user: false);

                AsSchema = Schema.Create(this);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // byte: default HiddenColumnOption value
                // int: number of raw column infos
                // for each raw column info
                //   int: id of output column name
                //   int: id of input column name
                //   byte: HiddenColumnOption
                Contracts.Assert((HiddenColumnOption)(byte)HidDefault == HidDefault);
                ctx.Writer.Write((byte)HidDefault);
                ctx.Writer.Write(RawInfos.Length);
                for (int i = 0; i < RawInfos.Length; i++)
                {
                    var raw = RawInfos[i];
                    ctx.SaveNonEmptyString(raw.Name);
                    ctx.SaveNonEmptyString(raw.Source);
                    Contracts.Assert((HiddenColumnOption)(byte)raw.Hid == raw.Hid);
                    ctx.Writer.Write((byte)raw.Hid);
                }
            }

            public int ColumnCount
            {
                get { return Infos.Length; }
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                Contracts.CheckValueOrNull(name);
                if (name == null)
                {
                    col = default(int);
                    return false;
                }
                return NameToInfoIndex.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Infos[col].Name;
            }

            public ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Infos[col].TypeSrc;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetMetadataTypes(Infos[col].Source);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetMetadataTypeOrNull(kind, Infos[col].Source);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Input.GetMetadata(kind, Infos[col].Source, ref value);
            }

            internal bool[] GetActive(Func<int, bool> predicate)
            {
                return Utils.BuildArray(ColumnCount, predicate);
            }

            internal Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);
                var active = new bool[Input.ColumnCount];
                for (int i = 0; i < Infos.Length; i++)
                {
                    if (predicate(i))
                        active[Infos[i].Source] = true;
                }
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public const string LoaderSignature = "ChooseColumnsTransform";
        internal const string LoaderSignatureOld = "ChooseColumnsFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CHSCOLSF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(ChooseColumnsTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "ChooseColumns";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columns">Names of the columns to choose.</param>
        public ChooseColumnsTransform(IHostEnvironment env, IDataView input, params string[] columns)
            : this(env, new Arguments(columns), input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ChooseColumnsTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));

            _bindings = new Bindings(args, Source.Schema);
        }

        private ChooseColumnsTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Float));
            _bindings = new Bindings(ctx, Source.Schema);
        }

        public static ChooseColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ChooseColumnsTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            ctx.Writer.Write(sizeof(Float));
            _bindings.Save(ctx);
        }

        public override Schema Schema => _bindings.AsSchema;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // Parallel doesn't matter to this transform.
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(Host, _bindings, input, active);
        }

        public sealed override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, _bindings, inputs[i], active);
            return cursors;
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;

            public RowCursor(IChannelProvider provider, Bindings bindings, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(bindings);
                Ch.Assert(active == null || active.Length == bindings.ColumnCount);

                _bindings = bindings;
                _active = active;
            }

            public Schema Schema => _bindings.AsSchema;

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                var info = _bindings.Infos[col];
                return Input.GetGetter<TValue>(info.Source);
            }
        }
    }
}
