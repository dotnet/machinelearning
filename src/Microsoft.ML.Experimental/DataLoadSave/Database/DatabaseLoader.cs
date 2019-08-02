// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(DatabaseLoader.Summary, typeof(DatabaseLoader), null, typeof(SignatureLoadModel),
    "Database Loader", DatabaseLoader.LoaderSignature)]

namespace Microsoft.ML.Data
{
    public sealed partial class DatabaseLoader : IDataLoader<DatabaseSource>
    {
        internal const string Summary = "Loads data from a DbDataReader.";
        internal const string LoaderSignature = "DatabaseLoader";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DBLOADER",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DatabaseLoader).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private readonly IHost _host;
        private const string RegistrationName = "DatabaseLoader";

        internal DatabaseLoader(IHostEnvironment env, Options options)
        {
            options = options ?? new Options();

            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(options, nameof(options));

            var cols = options.Columns;
            if (Utils.Size(cols) == 0)
            {
                throw _host.Except("DatabaseLoader requires at least one Column");
            }

            _bindings = new Bindings(this, cols);
        }

        private DatabaseLoader(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(ctx);

            _host = host;

            _bindings = new Bindings(ctx, this);
        }

        internal static DatabaseLoader Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new DatabaseLoader(h, ctx));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(ctx);
        }

        /// <summary>
        /// The output <see cref="DataViewSchema"/> that will be produced by the loader.
        /// </summary>
        public DataViewSchema GetOutputSchema() => _bindings.OutputSchema;

        /// <summary>
        /// Loads data from <paramref name="source"/> into an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="source">The source from which to load data.</param>
        public IDataView Load(DatabaseSource source) => new BoundLoader(this, source);

        /// <summary>
        /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
        /// </summary>
        public sealed class Column
        {
            /// <summary>
            /// Name of the column.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column")]
            public string Name;

            /// <summary>
            /// <see cref="DbType"/> of the items in the column.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Type of the items in the column")]
            public DbType Type = DbType.Single;

            /// <summary>
            /// Source index of the column.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Source index of the column", ShortName = "src")]
            public int? Source;

            /// <summary>
            /// For a key column, this defines the range of values.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyCount KeyCount;
        }

        /// <summary>
        /// The settings for <see cref="DatabaseLoader"/>
        /// </summary>
        public sealed class Options
        {
            /// <summary>
            /// Specifies the input columns that should be mapped to <see cref="IDataView"/> columns.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Column groups. Each group is specified as name:type:numeric-ranges, eg, col=Features:R4:1-17,26,35-40",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        /// <summary>
        /// Information for an output column.
        /// </summary>
        private sealed class ColInfo
        {
            public readonly string Name;
            public readonly int? SourceIndex;
            public readonly DataViewType ColType;

            public ColInfo(string name, int? sourceIndex, DataViewType colType)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.Assert(!sourceIndex.HasValue || sourceIndex >= 0);
                Contracts.AssertValue(colType);

                Name = name;
                SourceIndex = sourceIndex;
                ColType = colType;
            }
        }

        private sealed class Bindings
        {
            /// <summary>
            /// <see cref="Infos"/>[i] stores the i-th column's name and type. Columns are loaded from the input text file.
            /// </summary>
            public readonly ColInfo[] Infos;

            public DataViewSchema OutputSchema { get; }

            public Bindings(DatabaseLoader parent, Column[] cols)
            {
                Contracts.AssertNonEmpty(cols);

                using (var ch = parent._host.Start("Binding"))
                {
                    // Make sure all columns have at least one source range.
                    foreach (var col in cols)
                    {
                        if (col.Source < 0)
                            throw ch.ExceptUserArg(nameof(Column.Source), "Source column index must be non-negative");
                    }

                    Infos = new ColInfo[cols.Length];

                    // This dictionary is used only for detecting duplicated column names specified by user.
                    var nameToInfoIndex = new Dictionary<string, int>(Infos.Length);

                    for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    {
                        var col = cols[iinfo];

                        ch.CheckNonWhiteSpace(col.Name, nameof(col.Name));
                        string name = col.Name.Trim();
                        if (iinfo == nameToInfoIndex.Count && nameToInfoIndex.ContainsKey(name))
                            ch.Info("Duplicate name(s) specified - later columns will hide earlier ones");

                        PrimitiveDataViewType itemType;
                        if (col.KeyCount != null)
                        {
                            itemType = ConstructKeyType(col.Type, col.KeyCount);
                        }
                        else
                        {
                            itemType = ColumnTypeExtensions.PrimitiveTypeFromType(col.Type.ToType());
                        }

                        Infos[iinfo] = new ColInfo(name, col.Source, itemType);

                        nameToInfoIndex[name] = iinfo;
                    }
                }
                OutputSchema = ComputeOutputSchema();
            }

            public Bindings(ModelLoadContext ctx, DatabaseLoader parent)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     ulong: count for key range
                //   byte: bool of whether the source index is valid
                //   for a valid source index:
                //     int: source index
                int cinfo = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(cinfo > 0);
                Infos = new ColInfo[cinfo];

                for (int iinfo = 0; iinfo < cinfo; iinfo++)
                {
                    string name = ctx.LoadNonEmptyString();

                    PrimitiveDataViewType itemType;
                    var kind = (InternalDataKind)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(Enum.IsDefined(typeof(InternalDataKind), kind));
                    bool isKey = ctx.Reader.ReadBoolByte();
                    if (isKey)
                    {
                        ulong count;
                        Contracts.CheckDecode(KeyDataViewType.IsValidDataType(kind.ToType()));

                        count = ctx.Reader.ReadUInt64();
                        Contracts.CheckDecode(0 < count);

                        itemType = new KeyDataViewType(kind.ToType(), count);
                    }
                    else
                        itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);

                    int? sourceIndex = null;
                    bool hasSourceIndex = ctx.Reader.ReadBoolByte();
                    if (hasSourceIndex)
                    {
                        sourceIndex = ctx.Reader.ReadInt32();
                    }

                    Infos[iinfo] = new ColInfo(name, sourceIndex, itemType);
                }

                OutputSchema = ComputeOutputSchema();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     ulong: count for key range
                //   byte: bool of whether the source index is valid
                //   for a valid source index:
                //     int: source index
                ctx.Writer.Write(Infos.Length);
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    var info = Infos[iinfo];
                    ctx.SaveNonEmptyString(info.Name);
                    var type = info.ColType.GetItemType();
                    InternalDataKind rawKind = type.GetRawKind();
                    Contracts.Assert((InternalDataKind)(byte)rawKind == rawKind);
                    ctx.Writer.Write((byte)rawKind);
                    ctx.Writer.WriteBoolByte(type is KeyDataViewType);
                    if (type is KeyDataViewType key)
                        ctx.Writer.Write(key.Count);
                    ctx.Writer.WriteBoolByte(info.SourceIndex.HasValue);
                    if (info.SourceIndex.HasValue)
                        ctx.Writer.Write(info.SourceIndex.GetValueOrDefault());
                }
            }

            private DataViewSchema ComputeOutputSchema()
            {
                var schemaBuilder = new DataViewSchema.Builder();

                // Iterate through all loaded columns. The index i indicates the i-th column loaded.
                for (int i = 0; i < Infos.Length; ++i)
                {
                    var info = Infos[i];
                    schemaBuilder.AddColumn(info.Name, info.ColType);
                }

                return schemaBuilder.ToSchema();
            }

            /// <summary>
            /// Construct a <see cref="KeyDataViewType"/> out of the DbType and the keyCount.
            /// </summary>
            private static KeyDataViewType ConstructKeyType(DbType dbType, KeyCount keyCount)
            {
                Contracts.CheckValue(keyCount, nameof(keyCount));

                KeyDataViewType keyType;
                Type rawType = dbType.ToType();
                Contracts.CheckUserArg(KeyDataViewType.IsValidDataType(rawType), nameof(DatabaseLoader.Column.Type), "Bad item type for Key");

                if (keyCount.Count == null)
                    keyType = new KeyDataViewType(rawType, rawType.ToMaxInt());
                else
                    keyType = new KeyDataViewType(rawType, keyCount.Count.GetValueOrDefault());

                return keyType;
            }
        }

        private sealed class BoundLoader : IDataView
        {
            private readonly DatabaseLoader _loader;
            private readonly IHost _host;
            private readonly DatabaseSource _source;

            public BoundLoader(DatabaseLoader loader, DatabaseSource source)
            {
                _loader = loader;
                _host = loader._host.Register(nameof(BoundLoader));

                _host.CheckValue(source, nameof(source));
                _source = source;
            }

            public long? GetRowCount() => null;
            public bool CanShuffle => false;

            public DataViewSchema Schema => _loader._bindings.OutputSchema;

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.CheckValueOrNull(rand);
                var active = Utils.BuildArray(_loader._bindings.OutputSchema.Count, columnsNeeded);
                return Cursor.Create(_loader, _source, active);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
            }
        }
    }
}
