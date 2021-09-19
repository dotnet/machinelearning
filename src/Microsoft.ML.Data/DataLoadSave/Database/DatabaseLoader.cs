// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Reflection;
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

        internal static DatabaseLoader CreateDatabaseLoader<TInput>(IHostEnvironment host)
        {
            var userType = typeof(TInput);

            var fieldInfos = userType.GetFields(BindingFlags.Public | BindingFlags.Instance);

            var propertyInfos =
                userType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => x.CanRead && x.GetGetMethod() != null && x.GetIndexParameters().Length == 0);

            var memberInfos = (fieldInfos as IEnumerable<MemberInfo>).Concat(propertyInfos).ToArray();

            if (memberInfos.Length == 0)
                throw host.ExceptParam(nameof(TInput), $"Should define at least one public, readable field or property in {nameof(TInput)}.");

            var columns = new List<Column>();

            for (int index = 0; index < memberInfos.Length; index++)
            {
                var memberInfo = memberInfos[index];
                var mappingAttrName = memberInfo.GetCustomAttribute<ColumnNameAttribute>();

                var column = new Column();
                column.Name = mappingAttrName?.Name ?? memberInfo.Name;

                var indexMappingAttr = memberInfo.GetCustomAttribute<LoadColumnAttribute>();
                var nameMappingAttr = memberInfo.GetCustomAttribute<LoadColumnNameAttribute>();

                if (indexMappingAttr is object)
                {
                    if (nameMappingAttr is object)
                    {
                        throw Contracts.Except($"Cannot specify both {nameof(LoadColumnAttribute)} and {nameof(LoadColumnNameAttribute)}");
                    }

                    column.Source = indexMappingAttr.Sources.Select((source) => Range.FromTextLoaderRange(source)).ToArray();
                }
                else if (nameMappingAttr is object)
                {
                    column.Source = nameMappingAttr.Sources.Select((source) => new Range(source)).ToArray();
                }

                InternalDataKind dk;
                switch (memberInfo)
                {
                    case FieldInfo field:
                        if (!InternalDataKindExtensions.TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk))
                            throw Contracts.Except($"Field {memberInfo.Name} is of unsupported type.");

                        break;

                    case PropertyInfo property:
                        if (!InternalDataKindExtensions.TryGetDataKind(property.PropertyType.IsArray ? property.PropertyType.GetElementType() : property.PropertyType, out dk))
                            throw Contracts.Except($"Property {memberInfo.Name} is of unsupported type.");
                        break;

                    default:
                        Contracts.Assert(false);
                        throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
                }

                column.Type = dk.ToDbType();

                columns.Add(column);
            }

            var options = new Options
            {
                Columns = columns.ToArray()
            };
            return new DatabaseLoader(host, options);
        }

        /// <summary>
        /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
        /// </summary>
        public sealed class Column
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Column"/> class.
            /// </summary>
            public Column() { }

            /// <summary>
            /// Initializes a new instance of the <see cref="Column"/> class.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dbType"><see cref="DbType"/> of the items in the column.</param>
            /// <param name="index">Index of the column.</param>
            public Column(string name, DbType dbType, int index)
                : this(name, dbType, new[] { new Range(index) })
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="Column"/> class.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dbType"><see cref="DbType"/> of the items in the column.</param>
            /// <param name="minIndex">The minimum inclusive index of the column.</param>
            /// <param name="maxIndex">The maximum-inclusive index of the column.</param>
            public Column(string name, DbType dbType, int minIndex, int maxIndex)
                : this(name, dbType, new[] { new Range(minIndex, maxIndex) })
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="Column"/> class.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dbType"><see cref="DbType"/> of the items in the column.</param>
            /// <param name="source">Source index range(s) of the column.</param>
            /// <param name="keyCount">For a key column, this defines the range of values.</param>
            public Column(string name, DbType dbType, Range[] source, KeyCount keyCount = null)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValue(source, nameof(source));

                Name = name;
                Type = dbType;
                Source = source;
                KeyCount = keyCount;
            }

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
            /// Source index or name range(s) of the column.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Source index range(s) of the column", ShortName = "src")]
            public Range[] Source;

            /// <summary>
            /// For a key column, this defines the range of values.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyCount KeyCount;
        }

        /// <summary>
        /// Specifies the range of indices or names of input columns that should be mapped to an output column.
        /// </summary>
        public sealed class Range
        {
            public Range() { }

            /// <summary>
            /// A range representing a single value. Will result in a scalar column.
            /// </summary>
            /// <param name="index">The index of the field of the table to read.</param>
            public Range(int index)
            {
                Contracts.CheckParam(index >= 0, nameof(index), "Must be non-negative");
                Min = index;
                Max = index;
                Name = null;
            }

            /// <summary>
            /// A range representing a single value. Will result in a scalar column.
            /// </summary>
            /// <param name="name">The name of the field of the table to read.</param>
            public Range(string name)
            {
                Contracts.CheckValue(name, nameof(name));
                Min = -1;
                Max = -1;
                Name = name;
            }

            /// <summary>
            /// A range representing a set of values. Will result in a vector column.
            /// </summary>
            /// <param name="min">The minimum inclusive index of the column.</param>
            /// <param name="max">The maximum-inclusive index of the column.</param>
            public Range(int min, int max)
            {
                Contracts.CheckParam(min >= 0, nameof(min), "Must be non-negative");
                Contracts.CheckParam(max >= min, nameof(max), "Must be greater than or equal to " + nameof(min));

                Min = min;
                Max = max;
                // Note that without the following being set, in the case where there is a single range
                // where Min == Max, the result will not be a vector valued but a scalar column.
                ForceVector = true;
            }

            /// <summary>
            ///  The minimum index of the column, inclusive.
            /// </summary>
            /// <remarks>
            /// This value is ignored if <see cref="Name" /> is not <c>null</c>.
            /// </remarks>
            [Argument(ArgumentType.Required, HelpText = "First index in the range")]
            public int Min;

            /// <summary>
            /// The maximum index of the column, inclusive.
            /// </summary>
            /// <remarks>
            /// This value is ignored if <see cref="Name" /> is not <c>null</c>.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
            public int Max;

            /// <summary>
            /// The name of the input column.
            /// </summary>
            /// <remarks>
            /// This value, if non-<c>null</c>, overrides <see cref="Min" /> and <see cref="Max" />.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column")]
            public string Name;

            /// <summary>
            /// Force scalar columns to be treated as vectors of length one.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Force scalar columns to be treated as vectors of length one", ShortName = "vector")]
            public bool ForceVector;

            internal static Range FromTextLoaderRange(TextLoader.Range range)
            {
                Contracts.Assert(range.Max.HasValue);

                var dbRange = new Range(range.Min, range.Max.Value);
                dbRange.ForceVector = range.ForceVector;
                return dbRange;
            }
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
        /// Used as an input column range.
        /// </summary>
        internal readonly struct Segment
        {
            public readonly string Name;
            public readonly int Min;
            public readonly int Lim;
            public readonly bool ForceVector;

            public Segment(int min, int lim, bool forceVector)
            {
                Contracts.Assert(0 <= min && min < lim);
                Name = null;
                Min = min;
                Lim = lim;
                ForceVector = forceVector;
            }

            public Segment(string name, bool forceVector)
            {
                Contracts.Assert(name != null);
                Name = name;
                Min = -1;
                Lim = -1;
                ForceVector = forceVector;
            }
        }

        /// <summary>
        /// Information for an output column.
        /// </summary>
        private sealed class ColInfo
        {
            public readonly string Name;
            public readonly DataViewType ColType;
            public readonly Segment[] Segments;

            // BaseSize is the sum of the sizes of segments.
            public readonly int SizeBase;

            private ColInfo(string name, DataViewType colType, Segment[] segs, int sizeBase)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertValueOrNull(segs);
                Contracts.Assert(sizeBase > 0);

                Name = name;
                Contracts.Assert(colType.GetItemType().GetRawKind() != 0);
                ColType = colType;
                Segments = segs;
                SizeBase = sizeBase;
            }

            public static ColInfo Create(string name, PrimitiveDataViewType itemType, Segment[] segs, bool user)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertValue(itemType);
                Contracts.AssertValueOrNull(segs);

                int size = 0;
                DataViewType type = itemType;

                if (segs != null)
                {
                    var order = Utils.GetIdentityPermutation(segs.Length);

                    if ((segs.Length != 0) && (segs[0].Name is null))
                    {
                        Array.Sort(order, (x, y) => segs[x].Min.CompareTo(segs[y].Min));

                        // Check that the segments are disjoint.
                        for (int i = 1; i < order.Length; i++)
                        {
                            int a = order[i - 1];
                            int b = order[i];
                            Contracts.Assert(segs[a].Min <= segs[b].Min);
                            if (segs[a].Lim > segs[b].Min)
                            {
                                throw user ?
                                    Contracts.ExceptUserArg(nameof(Column.Source), "Intervals specified for column '{0}' overlap", name) :
                                    Contracts.ExceptDecode("Intervals specified for column '{0}' overlap", name);
                            }
                        }
                    }

                    // Note: since we know that the segments don't overlap, we're guaranteed that
                    // the sum of their sizes doesn't overflow.
                    for (int i = 0; i < segs.Length; i++)
                    {
                        var seg = segs[i];
                        size += (seg.Name is null) ? seg.Lim - seg.Min : 1;
                    }
                    Contracts.Assert(size >= segs.Length);

                    if (size > 1 || segs[0].ForceVector)
                        type = new VectorDataViewType(itemType, size);
                }
                else
                {
                    size++;
                }

                return new ColInfo(name, type, segs, size);
            }
        }

        private sealed class Bindings
        {
            /// <summary>
            /// <see cref="Infos"/>[i] stores the i-th column's name and type. Columns are loaded from the input database.
            /// </summary>
            public readonly ColInfo[] Infos;

            public DataViewSchema OutputSchema { get; }

            public Bindings(DatabaseLoader parent, Column[] cols)
            {
                Contracts.AssertNonEmpty(cols);

                using (var ch = parent._host.Start("Binding"))
                {
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
                            ch.CheckUserArg(Enum.IsDefined(typeof(DbType), col.Type), nameof(Column.Type), "Bad item type");
                            itemType = ColumnTypeExtensions.PrimitiveTypeFromType(col.Type.ToType());
                        }

                        Segment[] segs = null;

                        if (col.Source != null)
                        {
                            segs = new Segment[col.Source.Length];

                            for (int i = 0; i < segs.Length; i++)
                            {
                                var range = col.Source[i];
                                Segment seg;

                                if (range.Name is null)
                                {
                                    int min = range.Min;
                                    ch.CheckUserArg(0 <= min, nameof(range.Min));

                                    int max = range.Max;
                                    ch.CheckUserArg(min <= max, nameof(range.Max));
                                    seg = new Segment(min, max + 1, range.ForceVector);
                                }
                                else
                                {
                                    string columnName = range.Name;
                                    ch.CheckUserArg(columnName != null, nameof(range.Name));
                                    seg = new Segment(columnName, range.ForceVector);
                                }

                                segs[i] = seg;
                            }
                        }

                        Infos[iinfo] = ColInfo.Create(name, itemType, segs, true);

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
                //   int: number of segments
                //   foreach segment:
                //     string id: name
                //     int: min
                //     int: lim
                //     byte: force vector (verWrittenCur: verIsVectorSupported)
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

                    int cseg = ctx.Reader.ReadInt32();

                    Segment[] segs;

                    if (cseg == 0)
                    {
                        segs = null;
                    }
                    else
                    {
                        Contracts.CheckDecode(cseg > 0);
                        segs = new Segment[cseg];
                        for (int iseg = 0; iseg < cseg; iseg++)
                        {
                            string columnName = ctx.LoadStringOrNull();
                            int min = ctx.Reader.ReadInt32();
                            int lim = ctx.Reader.ReadInt32();
                            Contracts.CheckDecode(0 <= min && min < lim);
                            bool forceVector = ctx.Reader.ReadBoolByte();
                            segs[iseg] = (columnName is null) ? new Segment(min, lim, forceVector) : new Segment(columnName, forceVector);
                        }
                    }

                    // Note that this will throw if the segments are ill-structured, including the case
                    // of multiple variable segments (since those segments will overlap and overlapping
                    // segments are illegal).
                    Infos[iinfo] = ColInfo.Create(name, itemType, segs, false);
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
                //   int: number of segments
                //   foreach segment:
                //     string id: name
                //     int: min
                //     int: lim
                //     byte: force vector (verWrittenCur: verIsVectorSupported)
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

                    if (info.Segments is null)
                    {
                        ctx.Writer.Write(0);
                    }
                    else
                    {
                        ctx.Writer.Write(info.Segments.Length);
                        foreach (var seg in info.Segments)
                        {
                            ctx.SaveStringOrNull(seg.Name);
                            ctx.Writer.Write(seg.Min);
                            ctx.Writer.Write(seg.Lim);
                            ctx.Writer.WriteBoolByte(seg.ForceVector);
                        }
                    }
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
