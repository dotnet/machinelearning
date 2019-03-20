// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Data.Utilities;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(PartitionedFileLoader.Summary, typeof(PartitionedFileLoader), typeof(PartitionedFileLoader.Arguments), typeof(SignatureDataLoader),
    PartitionedFileLoader.UserName, PartitionedFileLoader.LoadName, PartitionedFileLoader.ShortName)]

[assembly: LoadableClass(PartitionedFileLoader.Summary, typeof(PartitionedFileLoader), null, typeof(SignatureLoadDataLoader),
    PartitionedFileLoader.UserName, PartitionedFileLoader.LoadName, PartitionedFileLoader.ShortName)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Loads a set of directory partitioned files into an IDataView.
    /// The directories of the file will treated as column data and the underlying files are loaded using the data loader.
    /// The first file will be used as the basis for all follow-up file paths and schemas. Any files that don't match
    /// the expected path or schema will be skipped.
    /// </summary>
    /// <example>
    /// Sample directory structure:
    ///
    /// Data/
    ///     Year=2017/
    ///         Month=01/
    ///             data1.parquet
    ///             data1.parquet
    ///         Month=02/
    ///             data1.parquet
    ///             data1.parquet
    ///     Year=2018/
    ///         Month=01/
    ///             data1.parquet
    ///             data1.parquet
    /// </example>
    [BestFriend]
    internal sealed class PartitionedFileLoader : ILegacyDataLoader
    {
        internal const string Summary = "Loads a horizontally partitioned file set.";
        internal const string UserName = "Partitioned Loader";
        public const string LoadName = "PartitionedLoader";
        public const string ShortName = "Part";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PARTLOAD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName,
                loaderAssemblyName: typeof(PartitionedFileLoader).Assembly.FullName);
        }

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Base path to the directory of your partitioned files.", ShortName = "bp")]
            public string BasePath;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Append a column with the file path.", ShortName = "path")]
            public bool IncludePathColumn = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Path parser to extract column name/value pairs from the file path.", ShortName = "parser")]
            public IPartitionedPathParserFactory PathParserFactory = new ParquetPartitionedPathParserFactory();

            [Argument(ArgumentType.Multiple, HelpText = "The data loader.", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, ILegacyDataLoader> Loader;
        }

        public sealed class Column
        {
            [Argument(ArgumentType.Required, HelpText = "Name of the column.")]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Data type of the column.")]
            public InternalDataKind? Type;

            [Argument(ArgumentType.Required, HelpText = "Index of the directory representing this column.")]
            public int Source;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                if (TryParse(str, out Column column))
                {
                    return column;
                }

                return null;
            }

            private static bool TryParse(string str, out Column column)
            {
                column = null;

                if (string.IsNullOrEmpty(str))
                {
                    return false;
                }

                if (!ColumnParsingUtils.TryParse(str, out string name, out string sourceStr, out string kindStr))
                {
                    return false;
                }

                InternalDataKind? kind = null;
                if (kindStr != null && TypeParsingUtils.TryParseDataKind(kindStr, out InternalDataKind parsedKind, out var keyCount))
                {
                    kind = parsedKind;
                }

                if (!int.TryParse(sourceStr, out int source))
                {
                    return false;
                }

                column = new Column()
                {
                    Name = name,
                    Source = source,
                    Type = kind
                };

                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);

                sb.Append($"{Name}");

                if (Type.HasValue)
                {
                    sb.Append($":{Type}");
                }

                sb.Append($":{Source}");

                return true;
            }
        }

        private readonly IHost _host;
        private readonly IMultiStreamSource _files;
        private readonly int[] _srcDirIndex;
        private readonly byte[] _subLoaderBytes;

        // Number of tailing directories to include.
        private readonly int _tailingDirCount;

        private readonly IPartitionedPathParser _pathParser;

        private const string RegistrationName = LoadName;
        private const string FilePathSpecCtxName = "FilePathSpec";
        private const string SchemaCtxName = "Schema.idv";
        private const int FilePathColIndex = -1;

        public PartitionedFileLoader(IHostEnvironment env, Arguments args, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(args.Loader, nameof(args.Loader));
            _host.CheckValue(files, nameof(files));

            _pathParser = args.PathParserFactory.CreateComponent(_host);
            _host.CheckUserArg(_pathParser != null, nameof(args.PathParserFactory), "Failed to create the FilePathSpec.");

            _files = files;

            var subLoader = args.Loader.CreateComponent(_host, _files);
            _subLoaderBytes = SaveLoaderToBytes(subLoader);

            string relativePath = GetRelativePath(args.BasePath, files);
            var columns = ParseColumns(relativePath).ToArray();
            _tailingDirCount = GetDirectoryCount(relativePath);

            if (args.IncludePathColumn)
            {
                var pathCol = new Column()
                {
                    Name = "Path",
                    Source = FilePathColIndex,
                    Type = InternalDataKind.Text
                };

                columns = columns.Concat(new[] { pathCol }).ToArray();
            }

            _srcDirIndex = columns.Select(c => c.Source).ToArray();
            Schema = CreateSchema(_host, columns, subLoader);
        }

        private PartitionedFileLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);
            _host.AssertValue(files);

            // ** Binary format **
            // int: tailing directory count
            // Schema of the loader
            // int[]: srcColumns
            // byte[]: subloader
            // model: file path spec

            _tailingDirCount = ctx.Reader.ReadInt32();

            // Load the schema
            byte[] buffer = null;
            if (!ctx.TryLoadBinaryStream(SchemaCtxName, r => buffer = r.ReadByteArray()))
                throw _host.ExceptDecode();
            BinaryLoader loader = null;
            var strm = new MemoryStream(buffer, writable: false);
            loader = new BinaryLoader(_host, new BinaryLoader.Arguments(), strm);
            Schema = loader.Schema;

            _srcDirIndex = ctx.Reader.ReadIntArray();
            _subLoaderBytes = ctx.Reader.ReadByteArray();

            ctx.LoadModel<IPartitionedPathParser, SignatureLoadModel>(_host, out _pathParser, FilePathSpecCtxName);

            _files = files;
        }

        public static PartitionedFileLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost host = env.Register(RegistrationName);

            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(files, nameof(files));

            return host.Apply("Loading Model",
                ch => new PartitionedFileLoader(host, ctx, files));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // ** Binary format **
            // int: tailing directory count
            // Schema of the loader
            // int[]: srcColumns
            // byte[]: subloader
            // model: file path spec

            ctx.Writer.Write(_tailingDirCount);

            // Save the schema
            var noRows = new EmptyDataView(_host, Schema);
            var saverArgs = new BinarySaver.Arguments();
            saverArgs.Silent = true;
            var saver = new BinarySaver(_host, saverArgs);
            using (var strm = new MemoryStream())
            {
                var allColumns = Enumerable.Range(0, Schema.Count).ToArray();
                saver.SaveData(strm, noRows, allColumns);
                ctx.SaveBinaryStream(SchemaCtxName, w => w.WriteByteArray(strm.ToArray()));
            }
            ctx.Writer.WriteIntArray(_srcDirIndex);

            ctx.Writer.WriteByteArray(_subLoaderBytes);
            ctx.SaveModel(_pathParser, FilePathSpecCtxName);
        }

        public bool CanShuffle => true;

        public DataViewSchema Schema { get; }

        public long? GetRowCount()
        {
            return null;
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return new Cursor(_host, this, _files, columnsNeeded, rand);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            var cursor = new Cursor(_host, this, _files, columnsNeeded, rand);
            return new DataViewRowCursor[] { cursor };
        }

        /// <summary>
        /// Create a composite schema of both the partitioned columns and the underlying loader columns.
        /// </summary>
        /// <param name="ectx">The exception context.</param>
        /// <param name="cols">The partitioned columns.</param>
        /// <param name="subLoader">The sub loader.</param>
        /// <returns>The resulting schema.</returns>
        private DataViewSchema CreateSchema(IExceptionContext ectx, Column[] cols, ILegacyDataLoader subLoader)
        {
            Contracts.AssertValue(cols);
            Contracts.AssertValue(subLoader);

            var builder = new DataViewSchema.Builder();
            builder.AddColumns(cols.Select(c => new DataViewSchema.DetachedColumn(c.Name, ColumnTypeExtensions.PrimitiveTypeFromKind(c.Type.Value), null)));
            var colSchema = builder.ToSchema();

            var subSchema = subLoader.Schema;

            if (subSchema.Count == 0)
            {
                return colSchema;
            }
            else
            {
                var schemas = new DataViewSchema[]
                {
                    subSchema,
                    colSchema
                };

                return new ZipBinding(schemas).OutputSchema;
            }
        }

        private byte[] SaveLoaderToBytes(ILegacyDataLoader loader)
        {
            Contracts.CheckValue(loader, nameof(loader));

            using (var stream = new MemoryStream())
            {
                LoaderUtils.SaveLoader(loader, stream);
                return stream.GetBuffer();
            }
        }

        private ILegacyDataLoader CreateLoaderFromBytes(byte[] loaderBytes, IMultiStreamSource files)
        {
            Contracts.CheckValue(loaderBytes, nameof(loaderBytes));
            Contracts.CheckValue(files, nameof(files));

            using (var stream = new MemoryStream(loaderBytes))
            using (var rep = RepositoryReader.Open(stream, _host))
            {
                return ModelFileUtils.LoadLoader(_host, rep, files, false);
            }
        }

        private sealed class Cursor : RootCursorBase
        {
            private PartitionedFileLoader _parent;

            private readonly bool[] _active;
            private readonly bool[] _subActive; // Active columns of the sub-cursor.
            private Delegate[] _getters;
            private Delegate[] _subGetters; // Cached getters of the sub-cursor.

            private readonly IEnumerable<DataViewSchema.Column> _columnsNeeded;
            private readonly IEnumerable<DataViewSchema.Column> _subActivecolumnsNeeded;

            private ReadOnlyMemory<char>[] _colValues; // Column values cached from the file path.
            private DataViewRowCursor _subCursor; // Sub cursor of the current file.

            private IEnumerator<int> _fileOrder;

            public Cursor(IChannelProvider provider, PartitionedFileLoader parent, IMultiStreamSource files, IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
                : base(provider)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(files);

                _parent = parent;
                _columnsNeeded = columnsNeeded;

                _active = Utils.BuildArray(Schema.Count, columnsNeeded);
                _subActive = _active.Take(SubColumnCount).ToArray();
                _colValues = new ReadOnlyMemory<char>[Schema.Count - SubColumnCount];

                _subActivecolumnsNeeded = Schema.Where(x => (_subActive?.Length > x.Index) && _subActive[x.Index]);

                _subGetters = new Delegate[SubColumnCount];
                _getters = CreateGetters();

                _fileOrder = CreateFileOrder(rand).GetEnumerator();
            }

            public override long Batch => 0;

            public override DataViewSchema Schema => _parent.Schema;

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

                var getter = _getters[column.Index] as ValueGetter<TValue>;
                if (getter == null)
                {
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                }

                return getter;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);

                        val = new DataViewRowId(0, (ulong)Position);
                    };
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < Schema.Count);
                return _active[column.Index];
            }

            protected override bool MoveNextCore()
            {
                // Iterate sub cursor or move to the next file.
                while (_subCursor == null || !_subCursor.MoveNext())
                {
                    // Cleanup old sub cursor
                    if (_subCursor != null)
                    {
                        _subCursor.Dispose();
                        _subCursor = null;
                    }

                    if (!TryGetNextPathAndValues(out string path, out string relativePath, out List<string> values))
                    {
                        return false;
                    }

                    ILegacyDataLoader loader = null;
                    try
                    {
                        // Load the sub cursor and reset the data.
                        loader = _parent.CreateLoaderFromBytes(_parent._subLoaderBytes, new MultiFileSource(path));
                    }
                    catch (Exception e)
                    {
                        Ch.Warning($"Failed to load file {path} due to a loader exception. Moving on to the next file. Ex: {e.Message}");
                        continue;
                    }

                    _subCursor = loader.GetRowCursor(_subActivecolumnsNeeded);

                    try
                    {
                        UpdateSubGetters();
                        UpdateColumnValues(relativePath, values);
                    }
                    catch (InvalidOperationException e)
                    {
                        // Failed to load this file so skip.
                        Ch.Warning(MessageSensitivity.Schema, e.Message);
                        if (_subCursor != null)
                        {
                            _subCursor.Dispose();
                            _subCursor = null;
                        }
                    }
                }

                return true;
            }

            private bool TryGetNextPathAndValues(out string path, out string relativePath, out List<string> values)
            {
                path = null;
                relativePath = null;
                values = null;

                do
                {
                    // No more files to load.
                    if (!_fileOrder.MoveNext())
                    {
                        return false;
                    }

                    // Get next file and parse the column values from the file path.
                    string curPath = _parent._files.GetPathOrNull(_fileOrder.Current);
                    if (String.IsNullOrEmpty(curPath))
                    {
                        Ch.Warning($"File at index {_fileOrder.Current} is missing a path. Loading of file is being skipped.");
                        continue;
                    }

                    if (!TryTruncatePath(_parent._tailingDirCount, curPath, out relativePath))
                    {
                        continue;
                    }

                    if (!TryParseValuesFromPath(relativePath, out values))
                    {
                        continue;
                    }

                    path = curPath;

                } while (String.IsNullOrEmpty(path));

                return true;
            }

            private void UpdateSubGetters()
            {
                // Reset getters for the subcursor.
                for (int i = 0; i < SubColumnCount; i++)
                {
                    if (_subActive[i])
                    {
                        var type = _subCursor.Schema[i].Type;
                        _subGetters[i] = MarshalGetter(_subCursor.GetGetter<DataViewSchema.Column>, type.RawType, _subCursor.Schema[i]);
                    }
                }
            }

            private void UpdateColumnValues(string path, List<string> values)
            {
                // Cache the column values for future Getter calls.
                for (int i = 0; i < _colValues.Length; i++)
                {
                    var source = _parent._srcDirIndex[i];
                    if (source >= 0 && source < values.Count)
                    {
                        _colValues[i] = values[source].AsMemory();
                    }
                    else if (source == FilePathColIndex)
                    {
                        // Force Unix path for consistency.
                        var cleanPath = path.Replace(@"\", @"/");
                        _colValues[i] = cleanPath.AsMemory();
                    }
                }
            }

            private Delegate[] CreateGetters()
            {
                Delegate[] getters = new Delegate[Schema.Count];
                for (int i = 0; i < getters.Length; i++)
                {
                    if (!_active[i])
                    {
                        continue;
                    }

                    var type = Schema[i].Type;

                    // Use sub-cursor for all sub-columns.
                    if (IsSubColumn(i))
                    {
                        getters[i] = Utils.MarshalInvoke(CreateSubGetterDelegateCore<int>, type.RawType, i);
                    }
                    else
                    {
                        int idx = i - SubColumnCount;
                        getters[i] = Utils.MarshalInvoke(CreateGetterDelegateCore<int>, type.RawType, idx, type);
                    }
                }

                return getters;
            }

            private Delegate CreateSubGetterDelegateCore<TValue>(int col)
            {
                return (Delegate)SubGetterDelegateCore<TValue>(col);
            }

            private ValueGetter<TValue> SubGetterDelegateCore<TValue>(int col)
            {
                Ch.Check(col >= 0 && col < SubColumnCount);

                return (ref TValue value) =>
                {
                    // SubCursor may change so always requery the getter.
                    ValueGetter<TValue> getter = _subGetters[col] as ValueGetter<TValue>;
                    getter?.Invoke(ref value);
                };
            }

            private Delegate CreateGetterDelegateCore<TValue>(int col, DataViewType type)
            {
                return (Delegate)GetterDelegateCore<TValue>(col, type);
            }

            private ValueGetter<TValue> GetterDelegateCore<TValue>(int col, DataViewType type)
            {
                Ch.Check(col >= 0 && col < _colValues.Length);
                Ch.AssertValue(type);

                var conv = Conversions.Instance.GetStandardConversion(TextDataViewType.Instance, type) as ValueMapper<ReadOnlyMemory<char>, TValue>;
                if (conv == null)
                {
                    throw Ch.Except("Invalid TValue: '{0}' of the conversion.", typeof(TValue));
                }

                return (ref TValue value) =>
                {
                    conv(in _colValues[col], ref value);
                };
            }

            private bool IsSubColumn(int col)
            {
                return col < SubColumnCount;
            }

            private int SubColumnCount => Schema.Count - _parent._srcDirIndex.Length;

            private IEnumerable<int> CreateFileOrder(Random rand)
            {
                if (rand == null)
                {
                    return Enumerable.Range(0, _parent._files.Count);
                }
                else
                {
                    return Utils.GetRandomPermutation(rand, _parent._files.Count);
                }
            }

            private bool SchemasMatch(DataViewSchema schema1, DataViewSchema schema2)
            {
                if (schema1.Count != schema2.Count)
                {
                    return false;
                }

                int colLim = schema1.Count;
                for (int col = 0; col < colLim; col++)
                {
                    var type1 = schema1[col].Type;
                    var type2 = schema2[col].Type;
                    if (!type1.Equals(type2))
                    {
                        return false;
                    }
                }

                return true;
            }

            private Delegate MarshalGetter(Func<DataViewSchema.Column, ValueGetter<DataViewSchema.Column>> func, Type type, DataViewSchema.Column column)
            {
                var returnType = typeof(ValueGetter<>).MakeGenericType(type);
                var meth = func.Method;

                var typedMeth = meth.GetGenericMethodDefinition().MakeGenericMethod(type);
                return (Delegate)typedMeth.Invoke(func.Target, new object[] { column });
            }

            /// <summary>
            /// Truncate path to the specified number of trailing directories.
            /// </summary>
            /// <param name="dirCount">Number of directories to retain.</param>
            /// <param name="path">Path to truncate.</param>
            /// <param name="truncPath">The resulting truncated path.</param>
            /// <returns>true if the truncation was successful.</returns>
            private bool TryTruncatePath(int dirCount, string path, out string truncPath)
            {
                truncPath = null;

                // Remove directories that shouldn't be parsed.
                var segments = PartitionedPathUtils.SplitDirectories(path);
                segments = segments.Skip(segments.Count() - dirCount - 1);

                if (segments.Count() < dirCount - 1)
                {
                    Ch.Warning($"Path {path} did not have {dirCount} directories necessary for parsing.");
                    return false;
                }

                // Rejoin segments to create a valid path.
                truncPath = String.Join(Path.DirectorySeparatorChar.ToString(), segments);
                return true;
            }

            /// <summary>
            /// Parse all column values from the directory path.
            /// </summary>
            /// <param name="path">The directory path to parse for name/value pairs.</param>
            /// <param name="results">The resulting name value pairs.</param>
            /// <returns>true if the parsing was successfull.</returns>
            private bool TryParseValuesFromPath(string path, out List<string> results)
            {
                Contracts.CheckNonWhiteSpace(path, nameof(path));

                results = null;

                try
                {
                    results = _parent._pathParser.ParseValues(path).ToList();
                    return true;
                }
                catch (InvalidOperationException e)
                {
                    Ch.Warning($"Could not parse column values from the path {path}. Ex: {e.Message}");
                    results = null;
                    return false;
                }
            }
        }

        /// <summary>
        /// Get a path relative to the base path.
        /// </summary>
        /// <param name="basepath">A base path.</param>
        /// <param name="files">A list of files under the base path.</param>
        /// <returns>A realtive file path.</returns>
        private string GetRelativePath(string basepath, IMultiStreamSource files)
        {
            Contracts.CheckNonEmpty(basepath, nameof(basepath));

            string path = files.GetPathOrNull(0);
            _host.CheckNonEmpty(path, nameof(path));

            var relativePath = PartitionedPathUtils.MakePathRelative(basepath, path);
            return relativePath;
        }

        /// <summary>
        /// Parse the column definitions using a path parser.
        /// </summary>
        /// <param name="path">The path to a file.</param>
        /// <returns>The resulting Columns.</returns>
        private IEnumerable<Column> ParseColumns(string path)
        {
            return _pathParser.ParseColumns(path).ToArray();
        }

        /// <summary>
        /// Get the number of directories in the file path.
        /// </summary>
        /// <param name="path">A file path.</param>
        /// <returns>The number of directories</returns>
        private int GetDirectoryCount(string path)
        {
            return PartitionedPathUtils.SplitDirectories(path).Count() - 1;
        }
    }
}
