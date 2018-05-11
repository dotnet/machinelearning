// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(PartitionedFileLoader.Summary, typeof(PartitionedFileLoader), typeof(PartitionedFileLoader.Arguments), typeof(SignatureDataLoader),
    PartitionedFileLoader.UserName, PartitionedFileLoader.LoadName, PartitionedFileLoader.ShortName)]

[assembly: LoadableClass(PartitionedFileLoader.Summary, typeof(PartitionedFileLoader), null, typeof(SignatureLoadDataLoader),
    PartitionedFileLoader.UserName, PartitionedFileLoader.LoadName, PartitionedFileLoader.ShortName)]

namespace Microsoft.ML.Runtime.Data
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
    public sealed class PartitionedFileLoader : IDataLoader
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
                loaderSignature: LoadName);
        }

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Base path to the directory of your partitioned files.", ShortName = "bp")]
            public string BasePath;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Append a column with the file path.", ShortName = "path")]
            public bool IncludePathColumn = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Path parser to extract column name/value pairs from the file path.", ShortName = "parser")]
            public IPartitionedPathParserFactory PathParserFactory = new ParquetPartitionedPathParserFactory();

            [Argument(ArgumentType.Multiple, HelpText = "The data loader.")]
            public SubComponent<IDataLoader, SignatureDataLoader> Loader;
        }

        public sealed class Column
        {
            [Argument(ArgumentType.Required, HelpText = "Name of the column.")]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Data type of the column.")]
            public DataKind? Type;

            [Argument(ArgumentType.Required, HelpText = "Source index of the column.")]
            public int Source;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                if (TryParse(str, out Column column))
                {
                    return column;
                }

                return null;
            }

            public static bool TryParse(string str, out Column column)
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

                DataKind? kind = null;
                if (kindStr != null && TypeParsingUtils.TryParseDataKind(kindStr, out DataKind parsedKind, out KeyRange range))
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

            public bool TryUnparse(StringBuilder sb)
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
        private readonly Column[] _columns;

        // Number of tailing directories to include.
        private readonly int _tailingDirCount;

        // An underlying loader used on each individual loader.
        private readonly SubComponent<IDataLoader, SignatureDataLoader> _subLoader;

        private readonly IPartitionedPathParser _pathParser;

        private const string RegistrationName = LoadName;
        private const string FilePathSpecName = "FilePathSpec";
        private const int FilePathColIndex = -1;

        public PartitionedFileLoader(IHostEnvironment env, Arguments args, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(files, nameof(files));

            _pathParser = args.PathParserFactory.CreateComponent(_host);
            _host.CheckValue(_pathParser, nameof(_pathParser), "Factory failed to create a FilePathSpec");

            _subLoader = args.Loader;
            _files = files;

            string relativePath = GetRelativePath(args.BasePath, files);
            _columns = ParseColumns(relativePath).ToArray();
            _tailingDirCount = GetDirectoryCount(relativePath);

            if (args.IncludePathColumn)
            {
                var pathCol = new Column()
                {
                    Name = "Path",
                    Source = FilePathColIndex,
                    Type = DataKind.Text
                };

                _columns = _columns.Concat(new[] { pathCol }).ToArray();
            }

            Schema = CreateSchema(_host, _columns, _subLoader);
        }

        private PartitionedFileLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);
            _host.AssertValue(files);

            // ** Binary format **
            // int: tailing directory count
            // int: number of columns
            // foreach column:
            //   string: column representation
            // string: subloader
            // model: file path spec

            _tailingDirCount = ctx.Reader.ReadInt32();

            int numColumns = ctx.Reader.ReadInt32();
            _host.CheckDecode(numColumns >= 0);

            _columns = new Column[numColumns];
            for (int i = 0; i < numColumns; i++)
            {
                var column = Column.Parse(ctx.LoadString());
                _host.CheckDecode(column != null);
                _columns[i] = column;
            }

            var loader = SubComponent.Parse(ctx.LoadString());
            _subLoader = new SubComponent<IDataLoader, SignatureDataLoader>(loader.Kind, loader.Settings);

            ctx.LoadModel<IPartitionedPathParser, SignatureLoadModel>(_host, out _pathParser, FilePathSpecName);

            _files = files;
            Schema = CreateSchema(_host, _columns, _subLoader);
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

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // ** Binary format **
            // int: tailing directory count
            // int: number of columns
            // foreach column:
            //   string: column representation
            // string: subloader
            // model: file path spec

            ctx.Writer.Write(_tailingDirCount);

            ctx.Writer.Write(_columns.Length);
            StringBuilder sb = new StringBuilder();
            foreach (var col in _columns)
            {
                sb.Clear();
                _host.Check(col.TryUnparse(sb));
                ctx.SaveString(sb.ToString());
            }

            ctx.SaveString(_subLoader.ToString());
            ctx.SaveModel(_pathParser, FilePathSpecName);
        }

        public bool CanShuffle => true;

        public ISchema Schema { get; }

        private ISchema SubSchema { get; set; }

        public long? GetRowCount(bool lazy = true)
        {
            return null;
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return new Cursor(_host, this, _files, needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            consolidator = null;
            var cursor = new Cursor(_host, this, _files, needCol, rand);
            return new IRowCursor[] { cursor };
        }

        /// <summary>
        /// Create a composite schema of both the partitioned columns and the underlying loader columns.
        /// </summary>
        /// <param name="ectx">The exception context.</param>
        /// <param name="cols">The partitioned columns.</param>
        /// <param name="subComponent">The sub loader.</param>
        /// <returns>The resulting schema.</returns>
        private ISchema CreateSchema(IExceptionContext ectx, Column[] cols, SubComponent<IDataLoader, SignatureDataLoader> subComponent)
        {
            Contracts.AssertValue(cols);
            Contracts.AssertValue(subComponent);

            var columnNameTypes = cols.Select((col) => new KeyValuePair<string, ColumnType>(col.Name, PrimitiveType.FromKind(col.Type.Value)));
            var colSchema = new SimpleSchema(ectx, columnNameTypes.ToArray());

            SubSchema = subComponent.CreateInstance(_host, _files).Schema;

            if (SubSchema.ColumnCount == 0)
            {
                return colSchema;
            }
            else
            {
                var schemas = new ISchema[]
                {
                    SubSchema,
                    colSchema
                };

                return new CompositeSchema(schemas);
            }
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private PartitionedFileLoader _parent;

            private bool[] _active;
            private bool[] _subActive; // Active columns of the sub-cursor.
            private Delegate[] _getters;
            private Delegate[] _subGetters; // Cached getters of the sub-cursor.

            private DvText[] _colValues; // Column values cached from the file path.
            private IRowCursor _subCursor; // Sub cursor of the current file.

            private IEnumerator<int> _fileOrder;

            public Cursor(IChannelProvider provider, PartitionedFileLoader parent, IMultiStreamSource files, Func<int, bool> predicate, IRandom rand)
                : base(provider)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(files);
                Contracts.AssertValue(predicate);

                _parent = parent;

                _active = Utils.BuildArray(Schema.ColumnCount, predicate);
                _subActive = _active.Take(SubColumnCount).ToArray();
                _colValues = new DvText[_parent._columns.Length];

                _subGetters = new Delegate[SubColumnCount];
                _getters = CreateGetters();

                _fileOrder = CreateFileOrder(rand).GetEnumerator();
            }

            public override long Batch => 0;

            public ISchema Schema => _parent.Schema;

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                var getter = _getters[col] as ValueGetter<TValue>;
                if (getter == null)
                {
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                }

                return getter;
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                    (ref UInt128 val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");

                        val = new UInt128(0, (ulong)Position);
                    };
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < Schema.ColumnCount);
                return _active[col];
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

                    IDataLoader loader = null;
                    try
                    {
                        // Load the sub cursor and reset the data.
                        loader = _parent._subLoader.CreateInstance(_parent._host, new MultiFileSource(path));
                    }
                    catch (Exception e)
                    {
                        Ch.Warning($"Failed to load file {path} due to a loader exception. Moving on to the next file. Ex: {e.Message}");
                        continue;
                    }

                    if (!SchemasMatch(_parent.SubSchema, loader.Schema))
                    {
                        Ch.Warning($"Schema of file {path} does not match.");
                        continue;
                    }

                    _subCursor = loader.GetRowCursor(col => _subActive[col]);

                    try
                    {
                        UpdateSubGetters();
                        UpdateColumnValues(relativePath, values);
                    }
                    catch (FormatException e)
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
                        var type = _parent.SubSchema.GetColumnType(i);
                        _subGetters[i] = MarshalGetter(_subCursor.GetGetter<int>, type.RawType, i);
                    }
                }
            }

            private void UpdateColumnValues(string path, List<string> values)
            {
                // Cache the column values for future Getter calls.
                for (int i = 0; i < _colValues.Length; i++)
                {
                    var col = _parent._columns[i];

                    var source = col.Source;
                    if (source >= 0 && source < values.Count)
                    {
                        _colValues[i] = new DvText(values[source]);
                    }
                    else if (source == FilePathColIndex)
                    {
                        _colValues[i] = new DvText(path);
                    }
                }
            }

            private Delegate[] CreateGetters()
            {
                Delegate[] getters = new Delegate[Schema.ColumnCount];
                for (int i = 0; i < getters.Length; i++)
                {
                    if (!_active[i])
                    {
                        continue;
                    }

                    var type = Schema.GetColumnType(i);

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

            private Delegate CreateGetterDelegateCore<TValue>(int col, ColumnType type)
            {
                return (Delegate)GetterDelegateCore<TValue>(col, type);
            }

            private ValueGetter<TValue> GetterDelegateCore<TValue>(int col, ColumnType type)
            {
                Ch.Check(col >= 0 && col < _colValues.Length);
                Ch.AssertValue(type);

                var conv = Conversions.Instance.GetStandardConversion(TextType.Instance, type) as ValueMapper<DvText, TValue>;
                if (conv == null)
                {
                    throw Ch.Except("Invalid TValue: '{0}' of the conversion.", typeof(TValue));
                }

                return (ref TValue value) =>
                {
                    conv(ref _colValues[col], ref value);
                };
            }

            private bool IsSubColumn(int col)
            {
                return col < SubColumnCount;
            }

            private int SubColumnCount => Schema.ColumnCount - _parent._columns.Length;

            private IEnumerable<int> CreateFileOrder(IRandom rand)
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

            private bool SchemasMatch(ISchema schema1, ISchema schema2)
            {
                if (schema1.ColumnCount != schema2.ColumnCount)
                {
                    return false;
                }

                int colLim = schema1.ColumnCount;
                for (int col = 0; col < colLim; col++)
                {
                    var type1 = schema1.GetColumnType(col);
                    var type2 = schema2.GetColumnType(col);
                    if (!type1.Equals(type2))
                    {
                        return false;
                    }
                }

                return true;
            }

            private Delegate MarshalGetter(Func<int, ValueGetter<int>> func, Type type, int col)
            {
                var returnType = typeof(ValueGetter<>).MakeGenericType(type);
                var meth = func.Method;

                var typedMeth = meth.GetGenericMethodDefinition().MakeGenericMethod(type);
                return (Delegate)typedMeth.Invoke(func.Target, new object[] { col });
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
                var segments = Utils.SplitDirectories(path);
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
                catch (FormatException e)
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

            var relativePath = Utils.MakePathRelative(basepath, path);
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
            return Utils.SplitDirectories(path).Count() - 1;
        }
    }
}
