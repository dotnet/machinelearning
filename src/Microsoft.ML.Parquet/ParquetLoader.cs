// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Parquet;
using Parquet.Data;
using Parquet.File.Values.Primitives;

[assembly: LoadableClass(ParquetLoader.Summary, typeof(ParquetLoader), typeof(ParquetLoader.Arguments), typeof(SignatureDataLoader),
    ParquetLoader.LoaderName, ParquetLoader.LoaderSignature, ParquetLoader.ShortName)]

[assembly: LoadableClass(ParquetLoader.Summary, typeof(ParquetLoader), null, typeof(SignatureLoadDataLoader),
    ParquetLoader.LoaderName, ParquetLoader.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Loads a parquet file into an IDataView. Supports basic mapping from Parquet input column data types to framework data types.
    /// </summary>
    public sealed class ParquetLoader : IDataLoader, IDisposable
    {
        /// <summary>
        /// A Column is a singular representation that consolidates all the related column chunks in the
        /// Parquet file. Information stored within the Column includes its name, raw type read from Parquet,
        /// its corresponding ColumnType, and index.
        /// Complex columns in Parquet like structs, maps, and lists are flattened into multiple columns.
        /// </summary>
        private sealed class Column
        {
            /// <summary>
            /// The name of the column.
            /// </summary>
            public readonly string Name;

            /// <summary>
            /// The column type of the column, translated from ParquetType.
            /// null when ParquetType is unsupported.
            /// </summary>
            public readonly ColumnType ColType;

            /// <summary>
            /// The DataField representation in the Parquet DataSet.
            /// </summary>
            public readonly DataField DataField;

            /// <summary>
            /// The DataType read from the Parquet file.
            /// </summary>
            public readonly DataType DataType;

            public Column(string name, ColumnType colType, DataField dataField, DataType dataType)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertValue(colType);
                Contracts.AssertValue(dataField);

                Name = name;
                ColType = colType;
                DataType = dataType;
                DataField = dataField;
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of column chunk values to cache while reading from parquet file", ShortName = "chunkSize")]
            public int ColumnChunkReadSize = _defaultColumnChunkReadSize;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, will read large numbers as dates", ShortName = "bigIntDates")]
            public bool TreatBigIntegersAsDates = true;
        }

        internal const string Summary = "IDataView loader for Parquet files.";
        internal const string LoaderName = "Parquet Loader";
        internal const string LoaderSignature = "ParquetLoader";
        internal const string ShortName = "Parquet";
        internal const string ModelSignature = "PARQELDR";

        private const string SchemaCtxName = "Schema.idv";

        private readonly IHost _host;
        private readonly Stream _parquetStream;
        private readonly ParquetOptions _parquetOptions;
        private readonly int _columnChunkReadSize;
        private readonly Column[] _columnsLoaded;
        private const int _defaultColumnChunkReadSize = 1000000;

        private bool _disposed;
        private long? _rowCount;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: ModelSignature,
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Add Schema to Model Context
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ParquetLoader).Assembly.FullName);
        }

        public ParquetLoader(IHostEnvironment env, Arguments args, IMultiStreamSource files)
            : this(env, args, OpenStream(files))
        {
        }

        public ParquetLoader(IHostEnvironment env, Arguments args, string filepath)
            : this(env, args, OpenStream(filepath))
        {
        }

        public ParquetLoader(IHostEnvironment env, Arguments args, Stream stream)
            : this(args, env.Register(LoaderSignature), stream)
        {
        }

        private ParquetLoader(Arguments args, IHost host, Stream stream)
        {
            Contracts.AssertValue(host, nameof(host));
            _host = host;

            _host.CheckValue(args, nameof(args));
            _host.CheckValue(stream, nameof(stream));
            _host.CheckParam(stream.CanRead, nameof(stream), "input stream must be readable");
            _host.CheckParam(stream.CanSeek, nameof(stream), "input stream must be seekable");
            _host.CheckParam(stream.Position == 0, nameof(stream), "input stream must be at head");

            using (var ch = _host.Start("Initializing host"))
            {
                _parquetStream = stream;
                _parquetOptions = new ParquetOptions()
                {
                    TreatByteArrayAsString = true,
                    TreatBigIntegersAsDates = args.TreatBigIntegersAsDates
                };

                DataSet schemaDataSet;

                try
                {
                    // We only care about the schema so ignore the rows.
                    ReaderOptions readerOptions = new ReaderOptions()
                    {
                        Count = 0,
                        Offset = 0
                    };
                    schemaDataSet = ParquetReader.Read(stream, _parquetOptions, readerOptions);
                    _rowCount = schemaDataSet.TotalRowCount;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException("Cannot read Parquet file", ex);
                }

                _columnChunkReadSize = args.ColumnChunkReadSize;
                _columnsLoaded = InitColumns(schemaDataSet);
                Schema = CreateSchema(_host, _columnsLoaded);
            }
        }

        private ParquetLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);
            _host.AssertValue(files);

            // *** Binary format ***
            // int: cached chunk size
            // bool: TreatBigIntegersAsDates flag
            // Schema of the loader (0x00010002)

            _columnChunkReadSize = ctx.Reader.ReadInt32();
            bool treatBigIntegersAsDates = ctx.Reader.ReadBoolean();

            if (ctx.Header.ModelVerWritten >= 0x00010002)
            {
                // Load the schema
                byte[] buffer = null;
                if (!ctx.TryLoadBinaryStream(SchemaCtxName, r => buffer = r.ReadByteArray()))
                    throw _host.ExceptDecode();
                var strm = new MemoryStream(buffer, writable: false);
                var loader = new BinaryLoader(_host, new BinaryLoader.Arguments(), strm);
                Schema = loader.Schema;
            }

            // Only load Parquest related data if a file is present. Otherwise, just the Schema is valid.
            if (files.Count > 0)
            {
                _parquetOptions = new ParquetOptions()
                {
                    TreatByteArrayAsString = true,
                    TreatBigIntegersAsDates = treatBigIntegersAsDates
                };

                _parquetStream = OpenStream(files);
                DataSet schemaDataSet;

                try
                {
                    // We only care about the schema so ignore the rows.
                    ReaderOptions readerOptions = new ReaderOptions()
                    {
                        Count = 0,
                        Offset = 0
                    };
                    schemaDataSet = ParquetReader.Read(_parquetStream, _parquetOptions, readerOptions);
                    _rowCount = schemaDataSet.TotalRowCount;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException("Cannot read Parquet file", ex);
                }

                _columnsLoaded = InitColumns(schemaDataSet);
                Schema = CreateSchema(_host, _columnsLoaded);
            }
            else if (Schema == null)
            {
                throw _host.Except("Parquet loader must be created with one file");
            }
        }

        public static ParquetLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost host = env.Register(LoaderName);

            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(files, nameof(files));

            return host.Apply("Loading Model",
                ch => new ParquetLoader(host, ctx, files));
        }

        /// <summary>
        /// Helper function called by the ParquetLoader constructor to initialize the Columns that belong in the Parquet file.
        /// Composite data fields are flattened; for example, a Map Field in Parquet is flattened into a Key column and a Value
        /// column.
        /// </summary>
        /// <param name="dataSet">The schema data set.</param>
        /// <returns>The array of flattened columns instantiated from the parquet file.</returns>
        private Column[] InitColumns(DataSet dataSet)
        {
            List<Column> columnsLoaded = new List<Column>();

            foreach (var parquetField in dataSet.Schema.Fields)
            {
                FlattenFields(parquetField, ref columnsLoaded, false);
            }
            return columnsLoaded.ToArray();
        }

        private void FlattenFields(Field field, ref List<Column> cols, bool isRepeatable)
        {
            if (field is DataField df)
            {
                if (isRepeatable)
                {
                    cols.Add(new Column(df.Path, ConvertFieldType(DataType.Unspecified), df, DataType.Unspecified));
                }
                else
                {
                    cols.Add(new Column(df.Path, ConvertFieldType(df.DataType), df, df.DataType));
                }
            }
            else if (field is MapField mf)
            {
                var key = mf.Key;
                cols.Add(new Column(key.Path, ConvertFieldType(DataType.Unspecified), key, DataType.Unspecified));

                var val = mf.Value;
                cols.Add(new Column(val.Path, ConvertFieldType(DataType.Unspecified), val, DataType.Unspecified));
            }
            else if (field is StructField sf)
            {
                foreach (var structField in sf.Fields)
                {
                    FlattenFields(structField, ref cols, isRepeatable);
                }
            }
            else if (field is ListField lf)
            {
                FlattenFields(lf.Item, ref cols, true);
            }
            else
            {
                throw _host.ExceptNotSupp("Encountered unknown Parquet field type(Currently recognizes data, map, list, and struct).");
            }
        }

        /// <summary>
        /// Create a new schema from the given columns.
        /// </summary>
        /// <param name="ectx">The exception context.</param>
        /// <param name="cols">The columns.</param>
        /// <returns>The resulting schema.</returns>
        private ISchema CreateSchema(IExceptionContext ectx, Column[] cols)
        {
            Contracts.AssertValue(ectx);
            Contracts.AssertValue(cols);

            var columnNameTypes = cols.Select((col) => new KeyValuePair<string, ColumnType>(col.Name, col.ColType));
            return new SimpleSchema(ectx, columnNameTypes.ToArray());
        }

        /// <summary>
        /// Translates Parquet types to ColumnTypes.
        /// </summary>
        private ColumnType ConvertFieldType(DataType parquetType)
        {
            switch (parquetType)
            {
                case DataType.Boolean:
                    return BoolType.Instance;
                case DataType.Byte:
                    return NumberType.U1;
                case DataType.SignedByte:
                    return NumberType.I1;
                case DataType.UnsignedByte:
                    return NumberType.U1;
                case DataType.Short:
                    return NumberType.I2;
                case DataType.UnsignedShort:
                    return NumberType.U2;
                case DataType.Int16:
                    return NumberType.I2;
                case DataType.UnsignedInt16:
                    return NumberType.U2;
                case DataType.Int32:
                    return NumberType.I4;
                case DataType.Int64:
                    return NumberType.I8;
                case DataType.Int96:
                    return NumberType.UG;
                case DataType.ByteArray:
                    return new VectorType(NumberType.U1);
                case DataType.String:
                    return TextType.Instance;
                case DataType.Float:
                    return NumberType.R4;
                case DataType.Double:
                    return NumberType.R8;
                case DataType.Decimal:
                    return NumberType.R8;
                case DataType.DateTimeOffset:
                    return DateTimeOffsetType.Instance;
                case DataType.Interval:
                    return TimeSpanType.Instance;
                default:
                    return TextType.Instance;
            }
        }

        private static Stream OpenStream(IMultiStreamSource files)
        {
            Contracts.CheckValue(files, nameof(files));
            Contracts.CheckParam(files.Count == 1, nameof(files), "Parquet loader must be created with one file");
            return files.Open(0);
        }

        private static Stream OpenStream(string filename)
        {
            Contracts.CheckNonEmpty(filename, nameof(filename));
            var files = new MultiFileSource(filename);
            return OpenStream(files);
        }

        public bool CanShuffle => true;

        public ISchema Schema { get; }

        public long? GetRowCount(bool lazy = true)
        {
            return _rowCount;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            return new Cursor(this, predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            consolidator = null;
            return new IRowCursor[] { GetRowCursor(predicate, rand) };
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: cached chunk size
            // bool: TreatBigIntegersAsDates flag
            // Schema of the loader

            ctx.Writer.Write(_columnChunkReadSize);
            ctx.Writer.Write(_parquetOptions.TreatBigIntegersAsDates);

            // Save the schema
            var noRows = new EmptyDataView(_host, Schema);
            var saverArgs = new BinarySaver.Arguments();
            saverArgs.Silent = true;
            var saver = new BinarySaver(_host, saverArgs);
            using (var strm = new MemoryStream())
            {
                var allColumns = Enumerable.Range(0, Schema.ColumnCount).ToArray();
                saver.SaveData(strm, noRows, allColumns);
                ctx.SaveBinaryStream(SchemaCtxName, w => w.WriteByteArray(strm.ToArray()));
            }
        }

        private sealed class Cursor : RootCursorBase, IRowCursor
        {
            private readonly ParquetLoader _loader;
            private readonly Stream _fileStream;
            private readonly ParquetConversions _parquetConversions;
            private readonly int[] _actives;
            private readonly int[] _colToActivesIndex;
            private readonly Delegate[] _getters;
            private readonly ReaderOptions _readerOptions;
            private int _curDataSetRow;
            private IEnumerator<int> _dataSetEnumerator;
            private IEnumerator<int> _blockEnumerator;
            private IList[] _columnValues;
            private IRandom _rand;

            public Cursor(ParquetLoader parent, Func<int, bool> predicate, IRandom rand)
               : base(parent._host)
            {
                Ch.AssertValue(predicate);
                Ch.AssertValue(parent._parquetStream);

                _loader = parent;
                _fileStream = parent._parquetStream;
                _parquetConversions = new ParquetConversions(Ch);
                _rand = rand;

                // Create Getter delegates
                Utils.BuildSubsetMaps(Schema.ColumnCount, predicate, out _actives, out _colToActivesIndex);
                _readerOptions = new ReaderOptions
                {
                    Count = _loader._columnChunkReadSize,
                    Columns = _loader._columnsLoaded.Select(i => i.Name).ToArray()
                };

                // The number of blocks is calculated based on the specified rows in a block (defaults to 1M).
                // Since we want to shuffle the blocks in addition to shuffling the rows in each block, checks
                // are put in place to ensure we can produce a shuffle order for the blocks.
                var numBlocks = MathUtils.DivisionCeiling((long)parent.GetRowCount(), _readerOptions.Count);
                if (numBlocks > int.MaxValue)
                {
                    throw _loader._host.ExceptParam(nameof(Arguments.ColumnChunkReadSize), "Error due to too many blocks. Try increasing block size.");
                }
                var blockOrder = CreateOrderSequence((int)numBlocks);
                _blockEnumerator = blockOrder.GetEnumerator();

                _dataSetEnumerator = Enumerable.Empty<int>().GetEnumerator();
                _columnValues = new IList[_actives.Length];
                _getters = new Delegate[_actives.Length];
                for (int i = 0; i < _actives.Length; ++i)
                {
                    int columnIndex = _actives[i];
                    _getters[i] = CreateGetterDelegate(columnIndex);
                }
            }

            #region CreateGetterDelegates
            private Delegate CreateGetterDelegate(int col)
            {
                Ch.CheckParam(IsColumnActive(col), nameof(col));

                var parquetType = _loader._columnsLoaded[col].DataType;
                switch (parquetType)
                {
                    case DataType.Boolean:
                        return CreateGetterDelegateCore<bool, bool>(col, _parquetConversions.Conv);
                    case DataType.Byte:
                        return CreateGetterDelegateCore<byte, byte>(col, _parquetConversions.Conv);
                    case DataType.SignedByte:
                        return CreateGetterDelegateCore<sbyte?, sbyte>(col, _parquetConversions.Conv);
                    case DataType.UnsignedByte:
                        return CreateGetterDelegateCore<byte, byte>(col, _parquetConversions.Conv);
                    case DataType.Short:
                        return CreateGetterDelegateCore<short?, short>(col, _parquetConversions.Conv);
                    case DataType.UnsignedShort:
                        return CreateGetterDelegateCore<ushort, ushort>(col, _parquetConversions.Conv);
                    case DataType.Int16:
                        return CreateGetterDelegateCore<short?, short>(col, _parquetConversions.Conv);
                    case DataType.UnsignedInt16:
                        return CreateGetterDelegateCore<ushort, ushort>(col, _parquetConversions.Conv);
                    case DataType.Int32:
                        return CreateGetterDelegateCore<int?, int>(col, _parquetConversions.Conv);
                    case DataType.Int64:
                        return CreateGetterDelegateCore<long?, long>(col, _parquetConversions.Conv);
                    case DataType.Int96:
                        return CreateGetterDelegateCore<BigInteger, UInt128>(col, _parquetConversions.Conv);
                    case DataType.ByteArray:
                        return CreateGetterDelegateCore<byte[], VBuffer<Byte>>(col, _parquetConversions.Conv);
                    case DataType.String:
                        return CreateGetterDelegateCore<string, ReadOnlyMemory<char>>(col, _parquetConversions.Conv);
                    case DataType.Float:
                        return CreateGetterDelegateCore<float?, Single>(col, _parquetConversions.Conv);
                    case DataType.Double:
                        return CreateGetterDelegateCore<double?, Double>(col, _parquetConversions.Conv);
                    case DataType.Decimal:
                        return CreateGetterDelegateCore<decimal?, Double>(col, _parquetConversions.Conv);
                    case DataType.DateTimeOffset:
                        return CreateGetterDelegateCore<DateTimeOffset, DateTimeOffset>(col, _parquetConversions.Conv);
                    case DataType.Interval:
                        return CreateGetterDelegateCore<Interval, TimeSpan>(col, _parquetConversions.Conv);
                    default:
                        return CreateGetterDelegateCore<IList, ReadOnlyMemory<char>>(col, _parquetConversions.Conv);
                }
            }

            private ValueGetter<TValue> CreateGetterDelegateCore<TSource, TValue>(int col, ValueMapper<TSource, TValue> valueConverter)
            {
                Ch.CheckParam(IsColumnActive(col), nameof(col));
                Ch.CheckValue(valueConverter, nameof(valueConverter));

                int activeIdx = _colToActivesIndex[col];

                return (ref TValue value) =>
                {
                    TSource val = (TSource)_columnValues[activeIdx][_curDataSetRow];
                    valueConverter(ref val, ref value);
                };
            }
            #endregion

            protected override bool MoveNextCore()
            {
                if (_dataSetEnumerator.MoveNext())
                {
                    _curDataSetRow = _dataSetEnumerator.Current;
                    return true;
                }
                else if (_blockEnumerator.MoveNext())
                {
                    _readerOptions.Offset = (long)_blockEnumerator.Current * _readerOptions.Count;

                    // When current dataset runs out, read the next portion of the parquet file.
                    DataSet ds;
                    lock (_loader._parquetStream)
                    {
                        ds = ParquetReader.Read(_loader._parquetStream, _loader._parquetOptions, _readerOptions);
                    }

                    var dataSetOrder = CreateOrderSequence(ds.RowCount);
                    _dataSetEnumerator = dataSetOrder.GetEnumerator();
                    _curDataSetRow = dataSetOrder.ElementAt(0);

                    // Cache list for each active column
                    for (int i = 0; i < _actives.Length; i++)
                    {
                        Column col = _loader._columnsLoaded[_actives[i]];
                        _columnValues[i] = ds.GetColumn(col.DataField);
                    }

                    return _dataSetEnumerator.MoveNext();
                }
                return false;
            }

            public ISchema Schema => _loader.Schema;

            public override long Batch => 0;

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(IsColumnActive(col), nameof(col), "requested column not active");

                var getter = _getters[_colToActivesIndex[col]] as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));

                return getter;
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                return
                   (ref UInt128 val) =>
                   {
                       // Unique row id consists of Position of cursor (how many times MoveNext has been called), and position in file
                       Ch.Check(IsGood, "Cannot call ID getter in current state");
                       val = new UInt128((ulong)(_readerOptions.Offset + _curDataSetRow), 0);
                   };
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _colToActivesIndex.Length, nameof(col));
                return _colToActivesIndex[col] >= 0;
            }

            /// <summary>
            /// Creates a in-order or shuffled sequence, based on whether _rand is specified.
            /// If unable to create a shuffle sequence, will default to sequential.
            /// </summary>
            /// <param name="size">Number of elements in the sequence.</param>
            /// <returns></returns>
            private IEnumerable<int> CreateOrderSequence(int size)
            {
                IEnumerable<int> order;
                try
                {
                    order = _rand == null ? Enumerable.Range(0, size) : Utils.GetRandomPermutation(_rand, size);
                }
                catch (OutOfMemoryException)
                {
                    order = Enumerable.Range(0, size);
                }
                return order;
            }
        }

        #region Dispose

        private void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                _parquetStream.Dispose();
            }
            _disposed = true;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~ParquetLoader()
        {
            Dispose(false);
        }

        #endregion

        /// <summary>
        /// Contains conversion functions to convert Parquet type values to framework type values.
        /// </summary>
        private sealed class ParquetConversions
        {
            private readonly IChannel _ch;

            public ParquetConversions(IChannel channel)
            {
                _ch = channel;
            }

            public void Conv(ref byte[] src, ref VBuffer<Byte> dst) => dst = src != null ? new VBuffer<byte>(src.Length, src) : new VBuffer<byte>(0, new byte[0]);

            public void Conv(ref sbyte? src, ref sbyte dst) => dst = (sbyte)src;

            public void Conv(ref byte src, ref byte dst) => dst = src;

            public void Conv(ref short? src, ref short dst) => dst = (short)src;

            public void Conv(ref ushort src, ref ushort dst) => dst = src;

            public void Conv(ref int? src, ref int dst) => dst = (int)src;

            public void Conv(ref long? src, ref long dst) => dst = (long)src;

            public void Conv(ref float? src, ref Single dst) => dst = src ?? Single.NaN;

            public void Conv(ref double? src, ref Double dst) => dst = src ?? Double.NaN;

            public void Conv(ref decimal? src, ref Double dst) => dst = src != null ? Decimal.ToDouble((decimal)src) : Double.NaN;

            public void Conv(ref string src, ref ReadOnlyMemory<char> dst) => dst = src.AsMemory();

            //Behavior for NA values is undefined.
            public void Conv(ref bool src, ref bool dst) => dst = src;

            public void Conv(ref DateTimeOffset src, ref DateTimeOffset dst) => dst = src;

            public void Conv(ref IList src, ref ReadOnlyMemory<char> dst) => dst = ConvertListToString(src).AsMemory();

            /// <summary>
            ///  Converts a System.Numerics.BigInteger value to a UInt128 data type value.
            /// </summary>
            /// <param name="src">BigInteger value.</param>
            /// <param name="dst">UInt128 object.</param>
            public void Conv(ref BigInteger src, ref UInt128 dst)
            {
                try
                {
                    byte[] arr = src.ToByteArray();
                    Array.Resize(ref arr, 16);
                    ulong lo = BitConverter.ToUInt64(arr, 0);
                    ulong hi = BitConverter.ToUInt64(arr, 8);
                    dst = new UInt128(lo, hi);
                }
                catch (Exception ex)
                {
                    _ch.Error("Cannot convert BigInteger to UInt128. Exception : '{0}'", ex.Message);
                    dst = default;
                }
            }

            /// <summary>
            /// Converts a Parquet Interval data type value to a TimeSpan data type value.
            /// </summary>
            /// <param name="src">Parquet Interval value (int : months, int : days, int : milliseconds).</param>
            /// <param name="dst">TimeSpan object.</param>
            public void Conv(ref Interval src, ref TimeSpan dst)
            {
                dst = TimeSpan.FromDays(src.Months * 30 + src.Days) + TimeSpan.FromMilliseconds(src.Millis);
            }

            private string ConvertListToString(IList list)
            {
                if (list == null)
                {
                    return String.Empty;
                }

                StringBuilder sb = new StringBuilder();
                var enu = list.GetEnumerator();
                while (enu.MoveNext())
                {
                    if (enu.Current is IList && enu.Current.GetType().IsGenericType)
                    {
                        sb.Append("[" + ConvertListToString((IList)enu.Current) + "],");
                    }
                    else
                    {
                        sb.Append(enu.Current?.ToString() + ",");
                    }
                }

                if (sb.Length > 0)
                    sb.Remove(sb.Length - 1, 1);

                return sb.ToString();
            }
        }
    }
}