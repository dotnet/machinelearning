// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.SchemaShape.Column;

[assembly: LoadableClass(typeof(RollingWindowTransformer), null, typeof(SignatureLoadModel),
    RollingWindowTransformer.UserName, RollingWindowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(RollingWindowTransformer), null, typeof(SignatureLoadRowMapper),
RollingWindowTransformer.UserName, RollingWindowTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(RollingWindowEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class RollingWindowExtensionClass
    {
        // TODO: review naming for public api before going into master.

        /// <summary>
        /// Creates a <see cref="RollingWindowEstimator"/> which computes rolling window calculations per grain. The currently supported window calculations are
        /// mean, min, and max. This also adds annotations to the output column to track the min/max window sizes, as well as what calculation was performed. The horizon
        /// is an initial offset, and the window calculation is performed starting at that initial offset, and then looping -1 until that offset is equal to 1.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="grainColumns">The list of grain columns</param>
        /// <param name="outputColumn">Where to store the result of the calculation</param>
        /// <param name="windowCalculation">The window calculation to perform</param>
        /// <param name="horizon">The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to the start of the window.</param>
        /// <param name="maxWindowSize">The maximum number of items in the window</param>
        /// <param name="minWindowSize">The minimum number of items required. If there are less, double.NaN is returned.</param>
        /// <param name="inputColumn">The source column.</param>
        /// <returns></returns>
        public static RollingWindowEstimator RollingWindow(this TransformsCatalog catalog, string[] grainColumns, string outputColumn, RollingWindowEstimator.RollingWindowCalculation windowCalculation,
            UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize = 1, string inputColumn = null)
        {
            var options = new RollingWindowEstimator.Options
            {
                GrainColumns = grainColumns,
                Column = new[] { new RollingWindowEstimator.Column() { Name = outputColumn, Source = inputColumn ?? outputColumn } },
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                MinWindowSize = minWindowSize,
                WindowCalculation = windowCalculation
            };

            return new RollingWindowEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        /// <summary>
        /// Creates a <see cref="RollingWindowEstimator"/> which computes rolling window calculations per grain. The currently supported window calculations are
        /// mean, min, and max. This also adds annotations to the output column to track the min/max window sizes, as well as what calculation was performed. The horizon
        /// is an initial offset, and the window calculation is performed starting at that initial offset, and then looping -1 until that offset is equal to 1.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="grainColumns">The list of grain columns</param>
        /// <param name="columns">List of columns mappings</param>
        /// <param name="windowCalculation">The window calculation to perform</param>
        /// <param name="horizon">The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to the start of the window.</param>
        /// <param name="maxWindowSize">The maximum number of items in the window</param>
        /// <param name="minWindowSize">The minimum number of items required. If there are less, double.NaN is returned.</param>
        /// <returns></returns>
        public static RollingWindowEstimator RollingWindow(this TransformsCatalog catalog, string[] grainColumns, InputOutputColumnPair[] columns, RollingWindowEstimator.RollingWindowCalculation windowCalculation,
            UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize = 1)
        {
            var options = new RollingWindowEstimator.Options
            {
                GrainColumns = grainColumns,
                Column = columns.Select(x => new RollingWindowEstimator.Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                MinWindowSize = minWindowSize,
                WindowCalculation = windowCalculation
            };

            return new RollingWindowEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// RollingWindow featurizer performs a rolling calculation over a window of data per grain. The currently supported window calculations are
    /// mean, min, and max. This also adds annotations to the output column to track the min/max window sizes, as well as what calculation was performed. The horizon
    /// is an initial offset, and the window calculation is performed starting at that initial offset, and then looping -1 until that offset is equal to 1.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | double |
    /// | Output column data type | vector of double. Size of vector is equal to the horizon |
    ///
    /// The <xref:Microsoft.ML.Transforms.RollingWindowEstimator> is a trivial estimator and doesn't need training.
    /// A simple example would be horizon = 1, maxWindowSize = 2, and we want to take the minimum.
    ///
    ///      +-----------+-------+-------------------+
    ///      | grain     | target| target_minimum    |
    ///      +===========+=======+===================+
    ///      | A         | 10    | [[NAN]]           |
    ///      +-----------+-------+-------------------+
    ///      | A         | 4     | [[10]]            |
    ///      +-----------+-------+-------------------+
    ///      | A         | 6     | [[4]]             |
    ///      +-----------+-------+-------------------+
    ///      | A         | 11    | [[4]]             |
    ///      +-----------+-------+-------------------+
    ///
    ///      A more complex example would be, assuming we have horizon = 2, maxWindowSize = 2, minWindowSize = 2, and we want the maximum value
    ///      +-----------+-------+-------------------+
    ///      | grain     | target| target_max        |
    ///      +===========+=======+===================+
    ///      | A         | 10    | [[NAN, NAN]]      |
    ///      +-----------+-------+-------------------+
    ///      | A         | 4     | [[NAN, NAN]]      |
    ///      +-----------+-------+-------------------+
    ///      | A         | 6     | [[NAN, 10]]       |
    ///      +-----------+-------+-------------------+
    ///      | A         | 11    | [[10, 6]]         |
    ///      +-----------+-------+-------------------+
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="RollingWindowExtensionClass.RollingWindow(TransformsCatalog, string[], string, RollingWindowEstimator.RollingWindowCalculation, UInt32, UInt32, UInt32, string)"/>
    /// <seealso cref="RollingWindowExtensionClass.RollingWindow(TransformsCatalog, string[], InputOutputColumnPair[], RollingWindowEstimator.RollingWindowCalculation, UInt32, UInt32, UInt32)"/>
    public class RollingWindowEstimator : IEstimator<RollingWindowTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        #region Options

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns",
                Name = "GrainColumn", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum horizon value",
                Name = "Horizon", ShortName = "hor", SortOrder = 2)]
            public UInt32 Horizon;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum window size",
                Name = "MaxWindowSize", ShortName = "maxsize", SortOrder = 3)]
            public UInt32 MaxWindowSize;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Minimum window size",
                Name = "MinWindowSize", ShortName = "minsize", SortOrder = 4)]
            public UInt32 MinWindowSize = 1;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "What window calculation to use",
                Name = "WindowCalculation", ShortName = "calc", SortOrder = 5)]
            public RollingWindowCalculation WindowCalculation;
        }

        #endregion

        #region Class Enums

        /// <summary>
        /// This is a representation of which RollingWindowCalculation to perform.
        /// Mean is the arithmatic mean of the window.
        /// Min is the minimum value in the window.
        /// Max is the maximum value in the window.
        /// </summary>
        public enum RollingWindowCalculation : byte
        {
            /// <summary>
            /// Mean is the arithmatic mean of the window.
            /// </summary>
            Mean = 1,

            /// <summary>
            /// Min is the minimum value in the window.
            /// </summary>
            Min = 2,

            /// <summary>
            /// Max is the maximum value in the window.
            /// </summary>
            Max = 3
        };

        #endregion

        internal RollingWindowEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");

            _host = env.Register(nameof(RollingWindowEstimator));
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");
            _host.CheckValue(options.Column, nameof(options.Column), "Columns should not be null.");
            _host.CheckNonEmpty(options.Column, nameof(options.Column), "Need at least one column pair.");
            _host.Check(options.Horizon > 0, "Can't have a horizon of 0.");
            _host.Check(options.MinWindowSize > 0, "Min window size must be greater then 0.");
            _host.Check(options.MaxWindowSize > 0, "Max window size must be greater then 0.");
            _host.Check(options.MaxWindowSize >= options.MinWindowSize, "Max window size must be greater or equal to min window size.");
            _host.Check(options.Horizon <= int.MaxValue, "Horizon must be less then or equal to int.max");

            _options = options;
        }

        public RollingWindowTransformer Fit(IDataView input)
        {
            return new RollingWindowTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            if (!AllGrainColumnsAreStrings(inputSchema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            var columns = inputSchema.ToDictionary(x => x.Name);

            // These are used in generating the column name, but don't need to be recreated in the loop.
            var calculationName = Enum.GetName(typeof(RollingWindowEstimator.RollingWindowCalculation), _options.WindowCalculation);
            var minWinName = $"MinWin{_options.MinWindowSize}";
            var maxWinName = $"MaxWin{_options.MaxWindowSize}";

            foreach (var column in _options.Column)
            {

                var inputColumn = columns[column.Source];

                if (!RollingWindowTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType} for column {column.Source} not a supported type.");

                // Create annotations
                // Since we can't get the value of the annotation from the schema shape, the current workaround is naming annotation with the value as well.
                // This workaround will need to be removed when the limitation is resolved.
                var sourceColName = column.Name;
                var annotations = new DataViewSchema.Annotations.Builder();

                ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = $"{sourceColName}_{calculationName}_{minWinName}_{maxWinName}".AsMemory();

                annotations.Add<ReadOnlyMemory<char>>($"ColumnNames={sourceColName}_{calculationName}_{minWinName}_{maxWinName}", TextDataViewType.Instance, nameValueGetter);

                columns[column.Name] = new SchemaShape.Column(column.Name, VectorKind.Vector,
                    NumberDataViewType.Double, false, SchemaShape.Create(annotations.ToAnnotations().Schema));
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class RollingWindowTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Performs a calculation over a rolling timeseries window";
        internal const string UserName = "Rolling Window Featurizer";
        internal const string ShortName = "RollingWindow";
        internal const string LoaderSignature = "RollingWindow";

        private TypedColumn[] _columns;
        private RollingWindowEstimator.Options _options;

        #endregion

        internal RollingWindowTransformer(IHostEnvironment host, IDataView input, RollingWindowEstimator.Options options) :
            base(host.Register(nameof(RollingWindowTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _columns = options.Column.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString(), _options)).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal RollingWindowTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(RollingWindowTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");

            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int length of grainColumns
            // string[] grainColumns
            // uint32 horizon
            // uint32 maxWindowSize
            // uint32 minWindowSize
            // byte windowCalculation
            // int number of column pairs
            // for each column pair:
            //      string output column name
            //      string source column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < grainColumns.Length; i++)
            {
                grainColumns[i] = ctx.Reader.ReadString();
            }

            var horizon = ctx.Reader.ReadUInt32();
            var maxWindowSize = ctx.Reader.ReadUInt32();
            var minWindowSize = ctx.Reader.ReadUInt32();
            var windowCalculation = ctx.Reader.ReadByte();

            var columnCount = ctx.Reader.ReadInt32();
            _columns = new TypedColumn[columnCount];

            _options = new RollingWindowEstimator.Options()
            {
                GrainColumns = grainColumns,
                Column = new RollingWindowEstimator.Column[columnCount],
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                MinWindowSize = minWindowSize,
                WindowCalculation = (RollingWindowEstimator.RollingWindowCalculation)windowCalculation
            };

            for (int i = 0; i < columnCount; i++)
            {
                var colName = ctx.Reader.ReadString();
                var sourceName = ctx.Reader.ReadString();
                _options.Column[i] = new RollingWindowEstimator.Column()
                {
                    Name = colName,
                    Source = sourceName
                };

                _columns[i] = TypedColumn.CreateTypedColumn(colName, sourceName, ctx.Reader.ReadString(), _options);

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new RollingWindowTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ROLWIN T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RollingWindowTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int length of grainColumns
            // string[] grainColumns
            // uint32 horizon
            // uint32 maxWindowSize
            // uint32 minWindowSize
            // byte windowCalculation
            // int number of column pairs
            // for each column pair:
            //      string output column name
            //      string source column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var grain in _options.GrainColumns)
            {
                ctx.Writer.Write(grain);
            }
            ctx.Writer.Write(_options.Horizon);
            ctx.Writer.Write(_options.MaxWindowSize);
            ctx.Writer.Write(_options.MinWindowSize);
            ctx.Writer.Write((byte)_options.WindowCalculation);

            // Save interop data.
            ctx.Writer.Write(_columns.Count());
            foreach (var column in _columns)
            {
                ctx.Writer.Write(column.Name);
                ctx.Writer.Write(column.Source);
                ctx.Writer.Write(column.Type);

                // Save C++ state
                var data = column.CreateTransformerSaveData();
                ctx.Writer.Write(data.Length);
                ctx.Writer.Write(data);
            }
        }

        public void Dispose()
        {
            foreach (var column in _columns)
            {
                column.Dispose();
            }
        }

        #region Native Safe handle classes
        internal delegate bool DestroyTransformedVectorDataNative(IntPtr columns, IntPtr rows, IntPtr items, out IntPtr errorHandle);
        internal class TransformedVectorDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private readonly DestroyTransformedVectorDataNative _destroyTransformedDataHandler;
            private readonly IntPtr _columns;
            private readonly IntPtr _rows;

            public TransformedVectorDataSafeHandle(IntPtr handle, IntPtr columns, IntPtr rows, DestroyTransformedVectorDataNative destroyTransformedDataHandler) : base(true)
            {
                SetHandle(handle);
                _destroyTransformedDataHandler = destroyTransformedDataHandler;
                _columns = columns;
                _rows = rows;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shouldn't ever be one though.
                var success = _destroyTransformedDataHandler(_columns, _rows, handle, out IntPtr errorHandle);
                return success;
            }
        }

        #endregion

        #region ColumnInfo

        #region BaseClass
        // TODO: The majority of this base class can probably be moved into the common.cs file. Look more into this before merging into master.
        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Source;
            internal readonly string Type;
            internal readonly string Name;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private static readonly Type[] _supportedTypes = new Type[] { typeof(double) };

            private protected string[] GrainColumns;

            internal TypedColumn(string name, string source, string type, string[] grainColumns)
            {
                Source = source;
                Type = type;
                Name = name;
                GrainColumns = grainColumns;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected abstract bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            public abstract void Dispose();

            public abstract Type ReturnType();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type, RollingWindowEstimator.Options options)
            {
                if (type == typeof(double).ToString() && options.WindowCalculation == RollingWindowEstimator.RollingWindowCalculation.Mean)
                {
                    return new AnalyticalDoubleTypedColumn(name, source, options);
                }
                else if (type == typeof(double).ToString() && (options.WindowCalculation == RollingWindowEstimator.RollingWindowCalculation.Min || options.WindowCalculation == RollingWindowEstimator.RollingWindowCalculation.Max))
                {
                    return new SimpleDoubleTypedColumn(name, source, options);
                }

                throw new InvalidOperationException($"Column {source} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            private protected DataViewRowCursor Cursor;
            private protected ValueGetter<ReadOnlyMemory<char>>[] GrainGetters;
            private protected readonly RollingWindowEstimator.Options Options;

            internal TypedColumn(string name, string source, string type, RollingWindowEstimator.Options options) :
                base(name, source, type, options.GrainColumns)
            {
                Options = options;

                // Initialize to the correct length
                GrainGetters = new ValueGetter<ReadOnlyMemory<char>>[GrainColumns.Length];
                Cursor = null;
            }

            internal abstract TOutputType Transform(IntPtr grainsArray, IntPtr grainsArraySize, TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected unsafe abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, TSourceType value, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    TrainingState trainingState;
                    FitResult fitResult;

                    // Declare these outside the loop so the size is only set once;
                    GCHandle[] grainHandles = new GCHandle[GrainColumns.Length];
                    IntPtr[] grainArray = new IntPtr[GrainColumns.Length];
                    GCHandle arrayHandle = default;

                    InitializeGrainGetters(input);

                    // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                    var data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                    data.MoveNext();
                    while (true)
                    {
                        // Get the state of the native estimator.
                        success = GetStateHelper(estimatorHandle, out trainingState, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If we are no longer training then exit loop.
                        if (trainingState != TrainingState.Training)
                            break;

                        // Build the grain string array
                        try
                        {
                            CreateGrainStringArrays(GrainGetters, ref grainHandles, ref arrayHandle, ref grainArray);

                            // Train the estimator
                            success = FitHelper(estimatorHandle, arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), data.Current, out fitResult, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        }
                        finally
                        {
                            FreeGrainStringArrays(ref grainHandles, ref arrayHandle);
                        }

                        // If we need to reset the data to the beginning.
                        if (fitResult == FitResult.ResetAndContinue)
                        {
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();

                            InitializeGrainGetters(input);
                        }

                        // If we are at the end of the data.
                        if (!data.MoveNext() && !Cursor.MoveNext())
                        {
                            OnDataCompletedHelper(estimatorHandle, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                            // Re-initialize the data
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                            data.MoveNext();

                            InitializeGrainGetters(input);
                        }
                    }

                    // When done training complete the estimator.
                    success = CompleteTrainingHelper(estimatorHandle, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Create the native transformer from the estimator;
                    success = CreateTransformerFromEstimatorHelper(estimatorHandle, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Manually dispose of the IEnumerator and Cursor since we dont have a using statement;
                    data.Dispose();
                    Cursor.Dispose();

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }

            private void InitializeGrainGetters(IDataView input)
            {
                // Create getters for the grain columns. Cant use using for the cursor because it may need to be reset.
                // Manually dispose of the cursor if its not null
                if (Cursor != null)
                    Cursor.Dispose();

                Cursor = input.GetRowCursor(input.Schema.Where(x => GrainColumns.Contains(x.Name)));

                for (int i = 0; i < GrainColumns.Length; i++)
                {
                    // Inititialize the enumerator and move it to a valid position.
                    GrainGetters[i] = Cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[GrainColumns[i]]);
                }

                // Move cursor to valid spot.
                Cursor.MoveNext();
            }

            public override Type ReturnType()
            {
                return typeof(TOutputType);
            }
        }

        #endregion

        // On the native side, these rolling windows are implemented as 2 separate featurizers.
        // We are only exposing 1 interface in ML.Net, but there needs to be interop code for both.
        #region AnalyticalDoubleTypedColumn

        internal sealed class AnalyticalDoubleTypedColumn : TypedColumn<double, VBuffer<double>>
        {
            internal AnalyticalDoubleTypedColumn(string name, string source, RollingWindowEstimator.Options options) :
                base(name, source, typeof(double).ToString(), options)
            {
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(RollingWindowEstimator.RollingWindowCalculation windowCalculation, UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, double value, out IntPtr outputCols, out IntPtr outputRows, out double* output, out IntPtr errorHandle);
            internal unsafe override VBuffer<double> Transform(IntPtr grainsArray, IntPtr grainsArraySize, double input)
            {
                var success = TransformDataNative(TransformerHandler, grainsArray, grainsArraySize, input, out IntPtr outputCols, out IntPtr outputRows, out double* output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using var handler = new TransformedVectorDataSafeHandle(new IntPtr(output), outputCols, outputRows, DestroyTransformedDataNative);

                // Not looping through the outputRows because we know for now there is only 1 row. If that changes will need to update the code here.
                var outputArray = new double[outputCols.ToInt32()];

                for (int i = 0; i < outputCols.ToInt32(); i++)
                {
                    outputArray[i] = *output++;
                }

                var buffer = new VBuffer<double>(outputCols.ToInt32(), outputArray);
                return buffer;
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool DestroyTransformedDataNative(IntPtr columns, IntPtr rows, IntPtr items, out IntPtr errorHandle);

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                return CreateEstimatorNative(Options.WindowCalculation, Options.Horizon, Options.MaxWindowSize, Options.MinWindowSize, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, grainsArray, grainsArraySize, value, out fitResult, out errorHandle);

            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }
        }

        #endregion

        // On the native side, these rolling windows are implemented as 2 separate featurizers.
        // We are only exposing 1 interface in ML.Net, but there needs to be interop code for both.
        #region SimpleDoubleTypedColumn

        internal sealed class SimpleDoubleTypedColumn : TypedColumn<double, VBuffer<double>>
        {
            internal SimpleDoubleTypedColumn(string name, string source, RollingWindowEstimator.Options options) :
                base(name, source, typeof(double).ToString(), options)
            {
            }

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(RollingWindowEstimator.RollingWindowCalculation windowCalculation, UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, double value, out IntPtr outputCols, out IntPtr outputRows, out double* output, out IntPtr errorHandle);
            internal unsafe override VBuffer<double> Transform(IntPtr grainsArray, IntPtr grainsArraySize, double input)
            {
                var success = TransformDataNative(TransformerHandler, grainsArray, grainsArraySize, input, out IntPtr outputCols, out IntPtr outputRows, out double* output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using var handler = new TransformedVectorDataSafeHandle(new IntPtr(output), outputCols, outputRows, DestroyTransformedDataNative);

                var outputArray = new double[outputCols.ToInt32()];

                for (int i = 0; i < outputCols.ToInt32(); i++)
                {
                    outputArray[i] = *output++;
                }

                var buffer = new VBuffer<double>(outputCols.ToInt32(), outputArray);
                return buffer;
            }

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool DestroyTransformedDataNative(IntPtr columns, IntPtr rows, IntPtr items, out IntPtr errorHandle);

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                // We are subtracting one from the window calculation because these are 2 different featurizers in the native code and both native enums
                // start at 1.
                return CreateEstimatorNative(Options.WindowCalculation - 1, Options.Horizon, Options.MaxWindowSize, Options.MinWindowSize, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, grainsArray, grainsArraySize, value, out fitResult, out errorHandle);

            }

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "SimpleRollingWindowFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }
        }

        #endregion

        #endregion

        private sealed class Mapper : MapperBase
        {
            private static readonly FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate> _makeGetterMethodInfo
                = FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate>.Create(target => target.MakeGetter<int, int>);

            #region Class members

            private readonly RollingWindowTransformer _parent;

            #endregion

            public Mapper(RollingWindowTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                // To add future support for when this will do multiple window sizes at once, output will be a 2d vector so nothing will need to change when that is implemented.

                // Create annotations
                // We create 4 annotations, these are used by the PivotFeaturizer.
                // We create annotations for the minWindowSize, maxWindowSize, this featurizerName, and which calculation was performed.
                // Since we can't get the value of the annotation from the schema shape, the current workaround is naming annotation with the value as well.
                // This workaround will need to be removed when the limitation is resolved.
                var outputColumns = new DataViewSchema.DetachedColumn[_parent._options.Column.Length];

                // These are used in generating the column name, but don't need to be recreated in the loop.
                var calculationName = Enum.GetName(typeof(RollingWindowEstimator.RollingWindowCalculation), _parent._options.WindowCalculation);
                var minWinName = $"MinWin{_parent._options.MinWindowSize}";
                var maxWinName = $"MaxWin{_parent._options.MaxWindowSize}";

                for (int i = 0; i < _parent._options.Column.Length; i++)
                {
                    var sourceColName = _parent._options.Column[i].Name;
                    var annotations = new DataViewSchema.Annotations.Builder();

                    ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = $"{sourceColName}_{calculationName}_{minWinName}_{maxWinName}".AsMemory();

                    annotations.Add<ReadOnlyMemory<char>>($"ColumnNames={sourceColName}_{calculationName}_{minWinName}_{maxWinName}", TextDataViewType.Instance, nameValueGetter);

                    outputColumns[i] = new DataViewSchema.DetachedColumn(_parent._options.Column[i].Name, new VectorDataViewType(NumberDataViewType.Double, 1, (int)_parent._options.Horizon), annotations.ToAnnotations());
                }

                return outputColumns;
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                // Initialize grain getters.
                int grainColumnCount = _parent._options.GrainColumns.Length;
                ValueGetter<ReadOnlyMemory<char>>[] grainGetters = new ValueGetter<ReadOnlyMemory<char>>[grainColumnCount];
                for (int i = 0; i < grainGetters.Length; i++)
                {
                    grainGetters[i] = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent._options.GrainColumns[i]]);
                }

                // Declaring these outside so they are only done once
                GCHandle[] grainHandles = new GCHandle[grainColumnCount];
                IntPtr[] grainArray = new IntPtr[grainHandles.Length];
                GCHandle arrayHandle = default;

                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    TSourceType value = default;

                    // Build the string array
                    try
                    {
                        CreateGrainStringArrays(grainGetters, ref grainHandles, ref arrayHandle, ref grainArray);

                        srcGetterScalar(ref value);

                        dst = ((TypedColumn<TSourceType, TOutputType>)_parent._columns[iinfo]).Transform(arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), value);
                    }
                    finally
                    {
                        FreeGrainStringArrays(ref grainHandles, ref arrayHandle);
                    }
                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Type inputType = input.Schema[_parent._columns[iinfo].Source].Type.RawType;
                Type outputType = _parent._columns[iinfo].ReturnType();

                return Utils.MarshalInvoke(_makeGetterMethodInfo, this, inputType, outputType, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (_parent._options.GrainColumns.Any(x => x == InputSchema[i].Name) || _parent._options.Column.Any(x => x.Source == InputSchema[i].Name))
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class RollingWindowEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.RollingWindow",
            Desc = RollingWindowTransformer.Summary,
            UserName = RollingWindowTransformer.UserName,
            ShortName = RollingWindowTransformer.ShortName)]
        public static CommonOutputs.TransformOutput AnalyticalRollingWindow(IHostEnvironment env, RollingWindowEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, RollingWindowTransformer.ShortName, input);
            var xf = new RollingWindowEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
