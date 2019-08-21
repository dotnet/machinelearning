// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Runtime.InteropServices;
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

[assembly: LoadableClass(typeof(ToStringTransformer), null, typeof(SignatureLoadModel),
    ToStringTransformer.UserName, ToStringTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ToStringTransformer), null, typeof(SignatureLoadRowMapper),
   ToStringTransformer.UserName, ToStringTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ToStringTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ToStringTransformerExtensionClass
    {
        /// <summary>
        /// Create a <see cref="ToStringTransformerEstimator"/>, which converts the input column specified by <paramref name="inputColumnName"/>
        /// into a string representation of its contents <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be of type <see cref="System.ReadOnlyMemory{Char}"/>
        /// <param name="inputColumnName">Name of column to convert to its string representation. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> </param>
        /// will be used as source. This column's data type can be scalar of numeric, text, and boolean</param>
        /// <seealso cref="ToStringTransformer(TransformsCatalog, InputOutputColumnPair[])"/>
        public static ToStringTransformerEstimator ToStringTransformer(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null)
            => ToStringTransformerEstimator.Create(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Create a <see cref="ToStringTransformerEstimator"/>, which converts each input column in <paramref name="columns"/> specified by <see cref="Microsoft.ML.InputOutputColumnPair.InputColumnName"/>
        /// into a string representation of its contents and stores it in the column specified by <see cref="Microsoft.ML.InputOutputColumnPair.InputColumnName"/>.
        /// The input column data type can be scalar of numeric, text, and boolean
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="columns">Array of <see cref="Microsoft.ML.InputOutputColumnPair"/>. The output column data type will be of type <see cref="System.ReadOnlyMemory{Char}"/> </param>
        /// <seealso cref="ToStringTransformer(TransformsCatalog, string, string)"/>
        public static ToStringTransformerEstimator ToStringTransformer(this TransformsCatalog catalog, params InputOutputColumnPair[] columns)
            => ToStringTransformerEstimator.Create(CatalogUtils.GetEnvironment(catalog), columns);
    }

    /// <summary>
    /// Converts one or more input columns into string representations of its contents. Supports input column's of data type numeric, text, and boolean
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Scalar of numeric, boolean, [text](xref:Microsoft.ML.Data.TextDataViewType) |
    /// | Output column data type | Scalar of [text](xref:Microsoft.ML.Data.TextDataViewType) type. |
    ///
    /// The <xref:Microsoft.ML.Transforms.ToStringTransformerEstimator> is a trivial estimator that doesn't need training.
    /// The resulting <xref:Microsoft.ML.Transforms.ToStringTransformer> converts one or more input columns into its appropriate string representation.
    ///
    /// The ToStringTransformer can be applied to one or more columns, in which case it turns each input type into its appropriate string represenation.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ToStringTransformerExtensionClass.ToStringTransformer(TransformsCatalog, InputOutputColumnPair[])"/>
    /// <seealso cref="ToStringTransformerExtensionClass.ToStringTransformer(TransformsCatalog, string, string)"/>
    public sealed class ToStringTransformerEstimator : IEstimator<ToStringTransformer>
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
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }
        #endregion

        internal static ToStringTransformerEstimator Create(IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            return new ToStringTransformerEstimator(env, outputColumnName, inputColumnName);
        }

        internal static ToStringTransformerEstimator Create(IHostEnvironment env, params InputOutputColumnPair[] columns)
        {
            var columnOptions = columns.Select(x => new Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray();
            return new ToStringTransformerEstimator(env, new Options { Columns = columnOptions });
        }

        internal ToStringTransformerEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ToStringTransformerEstimator));

            _options = new Options
            {
                Columns = new Column[1] { new Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } }
            };
        }

        internal ToStringTransformerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ToStringTransformerEstimator));

            foreach(var columnPair in options.Columns)
            {
                columnPair.Source = columnPair.Source ?? columnPair.Name;
            }
            _options = options;

        }

        public ToStringTransformer Fit(IDataView input)
        {
            return new ToStringTransformer(_host, _options.Columns, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Columns)
            {
                columns[column.Name] = new SchemaShape.Column(column.Name, SchemaShape.Column.VectorKind.Scalar,
                ColumnTypeExtensions.PrimitiveTypeFromType(typeof(string)), false, null);
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class ToStringTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Turns the given column into a column of its string representation";
        internal const string UserName = "ToString Transform";
        internal const string ShortName = "ToStringTransform";
        internal const string LoadName = "ToStringTransform";
        internal const string LoaderSignature = "ToStringTransform";

        private TypedColumn[] _columns;

        #endregion

        internal ToStringTransformer(IHostEnvironment host, ToStringTransformerEstimator.Column[] columns, IDataView input) :
            base(host.Register(nameof(ToStringTransformer)))
        {
            var schema = input.Schema;

            _columns = columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString())).ToArray();
            foreach (var column in _columns)
            {
                // No training is required so directly create the transformer
                column.CreateTransformerFromEstimator();
            }
        }

        // Factory method for SignatureLoadModel.
        internal ToStringTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(ToStringTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      string representation of type

            var columnCount = ctx.Reader.ReadInt32();

            _columns = new TypedColumn[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString());

                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);

            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new ToStringTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TOSTRI T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ToStringTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      string representation of type
            //      int c++ state array length

            ctx.Writer.Write(_columns.Count());
            foreach (var column in _columns)
            {
                ctx.Writer.Write(column.Name);
                ctx.Writer.Write(column.Source);
                ctx.Writer.Write(column.Type);

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

        #region C++ Safe handle classes

        internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private DestroyTransformedDataNative _destroySaveDataHandler;
            private IntPtr _dataSize;

            public TransformedDataSafeHandle(IntPtr handle, IntPtr dataSize, DestroyTransformedDataNative destroyCppTransformerEstimator) : base(true)
            {
                SetHandle(handle);
                _dataSize = dataSize;
                _destroySaveDataHandler = destroyCppTransformerEstimator;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shoudln't ever be one though.
                return _destroySaveDataHandler(handle, _dataSize, out IntPtr errorHandle);
            }
        }

        #endregion

        #region ColumnInfo

        // REVIEW: Since we can't do overloading on the native side due to the C style exports,
        // this was the best way I could think handle it to allow for any conversions needed based on the data type.

        #region BaseClass

        internal delegate bool DestroyCppTransformerEstimator(IntPtr estimator, out IntPtr errorHandle);
        internal delegate bool DestroyTransformerSaveData(IntPtr buffer, IntPtr bufferSize, out IntPtr errorHandle);
        internal delegate bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;
            internal TypedColumn(string name, string source, string type)
            {
                Name = name;
                Source = source;
                Type = type;
            }

            internal abstract void CreateTransformerFromEstimator();
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            public abstract void Dispose();

            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase()
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }

                using (var estimatorHandler = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {

                    success = CreateTransformerFromEstimatorHelper(estimatorHandler, out IntPtr transformer, out errorHandle);
                    if (!success)
                    {
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }

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

            internal static TypedColumn CreateTypedColumn(string name, string source, string type)
            {
                if (type == typeof(sbyte).ToString())
                {
                    return new Int8TypedColumn(name, source);
                }
                else if (type == typeof(short).ToString())
                {
                    return new Int16TypedColumn(name, source);
                }
                else if (type == typeof(int).ToString())
                {
                    return new Int32TypedColumn(name, source);
                }
                else if (type == typeof(long).ToString())
                {
                    return new Int64TypedColumn(name, source);
                }
                else if (type == typeof(byte).ToString())
                {
                    return new UInt8TypedColumn(name, source);
                }
                else if (type == typeof(ushort).ToString())
                {
                    return new UInt16TypedColumn(name, source);
                }
                else if (type == typeof(uint).ToString())
                {
                    return new UInt32TypedColumn(name, source);
                }
                else if (type == typeof(ulong).ToString())
                {
                    return new UInt64TypedColumn(name, source);
                }
                else if (type == typeof(float).ToString())
                {
                    return new FloatTypedColumn(name, source);
                }
                else if (type == typeof(double).ToString())
                {
                    return new DoubleTypedColumn(name, source);
                }
                else if (type == typeof(bool).ToString())
                {
                    return new BoolTypedColumn(name, source);
                }
                else if (type == typeof(string).ToString())
                {
                    return new StringTypedColumn(name, source);
                }
                else if (type == typeof(ReadOnlyMemory<char>).ToString())
                {
                    return new ReadOnlyCharTypedColumn(name, source);
                }

                throw new Exception($"Unsupported type {type}");
            }
        }

        internal abstract class TypedColumn<T> : TypedColumn
        {
            internal TypedColumn(string name, string source, string type) :
                base(name, source, type)
            {
            }

            internal abstract string Transform(T input);

        }

        #endregion

        #region Int8Column

        internal sealed class Int8TypedColumn : TypedColumn<sbyte>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal Int8TypedColumn(string name, string source) :
                base(name, source, typeof(sbyte).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, sbyte input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(sbyte input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int8_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region Int16Column

        internal sealed class Int16TypedColumn : TypedColumn<short>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal Int16TypedColumn(string name, string source) :
                base(name, source, typeof(short).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, short input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(short input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int16_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region Int32Column

        internal sealed class Int32TypedColumn : TypedColumn<int>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal Int32TypedColumn(string name, string source) :
                base(name, source, typeof(int).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, int input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(int input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }
            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int32_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region Int64Column

        internal sealed class Int64TypedColumn : TypedColumn<long>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal Int64TypedColumn(string name, string source) :
                base(name, source, typeof(long).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(long input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }
            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_int64_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region UInt8Column

        internal sealed class UInt8TypedColumn : TypedColumn<byte>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal UInt8TypedColumn(string name, string source) :
                base(name, source, typeof(byte).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(byte input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }
            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint8_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region UInt16Column

        internal sealed class UInt16TypedColumn : TypedColumn<ushort>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;
            internal UInt16TypedColumn(string name, string source) :
                base(name, source, typeof(ushort).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ushort input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(ushort input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint16_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region UInt32Column

        internal sealed class UInt32TypedColumn : TypedColumn<uint>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal UInt32TypedColumn(string name, string source) :
                base(name, source, typeof(uint).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, uint input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(uint input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint32_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region UInt64Column

        internal sealed class UInt64TypedColumn : TypedColumn<ulong>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal UInt64TypedColumn(string name, string source) :
                base(name, source, typeof(ulong).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ulong input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(ulong input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_uint64_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region FloatColumn

        internal sealed class FloatTypedColumn : TypedColumn<float>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal FloatTypedColumn(string name, string source) :
                base(name, source, typeof(float).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_Transform"), SuppressUnmanagedCodeSecurity]

            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, float input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(float input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_float_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region DoubleColumn

        internal sealed class DoubleTypedColumn : TypedColumn<double>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal DoubleTypedColumn(string name, string source) :
                base(name, source, typeof(double).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(double input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_double_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region BoolColumn

        internal sealed class BoolTypedColumn : TypedColumn<bool>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal BoolTypedColumn(string name, string source) :
                base(name, source, typeof(bool).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, bool input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(bool input)
            {
                var success = TransformDataNative(_transformerHandler, input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                {
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                }
                using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                {
                    byte[] buffer = new byte[outputSize.ToInt32()];
                    Marshal.Copy(output, buffer, 0, buffer.Length);
                    return Encoding.UTF8.GetString(buffer);
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_bool_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region StringColumn

        internal sealed class StringTypedColumn : TypedColumn<string>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal StringTypedColumn(string name, string source) :
                base(name, source, typeof(string).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(string input)
            {
                // Convert to byte array with NullPointer at end.
                var rawData = Encoding.UTF8.GetBytes(input + char.MinValue);
                bool result;
                GCHandle handle = GCHandle.Alloc(rawData, GCHandleType.Pinned);
                try
                {
                    IntPtr rawDataPtr = handle.AddrOfPinnedObject();
                    result = TransformDataNative(_transformerHandler, rawDataPtr, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);

                    if (!result)
                    {
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }

                    using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                    {
                        byte[] buffer = new byte[outputSize.ToInt32()];
                        Marshal.Copy(output, buffer, 0, buffer.Length);
                        return Encoding.UTF8.GetString(buffer);
                    }
                }
                finally
                {
                    handle.Free();
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region  ReadOnlyCharColumn

        internal sealed class ReadOnlyCharTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {
            private TransformerEstimatorSafeHandle _transformerHandler;

            internal ReadOnlyCharTypedColumn(string name, string source) :
                base(name, source, typeof(ReadOnlyMemory<char>).ToString())
            {
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateTransformerFromEstimator"), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator()
            {
                _transformerHandler = CreateTransformerFromEstimatorBase();
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyTransformer"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_Transform"), SuppressUnmanagedCodeSecurity]
            private static extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal override string Transform(ReadOnlyMemory<char> input)
            {
                var rawData = Encoding.UTF8.GetBytes(input.ToString() + char.MinValue);
                bool result;
                GCHandle handle = GCHandle.Alloc(rawData, GCHandleType.Pinned);
                try
                {
                    IntPtr rawDataPtr = handle.AddrOfPinnedObject();
                    result = TransformDataNative(_transformerHandler, rawDataPtr, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);

                    if (!result)
                    {
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }

                    if (outputSize.ToInt32() == 0)
                        return string.Empty;

                    using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                    {
                        byte[] buffer = new byte[outputSize.ToInt32()];
                        Marshal.Copy(output, buffer, 0, buffer.Length);
                        return Encoding.UTF8.GetString(buffer);
                    }
                }
                finally
                {
                    handle.Free();
                }
            }

            public override void Dispose()
            {
                if (!_transformerHandler.IsClosed)
                    _transformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "StringFeaturizer_string_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #endregion

        private sealed class Mapper : MapperBase
        {

            #region Class data members

            private readonly ToStringTransformer _parent;

            #endregion

            public Mapper(ToStringTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(typeof(string)))).ToArray();
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo)
            {
                ValueGetter<ReadOnlyMemory<char>> result = (ref ReadOnlyMemory<char> dst) =>
                {
                    var inputColumn = input.Schema[_parent._columns[iinfo].Source];

                    var srcGetter = input.GetGetter<T>(inputColumn);

                    T value = default;
                    srcGetter(ref value);
                    string transformed = ((TypedColumn<T>)_parent._columns[iinfo]).Transform(value);
                    dst = new ReadOnlyMemory<char>(transformed.ToArray());
                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Type inputType = input.Schema[_parent._columns[iinfo].Source].Type.RawType;
                return Utils.MarshalInvoke(MakeGetter<int>, inputType, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (_parent._columns.Any(x => x.Source == InputSchema[i].Name))
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class ToStringTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ToString",
            Desc = ToStringTransformer.Summary,
            UserName = ToStringTransformer.UserName,
            ShortName = ToStringTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ToString(IHostEnvironment env, ToStringTransformerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ToStringTransformer.ShortName, input);
            var xf = new ToStringTransformerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }

}
