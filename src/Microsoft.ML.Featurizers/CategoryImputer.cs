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
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(CategoryImputerTransformer), null, typeof(SignatureLoadModel),
    CategoryImputerTransformer.UserName, CategoryImputerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CategoryImputerTransformer), null, typeof(SignatureLoadRowMapper),
   CategoryImputerTransformer.UserName, CategoryImputerTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(CategoryImputerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class CategoryImputerExtensionClass
    {
        /// <summary>
        /// Create a <see cref="CategoryImputerEstimator"/>, which fills in the missing values in a column with the most frequent value.
        /// </summary>
        /// <param name="catalog">Transform Catalog</param>
        /// <param name="outputColumnName">Output column name</param>
        /// <param name="inputColumnName">Input column name, if null defaults to <paramref name="outputColumnName"/></param>
        /// <returns></returns>
        public static CategoryImputerEstimator CatagoryImputerTransformer(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null)
            => CategoryImputerEstimator.Create(CatalogUtils.GetEnvironment(catalog), outputColumnName, inputColumnName);

        /// <summary>
        /// Create a <see cref="CategoryImputerEstimator"/>, which fills in the missing values in a column with the most frequent value.
        /// </summary>
        /// <param name="catalog">Transform Catalog</param>
        /// <param name="columns">List of <see cref="InputOutputColumnPair"/> to fill in missing values</param>
        /// <returns></returns>
        public static CategoryImputerEstimator CatagoryImputerTransformer(this TransformsCatalog catalog, params InputOutputColumnPair[] columns)
            => CategoryImputerEstimator.Create(CatalogUtils.GetEnvironment(catalog), columns);
    }

    /// <summary>
    /// The CategoryImputer replaces missing values with the most common value in that column.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | All scalar types |
    /// | Output column data type | Same as input column |
    ///
    /// The <xref:Microsoft.ML.Transforms.CategoryImputerEstimator> is not a trivial estimator and needs training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="CategoryImputerExtensionClass.CatagoryImputerTransformer(TransformsCatalog, InputOutputColumnPair[])"/>
    /// <seealso cref="CategoryImputerExtensionClass.CatagoryImputerTransformer(TransformsCatalog, string, string)"/>
    public sealed class CategoryImputerEstimator : IEstimator<CategoryImputerTransformer>
    {
        private readonly Options _options;

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

        internal sealed class Options: TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        #endregion

        internal static CategoryImputerEstimator Create(IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            return new CategoryImputerEstimator(env, outputColumnName, inputColumnName);
        }

        internal static CategoryImputerEstimator Create(IHostEnvironment env, params InputOutputColumnPair[] columns)
        {
            var columnOptions = columns.Select(x => new Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray();
            return new CategoryImputerEstimator(env, new Options { Columns = columnOptions });
        }

        internal CategoryImputerEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CategoryImputerEstimator));
            _options = new Options
            {
                Columns = new Column[1] { new Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } }
            };
        }

        internal CategoryImputerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CategoryImputerEstimator));

            foreach (var columnPair in options.Columns)
            {
                columnPair.Source = columnPair.Source ?? columnPair.Name;
            }

            _options = options;
        }

        public CategoryImputerTransformer Fit(IDataView input)
        {
            return new CategoryImputerTransformer(_host, input, _options.Columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Columns)
            {
                var inputColumn = columns[column.Source];
                columns[column.Name] = new SchemaShape.Column(column.Name, inputColumn.Kind,
                inputColumn.ItemType, inputColumn.IsKey, inputColumn.Annotations);
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class CategoryImputerTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Fills in missing values in a column based on the most frequent value";
        internal const string UserName = "CategoryImputer";
        internal const string ShortName = "CategoryImputer";
        internal const string LoadName = "CategoryImputer";
        internal const string LoaderSignature = "CategoryImputer";

        private readonly TypedColumn[] _columns;

        #endregion

        internal CategoryImputerTransformer(IHostEnvironment host, IDataView input, CategoryImputerEstimator.Column[] columns) :
            base(host.Register(nameof(CategoryImputerEstimator)))
        {
            var schema = input.Schema;

            _columns = columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString())).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal CategoryImputerTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(CategoryImputerTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            var columnCount = ctx.Reader.ReadInt32();

            _columns = new TypedColumn[columnCount];
            for (int i = 0; i < columnCount; i++)
            {
                _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString());

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new CategoryImputerTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CATIMP T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CategoryImputerTransformer).Assembly.FullName);
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
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

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

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            public abstract void Dispose();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if(!success)
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
                fixed(byte* rawData = data)
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

            internal abstract T Transform(T input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, T input, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandler = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    var fitResult = FitResult.Continue;
                    while (fitResult != FitResult.Complete)
                    {
                        using (var data = input.GetColumn<T>(Source).GetEnumerator())
                        {
                            while (fitResult == FitResult.Continue && data.MoveNext())
                            {
                                {
                                    success = FitHelper(estimatorHandler, data.Current, out fitResult, out errorHandle);
                                    if (!success)
                                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                                }
                            }

                            success = CompleteTrainingHelper(estimatorHandler, out fitResult, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                        }
                    }

                    success = CreateTransformerFromEstimatorHelper(estimatorHandler, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, sbyte* input, out sbyte output, out IntPtr errorHandle);
            internal unsafe override sbyte Transform(sbyte input)
            {
                sbyte* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out sbyte output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, sbyte* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle)
            {
                sbyte* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int8_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, short* input, out short output, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal unsafe override short Transform(short input)
            {
                short* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out short output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, short* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle)
            {
                short* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int16_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, int* input, out int output, out IntPtr errorHandle);
            internal unsafe override int Transform(int input)
            {
                int* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out int output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, int* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle)
            {
                int* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int32_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long* input, out long output, out IntPtr errorHandle);
            internal unsafe override long Transform(long input)
            {
                long* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out long output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, long* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle)
            {
                long* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_int64_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out byte output, out IntPtr errorHandle);
            internal unsafe override byte Transform(byte input)
            {
                byte* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out byte output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle)
            {
                byte* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint8_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ushort* input, out ushort output, out IntPtr errorHandle);
            internal unsafe override ushort Transform(ushort input)
            {
                ushort* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out ushort output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ushort* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle)
            {
                ushort* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint16_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, uint* input, out uint output, out IntPtr errorHandle);
            internal unsafe override uint Transform(uint input)
            {
                uint* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out uint output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, uint* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle)
            {
                uint* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint32_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ulong* input, out ulong output, out IntPtr errorHandle);
            internal unsafe override ulong Transform(ulong input)
            {
                ulong* interopInput = input == 0 ? null : &input;
                var success = TransformDataNative(_transformerHandler, interopInput, out ulong output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ulong* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle)
            {
                ulong* interopInput = input == 0 ? null : &input;
                return FitNative(estimator, interopInput, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_uint64_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, in float input, out float output, out IntPtr errorHandle);
            internal override float Transform(float input)
            {
                var success = TransformDataNative(_transformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, in float input, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle) =>
                FitNative(estimator, input, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, in double input, out double output, out IntPtr errorHandle);
            internal override double Transform(double input)
            {
                var success = TransformDataNative(_transformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, in double input, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle) =>
                FitNative(estimator, input, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_t_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal unsafe override string Transform(string input)
            {
                // Convert to byte array with NullPointer at end or nullptr.
                fixed (byte* interopInput = input == null? null : Encoding.UTF8.GetBytes(input + char.MinValue))
                {
                    var result = TransformDataNative(_transformerHandler, interopInput, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);

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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, string input, out FitResult fitResult, out IntPtr errorHandle)
            {
                fixed (byte* interopInput = input == null ? null : Encoding.UTF8.GetBytes(input + char.MinValue))
                {
                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                }
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                _transformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                _transformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
            internal unsafe override ReadOnlyMemory<char> Transform(ReadOnlyMemory<char> input)
            {
                var inputAsString = input.ToString();
                fixed (byte* interopInput = string.IsNullOrEmpty(inputAsString) ? null : Encoding.UTF8.GetBytes(inputAsString + char.MinValue))
                {
                    var result = TransformDataNative(_transformerHandler, interopInput, out IntPtr output, out IntPtr outputSize, out IntPtr errorHandle);

                    if (!result)
                    {
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }

                    if (outputSize.ToInt32() == 0)
                        return new ReadOnlyMemory<char>(string.Empty.ToArray());

                    using (var handler = new TransformedDataSafeHandle(output, outputSize, DestroyTransformedDataNative))
                    {
                        byte[] buffer = new byte[outputSize.ToInt32()];
                        Marshal.Copy(output, buffer, 0, buffer.Length);
                        return new ReadOnlyMemory<char>(Encoding.UTF8.GetString(buffer).ToArray());
                    }
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

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte* input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ReadOnlyMemory<char> input, out FitResult fitResult, out IntPtr errorHandle)
            {
                var inputAsString = input.ToString();
                fixed (byte* interopInput = string.IsNullOrEmpty(inputAsString) ? null : Encoding.UTF8.GetBytes(inputAsString + char.MinValue))
                {
                    return FitNative(estimator, interopInput, out fitResult, out errorHandle);
                }
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out FitResult fitResult, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(_transformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #endregion

        private sealed class Mapper : MapperBase
        {

            #region Class data members

            private readonly CategoryImputerTransformer _parent;

            #endregion

            public Mapper(CategoryImputerTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(Type.GetType(x.Type)))).ToArray();
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo)
            {
                ValueGetter<T> result = (ref T dst) =>
                {
                    var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                    var srcGetterScalar = input.GetGetter<T>(inputColumn);

                    T value = default;
                    srcGetterScalar(ref value);

                    dst = ((TypedColumn<T>)_parent._columns[iinfo]).Transform(value);

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

    internal static class CategoryImputerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.CategoryImputer",
            Desc = CategoryImputerTransformer.Summary,
            UserName = CategoryImputerTransformer.UserName,
            ShortName = CategoryImputerTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ImputeToKey(IHostEnvironment env, CategoryImputerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, CategoryImputerTransformer.ShortName, input);
            var xf = new CategoryImputerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
