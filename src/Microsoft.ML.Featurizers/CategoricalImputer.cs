// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This is Auto Generated code used that is being used to unblock other teams.
// This functionality will be integrated into the MissingValueReplacer over the next several weeks.

using System;
using System.Collections.Generic;
using System.Diagnostics;
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
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(CategoricalImputerTransformer), null, typeof(SignatureLoadModel),
    CategoricalImputerTransformer.UserName, CategoricalImputerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CategoricalImputerTransformer), null, typeof(SignatureLoadRowMapper),
   CategoricalImputerTransformer.UserName, CategoricalImputerTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(CategoryImputerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class CategoricalImputerExtensionClass
    {
        /// <summary>
        /// Create a <see cref="CategoricalImputerEstimator"/>, which fills in the missing values in a column with the most frequent value.
        /// Supports Floats, Doubles, and Strings.
        /// A string is assumed "missing" if it is empty.
        /// </summary>
        /// <param name="catalog">Transform Catalog</param>
        /// <param name="outputColumnName">Output column name</param>
        /// <param name="inputColumnName">Input column name, if null defaults to <paramref name="outputColumnName"/></param>
        /// <returns><see cref="CategoricalImputerEstimator"/></returns>
        public static CategoricalImputerEstimator ImputeCategories(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null)
        {
            var options = new CategoricalImputerEstimator.Options
            {
                Columns = new CategoricalImputerEstimator.Column[1] { new CategoricalImputerEstimator.Column()
                    { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } }
            };

            return new CategoricalImputerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        /// <summary>
        /// Create a <see cref="CategoricalImputerEstimator"/>, which fills in the missing values in a column with the most frequent value.
        /// Supports Floats, Doubles, and Strings.
        /// A string is assumed "missing" if it is empty.
        /// </summary>
        /// <param name="catalog">Transform Catalog</param>
        /// <param name="columns">List of <see cref="InputOutputColumnPair"/> to fill in missing values</param>
        /// <returns><see cref="CategoricalImputerEstimator"/></returns>
        public static CategoricalImputerEstimator ImputeCategories(this TransformsCatalog catalog, params InputOutputColumnPair[] columns)
        {
            var options = new CategoricalImputerEstimator.Options
            {
                Columns = columns.Select(x => new CategoricalImputerEstimator.Column
                { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
            };

            return new CategoricalImputerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// The CategoryImputer replaces missing values with the most common value in that column. Supports Floats, Doubles, and Strings.
    /// A string is assumed "missing" if it is empty.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Float, Double, String |
    /// | Output column data type | Same as input column |
    /// | Exportable to ONNX | No |
    ///
    /// The <xref:Microsoft.ML.Transforms.CategoryImputerEstimator> is not a trivial estimator and needs training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="CategoricalImputerExtensionClass.ImputeCategories(TransformsCatalog, InputOutputColumnPair[])"/>
    /// <seealso cref="CategoricalImputerExtensionClass.ImputeCategories(TransformsCatalog, string, string)"/>
    public sealed class CategoricalImputerEstimator : IEstimator<CategoricalImputerTransformer>
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

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        #endregion

        internal CategoricalImputerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = env.Register(nameof(CategoricalImputerEstimator));

            foreach (var columnPair in options.Columns)
            {
                columnPair.Source = columnPair.Source ?? columnPair.Name;
            }

            _options = options;
        }

        public CategoricalImputerTransformer Fit(IDataView input)
        {
            return new CategoricalImputerTransformer(_host, input, _options.Columns);
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

    public sealed class CategoricalImputerTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Fills in missing values in a column based on the most frequent value";
        internal const string UserName = "CategoryImputer";
        internal const string ShortName = "CategoryImputer";
        internal const string LoadName = "CategoryImputer";
        internal const string LoaderSignature = "CategoryImputer";

        private readonly TypedColumn[] _columns;

        #endregion

        internal CategoricalImputerTransformer(IHostEnvironment host, IDataView input, CategoricalImputerEstimator.Column[] columns) :
            base(host.Register(nameof(CategoricalImputerEstimator)))
        {
            var schema = input.Schema;

            _columns = columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString())).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal CategoricalImputerTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(CategoricalImputerTransformer)))
        {
            host.CheckValue(ctx, nameof(ctx));
            host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
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
            => new CategoricalImputerTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CATIMP T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CategoricalImputerTransformer).Assembly.FullName);
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
                ctx.Writer.Write(column.TypeAsString);

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
            internal readonly string TypeAsString;
            internal readonly Type Type;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private static readonly Type[] _supportedTypes = new Type[] { typeof(float), typeof(double), typeof(ReadOnlyMemory<char>) };

            internal TypedColumn(string name, string source, Type type)
            {
                Name = name;
                Source = source;
                Type = type;
                TypeAsString = type.ToString();
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected abstract bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            public abstract void Dispose();

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
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
                    TransformerHandler = CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type)
            {
                if (type == typeof(float).ToString())
                {
                    return new FloatTypedColumn(name, source);
                }
                else if (type == typeof(double).ToString())
                {
                    return new DoubleTypedColumn(name, source);
                }
                else if (type == typeof(ReadOnlyMemory<char>).ToString())
                {
                    return new ReadOnlyCharTypedColumn(name, source);
                }

                throw new Exception($"Unsupported type {type}");
            }
        }

        internal abstract class TypedColumn<TSourceType> : TypedColumn
        {
            private protected TSourceType Result;

            internal TypedColumn(string name, string source, Type type) :
                base(name, source, type)
            {
            }

            internal abstract TSourceType Transform(TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, TSourceType input, out FitResult fitResult, out IntPtr errorHandle);
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

                    // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                    var data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                    var valid = data.MoveNext();

                    // Make sure its not an empty data frame
                    Debug.Assert(valid);

                    while (true)
                    {
                        // Get the state of the native estimator.
                        success = GetStateHelper(estimatorHandle, out trainingState, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If we are no longer training then exit loop.
                        if (trainingState != TrainingState.Training)
                            break;

                        // Fit the estimator
                        success = FitHelper(estimatorHandle, data.Current, out fitResult, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If we need to reset the data to the beginning.
                        if (fitResult == FitResult.ResetAndContinue)
                        {
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                        }

                        // If we are at the end of the data.
                        if (!data.MoveNext())
                        {
                            // If we get here fitResult should never be ResetAndContinue
                            Debug.Assert(fitResult != FitResult.ResetAndContinue);

                            OnDataCompletedHelper(estimatorHandle, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                            valid = data.MoveNext();
                            Debug.Assert(valid);
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

                    // Manually dispose of the IEnumerator since we dont have a using statement;
                    data.Dispose();

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }
        }

        #endregion

        #region FloatColumn

        internal sealed class FloatTypedColumn : TypedColumn<float>
        {
            internal FloatTypedColumn(string name, string source) :
                base(name, source, typeof(float))
            {
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);

                // Get the result of the transform and cache it so its cached before transform is called. Pass in float.NaN so we get the Mode back.
                var success = TransformDataNative(TransformerHandler, float.NaN, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                Result = output;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var success = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                var handle = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);

                // Get the result of the transform and cache it so its cached before transform is called. Pass in float.NaN so we get the Mode back.
                success = TransformDataNative(handle, float.NaN, out float output, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                Result = output;

                return handle;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, float input, out float output, out IntPtr errorHandle);
            internal override float Transform(float input)
            {
                if (!float.IsNaN(input))
                    return input;

                return Result;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle) =>
                FitNative(estimator, input, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_float_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region DoubleColumn

        internal sealed class DoubleTypedColumn : TypedColumn<double>
        {
            internal DoubleTypedColumn(string name, string source) :
                base(name, source, typeof(double))
            {
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);

                // Get the result of the transform and cache it so its cached when Transform is called. Pass in double.NaN so we get the Mode back.
                var success = TransformDataNative(TransformerHandler, double.NaN, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                Result = output;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var success = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                var handle = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);

                // Get the result of the transform and cache it. Pass in double.NaN so we get the Mode back.
                success = TransformDataNative(handle, double.NaN, out double output, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                Result = output;

                return handle;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double input, out double output, out IntPtr errorHandle);
            internal override double Transform(double input)
            {
                if (!double.IsNaN(input))
                    return input;

                return Result;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle) =>
                CreateEstimatorNative(out estimator, out errorHandle);

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle) =>
                FitNative(estimator, input, out fitResult, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #region  ReadOnlyCharColumn

        internal sealed class ReadOnlyCharTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {
            internal ReadOnlyCharTypedColumn(string name, string source) :
                base(name, source, typeof(ReadOnlyMemory<char>))
            {
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            internal override unsafe void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);

                // Get the result of the transform and cache it. Pass in null so we get the Mode back.
                var result = TransformDataNative(TransformerHandler, null, out IntPtr output, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var handler = new TransformedDataSafeHandle(output, DestroyTransformedDataNative))
                {
                    Result = PointerToString(output).AsMemory();
                }
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                var handle = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);

                // Get the result of the transform and cache it. Pass in null so we get the Mode back.
                result = TransformDataNative(handle, null, out IntPtr output, out errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var handler = new TransformedDataSafeHandle(output, DestroyTransformedDataNative))
                {
                    Result = PointerToString(output).AsMemory();
                }

                return handle;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte* input, out IntPtr output, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);
            internal unsafe override ReadOnlyMemory<char> Transform(ReadOnlyMemory<char> input)
            {
                if (!input.IsEmpty)
                    return input;

                return Result;
            }

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
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
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                   CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "CatImputerFeaturizer_string_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);
        }

        #endregion

        #endregion

        private sealed class Mapper : MapperBase
        {
            private static readonly FuncInstanceMethodInfo1<Mapper, DataViewRow, int, Delegate> _makeGetterMethodInfo
                = FuncInstanceMethodInfo1<Mapper, DataViewRow, int, Delegate>.Create(target => target.MakeGetter<int>);

            #region Class data members

            private readonly CategoricalImputerTransformer _parent;
            private readonly DataViewSchema _schema;
            #endregion

            public Mapper(CategoricalImputerTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _schema = inputSchema;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(x.Type))).ToArray();
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo)
            {
                var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                var srcGetterScalar = input.GetGetter<T>(inputColumn);

                ValueGetter<T> result = (ref T dst) =>
                {
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
                return Utils.MarshalInvoke(_makeGetterMethodInfo, this, inputType, input, iinfo);
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
            Desc = CategoricalImputerTransformer.Summary,
            UserName = CategoricalImputerTransformer.UserName,
            ShortName = CategoricalImputerTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ImputeToKey(IHostEnvironment env, CategoricalImputerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, CategoricalImputerTransformer.ShortName, input);
            var xf = new CategoricalImputerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
