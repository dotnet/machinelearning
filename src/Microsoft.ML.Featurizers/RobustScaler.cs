// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Featurizers.CommonExtensions;

[assembly: LoadableClass(typeof(RobustScalerTransformer), null, typeof(SignatureLoadModel),
    RobustScalerTransformer.UserName, RobustScalerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(RobustScalerTransformer), null, typeof(SignatureLoadRowMapper),
   RobustScalerTransformer.UserName, RobustScalerTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(RobustScalerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class RobustScalerExtensionClass
    {
        /// <summary>
        /// Creates a <see cref="RobustScalerEstimator"/> which scales features using statistics that are robust to outliers,
        /// by removing the median and scaling the data according to the quantile range (defaults to IQR: Interquartile Range).
        /// Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
        /// Median and interquartile range are then stored to be used on later data using the transform method.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="outputColumnName">Output Column Name</param>
        /// <param name="inputColumnName">Input Column Name. Defaults to <paramref name="outputColumnName"/> if not provided.</param>
        /// <param name="center">If true center the data</param>
        /// <param name="scale">If true scale the data</param>
        /// <param name="quantileMin">The lower bound to keep. Values range from 0 to 100.0. Must be lower then the <paramref name="quantileMax"/></param>
        /// <param name="quantileMax">The upper bound to keep. Values range from 0 to 100.0. Must be greater then the <paramref name="quantileMin"/></param>
        /// <returns><see cref="RobustScalerEstimator"/></returns>
        public static RobustScalerEstimator RobustScaler(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
            bool center = true, bool scale = true, float quantileMin = 25.0f, float quantileMax = 75.0f)
        {
            var options = new RobustScalerEstimator.Options
            {
                Columns = new RobustScalerEstimator.Column[1] { new RobustScalerEstimator.Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName } },
                Center = center,
                Scale = scale,
                QuantileMin = quantileMin,
                QuantileMax = quantileMax
            };

            return new RobustScalerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        /// <summary>
        /// Creates a <see cref="RobustScalerEstimator"/> which scales features using statistics that are robust to outliers,
        /// by removing the median and scaling the data according to the quantile range (defaults to IQR: Interquartile Range).
        /// Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
        /// Median and interquartile range are then stored to be used on later data using the transform method.
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="columns">List of columns mappings.</param>
        /// <param name="center">If true center the data</param>
        /// <param name="scale">If true scale the data</param>
        /// <param name="quantileMin">The lower bound to keep. Values range from 0 to 100.0. Must be lower then the <paramref name="quantileMax"/></param>
        /// <param name="quantileMax">The upper bound to keep. Values range from 0 to 100.0. Must be greater then the <paramref name="quantileMin"/></param>
        /// <returns><see cref="RobustScalerEstimator"/></returns>
        public static RobustScalerEstimator RobustScaler(this TransformsCatalog catalog, InputOutputColumnPair[] columns, bool center = true,
            bool scale = true, float quantileMin = 25.0f, float quantileMax = 75.0f)
        {
            var options = new RobustScalerEstimator.Options
            {
                Columns = columns.Select(x => new RobustScalerEstimator.Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
                Center = center,
                Scale = scale,
                QuantileMin = quantileMin,
                QuantileMax = quantileMax
            };

            return new RobustScalerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// RobustScalar Featurizer scales features using statistics that are robust to outliers, by removing the median and scaling the data according to the quantile range
    /// (defaults to IQR: Interquartile Range). Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
    /// Median and interquartile range are then stored to be used on later data using the transform method.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Scalar of numeric type |
    /// | Output column data type | If input is byte, sbyte, short, ushort, or float then returns Float, else Double |
    ///
    /// The <xref:Microsoft.ML.Transforms.RobustScalerEstimator> is not a trivial estimator and needs training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="RobustScalerExtensionClass.RobustScaler(TransformsCatalog, InputOutputColumnPair[], bool, bool, float, float)"/>
    /// <seealso cref="RobustScalerExtensionClass.RobustScaler(TransformsCatalog, string, string, bool, bool, float, float)"/>
    public sealed class RobustScalerEstimator : IEstimator<RobustScalerTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        // For determining what the output type is. These types return type is float, the rest will return as a double.
        private static readonly Type[] _floatTypes = new Type[] { typeof(byte), typeof(sbyte), typeof(short), typeof(ushort), typeof(float) };

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

            [Argument(ArgumentType.AtMostOnce, HelpText = "If True, center the data before scaling.",
                Name = "Center", ShortName = "ctr", SortOrder = 2)]
            public bool Center = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If True, scale the data to interquartile range.",
                Name = "Scale", ShortName = "sc", SortOrder = 3)]
            public bool Scale = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Min for the quantile range used to calculate scale.",
                Name = "QuantileMin", ShortName = "min", SortOrder = 4)]
            public float QuantileMin = 25.0f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max for the quantile range used to calculate scale.",
                Name = "QuantileMax", ShortName = "max", SortOrder = 5)]
            public float QuantileMax = 75.0f;
        }

        #endregion

        internal RobustScalerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host.Check(!CommonExtensions.OsIsCentOS7(), "CentOS7 is not supported");
            _host = env.Register(nameof(RobustScalerEstimator));
            Contracts.Check(options.QuantileMin >= 0.0f && options.QuantileMin < options.QuantileMax && options.QuantileMax <= 100.0f, "Invalid QuantileRange provided");
            Contracts.CheckNonEmpty(options.Columns, nameof(options.Columns));

            _options = options;
        }

        public RobustScalerTransformer Fit(IDataView input)
        {
            return new RobustScalerTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Columns)
            {
                var inputColumn = columns[column.Source];

                if (!RobustScalerTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType.ToString()} for column {column.Name} not a supported type.");

                if (_floatTypes.Contains(inputColumn.ItemType.RawType))
                {
                    columns[column.Name] = new SchemaShape.Column(column.Name, inputColumn.Kind,
                    ColumnTypeExtensions.PrimitiveTypeFromType(typeof(float)), inputColumn.IsKey, inputColumn.Annotations);
                }
                else
                {
                    columns[column.Name] = new SchemaShape.Column(column.Name, inputColumn.Kind,
                    ColumnTypeExtensions.PrimitiveTypeFromType(typeof(double)), inputColumn.IsKey, inputColumn.Annotations);
                }

            }
            return new SchemaShape(columns.Values);
        }
    }

    public sealed class RobustScalerTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Removes the median and scales the data according to the quantile range.";
        internal const string UserName = "RobustScalerTransformer";
        internal const string ShortName = "RobScalT";
        internal const string LoadName = "RobustScalerTransformer";
        internal const string LoaderSignature = "RobustScalerTransformer";

        private TypedColumn[] _columns;
        private RobustScalerEstimator.Options _options;

        #endregion

        internal RobustScalerTransformer(IHostEnvironment host, IDataView input, RobustScalerEstimator.Options options) :
            base(host.Register(nameof(RobustScalerTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _columns = options.Columns.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString(), this)).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal RobustScalerTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(RobustScalerTransformer)))
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
                _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString(), this);

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
        => new RobustScalerTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RbScal T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RobustScalerTransformer).Assembly.FullName);
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

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private static readonly Type[] _supportedTypes = new[] { typeof(sbyte), typeof(short), typeof(int), typeof(long), typeof(byte), typeof(ushort),
                typeof(uint), typeof(ulong), typeof(float), typeof(double) };

            internal TypedColumn(string name, string source, string type)
            {
                Name = name;
                Source = source;
                Type = type;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
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
                    TransformerHandler = CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type, RobustScalerTransformer parent)
            {
                if (type == typeof(sbyte).ToString())
                {
                    return new Int8TypedColumn(name, source, parent);
                }
                else if (type == typeof(short).ToString())
                {
                    return new Int16TypedColumn(name, source, parent);
                }
                else if (type == typeof(int).ToString())
                {
                    return new Int32TypedColumn(name, source, parent);
                }
                else if (type == typeof(long).ToString())
                {
                    return new Int64TypedColumn(name, source, parent);
                }
                else if (type == typeof(byte).ToString())
                {
                    return new UInt8TypedColumn(name, source, parent);
                }
                else if (type == typeof(ushort).ToString())
                {
                    return new UInt16TypedColumn(name, source, parent);
                }
                else if (type == typeof(uint).ToString())
                {
                    return new UInt32TypedColumn(name, source, parent);
                }
                else if (type == typeof(ulong).ToString())
                {
                    return new UInt64TypedColumn(name, source, parent);
                }
                else if (type == typeof(float).ToString())
                {
                    return new FloatTypedColumn(name, source, parent);
                }
                else if (type == typeof(double).ToString())
                {
                    return new DoubleTypedColumn(name, source, parent);
                }

                throw new InvalidOperationException($"Column {name} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            internal TypedColumn(string name, string source, string type) :
                base(name, source, type)
            {
            }

            internal abstract TOutputType Transform(TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, TSourceType input, out FitResult fitResult, out IntPtr errorHandle);

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

        #region Int8Column

        internal sealed class Int8TypedColumn : TypedColumn<sbyte, float>
        {
            private RobustScalerTransformer _parent;
            internal Int8TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(sbyte).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, sbyte input, out float output, out IntPtr errorHandle);
            internal unsafe override float Transform(sbyte input)
            {
                var success = TransformDataNative(TransformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int8_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(float);
            }

        }

        #endregion

        #region UInt8Column

        internal sealed class UInt8TypedColumn : TypedColumn<byte, float>
        {
            private RobustScalerTransformer _parent;
            internal UInt8TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(byte).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte input, out float output, out IntPtr errorHandle);
            internal unsafe override float Transform(byte input)
            {
                var success = TransformDataNative(TransformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint8_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(float);
            }
        }

        #endregion

        #region Int16Column

        internal sealed class Int16TypedColumn : TypedColumn<short, float>
        {
            private RobustScalerTransformer _parent;
            internal Int16TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(short).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, short input, out float output, out IntPtr errorHandle);
            internal unsafe override float Transform(short input)
            {
                var success = TransformDataNative(TransformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(float);
            }
        }

        #endregion

        #region UInt16Column

        internal sealed class UInt16TypedColumn : TypedColumn<ushort, float>
        {
            private RobustScalerTransformer _parent;
            internal UInt16TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(ushort).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ushort input, out float output, out IntPtr errorHandle);
            internal unsafe override float Transform(ushort input)
            {
                var success = TransformDataNative(TransformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint16_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int16_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(float);
            }
        }

        #endregion

        #region Int32Column

        internal sealed class Int32TypedColumn : TypedColumn<int, double>
        {
            private RobustScalerTransformer _parent;
            internal Int32TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(int).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, int input, out double output, out IntPtr errorHandle);
            internal unsafe override double Transform(int input)
            {
                var success = TransformDataNative(TransformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int32_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(double);
            }
        }

        #endregion

        #region UInt32Column

        internal sealed class UInt32TypedColumn : TypedColumn<uint, double>
        {
            private RobustScalerTransformer _parent;
            internal UInt32TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(uint).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, uint input, out double output, out IntPtr errorHandle);
            internal unsafe override double Transform(uint input)
            {
                var success = TransformDataNative(TransformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint32_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(double);
            }
        }

        #endregion

        #region Int64Column

        internal sealed class Int64TypedColumn : TypedColumn<long, double>
        {
            private RobustScalerTransformer _parent;
            internal Int64TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(long).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long input, out double output, out IntPtr errorHandle);
            internal unsafe override double Transform(long input)
            {
                var success = TransformDataNative(TransformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_int64_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(double);
            }
        }

        #endregion

        #region UInt64Column

        internal sealed class UInt64TypedColumn : TypedColumn<ulong, double>
        {
            private RobustScalerTransformer _parent;
            internal UInt64TypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(ulong).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ulong input, out double output, out IntPtr errorHandle);
            internal unsafe override double Transform(ulong input)
            {
                var success = TransformDataNative(TransformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, float.NaN, float.NaN, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_uint64_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(double);
            }
        }

        #endregion

        #region FloatColumn

        internal sealed class FloatTypedColumn : TypedColumn<float, float>
        {
            private RobustScalerTransformer _parent;
            internal FloatTypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(float).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, float input, out float output, out IntPtr errorHandle);
            internal unsafe override float Transform(float input)
            {
                var success = TransformDataNative(TransformerHandler, input, out float output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, float.NaN, float.NaN, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_IsTrainingComplete", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool IsTrainingCompleteNative(TransformerEstimatorSafeHandle transformer, out bool isTrainingComplete, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_float_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(float);
            }
        }

        #endregion

        #region DoubleColumn

        internal sealed class DoubleTypedColumn : TypedColumn<double, double>
        {

            private RobustScalerTransformer _parent;
            internal DoubleTypedColumn(string name, string source, RobustScalerTransformer parent) :
                base(name, source, typeof(double).ToString())
            {
                _parent = parent;
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern unsafe bool CreateEstimatorNative(bool withCentering, float qRangeMin, float qRangeMax, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double input, out double output, out IntPtr errorHandle);
            internal unsafe override double Transform(double input)
            {
                var success = TransformDataNative(TransformerHandler, input, out double output, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return output;
            }

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }

            private protected override unsafe bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                if (_parent._options.Scale)
                    return CreateEstimatorNative(_parent._options.Center, _parent._options.QuantileMin, _parent._options.QuantileMax, out estimator, out errorHandle);
                else
                    return CreateEstimatorNative(_parent._options.Center, -1f, -1f, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, input, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                    GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "RobustScalerFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override Type ReturnType()
            {
                return typeof(double);
            }
        }

        #endregion

        #endregion // Column Info

        private sealed class Mapper : MapperBase
        {
            private static readonly FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate> _makeGetterMethodInfo
                = FuncInstanceMethodInfo2<Mapper, DataViewRow, int, Delegate>.Create(target => target.MakeGetter<int, int>);

            #region Class data members

            private readonly RobustScalerTransformer _parent;
            private readonly DataViewSchema _schema;

            #endregion

            public Mapper(RobustScalerTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _schema = inputSchema;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(x.ReturnType()))).ToArray();
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    TSourceType value = default;
                    srcGetterScalar(ref value);

                    dst = ((TypedColumn<TSourceType, TOutputType>)_parent._columns[iinfo]).Transform(value);
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

    internal static class RobustScalerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.RobustScaler",
            Desc = RobustScalerTransformer.Summary,
            UserName = RobustScalerTransformer.UserName,
            ShortName = RobustScalerTransformer.ShortName)]
        public static CommonOutputs.TransformOutput RobustScaler(IHostEnvironment env, RobustScalerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, RobustScalerTransformer.ShortName, input);
            var xf = new RobustScalerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
