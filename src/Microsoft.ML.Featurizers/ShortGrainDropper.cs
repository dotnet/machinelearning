using System;
using System.Collections.Generic;
using System.IO;
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

[assembly: LoadableClass(typeof(ShortDropTransformer), null, typeof(SignatureLoadModel),
    ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(ShortDropTransformer), null, typeof(SignatureLoadDataTransform),
   ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ShortDropTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ShortGrainDropperExtensionClass
    {

        /// <summary>
        /// Creates a <see cref="ShortGrainDropperEstimator"/> that drops rows per grain when the number of rows for that grain is less then
        /// the <paramref name="minPoints"/>
        /// </summary>
        /// <param name="catalog">Transform catalog</param>
        /// <param name="grainColumns">List of the grain columns. The combination of these form the "unique key" for each row.</param>
        /// <param name="minPoints">The minimum number of occurances required for each "unique key". If less than this, the rows will be dropped.</param>
        /// <returns><see cref="ShortGrainDropperEstimator"/></returns>
        public static ShortGrainDropperEstimator DropShortGrains(this TransformsCatalog catalog, string[] grainColumns, UInt32 minPoints)
        {
            var options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                MinPoints = minPoints
            };

            return new ShortGrainDropperEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// ShortGrainDroppper Featurizer determines which grains have the minimum number of rows specified, and then drops all grains
    /// that don't have that minimum number.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Takes in all columns |
    /// | Output column data type | Same as the original input type. |
    ///
    /// The <xref:Microsoft.ML.Transforms.RobustScalerEstimator> is not a trivial estimator and needs training.
    ///
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ShortGrainDropperExtensionClass.DropShortGrains(TransformsCatalog, string[], UInt32)"/>
    public sealed class ShortGrainDropperEstimator : IEstimator<ShortDropTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns", Name = "GrainColumns", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Minimum number of values required",
                Name = "MinPoints", ShortName = "minp", SortOrder = 1)]
            public UInt32 MinPoints;
        }

        #endregion

        internal ShortGrainDropperEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("ShortDropEstimator");
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");
            Contracts.Check(options.MinPoints > 0, "Min points must be greater than 0");

            _options = options;
        }

        public ShortDropTransformer Fit(IDataView input)
        {
            if (!AllGrainColumnsAreStrings(input.Schema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            return new ShortDropTransformer(_host, _options, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // We dont change the schema, we just drop rows. Still validate grain columns are correct type
            if (!AllGrainColumnsAreStrings(inputSchema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            return inputSchema;
        }
    }

    public sealed class ShortDropTransformer : ITransformer, IDisposable
    {
        #region Class data members

        internal const string Summary = "Drops rows if there aren't enough values per grain.";
        internal const string UserName = "ShortDrop";
        internal const string ShortName = "sgd";
        internal const string LoadName = "ShortDrop";
        internal const string LoaderSignature = "ShortDrop";

        private readonly IHost _host;
        private readonly ShortGrainDropperEstimator.Options _options;
        internal TransformerEstimatorSafeHandle TransformerHandle;

        #endregion

        // Normal constructor.
        internal ShortDropTransformer(IHostEnvironment host, ShortGrainDropperEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ShortDropTransformer));
            _options = options;

            TransformerHandle = CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal ShortDropTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(ShortDropTransformer));

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // MinPoints
            // length of C++ state array
            // C++ byte state array

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < grainColumns.Length; i++)
                grainColumns[i] = ctx.Reader.ReadString();

            var minPoints = ctx.Reader.ReadUInt32();

            _options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                MinPoints = minPoints
            };

            var nativeState = ctx.Reader.ReadByteArray();
            TransformerHandle = CreateTransformerFromSavedData(nativeState);
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedData(byte[] nativeState)
        {
            fixed (byte* rawStatePointer = nativeState)
            {
                IntPtr dataSize = new IntPtr(nativeState.Count());
                var result = CreateTransformerFromSavedDataNative(rawStatePointer, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new ShortDropTransformer(env, ctx).Transform(input));
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimator(IDataView input)
        {
            IntPtr estimator;
            IntPtr errorHandle;
            bool success;

            success = CreateEstimatorNative(_options.MinPoints, out estimator, out errorHandle);
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorNative))
            {
                TrainingState trainingState;
                FitResult fitResult;

                // Declare these outside the loop so the size is only set once;
                GCHandle[] grainHandles = new GCHandle[_options.GrainColumns.Length];
                IntPtr[] grainArray = new IntPtr[_options.GrainColumns.Length];
                GCHandle arrayHandle = default;

                // These are initialized in InitializeGrainGetters
                ValueGetter<ReadOnlyMemory<char>>[] grainGetters = new ValueGetter<ReadOnlyMemory<char>>[_options.GrainColumns.Length];
                DataViewRowCursor cursor = null;

                // Initialize GrainGetters and put cursor in valid state.
                InitializeGrainGetters(input, ref cursor, ref grainGetters);

                // Start the loop with the cursor in a valid state already.
                while (true)
                {
                    // Get the state of the native estimator.
                    success = GetStateNative(estimatorHandle, out trainingState, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we are no longer training then exit loop.
                    if (trainingState != TrainingState.Training)
                        break;

                    // Build the grain string array
                    try
                    {
                        CreateGrainStringArrays(grainGetters, ref grainHandles, ref arrayHandle, ref grainArray);
                        // Fit the estimator
                        success = FitNative(estimatorHandle, arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), out fitResult, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    }
                    finally
                    {
                        FreeGrainStringArrays(ref grainHandles, ref arrayHandle);
                    }

                    // If we need to reset the data to the beginning.
                    if (fitResult == FitResult.ResetAndContinue)
                        InitializeGrainGetters(input, ref cursor, ref grainGetters);

                    // If we are at the end of the data.
                    if (!cursor.MoveNext())
                    {
                        OnDataCompletedNative(estimatorHandle, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        InitializeGrainGetters(input, ref cursor, ref grainGetters);
                    }
                }

                // When done training complete the estimator.
                success = CompleteTrainingNative(estimatorHandle, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Create the native transformer from the estimator;
                success = CreateTransformerFromEstimatorNative(estimatorHandle, out IntPtr transformer, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Manually dispose of the IEnumerator since we dont have a using statement;
                cursor.Dispose();

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        private bool InitializeGrainGetters(IDataView input, ref DataViewRowCursor cursor, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
        {
            // Create getters for the grain columns. Cant use using for the cursor because it may need to be reset.
            // Manually dispose of the cursor if its not null
            if (cursor != null)
                cursor.Dispose();

            cursor = input.GetRowCursor(input.Schema.Where(x => _options.GrainColumns.Contains(x.Name)));

            for (int i = 0; i < _options.GrainColumns.Length; i++)
            {
                // Inititialize the enumerator and move it to a valid position.
                grainGetters[i] = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_options.GrainColumns[i]]);
            }

            return cursor.MoveNext();
        }

        public bool IsRowToRowMapper => false;

        // Schema not changed
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            return inputSchema;
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SGDROP T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ShortDropTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // MinPoints
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var column in _options.GrainColumns)
                ctx.Writer.Write(column);

            ctx.Writer.Write(_options.MinPoints);

            var data = CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        private byte[] CreateTransformerSaveData()
        {
            var success = CreateTransformerSaveDataNative(TransformerHandle, out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
            {
                byte[] savedData = new byte[bufferSize.ToInt32()];
                Marshal.Copy(buffer, savedData, 0, savedData.Length);
                return savedData;
            }
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal ShortGrainDropperDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new ShortGrainDropperDataView(_host, input, _options.GrainColumns, this);
        }

        internal TransformerEstimatorSafeHandle CloneTransformer() => CreateTransformerFromSavedData(CreateTransformerSaveData());

        public void Dispose()
        {
            if (!TransformerHandle.IsClosed)
                TransformerHandle.Close();
        }

        #region C++ function declarations

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateEstimatorNative(UInt32 minPoints, out IntPtr estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, out FitResult fitResult, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_OnDataCompleted"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_GetState"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);

        #endregion
    }

    internal static class ShortDropTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ShortDrop",
            Desc = ShortDropTransformer.Summary,
            UserName = ShortDropTransformer.UserName,
            ShortName = ShortDropTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ShortDrop(IHostEnvironment env, ShortGrainDropperEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ShortDropTransformer.ShortName, input);
            var xf = new ShortGrainDropperEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
