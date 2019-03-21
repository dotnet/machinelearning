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
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(VectorWhiteningTransformer.Summary, typeof(IDataTransform), typeof(VectorWhiteningTransformer), typeof(VectorWhiteningTransformer.Options), typeof(SignatureDataTransform),
    VectorWhiteningTransformer.FriendlyName, VectorWhiteningTransformer.LoaderSignature, "Whitening")]

[assembly: LoadableClass(VectorWhiteningTransformer.Summary, typeof(IDataTransform), typeof(VectorWhiteningTransformer), null, typeof(SignatureLoadDataTransform),
    VectorWhiteningTransformer.FriendlyName, VectorWhiteningTransformer.LoaderSignature, VectorWhiteningTransformer.LoaderSignatureOld)]

[assembly: LoadableClass(VectorWhiteningTransformer.Summary, typeof(VectorWhiteningTransformer), null, typeof(SignatureLoadModel),
    VectorWhiteningTransformer.FriendlyName, VectorWhiteningTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(VectorWhiteningTransformer), null, typeof(SignatureLoadRowMapper),
   VectorWhiteningTransformer.FriendlyName, VectorWhiteningTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Which vector whitening technique to use. ZCA whitening ensures that the average covariance between whitened
    /// and original variables is maximal. In contrast, PCA whitening lead to maximally compressed whitened variables, as
    /// measured by squared covariance.
    /// </summary>
    public enum WhiteningKind
    {
        /// <summary> PCA whitening.</summary>
        [TGUI(Label = "PCA whitening")]
        PrincipalComponentAnalysis,

        /// <summary> ZCA whitening.</summary>
        [TGUI(Label = "ZCA whitening")]
        ZeroPhaseComponentAnalysis
    }

    /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
    public sealed class VectorWhiteningTransformer : OneToOneTransformerBase
    {
        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whitening kind (PCA/ZCA)")]
            public WhiteningKind Kind = VectorWhiteningEstimator.Defaults.Kind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling regularizer")]
            public float Eps = VectorWhiteningEstimator.Defaults.Epsilon;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of rows", ShortName = "rows")]
            public int MaxRows = VectorWhiteningEstimator.Defaults.MaximumNumberOfRows;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to save inverse (recovery) matrix", ShortName = "saveInv")]
            public bool SaveInverse = VectorWhiteningEstimator.Defaults.SaveInverse;

            [Argument(ArgumentType.AtMostOnce, HelpText = "PCA components to retain")]
            public int PcaNum = VectorWhiteningEstimator.Defaults.Rank;

            // REVIEW: add the following options:
            // 1. Currently there is no way to apply an inverse transform AFTER the the transform is trained.
            // 2. How many PCA components to retain/drop. Options: retain-first, drop-first, variance-threshold.
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whitening kind (PCA/ZCA)")]
            public WhiteningKind? Kind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling regularizer")]
            public float? Eps;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of rows", ShortName = "rows")]
            public int? MaxRows;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to save inverse (recovery) matrix", ShortName = "saveInv")]
            public bool? SaveInverse;

            [Argument(ArgumentType.AtMostOnce, HelpText = "PCA components to keep/drop")]
            public int? PcaNum;

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
                if (Kind != null || Eps != null || MaxRows != null || SaveInverse != null || PcaNum != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        private const Mkl.Layout Layout = Mkl.Layout.RowMajor;

        // Stores whitening matrix as float[] for each column. _models[i] is the whitening matrix of the i-th input column.
        private readonly float[][] _models;
        // Stores inverse ("recover") matrix as float[] for each column. Temporarily internal as it's used in unit test.
        // REVIEW: It doesn't look like this is used by non-test code. Should it be saved at all?
        private readonly float[][] _invModels;

        internal const string Summary = "Apply PCA or ZCA whitening algorithm to the input.";

        internal const string FriendlyName = "Whitening Transform";
        internal const string LoaderSignature = "WhiteningTransform";
        internal const string LoaderSignatureOld = "WhiteningFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WHITENTF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(VectorWhiteningTransformer).Assembly.FullName);
        }

        private readonly VectorWhiteningEstimator.ColumnOptions[] _columns;

        /// <summary>
        /// Initializes a new <see cref="VectorWhiteningTransformer"/> object.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="models">An array of whitening matrices where models[i] is learned from the i-th element of <paramref name="columns"/>.</param>
        /// <param name="invModels">An array of inverse whitening matrices, the i-th element being the inverse matrix of models[i].</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        internal VectorWhiteningTransformer(IHostEnvironment env, float[][] models, float[][] invModels, params VectorWhiteningEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorWhiteningTransformer)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            _columns = columns;
            _models = models;
            _invModels = invModels;
        }

        private VectorWhiteningTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorWhiteningTransformer)), ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // foreach column pair
            //   ColumnOptions
            // foreach model
            //   whitening matrix
            //   recovery matrix

            Host.AssertNonEmpty(ColumnPairs);
            _columns = new VectorWhiteningEstimator.ColumnOptions[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new VectorWhiteningEstimator.ColumnOptions(ctx);

            _models = new float[ColumnPairs.Length][];
            _invModels = new float[ColumnPairs.Length][];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                _models[i] = ctx.Reader.ReadFloatArray();
                if (_columns[i].SaveInv)
                    _invModels[i] = ctx.Reader.ReadFloatArray();
            }
        }

        // Factory method for SignatureLoadModel.
        internal static VectorWhiteningTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());
            return new VectorWhiteningTransformer(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            var infos = options.Columns.Select(colPair => new VectorWhiteningEstimator.ColumnOptions(colPair, options)).ToArray();
            (var models, var invModels) = TrainVectorWhiteningTransform(env, input, infos);
            return new VectorWhiteningTransformer(env, models, invModels, infos).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(VectorWhiteningEstimator.ColumnOptions[] columns)
            => columns.Select(c => (c.Name, c.InputColumnName ?? c.Name)).ToArray();

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema[srcCol].Type;
            var reason = TestColumn(inType);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        // Check if the input column's type is supported. Note that only float vector with a known shape is allowed.
        internal static string TestColumn(DataViewType type)
        {
            VectorType vectorType = type as VectorType;
            DataViewType itemType = vectorType?.ItemType ?? type;
            if ((vectorType != null && !vectorType.IsKnownSize && vectorType.Dimensions.Length > 1)
                || itemType != NumberDataViewType.Single)
                return "Expected float or float vector of known size";

            long valueCount = type.GetValueCount();
            if (valueCount * valueCount > Utils.ArrayMaxSize)
                return "Vector size exceeds maximum size for one dimensional array (2 146 435 071 elements)";

            return null;
        }

        private static void ValidateModel(IExceptionContext ectx, float[] model, DataViewType col)
        {
            long valueCount = col.GetValueCount();
            ectx.CheckDecode(Utils.Size(model) == valueCount * valueCount, "Invalid model size.");
            for (int i = 0; i < model.Length; i++)
                ectx.CheckDecode(FloatUtils.IsFinite(model[i]), "Found NaN or infinity in the model.");
        }

        // Sometime GetRowCount doesn't really return the number of rows in the associated IDataView.
        // A more reliable solution is to turely iterate through all rows via a RowCursor.
        private static long GetRowCount(IDataView inputData, params VectorWhiteningEstimator.ColumnOptions[] columns)
        {
            long? rows = inputData.GetRowCount();
            if (rows != null)
                return rows.GetValueOrDefault();

            int maxRows = columns.Max(i => i.MaximumNumberOfRows);
            long r = 0;
            using (var cursor = inputData.GetRowCursor())
            {
                while (r < maxRows && cursor.MoveNext())
                    r++;
            }
            return r;
        }

        // Computes the transformation matrices needed for whitening process from training data.
        internal static (float[][] models, float[][] invModels) TrainVectorWhiteningTransform(IHostEnvironment env, IDataView inputData, params VectorWhiteningEstimator.ColumnOptions[] columns)
        {
            var models = new float[columns.Length][];
            var invModels = new float[columns.Length][];
            // The training process will load all data into memory and perform whitening process
            // for each resulting column separately.
            using (var ch = env.Start("Training"))
            {
                GetColTypesAndIndex(env, inputData, columns, out DataViewType[] srcTypes, out int[] cols);
                var columnData = LoadDataAsDense(env, ch, inputData, out int[] rowCounts, srcTypes, cols, columns);
                TrainModels(env, ch, columnData, rowCounts, ref models, ref invModels, srcTypes, columns);
            }
            return (models, invModels);
        }

        // Extracts the indices and types of the input columns to the whitening transform.
        private static void GetColTypesAndIndex(IHostEnvironment env, IDataView inputData, VectorWhiteningEstimator.ColumnOptions[] columns, out DataViewType[] srcTypes, out int[] cols)
        {
            cols = new int[columns.Length];
            srcTypes = new DataViewType[columns.Length];
            var inputSchema = inputData.Schema;

            for (int i = 0; i < columns.Length; i++)
            {
                var col = inputSchema.GetColumnOrNull(columns[i].InputColumnName);
                if (!col.HasValue)
                    throw env.ExceptSchemaMismatch(nameof(inputSchema), "input", columns[i].InputColumnName);

                cols[i] = col.Value.Index;
                srcTypes[i] = col.Value.Type;
                var reason = TestColumn(srcTypes[i]);
                if (reason != null)
                    throw env.ExceptParam(nameof(inputData.Schema), reason);
            }
        }

        // Loads all relevant data for whitening training into memory.
        private static float[][] LoadDataAsDense(IHostEnvironment env, IChannel ch, IDataView inputData, out int[] actualRowCounts,
            DataViewType[] srcTypes, int[] cols, params VectorWhiteningEstimator.ColumnOptions[] columns)
        {
            long crowData = GetRowCount(inputData, columns);

            var columnData = new float[columns.Length][];
            actualRowCounts = new int[columns.Length];
            int maxActualRowCount = 0;

            for (int i = 0; i < columns.Length; i++)
            {
                VectorType vectorType = srcTypes[i] as VectorType;
                ch.Assert(vectorType != null && vectorType.IsKnownSize);
                // Use not more than MaxRow number of rows.
                var ex = columns[i];
                if (crowData <= ex.MaximumNumberOfRows)
                    actualRowCounts[i] = (int)crowData;
                else
                {
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", ex.MaximumNumberOfRows, columns[i].Name);
                    actualRowCounts[i] = ex.MaximumNumberOfRows;
                }

                int cslot = vectorType.Size;
                // Check that total number of values in matrix does not exceed int.MaxValue and adjust row count if necessary.
                if ((long)cslot * actualRowCounts[i] > int.MaxValue)
                {
                    actualRowCounts[i] = int.MaxValue / cslot;
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", actualRowCounts[i], columns[i].Name);
                }
                columnData[i] = new float[cslot * actualRowCounts[i]];
                if (actualRowCounts[i] > maxActualRowCount)
                    maxActualRowCount = actualRowCounts[i];
            }
            var idxDst = new int[columns.Length];

            using (var cursor = inputData.GetRowCursor(inputData.Schema.Where(c => cols.Any(col => c.Index == col))))
            {
                var getters = new ValueGetter<VBuffer<float>>[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                    getters[i] = cursor.GetGetter<VBuffer<float>>(cursor.Schema[cols[i]]);
                var val = default(VBuffer<float>);
                int irow = 0;
                while (irow < maxActualRowCount && cursor.MoveNext())
                {
                    for (int i = 0; i < columns.Length; i++)
                    {
                        if (irow >= actualRowCounts[i] || columnData[i].Length == 0)
                            continue;

                        getters[i](ref val);
                        val.CopyTo(columnData[i], idxDst[i]);
                        idxDst[i] += srcTypes[i].GetValueCount();
                    }
                    irow++;
                }
#if DEBUG
                for (int i = 0; i < columns.Length; i++)
                    ch.Assert(idxDst[i] == columnData[i].Length);
#endif
            }
            return columnData;
        }

        // Performs whitening training for each column separately. Notice that for both PCA and ZCA, _models and _invModels
        // will have dimension input_vec_size x input_vec_size. In the getter, the matrix will be truncated to only keep
        // PcaNum columns, and thus produce the desired output size.
        private static void TrainModels(IHostEnvironment env, IChannel ch, float[][] columnData, int[] rowCounts,
            ref float[][] models, ref float[][] invModels, DataViewType[] srcTypes, params VectorWhiteningEstimator.ColumnOptions[] columns)
        {
            ch.Assert(columnData.Length == rowCounts.Length);

            for (int iinfo = 0; iinfo < columns.Length; iinfo++)
            {
                var ex = columns[iinfo];
                var data = columnData[iinfo];
                int crow = rowCounts[iinfo];
                int ccol = srcTypes[iinfo].GetValueCount();

                // If there is no training data, simply initialize the model matrices to identity matrices.
                if (crow == 0)
                {
                    var matrixSize = ccol * ccol;
                    models[iinfo] = new float[matrixSize];
                    invModels[iinfo] = new float[matrixSize];
                    for (int i = 0; i < ccol; i++)
                    {
                        models[iinfo][i * ccol + i] = 1;
                        invModels[iinfo][i * ccol + i] = 1;
                    }
                    continue;
                }

                // Compute covariance matrix.
                var u = new float[ccol * ccol];
                ch.Info("Computing covariance matrix...");
                Mkl.Gemm(Layout, Mkl.Transpose.Trans, Mkl.Transpose.NoTrans,
                    ccol, ccol, crow, 1 / (float)crow, data, ccol, data, ccol, 0, u, ccol);

                ch.Info("Computing SVD...");
                var eigValues = new float[ccol]; // Eigenvalues.
                var unconv = new float[ccol]; // Superdiagonal unconverged values (if any). Not used but seems to be required by MKL.
                // After the next call, values in U will be ovewritten by left eigenvectors.
                // Each column in U will be an eigenvector.
                int r = Mkl.Svd(Layout, Mkl.SvdJob.MinOvr, Mkl.SvdJob.None,
                    ccol, ccol, u, ccol, eigValues, null, ccol, null, ccol, unconv);
                ch.Assert(r == 0);
                if (r > 0)
                    throw ch.Except("SVD did not converge.");
                if (r < 0)
                    throw ch.Except("Invalid arguments to LAPACK gesvd, error: {0}", r);

                ch.Info("Scaling eigenvectors...");
                // Scale eigenvalues first so we don't have to compute sqrt for every matrix element.
                // Scaled eigenvalues are used to compute inverse transformation matrix
                // while reciprocal (eigValuesRcp) values are used to compute whitening matrix.
                for (int i = 0; i < eigValues.Length; i++)
                    eigValues[i] = MathUtils.Sqrt(Math.Max(0, eigValues[i]) + ex.Epsilon);
                var eigValuesRcp = new float[eigValues.Length];
                for (int i = 0; i < eigValuesRcp.Length; i++)
                    eigValuesRcp[i] = 1 / eigValues[i];

                // Scale eigenvectors. Note that resulting matrix is transposed, so the scaled
                // eigenvectors are stored row-wise.
                var uScaled = new float[u.Length];
                var uInvScaled = new float[u.Length];
                int isrc = 0;
                for (int irowSrc = 0; irowSrc < ccol; irowSrc++)
                {
                    int idst = irowSrc;
                    for (int icolSrc = 0; icolSrc < ccol; icolSrc++)
                    {
                        uScaled[idst] = u[isrc] * eigValuesRcp[icolSrc];
                        uInvScaled[idst] = u[isrc] * eigValues[icolSrc];
                        isrc++;
                        idst += ccol;
                    }
                }

                // For ZCA need to do additional multiply by U.
                if (ex.Kind == WhiteningKind.PrincipalComponentAnalysis)
                {
                    // Save all components for PCA. Retained components will be selected during evaluation.
                    models[iinfo] = uScaled;
                    if (ex.SaveInv)
                        invModels[iinfo] = uInvScaled;
                }
                else if (ex.Kind == WhiteningKind.ZeroPhaseComponentAnalysis)
                {
                    models[iinfo] = new float[u.Length];
                    Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                        ccol, ccol, ccol, 1, u, ccol, uScaled, ccol, 0, models[iinfo], ccol);

                    if (ex.SaveInv)
                    {
                        invModels[iinfo] = new float[u.Length];
                        Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                            ccol, ccol, ccol, 1, u, ccol, uInvScaled, ccol, 0, invModels[iinfo], ccol);
                    }
                }
                else
                    ch.Assert(false);
            }
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // foreach column pair
            //   ColumnOptions
            // foreach model
            //   whitening matrix
            //   recovery matrix

            SaveColumns(ctx);

            Host.Assert(_columns.Length == ColumnPairs.Length);
            for (int i = 0; i < _columns.Length; i++)
                _columns[i].Save(ctx);
            for (int i = 0; i < _models.Length; i++)
            {
                ctx.Writer.WriteSingleArray(_models[i]);
                if (_columns[i].SaveInv)
                    ctx.Writer.WriteSingleArray(_invModels[i]);
            }
        }

        private static class Mkl
        {
            private const string MklPath = "MklImports";

            // The allowed value of Layout is specified in Intel's MLK library. See Layout parameter in this
            // [doc](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm) for details.
            public enum Layout
            {
                RowMajor = 101,
                ColMajor = 102
            }

            // The allowed value of Transpose is specified in Intel's MLK library. See transa parameter in this
            // [doc](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm) for details.
            public enum Transpose
            {
                NoTrans = 111,
                Trans = 112,
                ConjTrans = 113
            }

            // The allowed value of SvdJob is specified in Intel's MLK library. See jobvt parameter in this
            // [doc](https://software.intel.com/en-us/node/521150) for details.
            public enum SvdJob : byte
            {
                None = (byte)'N',
                All = (byte)'A',
                Min = (byte)'S',
                MinOvr = (byte)'O',
            }

            public static unsafe void Gemv(Layout layout, Transpose trans, int m, int n, float alpha,
                float[] a, int lda, ReadOnlySpan<float> x, int incx, float beta, Span<float> y, int incy)
            {
                fixed (float* pA = a)
                fixed (float* pX = x)
                fixed (float* pY = y)
                    Gemv(layout, trans, m, n, alpha, pA, lda, pX, incx, beta, pY, incy);
            }

            // See: https://software.intel.com/en-us/node/520750
            [DllImport(MklPath, CallingConvention = CallingConvention.Cdecl, EntryPoint = "cblas_sgemv"), SuppressUnmanagedCodeSecurity]
            private static extern unsafe void Gemv(Layout layout, Transpose trans, int m, int n, float alpha,
                float* a, int lda, float* x, int incx, float beta, float* y, int incy);

            // See: https://software.intel.com/en-us/node/520775
            [DllImport(MklPath, CallingConvention = CallingConvention.Cdecl, EntryPoint = "cblas_sgemm"), SuppressUnmanagedCodeSecurity]
            public static extern void Gemm(Layout layout, Transpose transA, Transpose transB, int m, int n, int k, float alpha,
                float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc);

            // See: https://software.intel.com/en-us/node/521150
            [DllImport(MklPath, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LAPACKE_sgesvd"), SuppressUnmanagedCodeSecurity]
            public static extern int Svd(Layout layout, SvdJob jobu, SvdJob jobvt,
                int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt, float[] superb);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly VectorWhiteningTransformer _parent;
            private readonly int[] _cols;
            private readonly DataViewType[] _srcTypes;

            public Mapper(VectorWhiteningTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _cols = new int[_parent.ColumnPairs.Length];
                _srcTypes = new DataViewType[_parent.ColumnPairs.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _cols[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    _srcTypes[i] = inputSchema[_cols[i]].Type;
                    ValidateModel(Host, _parent._models[i], _srcTypes[i]);
                    if (_parent._columns[i].SaveInv)
                        ValidateModel(Host, _parent._invModels[i], _srcTypes[i]);
                }
            }

            /// <summary>
            /// For PCA, the transform equation is y=U^Tx, where "^T" denotes matrix transpose, x is an 1-D vector (i.e., the input column), and U=[u_1, ..., u_PcaNum]
            /// is a n-by-PcaNum matrix. The symbol u_k is the k-th largest (in terms of the associated eigenvalue) eigenvector of (1/m)*\sum_{i=1}^m x_ix_i^T,
            /// where x_i is the whitened column at the i-th row and we have m rows in the training data.
            /// For ZCA, the transform equation is y = US^{-1/2}U^Tx, where U=[u_1, ..., u_n] (we retain all eigenvectors) and S is a diagonal matrix whose i-th
            /// diagonal element is the eigenvalues of u_i. The first U^Tx rotates x to another linear space (bases are u_1, ..., u_n), then S^{-1/2} is applied
            /// to ensure unit variance, and finally we rotate the scaled result back to the original space using U (note that UU^T is identity matrix so U is
            /// the inverse rotation of U^T).
            /// </summary>
            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _parent.ColumnPairs.Length; iinfo++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[iinfo].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var info = _parent._columns[iinfo];
                    DataViewType outType = (info.Kind == WhiteningKind.PrincipalComponentAnalysis && info.Rank > 0) ? new VectorType(NumberDataViewType.Single, info.Rank) : _srcTypes[iinfo];
                    result[iinfo] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[iinfo].outputColumnName, outType, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var ex = _parent._columns[iinfo];
                Host.Assert(ex.Kind == WhiteningKind.PrincipalComponentAnalysis || ex.Kind == WhiteningKind.ZeroPhaseComponentAnalysis);
                var getSrc = GetSrcGetter<VBuffer<float>>(input, iinfo);
                var src = default(VBuffer<float>);
                int cslotSrc = _srcTypes[iinfo].GetValueCount();
                // Notice that here that the learned matrices in _models will have the same size for both PCA and ZCA,
                // so we perform a truncation of the matrix in FillValues, that only keeps PcaNum columns.
                int cslotDst = (ex.Kind == WhiteningKind.PrincipalComponentAnalysis && ex.Rank > 0) ? ex.Rank : cslotSrc;
                var model = _parent._models[iinfo];
                ValueGetter<VBuffer<float>> del =
                    (ref VBuffer<float> dst) =>
                    {
                        getSrc(ref src);
                        Host.Check(src.Length == cslotSrc, "Invalid column size.");
                        FillValues(model, ref src, ref dst, cslotDst);
                    };
                return del;
            }

            private ValueGetter<T> GetSrcGetter<T>(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                var srcCol = input.Schema[_cols[iinfo]];
                Host.Assert(input.IsColumnActive(srcCol));
                return input.GetGetter<T>(srcCol);
            }

            private static void FillValues(float[] model, ref VBuffer<float> src, ref VBuffer<float> dst, int cdst)
            {
                var values = src.GetValues();
                int count = values.Length;
                int length = src.Length;

                // Since the whitening process produces dense vector, always use dense representation of dst.
                var editor = VBufferEditor.Create(ref dst, cdst);
                if (src.IsDense)
                {
                    Mkl.Gemv(Mkl.Layout.RowMajor, Mkl.Transpose.NoTrans, cdst, length,
                        1, model, length, values, 1, 0, editor.Values, 1);
                }
                else
                {
                    var indices = src.GetIndices();

                    int offs = 0;
                    for (int i = 0; i < cdst; i++)
                    {
                        // Returns a dot product of dense vector 'model' starting from offset 'offs' and sparse vector 'values'
                        // with first 'count' valid elements and their corresponding 'indices'.
                        editor.Values[i] = CpuMathUtils.DotProductSparse(model.AsSpan(offs), values, indices, count);
                        offs += length;
                    }
                }
                dst = editor.Commit();
            }

            private static float DotProduct(float[] a, int aOffset, ReadOnlySpan<float> b, ReadOnlySpan<int> indices, int count)
            {
                Contracts.Assert(count <= indices.Length);
                return CpuMathUtils.DotProductSparse(a.AsSpan(aOffset), b, indices, count);

            }
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
    public sealed class VectorWhiteningEstimator : IEstimator<VectorWhiteningTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const WhiteningKind Kind = WhiteningKind.ZeroPhaseComponentAnalysis;
            public const float Epsilon = 1e-5f;
            public const int MaximumNumberOfRows = 100 * 1000;
            public const bool SaveInverse = false;
            public const int Rank = 0;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <see cref="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of column to transform.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// Whitening kind (PCA/ZCA).
            /// </summary>
            public readonly WhiteningKind Kind;
            /// <summary>
            /// Whitening constant, prevents division by zero.
            /// </summary>
            public readonly float Epsilon;
            /// <summary>
            /// Maximum number of rows used to train the transform.
            /// </summary>
            public readonly int MaximumNumberOfRows;
            /// <summary>
            /// In case of PCA whitening, indicates the number of components to retain.
            /// </summary>
            public readonly int Rank;
            internal readonly bool SaveInv;

            /// <summary>
            /// Describes how the transformer handles one input-output column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="kind">Whitening kind (PCA/ZCA).</param>
            /// <param name="epsilon">Whitening constant, prevents division by zero.</param>
            /// <param name="maximumNumberOfRows">Maximum number of rows used to train the transform.</param>
            /// <param name="rank">In case of PCA whitening, indicates the number of components to retain.</param>
            public ColumnOptions(string name, string inputColumnName = null, WhiteningKind kind = Defaults.Kind, float epsilon = Defaults.Epsilon,
                int maximumNumberOfRows = Defaults.MaximumNumberOfRows, int rank = Defaults.Rank)
            {
                Name = name;
                Contracts.CheckValue(Name, nameof(Name));
                InputColumnName = inputColumnName ?? name;
                Contracts.CheckValue(InputColumnName, nameof(InputColumnName));
                Kind = kind;
                Contracts.CheckUserArg(Kind == WhiteningKind.PrincipalComponentAnalysis || Kind == WhiteningKind.ZeroPhaseComponentAnalysis, nameof(Kind));
                Epsilon = epsilon;
                Contracts.CheckUserArg(0 <= Epsilon && Epsilon < float.PositiveInfinity, nameof(Epsilon));
                MaximumNumberOfRows = maximumNumberOfRows;
                Contracts.CheckUserArg(MaximumNumberOfRows > 0, nameof(MaximumNumberOfRows));
                SaveInv = Defaults.SaveInverse;
                Rank = rank; // REVIEW: make it work with pcaNum == 1.
                Contracts.CheckUserArg(Rank >= 0, nameof(Rank));
            }

            internal ColumnOptions(VectorWhiteningTransformer.Column item, VectorWhiteningTransformer.Options options)
            {
                Name = item.Name;
                Contracts.CheckValue(Name, nameof(Name));
                InputColumnName = item.Source ?? item.Name;
                Contracts.CheckValue(InputColumnName, nameof(InputColumnName));
                Kind = item.Kind ?? options.Kind;
                Contracts.CheckUserArg(Kind == WhiteningKind.PrincipalComponentAnalysis || Kind == WhiteningKind.ZeroPhaseComponentAnalysis, nameof(item.Kind));
                Epsilon = item.Eps ?? options.Eps;
                Contracts.CheckUserArg(0 <= Epsilon && Epsilon < float.PositiveInfinity, nameof(item.Eps));
                MaximumNumberOfRows = item.MaxRows ?? options.MaxRows;
                Contracts.CheckUserArg(MaximumNumberOfRows > 0, nameof(item.MaxRows));
                SaveInv = item.SaveInverse ?? options.SaveInverse;
                Rank = item.PcaNum ?? options.PcaNum;
                Contracts.CheckUserArg(Rank >= 0, nameof(item.PcaNum));
            }

            internal ColumnOptions(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int:   kind
                // float: epsilon
                // int:   maxrow
                // byte:  saveInv
                // int:   pcaNum
                Kind = (WhiteningKind)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Kind == WhiteningKind.PrincipalComponentAnalysis || Kind == WhiteningKind.ZeroPhaseComponentAnalysis);
                Epsilon = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(0 <= Epsilon && Epsilon < float.PositiveInfinity);
                MaximumNumberOfRows = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(MaximumNumberOfRows > 0);
                SaveInv = ctx.Reader.ReadBoolByte();
                Rank = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Rank >= 0);
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int:   kind
                // float: epsilon
                // int:   maxrow
                // byte:  saveInv
                // int:   pcaNum
                Contracts.Assert(Kind == WhiteningKind.PrincipalComponentAnalysis || Kind == WhiteningKind.ZeroPhaseComponentAnalysis);
                ctx.Writer.Write((int)Kind);
                Contracts.Assert(0 <= Epsilon && Epsilon < float.PositiveInfinity);
                ctx.Writer.Write(Epsilon);
                Contracts.Assert(MaximumNumberOfRows > 0);
                ctx.Writer.Write(MaximumNumberOfRows);
                ctx.Writer.WriteBoolByte(SaveInv);
                Contracts.Assert(Rank >= 0);
                ctx.Writer.Write(Rank);
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptions[] _infos;

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
        internal VectorWhiteningEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorWhiteningEstimator));
            _infos = columns;
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="epsilon">Whitening constant, prevents division by zero when scaling the data by inverse of eigenvalues.</param>
        /// <param name="maximumNumberOfRows">Maximum number of rows used to train the transform.</param>
        /// <param name="rank">In case of PCA whitening, indicates the number of components to retain.</param>
        internal VectorWhiteningEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            WhiteningKind kind = Defaults.Kind,
            float epsilon = Defaults.Epsilon,
            int maximumNumberOfRows = Defaults.MaximumNumberOfRows,
            int rank = Defaults.Rank)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName, kind, epsilon, maximumNumberOfRows, rank))
        {
        }

        /// <summary>
        /// Trains and returns a <see cref="VectorWhiteningTransformer"/>.
        /// </summary>
        public VectorWhiteningTransformer Fit(IDataView input)
        {
            // Build transformation matrices for whitening process, then construct a trained transform.
            (var models, var invModels) = VectorWhiteningTransformer.TrainVectorWhiteningTransform(_host, input, _infos);
            return new VectorWhiteningTransformer(_host, models, invModels, _infos);
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in _infos)
            {
                if (!inputSchema.TryFindColumn(colPair.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.InputColumnName);
                var reason = VectorWhiteningTransformer.TestColumn(col.ItemType);
                if (reason != null)
                    throw _host.ExceptUserArg(nameof(inputSchema), reason);
                result[colPair.Name] = new SchemaShape.Column(colPair.Name, col.Kind, col.ItemType, col.IsKey, null);
            }
            return new SchemaShape(result.Values);
        }
    }
}
