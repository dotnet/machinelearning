// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

[assembly: LoadableClass(WhiteningTransform.Summary, typeof(IDataTransform), typeof(WhiteningTransform), typeof(WhiteningTransform.Arguments), typeof(SignatureDataTransform),
    WhiteningTransform.FriendlyName, WhiteningTransform.LoaderSignature, "Whitening")]

[assembly: LoadableClass(WhiteningTransform.Summary, typeof(IDataTransform), typeof(WhiteningTransform), null, typeof(SignatureLoadDataTransform),
    WhiteningTransform.FriendlyName, WhiteningTransform.LoaderSignature, WhiteningTransform.LoaderSignatureOld)]

[assembly: LoadableClass(WhiteningTransform.Summary, typeof(WhiteningTransform), null, typeof(SignatureLoadModel),
    WhiteningTransform.FriendlyName, WhiteningTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(WhiteningTransform), null, typeof(SignatureLoadRowMapper),
   WhiteningTransform.FriendlyName, WhiteningTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public enum WhiteningKind
    {
        [TGUI(Label = "PCA whitening")]
        Pca,

        [TGUI(Label = "ZCA whitening")]
        Zca
    }

    /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
    public sealed class WhiteningTransform : OneToOneTransformerBase
    {
        internal static class Defaults
        {
            public const WhiteningKind Kind = WhiteningKind.Zca;
            public const float Eps = 1e-5f;
            public const int MaxRows = 100 * 1000;
            public const bool SaveInverse = false;
            public const int PcaNum = 0;
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whitening kind (PCA/ZCA)")]
            public WhiteningKind Kind = Defaults.Kind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling regularizer")]
            public float Eps = Defaults.Eps;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of rows", ShortName = "rows")]
            public int MaxRows = Defaults.MaxRows;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to save inverse (recovery) matrix", ShortName = "saveInv")]
            public bool SaveInverse = Defaults.SaveInverse;

            [Argument(ArgumentType.AtMostOnce, HelpText = "PCA components to retain")]
            public int PcaNum = Defaults.PcaNum;

            // REVIEW: add the following options:
            // 1. Currently there is no way to apply an inverse transform AFTER the the transform is trained.
            // 2. How many PCA components to retain/drop. Options: retain-first, drop-first, variance-threshold.
        }

        public sealed class Column : OneToOneColumn
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

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Kind != null || Eps != null || MaxRows != null || SaveInverse != null || PcaNum != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class ColInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly WhiteningKind Kind;
            public readonly float Epsilon;
            public readonly int MaxRow;
            public readonly bool SaveInv;
            public readonly int PcaNum;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of the input column.</param>
            /// <param name="output">Name of the column resulting from the transformation of <paramref name="input"/>. Null means <paramref name="input"/> is replaced. </param>
            /// <param name="kind">Whitening kind (PCA/ZCA).</param>
            /// <param name="eps">Scaling regularizer.</param>
            /// <param name="maxRows">Max number of rows.</param>
            /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
            /// <param name="pcaNum">PCA components to retain.</param>
            public ColInfo(string input, string output = null, WhiteningKind kind = Defaults.Kind, float eps = Defaults.Eps,
                int maxRows = Defaults.MaxRows, bool saveInverse = Defaults.SaveInverse, int pcaNum = Defaults.PcaNum)
            {
                Input = input;
                Contracts.CheckValue(Input, nameof(Input));
                Output = output ?? input;
                Contracts.CheckValue(Output, nameof(Output));
                Kind = kind;
                Contracts.CheckUserArg(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca, nameof(Kind));
                Epsilon = eps;
                Contracts.CheckUserArg(0 <= Epsilon && Epsilon < float.PositiveInfinity, nameof(Epsilon));
                MaxRow = maxRows;
                Contracts.CheckUserArg(MaxRow > 0, nameof(MaxRow));
                SaveInv = saveInverse;
                PcaNum = pcaNum;
                Contracts.CheckUserArg(PcaNum >= 0, nameof(PcaNum));
            }

            internal ColInfo(Column item, Arguments args)
            {
                Input = item.Source;
                Contracts.CheckValue(Input, nameof(Input));
                Output = item.Name ?? item.Source;
                Contracts.CheckValue(Output, nameof(Output));
                Kind = item.Kind ?? args.Kind;
                Contracts.CheckUserArg(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca, nameof(item.Kind));
                Epsilon = item.Eps ?? args.Eps;
                Contracts.CheckUserArg(0 <= Epsilon && Epsilon < float.PositiveInfinity, nameof(item.Eps));
                MaxRow = item.MaxRows ?? args.MaxRows;
                Contracts.CheckUserArg(MaxRow > 0, nameof(item.MaxRows));
                SaveInv = item.SaveInverse ?? args.SaveInverse;
                PcaNum = item.PcaNum ?? args.PcaNum;
                Contracts.CheckUserArg(PcaNum >= 0, nameof(item.PcaNum));
            }

            internal ColInfo(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int:   kind
                // float: epsilon
                // int:   maxrow
                // byte:  saveInv
                // int:   pcaNum
                Kind = (WhiteningKind)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca);
                Epsilon = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(0 <= Epsilon && Epsilon < float.PositiveInfinity);
                MaxRow = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(MaxRow > 0);
                SaveInv = ctx.Reader.ReadBoolByte();
                PcaNum = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(PcaNum >= 0);
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
                Contracts.Assert(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca);
                ctx.Writer.Write((int)Kind);
                Contracts.Assert(0 <= Epsilon && Epsilon < float.PositiveInfinity);
                ctx.Writer.Write(Epsilon);
                Contracts.Assert(MaxRow > 0);
                ctx.Writer.Write(MaxRow);
                ctx.Writer.WriteBoolByte(SaveInv);
                Contracts.Assert(PcaNum >= 0);
                ctx.Writer.Write(PcaNum);
            }
        }

        private const Mkl.Layout Layout = Mkl.Layout.RowMajor;

        // Stores whitening matrix as float[] for each column.
        private readonly float[][] _models;
        // Stores inverse ("recover") matrix as float[] for each column. Temporarily internal as it's used in unit test.
        // REVIEW: It doesn't look like this is used by non-test code. Should it be saved at all?
        private readonly float[][] _invModels;

        private int[] _cols;
        private ColumnType[] _srcTypes;

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
                loaderAssemblyName: typeof(WhiteningTransform).Assembly.FullName);
        }

        private readonly ColInfo[] _infos;

        /// <summary>
        /// Initializes a new <see cref="WhiteningTransform"/> object.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputData">Input data.</param>
        /// <param name="columns">Describes the settings for the transformer.</param>
        internal WhiteningTransform(IHostEnvironment env, IDataView inputData, params ColInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WhiteningTransform)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            var inputSchema = inputData.Schema;

            _cols = new int[ColumnPairs.Length];
            _srcTypes = new ColumnType[ColumnPairs.Length];
            _infos = columns;
            for (int i = 0; i < _infos.Length; i++)
            {
                if (!inputSchema.TryGetColumnIndex(ColumnPairs[i].input, out _cols[i]))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[i].input);
                CheckInputColumn(inputSchema, i, _cols[i]);
                _srcTypes[i] = inputSchema.GetColumnType(_cols[i]);
            }

            using (var ch = Host.Start("Training"))
            {
                // The training process will load all data into memory and perform whitening process
                // for each resulting column separately.
                _models = new float[ColumnPairs.Length][];
                _invModels = new float[ColumnPairs.Length][];
                int[] rowCounts;
                var columnData = LoadDataAsDense(ch, inputData, out rowCounts);
                TrainModels(columnData, rowCounts, ch);
            }
        }

        private WhiteningTransform(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WhiteningTransform)), ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // foreach added column
            //   ColInfoEx
            // foreach model
            //   whitening matrix
            //   recovery matrix

            Host.AssertNonEmpty(ColumnPairs);
            _infos = new ColInfo[ColumnPairs.Length];
            for (int i = 0; i < _infos.Length; i++)
                _infos[i] = new ColInfo(ctx);

            _models = new float[ColumnPairs.Length][];
            _invModels = new float[ColumnPairs.Length][];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                _models[i] = ctx.Reader.ReadFloatArray();
                if (_infos[i].SaveInv)
                    _invModels[i] = ctx.Reader.ReadFloatArray();
            }
        }

        internal WhiteningTransform(IHostEnvironment env, Arguments args, IDataView inputData)
            : this(env, inputData, GetColumnInfos(args))
        {
        }

        // Factory method for SignatureLoadModel
        internal static WhiteningTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            ctx.CheckAtModel(GetVersionInfo());
            return new WhiteningTransform(env, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
            => new WhiteningTransform(env, input, GetColumnInfos(args)).MakeDataTransform(input);

        // Factory method for SignatureLoadDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static (string input, string output)[] GetColumnPairs(ColInfo[] columns)
            => columns.Select(c => (c.Input, c.Output ?? c.Input)).ToArray();

        private static ColInfo[] GetColumnInfos(Arguments args)
            => args.Column.Select(colPair => new ColInfo(colPair, args)).ToArray();

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var inType = inputSchema.GetColumnType(srcCol);
            var reason = TestColumn(inType);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        internal static string TestColumn(ColumnType t)
        {
            if ((t.IsVector && !t.IsKnownSizeVector) || t.ItemType != NumberType.R4)
                return "Expected float or float vector of known size";

            if ((long)t.ValueCount * t.ValueCount > Utils.ArrayMaxSize)
                return "Vector size exceeds limit";

            return null;
        }

        private static void ValidateModel(IExceptionContext ectx, float[] model, ColumnType col)
        {
            ectx.CheckDecode(Utils.Size(model) == (long)col.ValueCount * col.ValueCount, "Invalid model size.");
            for (int i = 0; i < model.Length; i++)
                ectx.CheckDecode(FloatUtils.IsFinite(model[i]), "Found NaN or infinity in the model.");
        }

        private long GetRowCount(IDataView inputData)
        {
            long? rows = inputData.GetRowCount(lazy: false);
            if (rows != null)
                return rows.GetValueOrDefault();

            int maxRows = _infos.Max(i => i.MaxRow);
            long r = 0;
            using (var cursor = inputData.GetRowCursor(col => false))
            {
                while (r < maxRows && cursor.MoveNext())
                    r++;
            }
            return r;
        }

        private float[][] LoadDataAsDense(IChannel ch, IDataView inputData, out int[] actualRowCounts)
        {
            long crowData = GetRowCount(inputData);

            var columnData = new float[ColumnPairs.Length][];
            actualRowCounts = new int[ColumnPairs.Length];
            int maxActualRowCount = 0;

            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                ch.Assert(_srcTypes[i].IsVector && _srcTypes[i].IsKnownSizeVector);
                // Use not more than MaxRow number of rows.
                var ex = _infos[i];
                if (crowData <= ex.MaxRow)
                    actualRowCounts[i] = (int)crowData;
                else
                {
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", ex.MaxRow, ColumnPairs[i].output);
                    actualRowCounts[i] = ex.MaxRow;
                }

                int cslot = _srcTypes[i].ValueCount;
                // Check that total number of values in matrix does not exceed int.MaxValue and adjust row count if necessary.
                if ((long)cslot * actualRowCounts[i] > int.MaxValue)
                {
                    actualRowCounts[i] = int.MaxValue / cslot;
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", actualRowCounts[i], ColumnPairs[i].output);
                }
                columnData[i] = new float[cslot * actualRowCounts[i]];
                if (actualRowCounts[i] > maxActualRowCount)
                    maxActualRowCount = actualRowCounts[i];
            }
            var idxDst = new int[ColumnPairs.Length];

            var colsSet = new HashSet<int>(_cols);
            using (var cursor = inputData.GetRowCursor(colsSet.Contains))
            {
                var getters = new ValueGetter<VBuffer<float>>[ColumnPairs.Length];
                for (int i = 0; i < ColumnPairs.Length; i++)
                    getters[i] = cursor.GetGetter<VBuffer<float>>(_cols[i]);
                var val = default(VBuffer<float>);
                int irow = 0;
                while (irow < maxActualRowCount && cursor.MoveNext())
                {
                    for (int i = 0; i < ColumnPairs.Length; i++)
                    {
                        if (irow >= actualRowCounts[i] || columnData[i].Length == 0)
                            continue;

                        getters[i](ref val);
                        val.CopyTo(columnData[i], idxDst[i]);
                        idxDst[i] += _srcTypes[i].ValueCount;
                    }
                    irow++;
                }
#if DEBUG
                for (int i = 0; i < ColumnPairs.Length; i++)
                    ch.Assert(idxDst[i] == columnData[i].Length);
#endif
            }
            return columnData;
        }

        private void TrainModels(float[][] columnData, int[] rowCounts, IChannel ch)
        {
            Host.AssertValue(ch);
            ch.Assert(columnData.Length == rowCounts.Length);

            for (int iinfo = 0; iinfo < ColumnPairs.Length; iinfo++)
            {
                var ex = _infos[iinfo];
                var data = columnData[iinfo];
                int crow = rowCounts[iinfo];
                int ccol = _srcTypes[iinfo].ValueCount;

                // Compute covariance matrix (sigma).
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
                if (ex.Kind == WhiteningKind.Pca)
                {
                    // Save all components for PCA. Retained components will be selected during evaluation.
                    _models[iinfo] = uScaled;
                    if (ex.SaveInv)
                        _invModels[iinfo] = uInvScaled;
                }
                else if (ex.Kind == WhiteningKind.Zca)
                {
                    _models[iinfo] = new float[u.Length];
                    Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                        ccol, ccol, ccol, 1, u, ccol, uScaled, ccol, 0, _models[iinfo], ccol);

                    if (ex.SaveInv)
                    {
                        _invModels[iinfo] = new float[u.Length];
                        Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                            ccol, ccol, ccol, 1, u, ccol, uInvScaled, ccol, 0, _invModels[iinfo], ccol);
                    }
                }
                else
                    ch.Assert(false);
            }
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(float)
            // <base>
            // foreach added column
            //   ColInfoEx
            // foreach model
            //   whitening matrix
            //   recovery matrix

            SaveColumns(ctx);

            Host.Assert(_infos.Length == ColumnPairs.Length);
            for (int i = 0; i < _infos.Length; i++)
                _infos[i].Save(ctx);
            for (int i = 0; i < _models.Length; i++)
            {
                ctx.Writer.WriteFloatArray(_models[i]);
                if (_infos[i].SaveInv)
                    ctx.Writer.WriteFloatArray(_invModels[i]);
            }
        }

        private static class Mkl
        {
            private const string DllName = "MklImports";

            public enum Layout
            {
                RowMajor = 101,
                ColMajor = 102
            }

            public enum Transpose
            {
                NoTrans = 111,
                Trans = 112,
                ConjTrans = 113
            }

            public enum SvdJob : byte
            {
                None = (byte)'N',
                All = (byte)'A',
                Min = (byte)'S',
                MinOvr = (byte)'O',
            }

            // See: https://software.intel.com/en-us/node/520750
            [DllImport(DllName, EntryPoint = "cblas_sgemv")]
            public static extern void Gemv(Layout layout, Transpose trans, int m, int n, float alpha,
                float[] a, int lda, float[] x, int incx, float beta, float[] y, int incy);

            // See: https://software.intel.com/en-us/node/520775
            [DllImport(DllName, EntryPoint = "cblas_sgemm")]
            public static extern void Gemm(Layout layout, Transpose transA, Transpose transB, int m, int n, int k, float alpha,
                float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc);

            // See: https://software.intel.com/en-us/node/521150
            [DllImport(DllName, EntryPoint = "LAPACKE_sgesvd")]
            public static extern int Svd(Layout layout, SvdJob jobu, SvdJob jobvt,
                int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt, float[] superb);
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly WhiteningTransform _parent;
            private readonly int[] _cols;
            private readonly ColumnType[] _srcTypes;

            public Mapper(WhiteningTransform parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _cols = new int[_parent.ColumnPairs.Length];
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var info = _parent._infos[i];
                    if (!InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _cols[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    _srcTypes[i] = inputSchema.GetColumnType(_cols[i]);
                    ColumnType outType = (info.Kind == WhiteningKind.Pca && info.PcaNum > 0) ? new VectorType(NumberType.Float, info.PcaNum) : _srcTypes[i];
                    ValidateModel(Host, _parent._models[i], outType);
                    if (info.SaveInv)
                        ValidateModel(Host, _parent._invModels[i], outType);
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _parent.ColumnPairs.Length; iinfo++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[iinfo].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var info = _parent._infos[iinfo];
                    ColumnType outType = (info.Kind == WhiteningKind.Pca && info.PcaNum > 0) ? new VectorType(NumberType.Float, info.PcaNum) : _srcTypes[iinfo];
                    result[iinfo] = new Schema.Column(_parent.ColumnPairs[iinfo].output, outType, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var ex = _parent._infos[iinfo];
                Host.Assert(ex.Kind == WhiteningKind.Pca || ex.Kind == WhiteningKind.Zca);
                var getSrc = GetSrcGetter<VBuffer<float>>(input, iinfo);
                var src = default(VBuffer<float>);
                int cslotSrc = _srcTypes[iinfo].ValueCount;
                int cslotDst = (ex.Kind == WhiteningKind.Pca && ex.PcaNum > 0) ? ex.PcaNum : _srcTypes[iinfo].ValueCount;
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

            private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                int src = _cols[iinfo];
                Host.Assert(input.IsColumnActive(src));
                return input.GetGetter<T>(src);
            }

            private static void FillValues(float[] model, ref VBuffer<float> src, ref VBuffer<float> dst, int cdst)
            {
                int count = src.Count;
                int length = src.Length;
                var values = src.Values;
                var indices = src.Indices;
                Contracts.Assert(Utils.Size(values) >= count);

                // Since the whitening process produces dense vector, always use dense representation of dst.
                var a = Utils.Size(dst.Values) >= cdst ? dst.Values : new float[cdst];
                if (src.IsDense)
                {
                    Mkl.Gemv(Mkl.Layout.RowMajor, Mkl.Transpose.NoTrans, cdst, length,
                        1, model, length, values, 1, 0, a, 1);
                }
                else
                {
                    Contracts.Assert(Utils.Size(indices) >= count);

                    int offs = 0;
                    for (int i = 0; i < cdst; i++)
                    {
                        a[i] = DotProduct(model, offs, values, indices, count);
                        offs += length;
                    }
                }
                dst = new VBuffer<float>(cdst, a, dst.Indices);
            }

            /// <summary>
            /// Returns a dot product of dense vector 'a' starting from offset 'aOffset' and sparse vector 'b'
            /// with first 'count' valid elements and their corresponding 'indices'.
            /// </summary>
            private static float DotProduct(float[] a, int aOffset, float[] b, int[] indices, int count)
            {
                Contracts.Assert(count <= indices.Length);
                return CpuMathUtils.DotProductSparse(a.AsSpan(aOffset), b, indices, count);

            }
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
    public sealed class WhiteningEstimator : IEstimator<WhiteningTransform>
    {
        private readonly IHost _host;
        private readonly WhiteningTransform.ColInfo[] _infos;

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="columns"> Describes the settings for the transformer.</param>
        public WhiteningEstimator(IHostEnvironment env, params WhiteningTransform.ColInfo[] columns)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(WhiteningTransform));
            _infos = columns;
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="kind">Whitening kind (PCA/ZCA).</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public WhiteningEstimator(IHostEnvironment env, string inputColumn, string outputColumn,
            WhiteningKind kind = WhiteningTransform.Defaults.Kind,
            float eps = WhiteningTransform.Defaults.Eps,
            int maxRows = WhiteningTransform.Defaults.MaxRows,
            bool saveInverse = WhiteningTransform.Defaults.SaveInverse,
            int pcaNum = WhiteningTransform.Defaults.PcaNum)
            : this(env, new WhiteningTransform.ColInfo(inputColumn, outputColumn, kind, eps, maxRows, saveInverse, pcaNum))
        {
        }

        public WhiteningTransform Fit(IDataView input)
            => new WhiteningTransform(_host, input, _infos);

        /// <summary>
        /// Returns the schema that would be produced by the transformation.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colPair in _infos)
            {
                if (!inputSchema.TryFindColumn(colPair.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.Input);
                var reason = WhiteningTransform.TestColumn(col.ItemType);
                if (reason != null)
                    throw _host.ExceptUserArg(nameof(inputSchema), reason);
                result[colPair.Output] = new SchemaShape.Column(colPair.Output, col.Kind, col.ItemType, col.IsKey, null);
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Extensions for statically typed Whitening estimator.
    /// </summary>
    public static class WhiteningExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, WhiteningKind kind, float eps, int maxRows, bool saveInverse, int pcaNum)
                : base(new Reconciler(kind, eps, maxRows, saveInverse, pcaNum), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly WhiteningKind _kind;
            private readonly float _eps;
            private readonly int _maxRows;
            private readonly bool _saveInverse;
            private readonly int _pcaNum;

            public Reconciler(WhiteningKind kind, float eps, int maxRows, bool saveInverse, int pcaNum)
            {
                _kind = kind;
                _eps = eps;
                _maxRows = maxRows;
                _saveInverse = saveInverse;
                _pcaNum = pcaNum;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var infos = new WhiteningTransform.ColInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; i++)
                    infos[i] = new WhiteningTransform.ColInfo(inputNames[((OutPipelineColumn)toOutput[i]).Input], outputNames[toOutput[i]], _kind, _eps, _maxRows, _saveInverse, _pcaNum);

                return new WhiteningEstimator(env, infos);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to which the transform will be applied.</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        /// <param name="pcaNum">PCA components to retain.</param>
        public static Vector<float> PcaWhitening(this Vector<float> input,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false,
            int pcaNum = 0) => new OutPipelineColumn(input, WhiteningKind.Pca, eps, maxRows, saveInverse, pcaNum);

        /// <include file='doc.xml' path='doc/members/member[@name="Whitening"]/*'/>
        /// <param name="input">The column to which the transform will be applied.</param>
        /// <param name="eps">Scaling regularizer.</param>
        /// <param name="maxRows">Max number of rows.</param>
        /// <param name="saveInverse">Whether to save inverse (recovery) matrix.</param>
        public static Vector<float> ZcaWhitening(this Vector<float> input,
            float eps = (float)1e-5,
            int maxRows = 100 * 1000,
            bool saveInverse = false) => new OutPipelineColumn(input, WhiteningKind.Zca, eps, maxRows, saveInverse, 0);
    }
}