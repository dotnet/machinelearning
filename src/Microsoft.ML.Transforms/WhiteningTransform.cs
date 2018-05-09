// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(WhiteningTransform.Summary, typeof(WhiteningTransform), typeof(WhiteningTransform.Arguments), typeof(SignatureDataTransform),
    "Whitening Transform", "WhiteningTransform", "Whitening")]

[assembly: LoadableClass(WhiteningTransform.Summary, typeof(WhiteningTransform), null, typeof(SignatureLoadDataTransform),
    "Whitening Transform", WhiteningTransform.LoaderSignature, WhiteningTransform.LoaderSignatureOld)]

namespace Microsoft.ML.Runtime.Data
{
    public enum WhiteningKind
    {
        [TGUI(Label = "PCA whitening")]
        Pca,

        [TGUI(Label = "ZCA whitening")]
        Zca
    }

    /// <summary>
    /// Implements PCA (Principal Component Analysis) and ZCA (Zero phase Component Analysis) whitening.
    /// The whitening process consists of 2 steps:
    /// 1. Decorrelation of the input data. Input data is assumed to have zero mean.
    /// 2. Rescale decorrelated features to have unit variance.
    /// That is, PCA whitening is essentially just a PCA + rescale.
    /// ZCA whitening tries to make resulting data to look more like input data by rotating it back to the 
    /// original input space.
    /// More information: <see href="http://ufldl.stanford.edu/wiki/index.php/Whitening"/>
    /// </summary>
    public sealed class WhiteningTransform : OneToOneTransformBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whitening kind (PCA/ZCA)")]
            public WhiteningKind Kind = WhiteningKind.Zca;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling regularizer")]
            public Float Eps = (Float)1e-5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of rows", ShortName = "rows")]
            public int MaxRows = 100 * 1000;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to save inverse (recovery) matrix", ShortName = "saveInv")]
            public bool SaveInverse = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "PCA components to retain")]
            public int PcaNum = 0;

            // REVIEW: add the following options:
            // 1. Currently there is no way to apply an inverse transform AFTER the the transform is trained.
            // 2. How many PCA components to retain/drop. Options: retain-first, drop-first, variance-threshold.
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whitening kind (PCA/ZCA)")]
            public WhiteningKind? Kind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling regularizer")]
            public Float? Eps;

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

        public sealed class ColInfoEx
        {
            public readonly WhiteningKind Kind;
            public readonly Float Epsilon;
            public readonly int MaxRow;
            public readonly bool SaveInv;
            public readonly int PcaNum;
            public readonly VectorType Type;

            public ColInfoEx(Column item, Arguments args, ColInfo info)
            {
                Kind = item.Kind ?? args.Kind;
                Contracts.CheckUserArg(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca, nameof(item.Kind));
                Epsilon = item.Eps ?? args.Eps;
                Contracts.CheckUserArg(0 <= Epsilon && Epsilon < Float.PositiveInfinity, nameof(item.Eps));
                MaxRow = item.MaxRows ?? args.MaxRows;
                Contracts.CheckUserArg(MaxRow > 0, nameof(item.MaxRows));
                SaveInv = item.SaveInverse ?? args.SaveInverse;
                PcaNum = item.PcaNum ?? args.PcaNum;
                Contracts.CheckUserArg(PcaNum >= 0, nameof(item.PcaNum));

                if (Kind == WhiteningKind.Zca || PcaNum == 0)
                    Type = info.TypeSrc.AsVector;
                else
                    Type = new VectorType(NumberType.Float, PcaNum); // REVIEW: make it work with pcaNum == 1.
            }

            public ColInfoEx(ModelLoadContext ctx, ColInfo info)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int:   kind
                // Float: epsilon
                // int:   maxrow
                // byte:  saveInv
                // int:   pcaNum
                Kind = (WhiteningKind)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca);
                Epsilon = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(0 <= Epsilon && Epsilon < Float.PositiveInfinity);
                MaxRow = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(MaxRow > 0);
                SaveInv = ctx.Reader.ReadBoolByte();
                PcaNum = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(PcaNum >= 0);

                if (Kind == WhiteningKind.Zca || PcaNum == 0)
                    Type = info.TypeSrc.AsVector;
                else
                    Type = new VectorType(NumberType.Float, PcaNum); // REVIEW: make it work with pcaNum == 1.
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int:   kind
                // Float: epsilon
                // int:   maxrow
                // byte:  saveInv
                // int:   pcaNum
                Contracts.Assert(Kind == WhiteningKind.Pca || Kind == WhiteningKind.Zca);
                ctx.Writer.Write((int)Kind);
                Contracts.Assert(0 <= Epsilon && Epsilon < Float.PositiveInfinity);
                ctx.Writer.Write(Epsilon);
                Contracts.Assert(MaxRow > 0);
                ctx.Writer.Write(MaxRow);
                ctx.Writer.WriteBoolByte(SaveInv);
                Contracts.Assert(PcaNum >= 0);
                ctx.Writer.Write(PcaNum);
            }
        }

        private const Mkl.Layout Layout = Mkl.Layout.RowMajor;

        // Stores whitening matrix as Float[] for each column.
        private readonly Float[][] _models;
        // Stores inverse ("recover") matrix as Float[] for each column. Temporarily internal as it's used in unit test.
        // REVIEW: It doesn't look like this is used by non-test code. Should it be saved at all?
        internal readonly Float[][] InvModels;

        internal const string Summary = "Apply PCA or ZCA whitening algorithm to the input.";

        public const string LoaderSignature = "WhiteningTransform";
        internal const string LoaderSignatureOld = "WhiteningFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WHITENTF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld);
        }

        private readonly ColInfoEx[] _exes;

        private const string RegistrationName = "Whitening";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public WhiteningTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestColumn)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(args.Column[i], args, Infos[i]);

            using (var ch = Host.Start("Training"))
            {
                // The training process will load all data into memory and perform whitening process
                // for each resulting column separately.
                _models = new Float[Infos.Length][];
                InvModels = new Float[Infos.Length][];
                int[] rowCounts;
                var columnData = LoadDataAsDense(ch, out rowCounts);
                TrainModels(columnData, rowCounts, ch);
                ch.Done();
            }
            Metadata.Seal();
        }

        private Float[][] LoadDataAsDense(IChannel ch, out int[] actualRowCounts)
        {
            long crowData = GetRowCount();

            var columnData = new Float[Infos.Length][];
            actualRowCounts = new int[Infos.Length];
            int maxActualRowCount = 0;
            for (int i = 0; i < Infos.Length; i++)
            {
                var type = Infos[i].TypeSrc;
                ch.Assert(type.IsVector && type.IsKnownSizeVector);
                // Use not more than MaxRow number of rows.
                var ex = _exes[i];
                if (crowData <= ex.MaxRow)
                    actualRowCounts[i] = (int)crowData;
                else
                {
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", ex.MaxRow, Infos[i].Name);
                    actualRowCounts[i] = ex.MaxRow;
                }

                int cslot = type.ValueCount;
                // Check that total number of values in matrix does not exceed int.MaxValue and adjust row count if necessary.
                if ((long)cslot * actualRowCounts[i] > int.MaxValue)
                {
                    actualRowCounts[i] = int.MaxValue / cslot;
                    ch.Info(MessageSensitivity.Schema, "Only {0:N0} rows of column '{1}' will be used for whitening transform.", actualRowCounts[i], Infos[i].Name);
                }
                columnData[i] = new Float[cslot * actualRowCounts[i]];
                if (actualRowCounts[i] > maxActualRowCount)
                    maxActualRowCount = actualRowCounts[i];
            }
            var idxDst = new int[Infos.Length];

            var cols = new HashSet<int>(Infos.Select(info => info.Source));
            using (var cursor = Source.GetRowCursor(cols.Contains))
            {
                var getters = new ValueGetter<VBuffer<Float>>[Infos.Length];
                for (int i = 0; i < Infos.Length; i++)
                    getters[i] = cursor.GetGetter<VBuffer<Float>>(Infos[i].Source);
                var val = default(VBuffer<Float>);
                int irow = 0;
                while (irow < maxActualRowCount && cursor.MoveNext())
                {
                    for (int i = 0; i < Infos.Length; i++)
                    {
                        if (irow >= actualRowCounts[i] || columnData[i].Length == 0)
                            continue;

                        getters[i](ref val);
                        val.CopyTo(columnData[i], idxDst[i]);
                        idxDst[i] += Infos[i].TypeSrc.ValueCount;
                    }
                    irow++;
                }
#if DEBUG
                for (int i = 0; i < Infos.Length; i++)
                    ch.Assert(idxDst[i] == columnData[i].Length);
#endif
            }

            return columnData;
        }

        private void TrainModels(Float[][] columnData, int[] rowCounts, IChannel ch)
        {
            Host.AssertValue(ch);
            ch.Assert(columnData.Length == rowCounts.Length);

            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var ex = _exes[iinfo];
                var data = columnData[iinfo];
                int crow = rowCounts[iinfo];
                int ccol = Infos[iinfo].TypeSrc.ValueCount;

                // Compute covariance matrix (sigma).
                var u = new Float[ccol * ccol];
                ch.Info("Computing covariance matrix...");
                Mkl.Gemm(Layout, Mkl.Transpose.Trans, Mkl.Transpose.NoTrans,
                    ccol, ccol, crow, 1 / (Float)crow, data, ccol, data, ccol, 0, u, ccol);

                ch.Info("Computing SVD...");
                var eigValues = new Float[ccol]; // Eigenvalues.
                var unconv = new Float[ccol]; // Superdiagonal unconverged values (if any). Not used but seems to be required by MKL.
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
                var eigValuesRcp = new Float[eigValues.Length];
                for (int i = 0; i < eigValuesRcp.Length; i++)
                    eigValuesRcp[i] = 1 / eigValues[i];

                // Scale eigenvectors. Note that resulting matrix is transposed, so the scaled
                // eigenvectors are stored row-wise.
                var uScaled = new Float[u.Length];
                var uInvScaled = new Float[u.Length];
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
                        InvModels[iinfo] = uInvScaled;
                }
                else if (ex.Kind == WhiteningKind.Zca)
                {
                    _models[iinfo] = new Float[u.Length];
                    Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                        ccol, ccol, ccol, 1, u, ccol, uScaled, ccol, 0, _models[iinfo], ccol);

                    if (ex.SaveInv)
                    {
                        InvModels[iinfo] = new Float[u.Length];
                        Mkl.Gemm(Layout, Mkl.Transpose.NoTrans, Mkl.Transpose.NoTrans,
                            ccol, ccol, ccol, 1, u, ccol, uInvScaled, ccol, 0, InvModels[iinfo], ccol);
                    }
                }
                else
                    ch.Assert(false);
            }
        }

        private WhiteningTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestColumn)
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

            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(ctx, Infos[i]);

            _models = new Float[Infos.Length][];
            InvModels = new Float[Infos.Length][];
            for (int i = 0; i < Infos.Length; i++)
            {
                _models[i] = ctx.Reader.ReadFloatArray();
                ValidateModel(Host, _models[i], Infos[i].TypeSrc);
                if (_exes[i].SaveInv)
                {
                    InvModels[i] = ctx.Reader.ReadFloatArray();
                    ValidateModel(Host, InvModels[i], Infos[i].TypeSrc);
                }
            }
            Metadata.Seal();
        }

        public static WhiteningTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <remainder handled in ctors>
            int cbFloat = ctx.Reader.ReadInt32();
            h.CheckDecode(cbFloat == sizeof(Float));
            return h.Apply("Loading Model", ch => new WhiteningTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // foreach added column
            //   ColInfoEx
            // foreach model
            //   whitening matrix
            //   recovery matrix
            ctx.Writer.Write(sizeof(Float));

            SaveBase(ctx);

            Host.Assert(_exes.Length == Infos.Length);
            for (int i = 0; i < _exes.Length; i++)
                _exes[i].Save(ctx);
            for (int i = 0; i < _models.Length; i++)
            {
                ctx.Writer.WriteFloatArray(_models[i]);
                if (_exes[i].SaveInv)
                    ctx.Writer.WriteFloatArray(InvModels[i]);
            }
        }

        private static string TestColumn(ColumnType t)
        {
            string reason = TestIsKnownSizeFloatVector(t);
            if (reason != null)
                return reason;

            if ((long)t.ValueCount * t.ValueCount > Utils.ArrayMaxSize)
                return "Vector size exceeds limit";

            return null;
        }

        private static void ValidateModel(IExceptionContext ectx, Float[] model, ColumnType col)
        {
            ectx.CheckDecode(Utils.Size(model) == (long)col.ValueCount * col.ValueCount, "Invalid model size.");
            for (int i = 0; i < model.Length; i++)
                ectx.CheckDecode(FloatUtils.IsFinite(model[i]), "Found NaN or infinity in the model.");
        }

        private long GetRowCount()
        {
            long? rows = Source.GetRowCount(lazy: false);
            if (rows != null)
                return rows.GetValueOrDefault();

            int maxRows = _exes.Max(i => i.MaxRow);
            long r = 0;
            using (var cursor = Source.GetRowCursor(col => false))
            {
                while (r < maxRows && cursor.MoveNext())
                    r++;
            }
            return r;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo & iinfo < Infos.Length);
            return _exes[iinfo].Type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var ex = _exes[iinfo];
            Host.Assert(ex.Kind == WhiteningKind.Pca || ex.Kind == WhiteningKind.Zca);
            var getSrc = GetSrcGetter<VBuffer<Float>>(input, iinfo);
            var src = default(VBuffer<Float>);
            int cslotSrc = Infos[iinfo].TypeSrc.ValueCount;
            int cslotDst = (ex.Kind == WhiteningKind.Pca && ex.PcaNum > 0) ? ex.PcaNum : Infos[iinfo].TypeSrc.ValueCount;
            var model = _models[iinfo];
            ValueGetter<VBuffer<Float>> del =
                (ref VBuffer<Float> dst) =>
                {
                    getSrc(ref src);
                    Host.Check(src.Length == cslotSrc, "Invalid column size.");
                    FillValues(model, ref src, ref dst, cslotDst);
                };
            return del;
        }

        private static void FillValues(Float[] model, ref VBuffer<Float> src, ref VBuffer<Float> dst, int cdst)
        {
            int count = src.Count;
            int length = src.Length;
            var values = src.Values;
            var indices = src.Indices;
            Contracts.Assert(Utils.Size(values) >= count);

            // Since the whitening process produces dense vector, always use dense representation of dst.
            var a = Utils.Size(dst.Values) >= cdst ? dst.Values : new Float[cdst];
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
            dst = new VBuffer<Float>(cdst, a, dst.Indices);
        }

        /// <summary>
        /// Returns a dot product of dense vector 'a' starting from offset 'aOffset' and sparse vector 'b'
        /// with first 'count' valid elements and their corresponding 'indices'.
        /// </summary>
        private static Float DotProduct(Float[] a, int aOffset, Float[] b, int[] indices, int count)
        {
            Contracts.Assert(count <= indices.Length);
            return SseUtils.DotProductSparse(a, aOffset, b, indices, count);

        }

        private static class Mkl
        {
            private const string DllName = "Microsoft.MachineLearning.MklImports.dll";

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
            public static extern void Gemv(Layout layout, Transpose trans, int m, int n, Float alpha,
                Float[] a, int lda, Float[] x, int incx, Float beta, Float[] y, int incy);

            // See: https://software.intel.com/en-us/node/520775
            [DllImport(DllName, EntryPoint = "cblas_sgemm")]
            public static extern void Gemm(Layout layout, Transpose transA, Transpose transB, int m, int n, int k, Float alpha,
                Float[] a, int lda, Float[] b, int ldb, Float beta, Float[] c, int ldc);

            // See: https://software.intel.com/en-us/node/521150
            [DllImport(DllName, EntryPoint = "LAPACKE_sgesvd")]
            public static extern int Svd(Layout layout, SvdJob jobu, SvdJob jobvt,
                int m, int n, Float[] a, int lda, Float[] s, Float[] u, int ldu, Float[] vt, int ldvt, Float[] superb);
        }
    }
}
