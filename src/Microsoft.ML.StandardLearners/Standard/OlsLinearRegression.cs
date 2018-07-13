// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Training;
using System.Runtime.InteropServices;

[assembly: LoadableClass(OlsLinearRegressionTrainer.Summary, typeof(OlsLinearRegressionTrainer), typeof(OlsLinearRegressionTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    OlsLinearRegressionTrainer.UserNameValue,
    OlsLinearRegressionTrainer.LoadNameValue,
    OlsLinearRegressionTrainer.ShortName)]

[assembly: LoadableClass(typeof(OlsLinearRegressionPredictor), null, typeof(SignatureLoadModel),
    "OLS Linear Regression Executor",
    OlsLinearRegressionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    public sealed class OlsLinearRegressionTrainer : TrainerBase<OlsLinearRegressionPredictor>
    {
        public sealed class Arguments : LearnerInputBaseWithWeight
        {
            // Adding L2 regularization turns this into a form of ridge regression,
            // rather than, strictly speaking, ordinary least squares. But it is an
            // incredibly uesful thing to have around.
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularization weight", ShortName = "l2", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "1e-6,0.1,1")]
            [TlcModule.SweepableDiscreteParamAttribute("L2Weight", new object[] { 1e-6f, 0.1f, 1f })]
            public Float L2Weight = 1e-6f;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether to calculate per parameter significance statistics", ShortName = "sig")]
            public bool PerParameterSignificance = true;
        }

        public const string LoadNameValue = "OLSLinearRegression";
        public const string UserNameValue = "Ordinary Least Squares (Regression)";
        public const string ShortName = "ols";
        internal const string Summary = "The ordinary least square regression fits the target function as a linear function of the numerical features "
            + "that minimizes the square loss function.";
        internal const string Remarks = @"<remarks>
<a href='https://en.wikipedia.org/wiki/Ordinary_least_squares'>Ordinary least squares (OLS)</a> is a parameterized regression method. 
It assumes that the conditional mean of the dependent variable follows a linear function of the dependent variables.
By minimizing the squares of the difference between observed values and the predictions, the parameters of the regressor can be estimated.
</remarks>";

        private readonly Float _l2Weight;
        private readonly bool _perParameterSignificance;

        public override PredictionKind PredictionKind => PredictionKind.Regression;
        public override TrainerInfo Info { get; }

        public OlsLinearRegressionTrainer(IHostEnvironment env, Arguments args)
            : base(env, LoadNameValue)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(args.L2Weight >= 0, nameof(args.L2Weight), "L2 regularization term cannot be negative");
            _l2Weight = args.L2Weight;
            _perParameterSignificance = args.PerParameterSignificance;
            // Two passes, only. Probably not worth caching.
            Info = new TrainerInfo(caching: false);
        }

        /// <summary>
        /// In several calculations, we calculate probabilities or other quantities that should range
        /// from 0 to 1, but because of numerical imprecision may, in entirely innocent circumstances,
        /// land outside that range. This is a helper function to "reclamp" this to sane ranges.
        /// </summary>
        /// <param name="p">The quantity that should be clamped from 0 to 1</param>
        /// <returns>Either p, or 0 or 1 if it was outside the range 0 to 1</returns>
        private static Double ProbClamp(Double p)
            => Math.Max(0, Math.Min(p, 1));

        public override OlsLinearRegressionPredictor Train(TrainContext context)
        {
            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(context, nameof(context));
                var examples = context.TrainingSet;
                ch.CheckParam(examples.Schema.Feature != null, nameof(examples), "Need a feature column");
                ch.CheckParam(examples.Schema.Label != null, nameof(examples), "Need a label column");

                // The label type must be either Float or a key type based on int (if allowKeyLabels is true).
                var typeLab = examples.Schema.Label.Type;
                if (typeLab != NumberType.Float)
                    throw ch.Except("Incompatible label column type {0}, must be {1}", typeLab, NumberType.Float);

                // The feature type must be a vector of Float.
                var typeFeat = examples.Schema.Feature.Type;
                if (!typeFeat.IsKnownSizeVector)
                    throw ch.Except("Incompatible feature column type {0}, must be known sized vector of {1}", typeFeat, NumberType.Float);
                if (typeFeat.ItemType != NumberType.Float)
                    throw ch.Except("Incompatible feature column type {0}, must be vector of {1}", typeFeat, NumberType.Float);

                var cursorFactory = new FloatLabelCursor.Factory(examples, CursOpt.Label | CursOpt.Features);

                var pred = TrainCore(ch, cursorFactory, typeFeat.VectorSize);
                ch.Done();
                return pred;
            }
        }

        private OlsLinearRegressionPredictor TrainCore(IChannel ch, FloatLabelCursor.Factory cursorFactory, int featureCount)
        {
            Host.AssertValue(ch);
            ch.AssertValue(cursorFactory);

            int m = featureCount + 1;

            // Check for memory conditions first.
            if ((long)m * (m + 1) / 2 > int.MaxValue)
                throw ch.Except("Cannot hold covariance matrix in memory with {0} features", m - 1);

            // Track the number of examples.
            long n = 0;
            // Since we are accumulating over many values, we use Double even for the single precision build.
            var xty = new Double[m];
            // The layout of this algorithm is a packed row-major lower triangular matrix.
            var xtx = new Double[m * (m + 1) / 2];

            // Build X'X (lower triangular) and X'y incrementally (X'X+=X'X_i; X'y+=X'y_i):
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    var yi = cursor.Label;
                    // Increment first element of X'y
                    xty[0] += yi;
                    // Increment first element of lower triangular X'X
                    xtx[0] += 1;
                    var values = cursor.Features.Values;

                    if (cursor.Features.IsDense)
                    {
                        int ioff = 1;
                        ch.Assert(cursor.Features.Count + 1 == m);
                        // Increment rest of first column of lower triangular X'X
                        for (int i = 1; i < m; i++)
                        {
                            ch.Assert(ioff == i * (i + 1) / 2);
                            var val = values[i - 1];
                            // Add the implicit first bias term to X'X
                            xtx[ioff++] += val;
                            // Add the remainder of X'X
                            for (int j = 0; j < i; j++)
                                xtx[ioff++] += val * values[j];
                            // X'y
                            xty[i] += val * yi;
                        }
                        ch.Assert(ioff == xtx.Length);
                    }
                    else
                    {
                        var fIndices = cursor.Features.Indices;
                        for (int ii = 0; ii < cursor.Features.Count; ++ii)
                        {
                            int i = fIndices[ii] + 1;
                            int ioff = i * (i + 1) / 2;
                            var val = values[ii];
                            // Add the implicit first bias term to X'X
                            xtx[ioff++] += val;
                            // Add the remainder of X'X
                            for (int jj = 0; jj <= ii; jj++)
                                xtx[ioff + fIndices[jj]] += val * values[jj];
                            // X'y
                            xty[i] += val * yi;
                        }
                    }
                    n++;
                }
                ch.Check(n > 0, "No training examples in dataset.");
                if (cursor.BadFeaturesRowCount > 0)
                    ch.Warning("Skipped {0} instances with missing features/label during training", cursor.SkippedRowCount);

                if (_l2Weight > 0)
                {
                    // Skip the bias term for regularization, in the ridge regression case.
                    // So start at [1,1] instead of [0,0].

                    // REVIEW: There are two ways to view this, firstly, it is more
                    // user friendly ot make this scaling factor behave similarly regardless
                    // of data size, so that if you have the same parameters, you get the same
                    // model if you feed in your data than if you duplicate your data 10 times.
                    // This is what I have now. The alternate point of view is to view this
                    // L2 regularization parameter as providing some sort of prior, in which
                    // case duplication 10 times should in fact be treated differently! (That
                    // is, we should not multiply by n below.) Both interpretations seem
                    // correct, in their way.
                    Double squared = _l2Weight * _l2Weight * n;
                    int ioff = 0;
                    for (int i = 1; i < m; ++i)
                        xtx[ioff += i + 1] += squared;
                    ch.Assert(ioff == xtx.Length - 1);
                }
            }

            if (!(_l2Weight > 0) && n < m)
                throw ch.Except("Ordinary least squares requires more examples than parameters. There are {0} parameters, but {1} examples. To enable training, use a positive L2 weight so this behaves as ridge regression.", m, n);

            Double yMean = n == 0 ? 0 : xty[0] / n;

            ch.Info("Trainer solving for {0} parameters across {1} examples", m, n);
            // Cholesky Decomposition of X'X into LL'
            try
            {
                Mkl.Pptrf(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, m, xtx);
            }
            catch (DllNotFoundException)
            {
                // REVIEW: Is there no better way?
                throw ch.ExceptNotSupp("The MKL library (Microsoft.ML.MklImports.dll) or one of its dependencies is missing.");
            }
            // Solve for beta in (LL')beta = X'y:
            Mkl.Pptrs(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, m, 1, xtx, xty, 1);
            // Note that the solver overwrote xty so it contains the solution. To be more clear,
            // we effectively change its name (through reassignment) so we don't get confused that
            // this is somehow xty in the remaining calculation.
            var beta = xty;
            xty = null;
            // Check that the solution is valid.
            for (int i = 0; i < beta.Length; ++i)
                ch.Check(FloatUtils.IsFinite(beta[i]), "Non-finite values detected in OLS solution");

            var weights = VBufferUtils.CreateDense<Float>(beta.Length - 1);
            for (int i = 1; i < beta.Length; ++i)
                weights.Values[i - 1] = (Float)beta[i];
            var bias = (Float)beta[0];
            if (!(_l2Weight > 0) && m == n)
            {
                // We would expect the solution to the problem to be exact in this case.
                ch.Info("Number of examples equals number of parameters, solution is exact but no statistics can be derived");
                return new OlsLinearRegressionPredictor(Host, ref weights, bias, null, null, null, 1, Float.NaN);
            }

            Double rss = 0; // residual sum of squares
            Double tss = 0; // total sum of squares
            using (var cursor = cursorFactory.Create())
            {
                var lrPredictor = new LinearRegressionPredictor(Host, ref weights, bias);
                var lrMap = lrPredictor.GetMapper<VBuffer<Float>, Float>();
                Float yh = default;
                while (cursor.MoveNext())
                {
                    var features = cursor.Features;
                    lrMap(ref features, ref yh);
                    var e = cursor.Label - yh;
                    rss += e * e;
                    var ydm = cursor.Label - yMean;
                    tss += ydm * ydm;
                }
            }
            var rSquared = ProbClamp(1 - (rss / tss));
            // R^2 adjusted differs from the normal formula on account of the bias term, by Said's reckoning.
            double rSquaredAdjusted;
            if (n > m)
            {
                rSquaredAdjusted = ProbClamp(1 - (1 - rSquared) * (n - 1) / (n - m));
                ch.Info("Coefficient of determination R2 = {0:g}, or {1:g} (adjusted)",
                    rSquared, rSquaredAdjusted);
            }
            else
                rSquaredAdjusted = Double.NaN;

            // The per parameter significance is compute intensive and may not be required for all practitioners.
            // Also we can't estimate it, unless we can estimate the variance, which requires more examples than
            // parameters.
            if (!_perParameterSignificance || m >= n)
                return new OlsLinearRegressionPredictor(Host, ref weights, bias, null, null, null, rSquared, rSquaredAdjusted);

            ch.Assert(!Double.IsNaN(rSquaredAdjusted));
            var standardErrors = new Double[m];
            var tValues = new Double[m];
            var pValues = new Double[m];
            // Invert X'X:
            Mkl.Pptri(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, m, xtx);
            var s2 = rss / (n - m); // estimate of variance of y

            for (int i = 0; i < m; i++)
            {
                // Initialize with inverse Hessian.
                standardErrors[i] = (Single)xtx[i * (i + 1) / 2 + i];
            }

            if (_l2Weight > 0)
            {
                // Iterate through all entries of inverse Hessian to make adjustment to variance.
                int ioffset = 1;
                Float reg = _l2Weight * _l2Weight * n;
                for (int iRow = 1; iRow < m; iRow++)
                {
                    for (int iCol = 0; iCol <= iRow; iCol++)
                    {
                        var entry = (Single)xtx[ioffset];
                        var adjustment = -reg * entry * entry;
                        standardErrors[iRow] -= adjustment;
                        if (0 < iCol && iCol < iRow)
                            standardErrors[iCol] -= adjustment;
                        ioffset++;
                    }
                }

                Contracts.Assert(ioffset == xtx.Length);
            }

            for (int i = 0; i < m; i++)
            {
                // sqrt of diagonal entries of s2 * inverse(X'X + reg * I) * X'X * inverse(X'X + reg * I).
                standardErrors[i] = Math.Sqrt(s2 * standardErrors[i]);
                ch.Check(FloatUtils.IsFinite(standardErrors[i]), "Non-finite standard error detected from OLS solution");
                tValues[i] = beta[i] / standardErrors[i];
                pValues[i] = (Float)MathUtils.TStatisticToPValue(tValues[i], n - m);
                ch.Check(0 <= pValues[i] && pValues[i] <= 1, "p-Value calculated outside expected [0,1] range");
            }

            return new OlsLinearRegressionPredictor(Host, ref weights, bias, standardErrors, tValues, pValues, rSquared, rSquaredAdjusted);
        }

        internal static class Mkl
        {
            private const string DllName = "Microsoft.ML.MklImports.dll";

            public enum Layout
            {
                RowMajor = 101,
                ColMajor = 102
            }

            public enum UpLo : byte
            {
                Up = (byte)'U',
                Lo = (byte)'L'
            }

            [DllImport(DllName, EntryPoint = "LAPACKE_dpptrf")]
            private static extern int PptrfInternal(Layout layout, UpLo uplo, int n, Double[] ap);

            /// <summary>
            /// Cholesky factorization of a symmetric positive-definite double matrix, using packed storage.
            /// The <c>pptrf</c> name comes from LAPACK, and means PositiveDefinitePackedTriangular(Cholesky)Factorize.
            /// </summary>
            /// <param name="layout">The storage order of this matrix</param>
            /// <param name="uplo">Whether the passed in matrix stores the upper or lower triangular part of the matrix</param>
            /// <param name="n">The order of the matrix</param>
            /// <param name="ap">An array with at least n*(n+1)/2 entries, containing the packed upper/lower part of the matrix.
            /// The triangular factorization is stored in this passed in matrix, when it returns. (U^T U or L L^T depending
            /// on whether this was upper or lower.)</param>
            public static void Pptrf(Layout layout, UpLo uplo, int n, Double[] ap)
            {
                Contracts.CheckParam((long)n * (n + 1) / 2 <= ap.Length, nameof(ap), "vector had insufficient length");
                int retval = PptrfInternal(layout, uplo, n, ap);
                if (retval == 0)
                    return;

                switch (-1 - retval)
                {
                    case 0:
                        throw Contracts.ExceptParam(nameof(layout));
                    case 1:
                        throw Contracts.ExceptParam(nameof(uplo));
                    case 2:
                        throw Contracts.ExceptParam(nameof(n));
                    case 3:
                        throw Contracts.ExceptParam(nameof(ap));
                    default:
                        throw Contracts.ExceptParam(nameof(ap), "Input matrix was not positive-definite. Try using a larger L2 regularization weight.");
                }
            }

            [DllImport(DllName, EntryPoint = "LAPACKE_dpptrs")]
            private static extern int PptrsInternal(Layout layout, UpLo uplo, int n, int nrhs, Double[] ap, Double[] b, int ldb);

            /// <summary>
            /// Solves a system of linear equations, using the Cholesky factorization of the <c>A</c> matrix,
            /// typically returned from <c>Pptrf</c>.
            /// The <c>pptrf</c> name comes from LAPACK, and means PositiveDefinitePackedTriangular(Cholesky)Solve.
            /// </summary>
            /// <param name="layout">The storage order of this matrix</param>
            /// <param name="uplo">Whether the passed in matrix stores the upper or lower triangular part of the matrix</param>
            /// <param name="n">The order of the matrix</param>
            /// <param name="nrhs">The number of columns in the right hand side matrix</param>
            /// <param name="ap">An array with at least n*(n+1)/2 entries, containing a Cholesky factorization
            /// of the matrix in the linear equation.</param>
            /// <param name="b">The right hand side</param>
            /// <param name="ldb">The major index step size (typically for row major order, the number of columns,
            /// or something larger)</param>
            public static void Pptrs(Layout layout, UpLo uplo, int n, int nrhs, Double[] ap, Double[] b, int ldb)
            {
                Contracts.CheckParam((long)n * (n + 1) / 2 <= ap.Length, nameof(ap), "vector had insufficient length");
                Contracts.CheckParam((long)n * ldb <= b.Length, nameof(b), "vector had insufficient length");
                int retval = PptrsInternal(layout, uplo, n, nrhs, ap, b, ldb);
                if (retval == 0)
                    return;

                switch (-1 - retval)
                {
                    case 0:
                        throw Contracts.ExceptParam(nameof(layout));
                    case 1:
                        throw Contracts.ExceptParam(nameof(uplo));
                    case 2:
                        throw Contracts.ExceptParam(nameof(n));
                    case 3:
                        throw Contracts.ExceptParam(nameof(nrhs));
                    case 4:
                        throw Contracts.ExceptParam(nameof(ap));
                    case 5:
                        throw Contracts.ExceptParam(nameof(b));
                    case 6:
                        throw Contracts.ExceptParam(nameof(ldb));
                    default:
                        throw Contracts.Except();
                }

            }

            [DllImport(DllName, EntryPoint = "LAPACKE_dpptri")]
            private static extern int PptriInternal(Layout layout, UpLo uplo, int n, Double[] ap);

            /// <summary>
            /// Compute the inverse of a matrix, using the Cholesky factorization of the <c>A</c> matrix,
            /// typically returned from <c>Pptrf</c>.
            /// The <c>pptrf</c> name comes from LAPACK, and means PositiveDefinitePackedTriangular(Cholesky)Invert.
            /// </summary>
            /// <param name="layout">The storage order of this matrix</param>
            /// <param name="uplo">Whether the passed in matrix stores the upper or lower triangular part of the matrix</param>
            /// <param name="n">The order of the matrix</param>
            /// <param name="ap">An array with at least n*(n+1)/2 entries, containing a Cholesky factorization
            /// of the matrix in the linear equation. The inverse is returned in this array.</param>
            public static void Pptri(Layout layout, UpLo uplo, int n, Double[] ap)
            {
                Contracts.CheckParam((long)n * (n + 1) / 2 <= ap.Length, nameof(ap), "vector had insufficient length");
                int retval = PptriInternal(layout, uplo, n, ap);

                if (retval == 0)
                    return;

                switch (-1 - retval)
                {
                    case 0:
                        throw Contracts.ExceptParam(nameof(layout));
                    case 1:
                        throw Contracts.ExceptParam(nameof(uplo));
                    case 2:
                        throw Contracts.ExceptParam(nameof(n));
                    case 3:
                        throw Contracts.ExceptParam(nameof(ap));
                    default:
                        throw Contracts.Except();
                }
            }
        }
    }

    /// <summary>
    /// A linear predictor for which per parameter significance statistics are available.
    /// </summary>
    public sealed class OlsLinearRegressionPredictor : RegressionPredictor
    {
        public const string LoaderSignature = "OlsLinearRegressionExec";
        public const string RegistrationName = "OlsLinearRegressionPredictor";

        /// <summary>
        /// Version information to be saved in binary format
        /// </summary>
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OLS RGRS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        // The following will be null iff RSquaredAdjusted is NaN.
        private readonly Double[] _standardErrors;
        private readonly Double[] _tValues;
        private readonly Double[] _pValues;
        private readonly Double _rSquared;
        private readonly Double _rSquaredAdjusted;

        /// <summary>
        /// The coefficient of determination.
        /// </summary>
        public Double RSquared
        { get { return _rSquared; } }

        /// <summary>
        /// The adjusted coefficient of determination. It is only possible to produce
        /// an adjusted R-squared if there are more examples than parameters in the model
        /// plus one. If this condition is not met, this value will be <c>NaN</c>.
        /// </summary>
        public Double RSquaredAdjusted
        { get { return _rSquaredAdjusted; } }

        /// <summary>
        /// Whether the model has per parameter statistics. This is false iff
        /// <see cref="StandardErrors"/>, <see cref="TValues"/>, and <see cref="PValues"/>
        /// are all null. A model may not have per parameter statistics because either
        /// there were not more examples than parameters in the model, or because they
        /// were explicitly suppressed in training by setting
        /// <see cref="OlsLinearRegressionTrainer.Arguments.PerParameterSignificance"/>
        /// to false.
        /// </summary>
        public bool HasStatistics
        { get { return _standardErrors != null; } }

        /// <summary>
        /// The standard error per model parameter, where the first corresponds to the bias,
        /// and all subsequent correspond to each weight in turn. This is <c>null</c> if and
        /// only if <see cref="HasStatistics"/> is <c>false</c>.
        /// </summary>
        public IReadOnlyCollection<Double> StandardErrors
        { get { return _standardErrors.AsReadOnly(); } }

        /// <summary>
        /// t-Statistic values corresponding to each of the model standard errors. This is
        /// <c>null</c> if and only if <see cref="HasStatistics"/> is <c>false</c>.
        /// </summary>
        public IReadOnlyCollection<Double> TValues
        { get { return _tValues.AsReadOnly(); } }

        /// <summary>
        /// p-values corresponding to each of the model standard errors. This is <c>null</c>
        /// if and only if <see cref="HasStatistics"/> is <c>false</c>.
        /// </summary>
        public IReadOnlyCollection<Double> PValues
        { get { return _pValues.AsReadOnly(); } }

        internal OlsLinearRegressionPredictor(IHostEnvironment env, ref VBuffer<Float> weights, Float bias,
            Double[] standardErrors, Double[] tValues, Double[] pValues, Double rSquared, Double rSquaredAdjusted)
            : base(env, RegistrationName, ref weights, bias)
        {
            Contracts.AssertValueOrNull(standardErrors);
            Contracts.AssertValueOrNull(tValues);
            Contracts.AssertValueOrNull(pValues);
            // If r-squared is NaN then the other statistics must be null, however, if r-rsquared is not NaN,
            // then the statistics may be null if creation of statistics was suppressed.
            Contracts.Assert(!Double.IsNaN(rSquaredAdjusted) || standardErrors == null);
            // Nullity or not must be consistent between the statistics.
            Contracts.Assert((standardErrors == null) == (tValues == null) && (tValues == null) == (pValues == null));
            Contracts.Assert(0 <= rSquared & rSquared <= 1);
            Contracts.Assert(Double.IsNaN(rSquaredAdjusted) | (0 <= rSquaredAdjusted & rSquaredAdjusted <= 1));
            if (standardErrors != null)
            {
                // If not null, the input arrays must have one value for each parameter.
                Contracts.Assert(Utils.Size(standardErrors) == weights.Length + 1);
                Contracts.Assert(Utils.Size(tValues) == weights.Length + 1);
                Contracts.Assert(Utils.Size(pValues) == weights.Length + 1);
#if DEBUG
                for (int i = 0; i <= weights.Length; ++i)
                {
                    Contracts.Assert(FloatUtils.IsFinite(standardErrors[i]));
                    Contracts.Assert(FloatUtils.IsFinite(tValues[i]));
                    Contracts.Assert(FloatUtils.IsFinite(pValues[i]));
                }
#endif
            }

            _standardErrors = standardErrors;
            _tValues = tValues;
            _pValues = pValues;
            _rSquared = rSquared;
            _rSquaredAdjusted = rSquaredAdjusted;
        }

        private OlsLinearRegressionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // double: r-squared value
            // double: adjusted r-squared value, or NaN
            // The remaining are only present if the previous value was true
            //     double[#parameters]: standard errors per model parameter (each weight and the bias)
            //     double[#parameters]: t-statistics per parameter
            //     double[#parameters]: p-values per parameter

            Host.CheckDecode(Weight.IsDense);
            int m = Weight.Length + 1;

            _rSquared = ctx.Reader.ReadDouble();
            ProbCheckDecode(RSquared);
            _rSquaredAdjusted = ctx.Reader.ReadDouble();
            if (!Double.IsNaN(RSquaredAdjusted))
                ProbCheckDecode(RSquaredAdjusted);
            bool hasStats = ctx.Reader.ReadBoolByte();
            Host.CheckDecode(!Double.IsNaN(RSquaredAdjusted) || !hasStats);
            if (!hasStats)
                return;

            _standardErrors = ctx.Reader.ReadDoubleArray(m);
            for (int i = 0; i < m; ++i)
                Host.CheckDecode(FloatUtils.IsFinite(_standardErrors[i]) && _standardErrors[i] >= 0);

            _tValues = ctx.Reader.ReadDoubleArray(m);
            TValueCheckDecode(Bias, _tValues[0]);
            for (int i = 1; i < m; ++i)
                TValueCheckDecode(Weight.Values[i - 1], _tValues[i]);

            _pValues = ctx.Reader.ReadDoubleArray(m);
            for (int i = 0; i < m; ++i)
                ProbCheckDecode(_pValues[i]);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            int m = Weight.Length + 1;

            // *** Binary format ***
            // double: r-squared value
            // double: adjusted r-squared value
            // bool: whether per parameter statistics were calculated
            // The remaining are only present if the previous value was true
            //     double[#parameters]: standard errors per model parameter (each weight and the bias)
            //     double[#parameters]: t-statistics per parameter
            //     double[#parameters]: p-values per parameter

            Contracts.Assert(0 <= _rSquared & _rSquared <= 1);
            ctx.Writer.Write(_rSquared);
            Contracts.Assert(Double.IsNaN(_rSquaredAdjusted) | (0 <= _rSquaredAdjusted & _rSquaredAdjusted <= 1));
            ctx.Writer.Write(_rSquaredAdjusted);
            Contracts.Assert(!Double.IsNaN(_rSquaredAdjusted) | !HasStatistics);
            ctx.Writer.WriteBoolByte(HasStatistics);
            if (!HasStatistics)
            {
                Contracts.Assert(_standardErrors == null & _tValues == null & _pValues == null);
                return;
            }
            Contracts.Assert(Weight.Length + 1 == _standardErrors.Length);
            Contracts.Assert(Weight.Length + 1 == _tValues.Length);
            Contracts.Assert(Weight.Length + 1 == _pValues.Length);
            ctx.Writer.WriteDoublesNoCount(_standardErrors, m);
            ctx.Writer.WriteDoublesNoCount(_tValues, m);
            ctx.Writer.WriteDoublesNoCount(_pValues, m);
        }

        private static void TValueCheckDecode(Double param, Double tvalue)
        {
            Contracts.CheckDecode(Math.Sign(param) == Math.Sign(tvalue));
        }

        private static void ProbCheckDecode(Double p)
        {
            Contracts.CheckDecode(0 <= p && p <= 1);
        }

        public static OlsLinearRegressionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new OlsLinearRegressionPredictor(env, ctx);
        }

        public override void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            var names = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, Weight.Length, ref names);

            writer.WriteLine("Ordinary Least Squares Model Summary");
            writer.WriteLine("R-squared: {0:g4}", RSquared);

            if (HasStatistics)
            {
                writer.WriteLine("Adjusted R-squared: {0:g4}", RSquaredAdjusted);
                writer.WriteLine();
                writer.WriteLine("Index\tName\tWeight\tStdErr\tt-Value\tp-Value");
                const string format = "{0}\t{1}\t{2}\t{3:g4}\t{4:g4}\t{5:e4}";
                writer.WriteLine(format, "", "Bias", Bias, _standardErrors[0], _tValues[0], _pValues[0]);
                Contracts.Assert(Weight.IsDense);
                var coeffs = Weight.Values;
                for (int i = 0; i < coeffs.Length; i++)
                {
                    var name = names.GetItemOrDefault(i);
                    writer.WriteLine(format, i, DvText.Identical(name, DvText.Empty) ? $"f{i}" : name.ToString(),
                        coeffs[i], _standardErrors[i + 1], _tValues[i + 1], _pValues[i + 1]);
                }
            }
            else
            {
                writer.WriteLine();
                writer.WriteLine("Index\tName\tWeight");
                const string format = "{0}\t{1}\t{2}";
                writer.WriteLine(format, "", "Bias", Bias);
                Contracts.Assert(Weight.IsDense);
                var coeffs = Weight.Values;
                for (int i = 0; i < coeffs.Length; i++)
                {
                    var name = names.GetItemOrDefault(i);
                    writer.WriteLine(format, i, DvText.Identical(name, DvText.Empty) ? $"f{i}" : name.ToString(), coeffs[i]);
                }
            }
        }

        public override void GetFeatureWeights(ref VBuffer<Float> weights)
        {
            if (_pValues == null)
            {
                base.GetFeatureWeights(ref weights);
                return;
            }

            var values = weights.Values;
            var size = _pValues.Length - 1;
            if (Utils.Size(values) < size)
                values = new Float[size];
            for (int i = 0; i < size; i++)
            {
                var score = -(Float)Math.Log(_pValues[i + 1]);
                if (score > Float.MaxValue)
                    score = Float.MaxValue;
                values[i] = score;
            }
            weights = new VBuffer<Float>(size, values, weights.Indices);
        }
    }
}

