// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Trainers.HalLearners;
using System;

namespace Microsoft.ML.Runtime.Learners
{
    using Mkl = OlsLinearRegressionTrainer.Mkl;

    /// <include file='doc.xml' path='doc/members/member[@name="LBFGS"]/*' />
    /// <include file='doc.xml' path='docs/members/example[@name="LogisticRegressionBinaryClassifier"]/*' />
    public static class LogisticRegressionTrainingStats
    {

        public static void ComputeExtendedTrainingStatistics(this LinearBinaryPredictor model, IChannel ch, float l2Weight = LogisticRegression.Arguments.Defaults.L2Weight)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(model.Statistics, $"Training Statistics can get generated after training finishes. Train with setting: ShowTrainigStats set to true.");
            Contracts.Assert(l2Weight > 0);

            int numSelectedParams = model.Statistics.ParametersCount;

            // Apply Cholesky Decomposition to find the inverse of the Hessian.
            Double[] invHessian = null;
            try
            {
                // First, find the Cholesky decomposition LL' of the Hessian.
                Mkl.Pptrf(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numSelectedParams, model.Statistics.Hessian);
                // Note that hessian is already modified at this point. It is no longer the original Hessian,
                // but instead represents the Cholesky decomposition L.
                // Also note that the following routine is supposed to consume the Cholesky decomposition L instead
                // of the original information matrix.
                Mkl.Pptri(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numSelectedParams, model.Statistics.Hessian);
                // At this point, hessian should contain the inverse of the original Hessian matrix.
                // Swap hessian with invHessian to avoid confusion in the following context.
                Utils.Swap(ref model.Statistics.Hessian, ref invHessian);
                Contracts.Assert(model.Statistics.Hessian == null);
            }
            catch (DllNotFoundException)
            {
                throw ch.ExceptNotSupp("The MKL library (MklImports.dll) or one of its dependencies is missing.");
            }

            float[] stdErrorValues = new float[numSelectedParams];
            stdErrorValues[0] = (float)Math.Sqrt(invHessian[0]);

            for (int i = 1; i < numSelectedParams; i++)
            {
                // Initialize with inverse Hessian.
                stdErrorValues[i] = (Single)invHessian[i * (i + 1) / 2 + i];
            }

            if (l2Weight > 0)
            {
                // Iterate through all entries of inverse Hessian to make adjustment to variance.
                // A discussion on ridge regularized LR coefficient covariance matrix can be found here:
                // http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
                // http://www.inf.unibz.it/dis/teaching/DWDM/project2010/LogisticRegression.pdf
                int ioffset = 1;
                for (int iRow = 1; iRow < numSelectedParams; iRow++)
                {
                    for (int iCol = 0; iCol <= iRow; iCol++)
                    {
                        var entry = (Single)invHessian[ioffset];
                        var adjustment = -l2Weight * entry * entry;
                        stdErrorValues[iRow] -= adjustment;
                        if (0 < iCol && iCol < iRow)
                            stdErrorValues[iCol] -= adjustment;
                        ioffset++;
                    }
                }

                Contracts.Assert(ioffset == invHessian.Length);
            }

            for (int i = 1; i < numSelectedParams; i++)
                stdErrorValues[i] = (float)Math.Sqrt(stdErrorValues[i]);

           VBuffer<float> stdErrors = new VBuffer<float>(model.Weights2.Count, numSelectedParams, stdErrorValues, model.Statistics.WeightIndices);
            model.Statistics.SetCoeffStdError(stdErrors);
        }
    }
}
