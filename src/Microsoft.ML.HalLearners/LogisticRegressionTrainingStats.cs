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

        public static void ComputeExtendedTrainingStatistics(this LogisticRegression trainer, IChannel ch)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(trainer.Stats, $"Training Statistics can get generated after training finishes. Train with setting: ShowTrainigStats set to true.");
            Contracts.Assert(trainer.GetL2Weight > 0);
            Contracts.Assert(trainer.GetNumGoodRows > 0);

            ch.Info("Model trained with {0} training examples.", trainer.GetNumGoodRows);

            // Apply Cholesky Decomposition to find the inverse of the Hessian.
            Double[] invHessian = null;
            try
            {
                // First, find the Cholesky decomposition LL' of the Hessian.
                Mkl.Pptrf(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, trainer.GetNumSelectedParams, trainer.Stats.Hessian);
                // Note that hessian is already modified at this point. It is no longer the original Hessian,
                // but instead represents the Cholesky decomposition L.
                // Also note that the following routine is supposed to consume the Cholesky decomposition L instead
                // of the original information matrix.
                Mkl.Pptri(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, trainer.GetNumSelectedParams, trainer.Stats.Hessian);
                // At this point, hessian should contain the inverse of the original Hessian matrix.
                // Swap hessian with invHessian to avoid confusion in the following context.
                Utils.Swap(ref trainer.Stats.Hessian, ref invHessian);
                Contracts.Assert(trainer.Stats.Hessian == null);
            }
            catch (DllNotFoundException)
            {
                throw ch.ExceptNotSupp("The MKL library (MklImports.dll) or one of its dependencies is missing.");
            }

            float[] stdErrorValues = new float[trainer.GetNumSelectedParams];
            stdErrorValues[0] = (float)Math.Sqrt(invHessian[0]);

            for (int i = 1; i < trainer.GetNumSelectedParams; i++)
            {
                // Initialize with inverse Hessian.
                stdErrorValues[i] = (Single)invHessian[i * (i + 1) / 2 + i];
            }

            if (trainer.GetL2Weight > 0)
            {
                // Iterate through all entries of inverse Hessian to make adjustment to variance.
                // A discussion on ridge regularized LR coefficient covariance matrix can be found here:
                // http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
                // http://www.inf.unibz.it/dis/teaching/DWDM/project2010/LogisticRegression.pdf
                int ioffset = 1;
                for (int iRow = 1; iRow < trainer.GetNumSelectedParams; iRow++)
                {
                    for (int iCol = 0; iCol <= iRow; iCol++)
                    {
                        var entry = (Single)invHessian[ioffset];
                        var adjustment = -trainer.GetL2Weight * entry * entry;
                        stdErrorValues[iRow] -= adjustment;
                        if (0 < iCol && iCol < iRow)
                            stdErrorValues[iCol] -= adjustment;
                        ioffset++;
                    }
                }

                Contracts.Assert(ioffset == invHessian.Length);
            }

            for (int i = 1; i < trainer.GetNumSelectedParams; i++)
                stdErrorValues[i] = (float)Math.Sqrt(stdErrorValues[i]);

           VBuffer<float> stdErrors = new VBuffer<float>(trainer.GetWeights.Length, trainer.GetNumSelectedParams, stdErrorValues, trainer.Stats.WeightIndices);
           trainer.Stats.SetCoeffStdError(stdErrors);
        }
    }
}
