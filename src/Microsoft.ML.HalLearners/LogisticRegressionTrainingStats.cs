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

    public static class LogisticRegressionTrainingStats
    {
        /// <summary>
        /// Computes the standart deviation matrix of each of the non-zero training weights, needed to calculate further the standart deviation,
        /// p-value and z-Score.
        /// This function performs the same calculations as <see cref="ComputeStd(LinearBinaryPredictor, IChannel, float)"/> but it is faster than it, because it makes use of Intel's MKL.
        /// </summary>
        /// <param name="model">A <see cref="LinearBinaryPredictor"/> obtained as a result of training with <see cref="LogisticRegression"/>.</param>
        /// <param name="ch">The <see cref="IChannel"/> used for messaging.</param>
        /// <param name="l2Weight">The L2Weight used for training. (Supply the same one that got used during training.)</param>
        public static void ComputeStd(LinearBinaryPredictor model, IChannel ch, float l2Weight = LogisticRegression.Arguments.Defaults.L2Weight)
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
                // http://www.aloki.hu/pdf/0402_171179.pdf (Equations 11 and 25)
                // http://www.inf.unibz.it/dis/teaching/DWDM/project2010/LogisticRegression.pdf (Section "Significance testing in ridge logistic regression")
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

            // currentWeights vector size is Weights2 + the bias
            var currentWeightsCount = model.Weights2.Count + 1;
            VBuffer<float> stdErrors = new VBuffer<float>(currentWeightsCount, numSelectedParams, stdErrorValues, model.Statistics.WeightIndices);
            model.Statistics.SetCoeffStdError(stdErrors);
        }
    }
}
