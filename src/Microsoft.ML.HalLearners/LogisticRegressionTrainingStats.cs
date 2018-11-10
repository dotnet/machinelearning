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

    public sealed class ComputeLRTrainingStdThroughHal : IComputeLRTrainingStd
    {
        /// <summary>
        /// Computes the standart deviation matrix of each of the non-zero training weights, needed to calculate further the standart deviation,
        /// p-value and z-Score.
        /// If you need faster calculations, use the ComputeStd method from the Microsoft.ML.HALLearners package, which makes use of hardware acceleration.
        /// Due to the existence of regularization, an approximation is used to compute the variances of the trained linear coefficients.
        /// </summary>
        /// <param name="hessian"></param>
        /// <param name="weightIndices"></param>
        /// <param name="numSelectedParams"></param>
        /// <param name="currentWeightsCount"></param>
        /// <param name="ch">The <see cref="IChannel"/> used for messaging.</param>
        /// <param name="l2Weight">The L2Weight used for training. (Supply the same one that got used during training.)</param>
        public VBuffer<float> ComputeStd(double[] hessian, int[] weightIndices, int numSelectedParams, int currentWeightsCount, IChannel ch, float l2Weight)
        {
            Contracts.AssertValue(ch);
            Contracts.AssertValue(hessian, nameof(hessian));
            Contracts.Assert(numSelectedParams > 0);
            Contracts.Assert(currentWeightsCount > 0);
            Contracts.Assert(l2Weight > 0);

            // Apply Cholesky Decomposition to find the inverse of the Hessian.
            Double[] invHessian = null;
            try
            {
                // First, find the Cholesky decomposition LL' of the Hessian.
                Mkl.Pptrf(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numSelectedParams, hessian);
                // Note that hessian is already modified at this point. It is no longer the original Hessian,
                // but instead represents the Cholesky decomposition L.
                // Also note that the following routine is supposed to consume the Cholesky decomposition L instead
                // of the original information matrix.
                Mkl.Pptri(Mkl.Layout.RowMajor, Mkl.UpLo.Lo, numSelectedParams, hessian);
                // At this point, hessian should contain the inverse of the original Hessian matrix.
                // Swap hessian with invHessian to avoid confusion in the following context.
                Utils.Swap(ref hessian, ref invHessian);
                Contracts.Assert(hessian == null);
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
                stdErrorValues[i] = (float)invHessian[i * (i + 1) / 2 + i];
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
                        var entry = (float)invHessian[ioffset];
                        var adjustment = l2Weight * entry * entry;
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
            return new VBuffer<float>(currentWeightsCount, numSelectedParams, stdErrorValues, weightIndices);
        }
    }
}
