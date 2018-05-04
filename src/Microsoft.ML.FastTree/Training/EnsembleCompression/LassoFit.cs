// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public sealed class LassoFit
    {
        // Number of lambda values
        public int NumberOfLambdas;

        // Intercepts
        public double[] Intercepts;

        // Compressed weights for each solution
        public double[][] CompressedWeights;

        // Pointers to compressed weights
        public int[] Indices;

        // Number of weights for each solution
        public int[] NumberOfWeights;

        // Number of non-zero weights for each solution
        public int[] NonZeroWeights;

        // The value of lambdas for each solution
        public double[] Lambdas;

        // R^2 value for each solution
        public double[] Rsquared;

        // Total number of passes over data
        public int NumberOfPasses;

        private int _numFeatures;

        public LassoFit(int numberOfLambdas, int maxAllowedFeaturesAlongPath, int numFeatures)
        {
            Intercepts = new double[numberOfLambdas];
            CompressedWeights = VectorUtils.AllocateDoubleMatrix(numberOfLambdas, maxAllowedFeaturesAlongPath);
            Indices = new int[maxAllowedFeaturesAlongPath];
            NumberOfWeights = new int[numberOfLambdas];
            Lambdas = new double[numberOfLambdas];
            Rsquared = new double[numberOfLambdas];
            NonZeroWeights = new int[numberOfLambdas];
            _numFeatures = numFeatures;
        }

        public double[] GetWeights(int lambdaIdx)
        {
            double[] weights = new double[_numFeatures];
            for (int i = 0; i < NumberOfWeights[lambdaIdx]; i++)
            {
                weights[Indices[i]] = CompressedWeights[lambdaIdx][i];
            }
            return weights;
        }
    }
}
