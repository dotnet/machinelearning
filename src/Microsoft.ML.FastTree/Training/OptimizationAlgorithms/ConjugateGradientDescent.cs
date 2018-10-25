// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    // Conjugate gradient descent
    public class ConjugateGradientDescent : GradientDescent
    {
        private double[] _previousGradient;
        private double[] _currentGradient;
        private double[] _currentDk;

        public ConjugateGradientDescent(Ensemble ensemble, Dataset trainData, double[] initTrainScores, IGradientAdjuster gradientWrapper)
            : base(ensemble, trainData, initTrainScores, gradientWrapper)
        {
            _currentDk = new double[trainData.NumDocs];
        }

        protected override double[] GetGradient(IChannel ch)
        {
            Contracts.AssertValue(ch);
            _previousGradient = _currentGradient;

            _currentGradient = ObjectiveFunction.GetGradient(ch, TrainingScores.Scores);
            // We need to make a copy of gradient coz the reference returned is private structare of ObejctiveFunctionBase is valid only till next GetGradient call
            _currentGradient = (double[])_currentGradient.Clone();

            double[] previousDk = _currentDk;

            //First iteration
            if (_previousGradient == null)
                _previousGradient = _currentGradient;
#if !POLAK_RIBIERE_STEP
            // Compute Beta[k] = curG[k] * (curG[k] - prevG[k])
            // TODO: this can be optimized for speed. Keeping it slow but simple for now
            double beta = VectorUtils.GetDotProduct(_currentGradient, VectorUtils.Subtract(_currentGradient, _previousGradient)) / VectorUtils.GetDotProduct(_previousGradient, _previousGradient);
#else //Fletcher Reeves step
            // Compute Beta[k] = (curG[k]*cutG[k]) / (prevG[k] * prevG[k])
            double beta = VectorUtils.GetDotProduct(currentGradient, currentGradient) / VectorUtils.GetDotProduct(previousGradient, previousGradient);
#endif
            if (beta < 0)
                beta = 0;

            ch.Info("beta: {0}", beta);
            VectorUtils.MutiplyInPlace(previousDk, beta);
            VectorUtils.AddInPlace(previousDk, _currentGradient);
            _currentDk = previousDk; // Reallay no-op opration

            // We know that LeastSquaresRegressionTreeLearner does not destroy gradients so we can return our reference that we will need in next iter.
            if (TreeLearner is LeastSquaresRegressionTreeLearner)
                return _currentDk;
            // Assume that other treLearners destroy the gradient array so return a copy.
            else
                return (double[])_currentDk.Clone();
        }
    }
}

