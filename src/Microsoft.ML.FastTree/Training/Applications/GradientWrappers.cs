// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// Trivial weights wrapper. Creates proxy class holding the targets.
    /// </summary>
    public class TrivialGradientWrapper : IGradientAdjuster
    {
        public string TargetWeightsSetName => "";

        public TrivialGradientWrapper() { }

        public virtual double[] AdjustTargetAndSetWeights(double[] gradient, ObjectiveFunctionBase objFunction, out double[] targetWeights)
        {
            targetWeights = null;
            return gradient;
        }
    }

    /// <summary>
    /// Provides weights used when best regression step option is on.
    /// </summary>
    /// Second-derivatives used as weights in a leaf when one makes Newton-Raphson step (taken in account when regression tree is trained).
    public class BestStepRegressionGradientWrapper : IGradientAdjuster
    {
        public BestStepRegressionGradientWrapper() { }

        public virtual double[] AdjustTargetAndSetWeights(double[] gradient, ObjectiveFunctionBase objFunction, out double[] targetWeights)
        {
            targetWeights = objFunction.Weights;
            return gradient;
        }
    }

    /// <summary>
    /// Wraps targets with query weights. Regression tree is built for weighted data, and weights are used for mean
    /// calculation at Newton-Raphson step.
    /// </summary>
    public class QueryWeightsGradientWrapper : IGradientAdjuster
    {
        public QueryWeightsGradientWrapper()
        {
        }

        public virtual double[] AdjustTargetAndSetWeights(double[] gradient, ObjectiveFunctionBase objFunction, out double[] targetWeights)
        {
            double[] weightedTargets = new double[gradient.Length];
            double[] sampleWeights = objFunction.Dataset.SampleWeights;
            for (int i = 0; i < gradient.Length; ++i)
                weightedTargets[i] = gradient[i] * sampleWeights[i];
            targetWeights = sampleWeights;
            return weightedTargets;
        }
    }

    /// <summary>
    /// Wraps targets whan both query weights and best step regression options are active.
    /// </summary>
    public class QueryWeightsBestResressionStepGradientWrapper : IGradientAdjuster
    {
        public QueryWeightsBestResressionStepGradientWrapper()
        {
        }

        public virtual double[] AdjustTargetAndSetWeights(double[] gradient, ObjectiveFunctionBase objFunction, out double[] targetWeights)
        {
            double[] weightedTargets = new double[gradient.Length];
            double[] weights = new double[gradient.Length];
            double[] sampleWeights = objFunction.Dataset.SampleWeights;
            for (int i = 0; i < gradient.Length; ++i)
            {
                double queryWeight = sampleWeights[i];
                weightedTargets[i] = gradient[i] * queryWeight;
                weights[i] = objFunction.Weights[i] * queryWeight;
            }
            targetWeights = weights;
            return weightedTargets;
        }
    }
}
