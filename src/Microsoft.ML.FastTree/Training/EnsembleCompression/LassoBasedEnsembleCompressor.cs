// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// This implementation is based on:
    /// Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization
    /// Paths for Generalized Linear Models via Coordinate Descent.
    /// http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf
    /// </summary>
    /// <remarks>Author was Yasser Ganjisaffar during his internship.</remarks>
    public class LassoBasedEnsembleCompressor : IEnsembleCompressor<short>
    {
        // This module shouldn't consume more than 4GB of memory
        private const long MaxAvailableMemory = 4L * 1024 * 1024 * 1024;

        // In order to speed up the compression, we limit the number of observations,
        // but this limit is dependent on the number of features that we should learn
        // their weights. In other words, for learning weights of more features, we
        // need more observations.
        private const int MaxObservationsTOFeaturesRatio = 100;

        // Number of relevance levels: Perfect, Excellent, Good, Fair, Bad
        private const int NumRelevanceLevels = 5;

        private const double Epsilon = 1.0e-6;

        // The default number of lambda values to use
        private const int DefaultNumberOFLambdas = 100;

        // Convergence threshold for coordinate descent
        // Each inner coordination loop continues until the relative change
        // in any coefficient is less than this threshold
        private const double ConvergenceThreshold = 1.0e-4;

        private const double Small = 1.0e-5;
        private const int MinNumberOFLambdas = 5;
        private const double MaxRSquared = 0.999;

        private float[] _targets;
        private float[][] _observations;
        private int _numFeatures;
        private int _numObservations;

        private Dataset _trainSet;
        private short[] _labels;
        private Ensemble _compressedEnsemble;
        private int[] _sampleObservationIndices;
        private Random _rnd;

        public void Initialize(int numTrees, Dataset trainSet, short[] labels, int randomSeed)
        {
            _numFeatures = numTrees;
            _trainSet = trainSet;
            _labels = labels;
            _rnd = new Random(randomSeed);

            int maxObservations = (int)(MaxAvailableMemory / _numFeatures / sizeof(float));
            _numObservations = Math.Min(_trainSet.NumDocs, maxObservations);
            if (_numObservations > MaxObservationsTOFeaturesRatio * _numFeatures)
            {
                _numObservations = MaxObservationsTOFeaturesRatio * _numFeatures;
            }
            DoLabelBasedSampling();

            _observations = new float[_numFeatures][];
            for (int t = 0; t < _numFeatures; t++)
            {
                _observations[t] = new float[_numObservations];
            }
        }

        private void DoLabelBasedSampling()
        {
            if (_numObservations == _trainSet.NumDocs)
            {
                // No sampling
                _sampleObservationIndices = null;
            }
            else
            {
                _sampleObservationIndices = new int[_numObservations];
                int[] perLabelDocCount = new int[NumRelevanceLevels];
                for (int d = 0; d < _trainSet.NumDocs; d++)
                {
                    perLabelDocCount[_labels[d]]++;
                }
                List<KeyValuePair<short, int>> labelFreqList = new List<KeyValuePair<short, int>>();
                for (short i = 0; i < NumRelevanceLevels; i++)
                {
                    labelFreqList.Add(new KeyValuePair<short, int>(i, perLabelDocCount[i]));
                }
                labelFreqList.Sort(delegate (KeyValuePair<short, int> c1, KeyValuePair<short, int> c2)
                {
                    return Comparer<double>.Default.Compare(c1.Value, c2.Value);
                });
                int remainedDocs = _numObservations;
                double[] perLabelSampleRate = new double[NumRelevanceLevels];
                for (short i = 0; i < NumRelevanceLevels; i++)
                {
                    short curLabel = labelFreqList[i].Key;
                    int currentMax = remainedDocs / (NumRelevanceLevels - i);
                    int selectedDocs = Math.Min(perLabelDocCount[curLabel], currentMax);
                    perLabelSampleRate[curLabel] = (double)selectedDocs / perLabelDocCount[curLabel];
                    remainedDocs -= selectedDocs;
                }
                int obsCount = 0;
                for (int d = 0; d < _trainSet.NumDocs; d++)
                {
                    if (_rnd.NextDouble() <= perLabelSampleRate[_labels[d]])
                    {
                        _sampleObservationIndices[obsCount] = d;
                        obsCount++;
                        if (obsCount == _numObservations)
                        {
                            break;
                        }
                    }
                }
                // Since it's a random process, the generated number of observations might be
                // slightly different. So, we make them the same.
                _numObservations = obsCount;
            }
        }

        public unsafe void SetTreeScores(int idx, double[] scores)
        {
            if (_sampleObservationIndices == null)
            {
                int length = scores.Length;
                unsafe
                {
                    fixed (double* pScores = scores)
                    fixed (float* pCurObservations = _observations[idx])
                    {
                        for (int i = 0; i < length; i++)
                        {
                            pCurObservations[i] = (float)pScores[i];
                        }
                    }
                }
            }
            else
            {
                unsafe
                {
                    fixed (double* pScores = scores)
                    fixed (float* pCurObservations = _observations[idx])
                    fixed (int* pSampleObservationIndices = _sampleObservationIndices)
                    {
                        for (int i = 0; i < _numObservations; i++)
                        {
                            pCurObservations[i] = (float)pScores[pSampleObservationIndices[i]];
                        }
                    }
                }
            }
        }

        private LassoFit GetLassoFit(IChannel ch, int maxAllowedFeaturesPerModel)
        {
            DateTime startTime = DateTimeOffset.Now.UtcDateTime;

            if (maxAllowedFeaturesPerModel < 0)
            {
                maxAllowedFeaturesPerModel = _numFeatures;
            }
            int numberOfLambdas = DefaultNumberOFLambdas;
            int maxAllowedFeaturesAlongPath = (int)Math.Min(maxAllowedFeaturesPerModel * 1.2, _numFeatures);

            ch.Info("Lasso Compression uses {0} observations.", _numObservations);

            // lambdaMin = flmin * lambdaMax
            double flmin = (_numObservations < _numFeatures ? 5e-2 : 1e-4);

            /********************************
            * Standardize predictors and target:
            * Center the target and features (mean 0) and normalize their vectors to have the same
            * standard deviation
            */
            double[] featureMeans = new double[_numFeatures];
            double[] featureStds = new double[_numFeatures];
            double[] feature2residualCorrelations = new double[_numFeatures];

            float factor = (float)(1.0 / Math.Sqrt(_numObservations));
            for (int j = 0; j < _numFeatures; j++)
            {
                double mean = VectorUtils.GetMean(_observations[j]);
                featureMeans[j] = mean;
                unsafe
                {
                    fixed (float* pVector = _observations[j])
                    {
                        for (int i = 0; i < _numObservations; i++)
                        {
                            pVector[i] = (float)(factor * (pVector[i] - mean));
                        }
                    }
                }
                featureStds[j] = Math.Sqrt(VectorUtils.GetDotProduct(_observations[j], _observations[j]));

                VectorUtils.DivideInPlace(_observations[j], (float)featureStds[j]);
            }

            float targetMean = (float)VectorUtils.GetMean(_targets);
            unsafe
            {
                fixed (float* pVector = _targets)
                {
                    for (int i = 0; i < _numObservations; i++)
                    {
                        pVector[i] = factor * (pVector[i] - targetMean);
                    }
                }
            }
            float targetStd = (float)Math.Sqrt(VectorUtils.GetDotProduct(_targets, _targets));
            VectorUtils.DivideInPlace(_targets, targetStd);

            for (int j = 0; j < _numFeatures; j++)
            {
                feature2residualCorrelations[j] = VectorUtils.GetDotProduct(_targets, _observations[j]);
            }

            double[][] feature2featureCorrelations = VectorUtils.AllocateDoubleMatrix(_numFeatures, maxAllowedFeaturesAlongPath);
            double[] activeWeights = new double[_numFeatures];
            int[] correlationCacheIndices = new int[_numFeatures];
            double[] denseActiveSet = new double[_numFeatures];

            LassoFit fit = new LassoFit(numberOfLambdas, maxAllowedFeaturesAlongPath, _numFeatures);
            fit.NumberOfLambdas = 0;

            double alf = Math.Pow(Math.Max(Epsilon, flmin), 1.0 / (numberOfLambdas - 1));
            double rsquared = 0.0;
            fit.NumberOfPasses = 0;
            int numberOfInputs = 0;
            int minimumNumberOfLambdas = Math.Min(MinNumberOFLambdas, numberOfLambdas);

            double curLambda = 0;
            double maxDelta;
            for (int iteration = 1; iteration <= numberOfLambdas; iteration++)
            {
                ch.Info("Starting iteration {0}: R2={1}", iteration, rsquared);

                /**********
                * Compute lambda for this round
                */
                if (iteration == 1)
                {
                    curLambda = Double.MaxValue; // first lambda is infinity
                }
                else if (iteration == 2)
                {
                    curLambda = 0.0;
                    for (int j = 0; j < _numFeatures; j++)
                    {
                        curLambda = Math.Max(curLambda, Math.Abs(feature2residualCorrelations[j]));
                    }
                    curLambda = alf * curLambda;
                }
                else
                {
                    curLambda = curLambda * alf;
                }

                double prevRsq = rsquared;
                double v;
                unsafe
                {
                    fixed (double* pActiveWeights = activeWeights)
                    fixed (double* pFeature2residualCorrelations = feature2residualCorrelations)
                    fixed (int* pIndices = fit.Indices)
                    fixed (int* pCorrelationCacheIndices = correlationCacheIndices)
                    {
                        while (true)
                        {
                            fit.NumberOfPasses++;
                            maxDelta = 0.0;
                            for (int k = 0; k < _numFeatures; k++)
                            {
                                double prevWeight = pActiveWeights[k];
                                double u = pFeature2residualCorrelations[k] + prevWeight;
                                v = (u >= 0 ? u : -u) - curLambda;
                                // Computes sign(u)(|u| - curLambda)+
                                pActiveWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);

                                // Is the weight of this variable changed?
                                // If not, we go to the next one
                                if (pActiveWeights[k] == prevWeight)
                                {
                                    continue;
                                }

                                // If we have not computed the correlations of this
                                // variable with other variables, we do this now and
                                // cache the result
                                if (pCorrelationCacheIndices[k] == 0)
                                {
                                    numberOfInputs++;
                                    if (numberOfInputs > maxAllowedFeaturesAlongPath)
                                    {
                                        // we have reached the maximum
                                        break;
                                    }
                                    for (int j = 0; j < _numFeatures; j++)
                                    {
                                        // if we have already computed correlations for
                                        // the jth variable, we will reuse it here.
                                        if (pCorrelationCacheIndices[j] != 0)
                                        {
                                            feature2featureCorrelations[j][numberOfInputs - 1] = feature2featureCorrelations[k][pCorrelationCacheIndices[j] - 1];
                                        }
                                        else
                                        {
                                            // Correlation of variable with itself if one
                                            if (j == k)
                                            {
                                                feature2featureCorrelations[j][numberOfInputs - 1] = 1.0;
                                            }
                                            else
                                            {
                                                feature2featureCorrelations[j][numberOfInputs - 1] = VectorUtils.GetDotProduct(_observations[j], _observations[k]);
                                            }
                                        }
                                    }
                                    pCorrelationCacheIndices[k] = numberOfInputs;
                                    pIndices[numberOfInputs - 1] = k;
                                }

                                // How much is the weight changed?
                                double delta = pActiveWeights[k] - prevWeight;
                                rsquared += delta * (2.0 * pFeature2residualCorrelations[k] - delta);
                                maxDelta = Math.Max((delta >= 0 ? delta : -delta), maxDelta);

                                for (int j = 0; j < _numFeatures; j++)
                                {
                                    pFeature2residualCorrelations[j] -= feature2featureCorrelations[j][pCorrelationCacheIndices[k] - 1] * delta;
                                }
                            }

                            if (maxDelta < ConvergenceThreshold || numberOfInputs > maxAllowedFeaturesAlongPath)
                            {
                                break;
                            }

                            for (int ii = 0; ii < numberOfInputs; ii++)
                            {
                                denseActiveSet[ii] = activeWeights[pIndices[ii]];
                            }

                            do
                            {
                                fit.NumberOfPasses++;
                                maxDelta = 0.0;
                                for (int l = 0; l < numberOfInputs; l++)
                                {
                                    int k = pIndices[l];
                                    double prevWeight = pActiveWeights[k];
                                    double u = pFeature2residualCorrelations[k] + prevWeight;
                                    v = (u >= 0 ? u : -u) - curLambda;
                                    pActiveWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);
                                    if (activeWeights[k] == prevWeight)
                                    {
                                        continue;
                                    }
                                    double delta = pActiveWeights[k] - prevWeight;
                                    rsquared += delta * (2.0 * pFeature2residualCorrelations[k] - delta);
                                    maxDelta = Math.Max((delta >= 0 ? delta : -delta), maxDelta);
                                    for (int j = 0; j < numberOfInputs; j++)
                                    {
                                        pFeature2residualCorrelations[pIndices[j]] -= feature2featureCorrelations[pIndices[j]][pCorrelationCacheIndices[k] - 1] * delta;
                                    }
                                }
                            } while (maxDelta >= ConvergenceThreshold);

                            for (int ii = 0; ii < numberOfInputs; ii++)
                            {
                                denseActiveSet[ii] = pActiveWeights[pIndices[ii]] - denseActiveSet[ii];
                            }
                            for (int j = 0; j < _numFeatures; j++)
                            {
                                if (pCorrelationCacheIndices[j] == 0)
                                {
                                    pFeature2residualCorrelations[j] -= VectorUtils.GetDotProduct(denseActiveSet, feature2featureCorrelations[j], numberOfInputs);
                                }
                            }
                        }

                        if (numberOfInputs > maxAllowedFeaturesAlongPath)
                        {
                            break;
                        }
                        if (numberOfInputs > 0)
                        {
                            for (int ii = 0; ii < numberOfInputs; ii++)
                            {
                                fit.CompressedWeights[iteration - 1][ii] = pActiveWeights[pIndices[ii]];
                            }
                        }
                        fit.NumberOfWeights[iteration - 1] = numberOfInputs;
                        fit.Rsquared[iteration - 1] = rsquared;
                        fit.Lambdas[iteration - 1] = curLambda;
                        fit.NumberOfLambdas = iteration;

                        if (iteration < minimumNumberOfLambdas)
                        {
                            continue;
                        }

                        int me = 0;
                        for (int j = 0; j < numberOfInputs; j++)
                        {
                            if (fit.CompressedWeights[iteration - 1][j] != 0.0)
                            {
                                me++;
                            }
                        }
                        if (me > maxAllowedFeaturesPerModel || ((rsquared - prevRsq) < (Small * rsquared)) || rsquared > MaxRSquared)
                        {
                            break;
                        }
                    }
                }
            }

            for (int k = 0; k < fit.NumberOfLambdas; k++)
            {
                fit.Lambdas[k] = targetStd * fit.Lambdas[k];
                int nk = fit.NumberOfWeights[k];
                for (int l = 0; l < nk; l++)
                {
                    fit.CompressedWeights[k][l] = targetStd * fit.CompressedWeights[k][l] / featureStds[fit.Indices[l]];
                    if (fit.CompressedWeights[k][l] != 0)
                    {
                        fit.NonZeroWeights[k]++;
                    }
                }
                double product = 0;
                for (int i = 0; i < nk; i++)
                {
                    product += fit.CompressedWeights[k][i] * featureMeans[fit.Indices[i]];
                }
                fit.Intercepts[k] = targetMean - product;
            }

            // First lambda was infinity; fixing it
            fit.Lambdas[0] = Math.Exp(2 * Math.Log(fit.Lambdas[1]) - Math.Log(fit.Lambdas[2]));

            TimeSpan duration = DateTimeOffset.Now.UtcDateTime - startTime;
            ch.Info("Elapsed time for compression: {0}", duration);

            return fit;
        }

        private Ensemble GetEnsembleFromSolution(LassoFit fit, int solutionIdx, Ensemble originalEnsemble)
        {
            Ensemble ensemble = new Ensemble();

            int weightsCount = fit.NumberOfWeights[solutionIdx];
            for (int i = 0; i < weightsCount; i++)
            {
                double weight = fit.CompressedWeights[solutionIdx][i];
                if (weight != 0)
                {
                    RegressionTree tree = originalEnsemble.GetTreeAt(fit.Indices[i]);
                    tree.Weight = weight;
                    ensemble.AddTree(tree);
                }
            }

            ensemble.Bias = fit.Intercepts[solutionIdx];
            return ensemble;
        }

        private unsafe void LoadTargets(double[] trainScores, int bestIteration)
        {
            if (bestIteration == -1)
            {
                bestIteration = _numFeatures;
            }
            double[] targetScores;
            if (bestIteration == _numFeatures)
            {
                // If best iteration is the last one, train scores will be our targets
                targetScores = trainScores;
            }
            else
            {
                // We need to sum up scores of trees before best iteration to find targets
                targetScores = new double[_numObservations];
                for (int d = 0; d < _numObservations; d++)
                {
                    for (int t = 0; t < bestIteration; t++)
                    {
                        targetScores[d] += _observations[t][d];
                    }
                }
            }
            _targets = new float[_numObservations];
            if (_sampleObservationIndices == null || bestIteration != _numFeatures)
            {
                unsafe
                {
                    fixed (double* pScores = targetScores)
                    fixed (float* pTargets = _targets)
                    {
                        for (int i = 0; i < _numObservations; i++)
                        {
                            pTargets[i] = (float)pScores[i];
                        }
                    }
                }
            }
            else
            {
                unsafe
                {
                    fixed (double* pScores = targetScores)
                    fixed (float* pTargets = _targets)
                    fixed (int* pSampleObservationIndices = _sampleObservationIndices)
                    {
                        for (int i = 0; i < _numObservations; i++)
                        {
                            pTargets[i] = (float)pScores[pSampleObservationIndices[i]];
                        }
                    }
                }
            }
        }

        public bool Compress(IChannel ch, Ensemble ensemble, double[] trainScores, int bestIteration, int maxTreesAfterCompression)
        {
            LoadTargets(trainScores, bestIteration);

            LassoFit fit = GetLassoFit(ch, maxTreesAfterCompression);
            int numberOfSolutions = fit.NumberOfLambdas;
            int bestSolutionIdx = 0;

            ch.Info("Compression R2 values:");
            for (int i = 0; i < numberOfSolutions; i++)
            {
                ch.Info("Solution {0}:\t{1}\t{2}", i + 1, fit.NonZeroWeights[i], fit.Rsquared[i]);
            }
            bestSolutionIdx = numberOfSolutions - 1;
            _compressedEnsemble = GetEnsembleFromSolution(fit, bestSolutionIdx, ensemble);
            return true;
        }

        public Ensemble GetCompressedEnsemble()
        {
            return _compressedEnsemble;
        }
    }
}