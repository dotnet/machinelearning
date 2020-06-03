// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// this class is used to detect the periodicity automatically
    /// </summary>
    public class PeriodDetectUtils
    {
        /// <summary>
        /// the minimum value that is considered as a valid period.
        /// </summary>
        private const int MinPeriod = 4;

        /// <summary>
        /// in practice, the max lag very rarely exceed 365, which lacks of strong interpretation, and which also brings performance overhead.
        /// </summary>
        private const int MaxLag = 400;

        /// <summary>
        /// suppose the length of time series is 651, now we found an period is 128, then 651/128 = 5, which means there are at most 5 recurrent period. this is too small, the significance build upon this is not trustable.
        /// </summary>
        private const int MinRecurrentCount = 8;

        /// <summary>
        /// when input time series is with very close values (i.e., different is smaller than E-20), the accuracy of double could distort the
        /// final trend signal. any seasonal signal under such circumstance becomes unreliable.
        /// so use this threshold to eliminate such kind of time series. here set to 1e-10 is for conservative consideration.
        /// </summary>
        private const double MinEnergyThreshold = 1e-10;

        public static int DetectSeasonality(double[] y)
        {
            int length = y.Length;
            int newLength = Get2Power(y.Length);
            double[] fftRe = new double[newLength];
            double[] fftIm = new double[newLength];
            double[] inputRe = new double[newLength];

            double mean = 0;
            double std = 0;

            foreach (double value in y)
                mean += value;
            mean /= length;

            for (int i = 0; i < length; ++i)
            {
                inputRe[i] = y[i] - mean;
                std += inputRe[i] * inputRe[i];
            }
            if (std / length < 1e-8)
            {
                return -1;
            }

            for (int i = length; i < newLength; ++i)
            {
                inputRe[i] = 0;
            }

            FftUtils.ComputeForwardFft(inputRe, Enumerable.Repeat(0.0, newLength).ToArray(), fftRe, fftIm, newLength);

            var z = fftRe.Select((m, i) => new Complex(m, fftIm[i])).ToArray();
            var w = z.Select((t, i) => t * Complex.Conjugate(t)).ToArray();
            FindBestTwoFrequencies(w, length, out var bestFreq, out var secondFreq);

            double[] ifftRe = new double[newLength];
            double[] ifftIm = new double[newLength];
            FftUtils.ComputeBackwardFft(
                w.Select(t => (double)t.Real).ToArray(),
                w.Select(t => (double)t.Imaginary).ToArray(), ifftRe, ifftIm, newLength);
            var r = ifftRe.Select((t, i) => new Complex(t, ifftIm[i])).ToArray();
            int period = FindTruePeriod(r, bestFreq, secondFreq, newLength);

            if (period < MinPeriod)
            {
                period = -1;
            }

            return period;
        }

        private static int FindTruePeriod(Complex[] r, int bestFreq, int secondFreq, int timeSeriesLength)
        {
            int firstPeriod = -1;
            int secondPeriod = -1;
            double firstTimeDomainEnergy = -1;
            double secondTimeDomainEnergy = -1;
            firstPeriod = FindBestPeriod(r, bestFreq, timeSeriesLength, out firstTimeDomainEnergy);
            if (secondFreq != -1)
            {
                secondPeriod = FindBestPeriod(r, secondFreq, timeSeriesLength, out secondTimeDomainEnergy);
            }
            if (firstPeriod == -1 && secondPeriod == -1)
                return -1;
            int truePeriod;
            double trueTimeDomainEnergy;
            if (firstPeriod == -1)
            {
                truePeriod = secondPeriod;
                trueTimeDomainEnergy = secondTimeDomainEnergy;
            }
            else if (secondPeriod == -1)
            {
                truePeriod = firstPeriod;
                trueTimeDomainEnergy = firstTimeDomainEnergy;
            }
            else
            {
                if (firstPeriod == secondPeriod)
                {
                    truePeriod = firstPeriod;
                    trueTimeDomainEnergy = firstTimeDomainEnergy;
                }
                else
                {
                    // hueristic: if the second frequency is with somewhat higher energy in time domain, we think it is a better candidate
                    if (secondTimeDomainEnergy > firstTimeDomainEnergy * 1.05)
                    {
                        truePeriod = secondPeriod;
                        trueTimeDomainEnergy = secondTimeDomainEnergy;
                    }
                    else
                    {
                        truePeriod = firstPeriod;
                        trueTimeDomainEnergy = firstTimeDomainEnergy;
                    }
                }
            }
            trueTimeDomainEnergy /= r[0].Real;

            // this is a key equation, which is named the "testing for randomness with the correlogram". /ref: http://www.ltrr.arizona.edu/~dmeko/notes_3.pdf
            // actually, 1.96 is for the 2-sigma, which has 95% statistical confidence. 2.58 is for 99% confidence, 2.85 for 99.5% confidence
            /* increasing the threshold aims to mitigate the fake seasonal component caused by outliers. in practice, if there exist true seasonal component,
             * such as BirdStrike/Appdownloads, the energy is far larger than threshold, hence change threshold from 2.85 to 4.0 have no impact (tested);
             */

            double threshold = 4 / Math.Sqrt(timeSeriesLength);

            if (trueTimeDomainEnergy < threshold || r[truePeriod].Real < MinEnergyThreshold)
                return -1;

            return truePeriod;
        }

        /// <summary>
        /// in order to pick up a proper frequency robustly (this is useful especially for large frequency, or small period, e.g., period = 2),
        /// this method aims to pick up the top two frequencies for further evaluation.
        /// of course, the energy of the second frequency (in frequency domain) must be at similar magnitude compared with the energy of the first
        /// frequency.
        /// </summary>
        /// <param name="w">the energy list in the frequency domain, the index is the frequency.</param>
        /// <param name="timeSeriesLength">the original time series length</param>
        /// <param name="bestFreq">the frequency with highest energy</param>
        /// <param name="secondFreq">the frequency with second highest energy</param>
        private static void FindBestTwoFrequencies(Complex[] w, int timeSeriesLength, out int bestFreq, out int secondFreq)
        {
            bestFreq = -1;
            double bestEnergy = -1.0;
            secondFreq = -1;
            double secondEnergy = -1.0;

            if (w.Length < 2)
                return;

            List<double> energies = new List<double>();

            /* length of time series divided by frequency is period. it is obvious that the period should be larger than 1 and smaller than the total length, and is an integer */
            for (int i = w.Length / timeSeriesLength; i < w.Length / 2 + 1; i++)
            {
                double nextWeight = w[i].Magnitude;
                energies.Add(nextWeight);

                if (nextWeight > bestEnergy)
                {
                    bestEnergy = nextWeight;
                    bestFreq = i;
                }
            }

            // once we found a best frequency, the region formed by lower bound to upper bound corresponding to this frequency will not be inspected anymore. because they all share the same period.
            int period = w.Length / bestFreq;
            double lowerBound = w.Length * 1.0 / (period + 1);
            double upperBound = w.Length * 1.0 / (period - 1);

            for (int i = w.Length / timeSeriesLength; i < w.Length / 2 + 1; i++)
            {
                if (i > lowerBound && i < upperBound)
                    continue;
                double weight = w[i].Magnitude;
                if (weight > secondEnergy)
                {
                    double prevWeight = 0;
                    if (i > 0)
                        prevWeight = w[i - 1].Magnitude;
                    double nextWeight = 0;
                    if (i < w.Length - 1)
                        nextWeight = w[i + 1].Magnitude;

                    // should be a local maximum
                    if (weight >= prevWeight && weight >= nextWeight)
                    {
                        secondEnergy = nextWeight;
                        secondFreq = i;
                    }
                }
            }
            double typycalEnergy = MathUtils.QuickMedian(energies);

            // the second energy must be at least significantly large enough than typical energies, and also similar to best energy at magnitude level.
            if (typycalEnergy * 6.0 < secondEnergy && secondEnergy * 10.0 > bestEnergy)
                return;

            // set the second frequency to -1, since it is obviously not strong enought to compete with the best energy.
            secondFreq = -1;
        }

        /// <summary>
        /// given a frequency F represented by an integer, we aim to find the best period by inspecting the auto-correlation function in time domain.
        /// since either frequency or the period is an integer, so the possible period located within
        /// [N/(F+1), N/(F-1)], we need to check this domain, and pick the best one. where N is the length of the augmented time series
        /// </summary>
        /// <param name="r">the auto-correlation function of the augmented time series</param>
        /// <param name="frequency">the input frequency candidate</param>
        /// <param name="timeSeriesLength">the length of the original time series, this is used for post processing to reduce false positive</param>
        /// <param name="energy">output the energy on the auto-correlation function</param>
        /// <returns>return the best period estimated</returns>
        private static int FindBestPeriod(Complex[] r, int frequency, int timeSeriesLength, out double energy)
        {
            energy = -1;

            // this will never make sense of a seasonal signal
            if (frequency <= 1)
                return -1;

            int lowerBound = r.Length / (frequency + 1);
            int upperBound = r.Length / (frequency - 1);
            int bestPeriod = -1;
            for (int i = lowerBound; i <= upperBound && i < r.Length; i++)
            {
                var currentEnergy = r[i].Real;
                if (currentEnergy > energy)
                {
                    energy = currentEnergy;
                    bestPeriod = i;
                }
            }

            /* condition1: period does not make sense, since the corresponding zone in the auto-correlation energy list are all negative.
             * condition2: for real dataset, we do not think there will exist such long period. this is used to reduce false-positive
             * condition3: the number of repeats under this period is too few. this is used to reduce false-positive
             */
            if (bestPeriod <= 1 || bestPeriod > MaxLag || timeSeriesLength < MinRecurrentCount * bestPeriod)
            {
                energy = -1;
                return -1;
            }
            return bestPeriod;
        }

        /// <summary>
        /// get the smallest 2^k which is equal or greater than n
        /// </summary>
        private static int Get2Power(int n)
        {
            int result = 1;
            bool meet1 = false; // check is n is just equals to 2^k for some k
            while (n > 1)
            {
                if ((n & 1) != 0)
                    meet1 = true;
                result <<= 1;
                n >>= 1;
            }
            if (meet1)
                result <<= 1;
            return result;
        }
    }
}
