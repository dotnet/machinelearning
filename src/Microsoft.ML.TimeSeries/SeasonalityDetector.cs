// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// This class is used to detect the periodicity.
    /// </summary>
    internal class SeasonalityDetector
    {
        /// <summary>
        /// In practice, the max lag very rarely exceed 365, which lacks of strong interpretation, and which also brings performance overhead.
        /// </summary>
        private const int MaxLag = 400;

        /// <summary>
        /// Suppose the length of time series is 651, now we found an period is 128, then 651/128 = 5, which means there are at most 5 recurrent period. this is too small, the significance build upon this is not trustable.
        /// </summary>
        private const int MinRecurrentCount = 8;

        /// <summary>
        /// When input time series is with very close values (i.e., different is smaller than E-20), the accuracy of double could distort the
        /// final trend signal. Any seasonal signal under such circumstance becomes unreliable.
        /// So use this threshold to eliminate such kind of time series. Here set to 1e-10 is for conservative consideration.
        /// </summary>
        private const double MinEnergyThreshold = 1e-10;

        /// <summary>
        /// This method detects this predictable interval (or period) by adopting techniques of fourier analysis.
        /// Returns -1 if no such pattern is found, that is, the input values do not follow a seasonal fluctuation.
        /// </summary>
        /// <param name="host">The detect seasonality host environment.</param>
        /// <param name="input">Input DataView.The data is an instance of <see cref="Microsoft.ML.IDataView"/>.</param>
        /// <param name="inputColumnName">Name of column to process. The column data must be <see cref="System.Double"/>.</param>
        /// <param name="seasonalityWindowSize">An upper bound on the number of values to be considered in the input values.
        /// When set to -1, use the whole input to fit model; when set to a positive integer, use this number as batch size.
        /// Default value is -1.</param>
        /// <param name="randomessThreshold">Randomness threshold, ranging from [0, 1]. It specifies how confidence the input
        /// follows a predictable pattern recurring as seasonal data. By default, it is set as 0.95.
        /// </param>
        /// <returns>The detected period if seasonality period exists, otherwise return -1.</returns>
        public int DetectSeasonality(
            IHostEnvironment host,
            IDataView input,
            string inputColumnName,
            int seasonalityWindowSize,
            double randomessThreshold)
        {
            host.CheckValue(input, nameof(input));
            host.CheckValue(inputColumnName, nameof(inputColumnName));
            host.CheckUserArg(seasonalityWindowSize == -1 || seasonalityWindowSize >= 0, nameof(seasonalityWindowSize));

            var column = input.Schema.GetColumnOrNull(inputColumnName);
            host.CheckUserArg(column.HasValue, nameof(inputColumnName));

            var rowCursor = input.GetRowCursor(new List<DataViewSchema.Column>() { column.Value });
            var valueDelegate = rowCursor.GetGetter<double>(column.Value);

            int length = 0;
            double valueRef = 0;
            var valueCache = seasonalityWindowSize == -1 ? new List<double>() : new List<double>(seasonalityWindowSize);

            while (rowCursor.MoveNext())
            {
                valueDelegate.Invoke(ref valueRef);
                length++;
                valueCache.Add(valueRef);
                if (seasonalityWindowSize != -1 && length >= seasonalityWindowSize)
                    break;
            }

            double[] fftRe = new double[length];
            double[] fftIm = new double[length];
            double[] inputRe = valueCache.ToArray();

            FftUtils.ComputeForwardFft(inputRe, Enumerable.Repeat(0.0, length).ToArray(), fftRe, fftIm, length);

            var energies = fftRe.Select((m, i) => new Complex(m, fftIm[i])).ToArray();

            /* Periodogram indicates the square of "energy" on the  frequency domain.
             * Specifically, periodogram[j] = a[j]^2+b[j]^2, where a and b are Fourier Coefficients for cosine and sine,
             * x(t) = a0+sum(a[j]cos(2Pi * f[j]t)+b[j]sin(2Pi * f[j]t)
             */
            var periodogram = energies.Select((t, i) => t * Complex.Conjugate(t)).ToArray();
            FindBestTwoFrequencies(periodogram, length, out var bestFreq, out var secondFreq);

            double[] ifftRe = new double[length];
            double[] ifftIm = new double[length];
            FftUtils.ComputeBackwardFft(
                periodogram.Select(t => t.Real).ToArray(),
                periodogram.Select(t => t.Imaginary).ToArray(),
                ifftRe,
                ifftIm,
                length);
            var values = ifftRe.Select((t, i) => new Complex(t, ifftIm[i])).ToArray();

            int period = FindActualPeriod(values, bestFreq, secondFreq, length, randomessThreshold);

            return period < 0 ? -1 : period;
        }

        /// <summary>
        /// Find the actual period based on best frequency and second best frequency:
        /// Pick the best frequency by inspecting the auto-correlation energy (pick the highest) in time-domain.
        /// In the normal case, usually, when the time series is with period T, then the best frequency is N/T,
        /// while the second frequency would be N/2T, because period = T implies period = nT, where n is an integer.
        /// In such a case, smaller period will win out on the autu-correlation energy list, due to the property
        /// of auto-correlation.
        /// </summary>
        /// <param name="values">The auto-correlation function of the augmented time series</param>
        /// <param name="bestFrequency">The best frequency candidate</param>
        /// <param name="secondFrequency">The second best frequency candidate</param>
        /// <param name="timeSeriesLength">The length of the original time series, this is used for post
        /// processing to reduce false positive
        /// </param>
        /// <param name="randomnessThreshold">Randomness threshold that specifies how confidently the input
        /// values follow a predictable pattern recurring as seasonal data.
        /// </param>
        /// <returns>The period detected by check best frequency and second best frequency</returns>
        private static int FindActualPeriod(Complex[] values, int bestFrequency, int secondFrequency, int timeSeriesLength, double randomnessThreshold)
        {
            int firstPeriod = -1;
            int secondPeriod = -1;
            double firstTimeDomainEnergy = -1;
            double secondTimeDomainEnergy = -1;
            firstPeriod = FindBestPeriod(values, bestFrequency, timeSeriesLength, out firstTimeDomainEnergy);

            if (secondFrequency != -1)
            {
                secondPeriod = FindBestPeriod(values, secondFrequency, timeSeriesLength, out secondTimeDomainEnergy);
            }

            if (firstPeriod == -1 && secondPeriod == -1)
            {
                return -1;
            }

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
                    // hueristic: if the second frequency is with higher energy in time domain, we think it is a better candidate
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

            trueTimeDomainEnergy /= values[0].Real;

            /* This is a key equation, which is named the "testing for randomness with the correlogram". /ref: http://www.ltrr.arizona.edu/~dmeko/notes_3.pdf
             * 1.96 is for the 2-sigma, which has 95% statistical confidence. 2.58 is for 99% confidence, 2.85 for 99.5% confidence
             * increasing the threshold aims to mitigate the fake seasonal component caused by outliers. in practice, if there exist true seasonal component,
             * such as BirdStrike/Appdownloads, the energy is far larger than threshold, hence change threshold from 2.85 to 4.0 have no impact (tested);
             */
            double randomnessValue = ProbabilityFunctions.Probit(randomnessThreshold);
            double threshold = randomnessValue / Math.Sqrt(timeSeriesLength);

            if (trueTimeDomainEnergy < threshold || values[truePeriod].Real < MinEnergyThreshold)
            {
                return -1;
            }

            return truePeriod;
        }

        /// <summary>
        /// In order to pick up a proper frequency robustly (this is useful especially for large frequency,
        /// or small period, e.g., period = 2), this method aims to pick up the top two frequencies for
        /// further evaluation. The energy of the second frequency (in frequency domain) must be at similar
        /// magnitude compared with the energy of the first frequency.
        /// </summary>
        /// <param name="values">the energy list in the frequency domain, the index is the frequency.</param>
        /// <param name="timeSeriesLength">the original time series length</param>
        /// <param name="bestFrequency">the frequency with highest energy</param>
        /// <param name="secondFrequency">the frequency with second highest energy</param>
        private static void FindBestTwoFrequencies(Complex[] values, int timeSeriesLength, out int bestFrequency, out int secondFrequency)
        {
            bestFrequency = -1;
            double bestEnergy = -1.0;
            secondFrequency = -1;
            double secondEnergy = -1.0;

            if (values.Length < 2)
            {
                return;
            }

            var medianAggregator = new MedianDblAggregator(values.Length / 2 + 1 - values.Length / timeSeriesLength);

            /* Length of time series divided by frequency is period.
             * It is obvious that the period should be larger than 1 and smaller than the total length, and is an integer.
             */
            for (int i = values.Length / timeSeriesLength; i < values.Length / 2 + 1; i++)
            {
                double nextWeight = values[i].Magnitude;
                medianAggregator.ProcessValue(nextWeight);

                if (nextWeight > bestEnergy)
                {
                    bestEnergy = nextWeight;
                    bestFrequency = i;
                }
            }

            /* Once we found a best frequency, the region formed by lower bound to upper bound corresponding to this frequency
             * will not be inspected anymore. because they all share the same period.
            */
            int period = values.Length / bestFrequency;
            double lowerBound = values.Length * 1.0 / (period + 1);
            double upperBound = values.Length * 1.0 / (period - 1);

            for (int i = values.Length / timeSeriesLength; i < values.Length / 2 + 1; i++)
            {
                if (i > lowerBound && i < upperBound)
                {
                    continue;
                }

                double weight = values[i].Magnitude;
                if (weight > secondEnergy)
                {
                    double prevWeight = 0;
                    if (i > 0)
                    {
                        prevWeight = values[i - 1].Magnitude;
                    }

                    double nextWeight = 0;
                    if (i < values.Length - 1)
                    {
                        nextWeight = values[i + 1].Magnitude;
                    }

                    // should be a local maximum
                    if (weight >= prevWeight && weight >= nextWeight)
                    {
                        secondEnergy = nextWeight;
                        secondFrequency = i;
                    }
                }
            }

            var typicalEnergy = medianAggregator.Median;

            /* The second energy must be at least significantly large enough than typical energies,
             * and also similar to best energy at magnitude level.
            */
            if (typicalEnergy * 6.0 < secondEnergy && secondEnergy * 10.0 > bestEnergy)
            {
                return;
            }

            // set the second frequency to -1, since it is obviously not strong enought to compete with the best energy.
            secondFrequency = -1;
        }

        /// <summary>
        /// Given a frequency F represented by an integer, we aim to find the best period by inspecting the
        /// auto-correlation function in time domain. Since either frequency or the period is an integer,
        /// the possible period located within [N/(F+1), N/(F-1)], we need to check this domain, and pick
        /// the best one, where N is the length of the augmented time series.
        /// </summary>
        /// <param name="values">The auto-correlation function of the augmented time series</param>
        /// <param name="frequency">The input frequency candidate</param>
        /// <param name="timeSeriesLength">The length of the original time series, this is used for post processing to reduce false positive</param>
        /// <param name="energy">Output the energy on the auto-correlation function</param>
        /// <returns>The best period estimated</returns>
        private static int FindBestPeriod(Complex[] values, int frequency, int timeSeriesLength, out double energy)
        {
            energy = -1;

            // this will never make sense of a seasonal signal
            if (frequency <= 1)
            {
                return -1;
            }

            int lowerBound = values.Length / (frequency + 1);
            int upperBound = values.Length / (frequency - 1);
            int bestPeriod = -1;
            for (int i = lowerBound; i <= upperBound && i < values.Length; i++)
            {
                var currentEnergy = values[i].Real;
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
    }
}
