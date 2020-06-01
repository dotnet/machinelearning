using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class MathUtility
    {
        private const int PearsonCoeffMinLength = 5;

        private const double SquareRootOfTwo = 1.414213562373095;

        /// <summary>
        /// Returns the reciprocal of <paramref name="value"/>, guarding against division by zero.
        /// If the input value is less than the given <paramref name="precision"/>, this method does not perform
        /// the division and simply returns zero.
        /// </summary>
        public static double ReciprocalOrZero(double value, double precision)
        {
            if (Math.Abs(value) < precision)
                return 0.0;

            return 1.0 / value;
        }

        /// <summary>
        /// efficient method (O(m) complexity) for randomly sample m data points from total n data points. with complexity O(m)
        /// these n data points are indexed from 0 to n-1.
        /// </summary>
        /// <param name="n">total number of data points</param>
        /// <param name="m">number of points be sampled</param>
        /// <returns>return the indexes of the m randomly sampled points</returns>
        public static int[] RandomSampling(int n, int m)
        {
            if (n <= 0 || m > n || m <= 0)
                return null;
            int[] result = new int[m];
            for (int i = 0; i < m; i++)
                result[i] = i;
            if (m == n)
            {
                return result;
            }

            // we use fixed seed, to make the results stable.
            Random rd = new Random(0);
            for (int i = m + 1; i < n; i++)
            {
                var value = rd.Next(i);

                // probability m/i is hit, will use i to replace
                if (value < m)
                {
                    var chosenIndex = rd.Next(m);
                    result[chosenIndex] = i;
                }
            }
            return result;
        }

        /// <summary>
        /// calculate the cosine of two input 2-dimensional vectors. return false if either vector is a zero vector, where
        /// cosine is undefined there.
        /// </summary>
        public static bool Cosine(double vector1X, double vector1Y, double vector2X, double vector2Y, out double cosine)
        {
            cosine = double.NaN;
            double norm = Math.Sqrt((vector1X * vector1X + vector1Y * vector1Y) * (vector2X * vector2X + vector2Y * vector2Y));
            if (norm == 0.0)
                return false;
            cosine = (vector1X * vector2X + vector1Y * vector2Y) / norm;
            return true;
        }

        /// <summary>
        /// calculate the cosine of two input vectors. return false if either vector is a zero vector, where
        /// cosine is undefined there.
        /// </summary>
        public static bool Cosine(double[] vector1, double[] vector2, out double cosine)
        {
            cosine = double.NaN;
            if (vector1 == null || vector2 == null || vector1.Length == vector2.Length || vector1.Length == 0)
                return false;
            double norm1 = 0.0;
            double norm2 = 0.0;
            double innerProduct = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                norm1 += vector1[i] * vector1[i];
                norm2 += vector2[i] * vector2[i];
                innerProduct += vector1[i] * vector2[i];
            }
            if (norm1 == 0.0 || norm2 == 0.0)
                return false;
            cosine = innerProduct / Math.Sqrt(norm1 * norm2);
            return true;
        }

        /// <summary>
        /// error function.
        /// </summary>
        /// <param name="x">given the input x</param>
        public static double Erf(double x)
        {
            // handle either positive or negative x. because error function is negatively symmetric of x
            double a = 0.140012;
            double b = x * x;
            double item = -b * (4 / Math.PI + a * b) / (1 + a * b);
            double result = Math.Sqrt(1 - Math.Exp(item));
            if (x >= 0)
                return result;
            return -result;
        }

        /// <summary>
        /// calculate the standard cumulative distribution function F(x) = P(Z less or equal than x), where Z follows
        /// a standard normal distribution.
        /// </summary>
        public static double StandardCdf(double x)
        {
            return (1.0 + Erf(x / SquareRootOfTwo)) / 2;
        }

        /// <summary>
        /// given a confidence level <param name="alpha"/> as input, we calculate the Z such that P(Z greater than alpha) = alpha
        /// </summary>
        public static double ReverseAlpha(double alpha)
        {
            var p = 1.0 - alpha;

            // for a standard normal distribution, the probability that x is smaller than lower or x is larger than upper is almost zero.
            // we can set a larger value but already has no gain.
            double lower = -5.0;
            double upper = 5.0;
            double middle;
            while (true)
            {
                middle = (lower + upper) / 2;
                var estimate = StandardCdf(middle);
                if (Math.Abs(estimate - p) < 0.00000001)
                    break;

                // because standard CDF is monotonic, thus we can use binary search
                if (estimate > p)
                {
                    upper = middle;
                }
                else
                {
                    lower = middle;
                }
            }
            return middle;
        }

        /// <summary>
        /// calculate the statistical significance for a gaussian distribution.
        /// </summary>
        /// <param name="x">the observed x value</param>
        /// <param name="u">mean value</param>
        /// <param name="sigma">the standard deviation</param>
        public static double GaussianSignificance(double x, double u, double sigma)
        {
            double x1 = Math.Abs(x - u);

            // 1.414213562373095 is sqrt(2)
            double cdf = 0.5 + 0.5 * Erf(x1 / sigma / 1.414213562373095);
            return 2 * cdf - 1;
        }

        /// <summary>
        /// calculate the standard sigmoid function
        /// </summary>
        /// <param name="x">the input value</param>
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// use quick-sort like method to obtain the median value.
        /// the complexity in expectation is O(n), which is faster than using quickSort.
        /// </summary>
        /// <param name="values">the input list of values. note that this list will be modified after calling this method</param>
        /// <returns>returns the median value</returns>
        public static double QuickMedian(List<double> values)
        {
            if (values == null || values.Count == 0)
                return double.NaN;

            // here the third parameter is start from 1. so we need to plus 1 to compliant.
            return QuickSelect(values, 0, values.Count - 1, values.Count / 2 + 1);
        }

        private static double QuickSelect(List<double> values, int start, int end, int k)
        {
            if (start == end)
                return values[start];
            int q = Partition(values, start, end);
            int index = q - start + 1;
            if (index == k)
                return values[q];
            else if (k < index)
                return QuickSelect(values, start, q - 1, k);
            return QuickSelect(values, q + 1, end, k - index);
        }

        /// <summary>
        /// This version of QuickSelect follows a similar idea as the one above, except that this method
        /// does not use the Partition() method, and, therefore, does not modify the original values.
        /// On average, this version is faster (~40% faster) and has better memory allocation (~60% less).
        /// </summary>
        /// <param name="values">The list of values</param>
        /// <param name="k">The k smallest value in the list</param>
        public static double QuickSelect(IReadOnlyList<double> values, int k)
        {
            var nums = values;
            double[] left = new double[values.Count];
            double[] right = new double[values.Count];
            int numsCount = nums.Count;

            while (true)
            {
                if (numsCount == 1)
                    return nums[0];

                int idx = FindMedianIndex(nums, 0, numsCount - 1);
                double key = nums[idx];

                int leftIdx = 0;
                int rightIdx = 0;
                for (int i = 0; i < numsCount; i++)
                {
                    if (i == idx)
                        continue;

                    if (nums[i] < key)
                        left[leftIdx++] = nums[i];
                    else
                        right[rightIdx++] = nums[i];
                }

                if (leftIdx == k - 1)
                    return key;

                if (leftIdx >= k)
                {
                    nums = left;
                    numsCount = leftIdx;
                }
                else
                {
                    nums = right;
                    k = k - leftIdx - 1;
                    numsCount = rightIdx;
                }
            }
        }

        private static int Partition(List<double> values, int start, int end)
        {
            int medianIndex = FindMedianIndex(values, start, end);
            if (medianIndex != end)
            {
                double temp = values[end];
                values[end] = values[medianIndex];
                values[medianIndex] = temp;
            }
            double pivot = values[end];
            int startIndex = start;
            int endIndex = end - 1;
            while (startIndex < endIndex)
            {
                while (values[startIndex] < pivot && startIndex < endIndex)
                {
                    startIndex++;
                }
                while (values[endIndex] > pivot && endIndex > startIndex)
                {
                    endIndex--;
                }
                if (startIndex == endIndex)
                    break;
                double temp = values[startIndex];
                values[startIndex] = values[endIndex];
                values[endIndex] = temp;
                startIndex++;
                endIndex--;
            }
            if (values[startIndex] > pivot)
            {
                double temp = values[startIndex];
                values[startIndex] = pivot;
                values[end] = temp;
            }
            else
            {
                startIndex++;
                double temp = values[startIndex];
                values[startIndex] = pivot;
                values[end] = temp;
            }
            return startIndex;
        }

        public static int FindMedianIndex(IReadOnlyList<double> values, int start, int end)
        {
            // use the middle value among first/middle/end as the guard value, to make sure the average performance good.
            // according to unit test, this fix will improve the average performance 10%. and works normally when input list is ordered.
            double first = values[start];
            double last = values[end];
            int midIndex = (start + end) / 2;
            int medianIndex = -1;
            double middleValue = values[midIndex];
            if (first < last)
            {
                if (middleValue > last)
                {
                    // last is the middle value
                    medianIndex = end;
                }
                else if (middleValue > first)
                {
                    // middleValue is the middle value
                    medianIndex = midIndex;
                }
                else
                {
                    // first is the middle value
                    medianIndex = start;
                }
            }
            else
            {
                if (middleValue > first)
                {
                    // first is the middle value
                    medianIndex = start;
                }
                else if (middleValue < last)
                {
                    // last is the middle value
                    medianIndex = end;
                }
                else
                {
                    // middleValue is the middle value
                    medianIndex = midIndex;
                }
            }
            return medianIndex;
        }

        /// <summary>
        /// Calculate the Pearson correlation of two series of real numbers with filter conditions.
        /// The Pearson score is in [-1, 1].
        /// NOTE: We have 2 versions of this function in order to avoid allocating extra memory. This method should be
        /// removed once we change everything to k-dimension. In the mean time, any changes made here should also be made
        /// in the other version of this method.
        /// </summary>
        /// <param name="vector1">The first numerical vector</param>
        /// <param name="vector2">The second numerical vector</param>
        /// <param name="validIndexes">The indexes that are accountable for the pearson correlation calculation</param>
        public static double PearsonCoeff(
            IReadOnlyList<double> vector1,
            IReadOnlyList<double> vector2,
            IReadOnlyList<int> validIndexes = null)
        {
            // Conduct pearson correlation only when the two series are equal length, and also with at least 5 data points.
            // Otherwise, the correlation may not be statistically significant
            if (vector1.Count <= PearsonCoeffMinLength ||
                vector1.Count != vector2.Count ||
                (validIndexes != null && validIndexes.Count <= PearsonCoeffMinLength))
                return 0;

            bool hasIndexes = validIndexes != null && validIndexes.Count > 0;
            int length = hasIndexes ? validIndexes.Count : vector1.Count;

            double averageX = 0;
            double averageY = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    averageX += vector1[idx];
                    averageY += vector2[idx];
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    averageX += vector1[i];
                    averageY += vector2[i];
                }
            }
            averageX /= length;
            averageY /= length;

            double stdvX = 0;
            double stdvY = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    double deltaX = vector1[idx] - averageX;
                    double deltaY = vector2[idx] - averageY;
                    stdvX += deltaX * deltaX;
                    stdvY += deltaY * deltaY;
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    double deltaX = vector1[i] - averageX;
                    double deltaY = vector2[i] - averageY;
                    stdvX += deltaX * deltaX;
                    stdvY += deltaY * deltaY;
                }
            }

            // This happens when one vector with identical values.
            if (stdvX == 0.0 || stdvY == 0.0)
                return 0;

            stdvX = Math.Sqrt(stdvX / (length - 1));
            stdvY = Math.Sqrt(stdvY / (length - 1));

            double coeff = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    double item1 = (vector1[idx] - averageX) / stdvX;
                    double item2 = (vector2[idx] - averageY) / stdvY;
                    coeff += item1 * item2;
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    double item1 = (vector1[i] - averageX) / stdvX;
                    double item2 = (vector2[i] - averageY) / stdvY;
                    coeff += item1 * item2;
                }
            }

            return coeff / (length - 1);
        }

        public static double PearsonCoeff(
            IReadOnlyList<double[]> values,
            IReadOnlyList<int> validIndexes = null)
        {
            // Conduct pearson correlation only when the two series are equal length, and also with at least 5 data points.
            // Otherwise, the correlation may not be statistically significant
            if (values.Count <= PearsonCoeffMinLength ||
                (validIndexes != null && validIndexes.Count <= PearsonCoeffMinLength))
            {
                return 0;
            }

            bool hasIndexes = validIndexes != null && validIndexes.Count > 0;
            int length = hasIndexes ? validIndexes.Count : values.Count;

            double averageX = 0;
            double averageY = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    averageX += values[idx][0];
                    averageY += values[idx][1];
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    averageX += values[i][0];
                    averageY += values[i][1];
                }
            }
            averageX /= length;
            averageY /= length;

            double stdvX = 0;
            double stdvY = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    double deltaX = values[idx][0] - averageX;
                    double deltaY = values[idx][1] - averageY;
                    stdvX += deltaX * deltaX;
                    stdvY += deltaY * deltaY;
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    double deltaX = values[i][0] - averageX;
                    double deltaY = values[i][1] - averageY;
                    stdvX += deltaX * deltaX;
                    stdvY += deltaY * deltaY;
                }
            }

            // This happens when one vector with identical values.
            if (stdvX == 0.0 || stdvY == 0.0)
                return 0;

            stdvX = Math.Sqrt(stdvX / (length - 1));
            stdvY = Math.Sqrt(stdvY / (length - 1));

            double coeff = 0;
            if (hasIndexes)
            {
                for (int i = 0; i < length; i++)
                {
                    int idx = validIndexes[i];
                    double item1 = (values[idx][0] - averageX) / stdvX;
                    double item2 = (values[idx][1] - averageY) / stdvY;
                    coeff += item1 * item2;
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    double item1 = (values[i][0] - averageX) / stdvX;
                    double item2 = (values[i][1] - averageY) / stdvY;
                    coeff += item1 * item2;
                }
            }

            return coeff / (length - 1);
        }

        /// <summary>
        /// Compute the square of the euclidean distance between 2 k-dimensional points.
        /// </summary>
        /// <param name="point1">The first point</param>
        /// <param name="point2">The second point</param>
        public static double SquareDistance(double[] point1, double[] point2)
        {
            if (point1.Length == 0 || point1.Length != point2.Length)
                return double.NaN;

            double result = 0;
            for (int i = 0; i < point1.Length; i++)
            {
                var delta = point1[i] - point2[i];
                result += delta * delta;
            }

            return result;
        }
    }

    /// <summary>
    /// this class is used to calculate the common mathematical properties of time series.
    /// </summary>
    public class TimeSeriesProperty
    {
        /// <summary>
        /// 0.1 ~ 0.3 * 0.3, which means, when the fluctuation is more than 30% of the base signal, we think it is noisy.
        /// </summary>
        private const double NoisyLevelThreshold = 0.1;

        /// <summary>
        /// when the different between the typical distance (represented by median) and max/min distance is smaller than this threshold,
        /// we think the equal distance property is approximately hold.
        /// [6821022]: parameter is tuned to fix this bug.
        /// </summary>
        private const double AlmostEqualDistanceThreshold = 0.1;

        /// <summary>
        /// when LargestGap/Median exceeds this threshold, we think the non-equal distance becomes illness.
        /// this threshold is a magnitude-level threshold
        /// </summary>
        private const double MaxNonEqualDistanceThreshold = 6.0;

        /// <summary>
        /// one data point is defined as spatial outlier is its gap:
        /// gap/median exceeds this threshold.
        /// </summary>
        private const double SpatialOutlierThreshold = 1.5;

        /// <summary>
        /// when the population rate of spatial outliers exceeds this threshold, we think the non-equal distance becomes illness.
        /// </summary>
        private const double SpatialOutlierRateThreshold = 0.1;

        /// <summary>
        /// we are using Gaussian kernel smoothing function. here this value refers to the weight of the nearest neighbor of a given point.
        /// NOTE: this parameter effects the smoothing radius, which relates to the overall smoothing complexity. we should carefully tune this parameter.
        /// </summary>
        private const double KernelSmoothingParameter = 0.6;

        /// <summary>
        /// when the weight is smaller than this threshold, it will not be taken into account for weighted smoothing.
        /// </summary>
        private const double MinSmoothEffectedWeight = 0.01;

        /// <summary>
        /// this is the threshold of the angle of the corner point, to check if it is significant or not.
        /// this threshold is carefully tuned. since we are using Gaussian kernel for smoothing, the smoothing window size is smaller than previous method,
        /// which cause the corner point threshold should be changed as well.
        /// </summary>
        private const double CornerPointThreshold = 121;

        /// <summary>
        /// for performance optimization.
        /// the smoothing radius can be pre-calculated, so that we do not need to calculate it on-the-fly.
        /// NOTE: the overall complexity of smoothing a whole time series is O(n * smoothing radius)
        /// </summary>
        private static readonly int _smoothingRadius;

        /// <summary>
        /// the smoothing weights for the data points near the given data point (with the range of radius)
        /// </summary>
        private static readonly double[] _smoothingWeights;

        /// <summary>
        /// this is used to boost the normalization. this is a cumulative array of the weights within the range of [-radius, radius]. specifically, this is with
        /// length equals to 2*radius + 2, i.e., c0=0, c1, c2, ... c(r), c(r+1), c(r+2), ... c(2r+1), where c(r) is the center.
        /// </summary>
        private static readonly double[] _cumulativeDenominator;

        static TimeSeriesProperty()
        {
            _smoothingRadius = (int)Math.Sqrt(Math.Log(MinSmoothEffectedWeight) / Math.Log(KernelSmoothingParameter));
            if (_smoothingRadius >= 0)
            {
                _smoothingWeights = new double[_smoothingRadius + 1];
                _smoothingWeights[0] = 1.0;
                for (int i = 1; i < _smoothingWeights.Length; i++)
                {
                    _smoothingWeights[i] = Math.Pow(KernelSmoothingParameter, i * i);
                }

                _cumulativeDenominator = new double[2 * _smoothingRadius + 2];
                _cumulativeDenominator[0] = 0.0;
                for (int i = 1; i < _cumulativeDenominator.Length; i++)
                {
                    int dis = Math.Abs(_smoothingRadius + 1 - i);
                    _cumulativeDenominator[i] = _cumulativeDenominator[i - 1] + _smoothingWeights[dis];
                }
            }
        }

        /// <summary>
        /// given a pure seasonal series, calculate the average amplitude.
        /// </summary>
        /// <param name="seasonalSeries">the input time series, which should be pure seasonal</param>
        /// <param name="period">the period of the seasonal time series</param>
        public static double Amplitude(IReadOnlyList<double> seasonalSeries, int period)
        {
            double result = 0;
            int numberOfPeriods = 0;
            for (int i = 0; i + period <= seasonalSeries.Count; i += period)
            {
                numberOfPeriods++;
                double max = double.MinValue;
                double min = double.MaxValue;
                for (int j = i; j < i + period; j++)
                {
                    max = Math.Max(max, seasonalSeries[j]);
                    min = Math.Min(min, seasonalSeries[j]);
                }
                result += max - min;
            }

            // divde by 2, is by definition of amplitude.
            return result / numberOfPeriods / 2;
        }

        /// <summary>
        /// obtain the outliers by using 6-sigma methods.
        /// </summary>
        /// <param name="residual">the residual that the regression curve is eliminated</param>
        /// <param name="outlierSeverity">outlier Severity</param>
        /// <param name="mrs">the mean residual squares is an additional output</param>
        /// <param name="significance">significance</param>
        /// <param name="severityThreshold">measure the strictness of the outlier. default value is 6, which is very strict for selecting outliers.</param>
        public static int[] DetermineOutliers(IReadOnlyList<double> residual, out double[] outlierSeverity, out double mrs, out double significance, double severityThreshold = 6)
        {
            int length = residual.Count;

            // epsilon is used to represents a very small positive real value.
            const double epsilon = 0.0001;
            mrs = 0;
            significance = 0;
            outlierSeverity = new double[length];
            var outlierIndexes = new int[length];

            var absResiduals = new double[length];
            double nonzeroMin = double.MaxValue;
            for (int i = 0; i < length; i++)
            {
                absResiduals[i] = Math.Abs(residual[i]);
                if (absResiduals[i] > epsilon && nonzeroMin > absResiduals[i])
                    nonzeroMin = absResiduals[i];
            }

            double median = MathUtility.QuickSelect(absResiduals, length / 2);

            // When the median is 0, this is possible when there are more than half 0s.
            // In such case, it is not perfect to mark all the non-zero points as outliers.
            // Therefore, we need to find the first value that is not zero.
            if (median < epsilon)
            {
                median = nonzeroMin;
            }

            // this implies that all the values are smaller than epsilon, therefore, there's no outliers at all.
            if (median < epsilon)
            {
                for (int i = 0; i < length; i++)
                {
                    outlierIndexes[i] = 0;
                    outlierSeverity[i] = 0;
                }
            }
            else
            {
                // when median is very close to 0, which means the regularity of the serial is strong, so that no data points is outlier.
                for (int i = 0; i < length; i++)
                {
                    double severity = Math.Abs(residual[i]) / median;

                    // this is the key criteria
                    if (severity > severityThreshold)
                    {
                        double newSignificance = MathUtility.GaussianSignificance(residual[i], median, 3 * median);
                        if (newSignificance > significance)
                            significance = newSignificance;
                        outlierIndexes[i] = 1;
                        outlierSeverity[i] = severity;
                    }
                    else
                    {
                        outlierIndexes[i] = 0;
                        outlierSeverity[i] = 0;
                    }

                    // Here, we estimate the Mrs by adding all the residuals, rather than the residuals of non-outliers.
                    // Such compromise from robust statistic is for user friendly consideration.
                    mrs += residual[i] * residual[i];
                }
                mrs /= length;
            }
            return outlierIndexes;
        }

        /// <summary>
        /// this is a method to calculate the CCF lower bound of two time series, by given the first time series, and partial  information of the second
        /// time series, which is indicated by alpha.
        /// specifically, the second time series is the child of the first one from subspace perspective.
        /// both should be non-negative, and the first one is larger than the second one for each time epoch. let sum1 is the sum of all values from series1,
        /// and sum2 is the sum of all values from series2, then sum2=sum1*(1-alpha). when alpha is a small value, the two time series naturally be correlated, which is not interesting.
        /// </summary>
        /// <param name="series">represents the first time series</param>
        /// <param name="alpha">represents the overall different from the second time series. which should be between 0 and 1. this method works well only when alpha largely smaller than 1</param>
        /// <returns>returns the strict lower bound of the CCF of the two time series.</returns>
        public static double CcfLowerBound(List<double> series, double alpha)
        {
            int n = series.Count;
            double sum = 0;
            double max = 0;
            double min = double.MaxValue;
            foreach (double item in series)
            {
                sum += item;
                max = Math.Max(max, item);
                min = Math.Min(min, item);
            }
            double mean = sum / n;
            double var = 0;
            foreach (double item in series)
            {
                var += (item - mean) * (item - mean);
            }
            double numerator = var + n * alpha * mean * (mean - max);
            double denominator = Math.Sqrt(var * (var + alpha * (n * n * alpha * mean * mean - n * alpha * mean * mean + 2 * n * mean * mean - 2 * n * mean * min)));
            return numerator / denominator;
        }

        /// <summary>
        /// obtain the outliers by inspecting the residual distribution.
        /// this is a more sophisticated method rather than simply set the 6-median, or 3-sigma to determine outliers.
        /// </summary>
        /// <param name="residual">the input residuals after trending</param>
        /// <param name="cornerPointThreshold">in the problem of why-analysis, the suitable threshold of corner point is difference than default, thus need to make this as a parameter rather than constant</param>
        /// <returns>returns the index of the potential outlier candidates. the index is aligned with the input residual</returns>
        public static List<int> DetermineOutliersEx(IReadOnlyList<double> residual, double cornerPointThreshold = CornerPointThreshold)
        {
            List<int> result = new List<int>();
            int length = residual.Count;
            if (length < 3)
                return result;

            List<KeyValuePair<double, int>> absResiduals = new List<KeyValuePair<double, int>>();
            for (int i = 0; i < residual.Count; i++)
            {
                double absValue = Math.Abs(residual[i]);
                absResiduals.Add(new KeyValuePair<double, int>(absValue, i));
            }
            absResiduals.Sort(ReverseComparer);

            // smoothing. the reason is after we find the corner point, we still eliminate the extra points which are close to the corner point.
            var rawY = new double[absResiduals.Count];
            for (int i = 0; i < absResiduals.Count; i++)
            {
                rawY[i] = absResiduals[i].Key;
            }
            var smoothedY = KernelSmooth(rawY);

            // normalization. because we use geometric way to identify the corner point.
            double yMax = smoothedY[0];
            double yMin = smoothedY[length - 1];
            double xMax = length - 1;
            double xMin = 0.0;
            List<double> normX = new List<double>();
            List<double> normY = new List<double>();
            for (int i = 0; i < length; i++)
            {
                normX.Add((i - xMin) / (xMax - xMin));
                normY.Add((smoothedY[i] - yMin) / (yMax - yMin));
            }

            // inspecting the inner product and the angle of each check point
            int cornerPointIndex = -1;
            double bestAngle = double.MaxValue;
            for (int i = 1; i < length - 1; i++)
            {
                double angle;
                double x = normX[i];
                double y = normY[i];

                if (!EvaluateCornerPoint(x, y, out angle))
                    continue;

                if (angle < bestAngle)
                {
                    bestAngle = angle;
                    cornerPointIndex = i;
                }
            }
            if (cornerPointIndex == -1)
                return result;
            double normRawX = cornerPointIndex * 1.0 / (length - 1);
            double normRawY = (rawY[cornerPointIndex] - rawY[length - 1]) / (rawY[0] - rawY[length - 1]);
            double bestRawAngle;

            // we use smoothing curve to obtain the reasonable corner point, then we use the raw (also normalized) curve to obtain the true angle
            // of the corner point. this makes best sense: smoothing makes sure robust corner point identification, while angle on the raw curve
            // reflects how the original residuals behave.
            if (!EvaluateCornerPoint(normRawX, normRawY, out bestRawAngle))
                return result;

            if (bestRawAngle > cornerPointThreshold)
                return result;

            // further absort the data points close to the corner point, that doesnt look like outliers
            double slope = (1 - normY[cornerPointIndex]) / normX[cornerPointIndex];
            int absorbIndex = -1;
            for (absorbIndex = cornerPointIndex - 1; absorbIndex >= 0; absorbIndex--)
            {
                double currentSlope = (normY[absorbIndex] - normY[absorbIndex + 1]) / (normX[absorbIndex + 1] - normX[absorbIndex]);
                if (currentSlope > slope)
                    break;
            }

            if (absorbIndex != -1)
                cornerPointIndex = absorbIndex + 1;

            for (int i = 0; i < cornerPointIndex; i++)
            {
                result.Add(absResiduals[i].Value);
            }

            return result;
        }

        /// <summary>
        /// check if the (almost) equal distance property is hold
        /// </summary>
        /// <param name="xValues">the x-axis values</param>
        /// <param name="typicalGap">the typical gap/distance between two adjacent data points</param>
        /// <returns>return true if the x-axis values are equal or almost equal distance</returns>
        public static bool IsEqualDistance(IReadOnlyList<double> xValues, out double typicalGap)
        {
            typicalGap = double.NaN;
            int length = xValues.Count;

            // equal distance is undefined under such condition
            if (xValues == null || length < 2)
            {
                return false;
            }
            else if (length == 2)
            {
                typicalGap = xValues[1] - xValues[0];
                return true;
            }

            // if this is indeed a regular distance (i.e., equal-distance segments divided by regular gaps)
            List<double> gaps;
            double regularGap;
            double minGap;
            double maxGap;
            if (IsRegularEqualDistance(xValues, out typicalGap, out regularGap, out gaps, out minGap, out maxGap))
                return true;

            // will inspect the x values only when the gaps is null after checking regular equal-distance.
            if (gaps == null)
            {
                gaps = new List<double>(xValues.Count - 1);
                for (int i = 1; i < length; i++)
                {
                    double gap = xValues[i] - xValues[i - 1];
                    maxGap = Math.Max(maxGap, gap);
                    minGap = Math.Min(minGap, gap);
                    gaps.Add(gap);
                }
            }
            typicalGap = MathUtility.QuickMedian(gaps);

            // the min gap should not be too small
            if (typicalGap - minGap > AlmostEqualDistanceThreshold * typicalGap)
                return false;

            // the max gap should not be too large
            if (maxGap - typicalGap > AlmostEqualDistanceThreshold * typicalGap)
                return false;
            return true;
        }

        /// <summary>
        /// considering there would be the case that final visual effect is confusing, although underlying algorithm could deal with non-equal distance.
        /// Therefore, we need to identify the strong non-equal distance datasets, and avoid analyze or output it.
        /// Here I propose several rules to define what is a strong non-equal distance behavior:
        /// 1 - a point is an outlier (spatial outlier) if and only if its gap is SpatialOutlierThreshold larger than typical distance (represented by median gap)
        /// 2 - an outlier is illness if its gap is MaxNonEqualDistanceThreshold larger from typical distance. when there exist illness outlier, we bypass time series analysis
        /// 3 - the ratio of the outliers cannot exceed SpatialOutlierRateThreshold of total number of points
        /// 4 - any two outliers must NOT be adjacent (otherwise, the analysis results may differ from user intuition! think about it)
        /// </summary>
        /// <param name="xValues">the input x-axis values</param>
        /// <returns>returns true if the input series is an illness/strongly non-equal distance series.</returns>
        public static bool IsStrongNonEqualDistance(List<double> xValues)
        {
            if (xValues == null)
                return false;

            int length = xValues.Count;
            if (length <= 2)
                return false;

            // if it is a regular equal-distance, then strong non-equal distance is false.
            double typicalGap;
            double regularGap;
            List<double> gaps;
            double minGap;
            double maxGap;
            if (IsRegularEqualDistance(xValues, out typicalGap, out regularGap, out gaps, out minGap, out maxGap))
                return false;

            if (gaps == null)
            {
                gaps = new List<double>(xValues.Count - 1);
                maxGap = double.MinValue;
                for (int i = 1; i < length; i++)
                {
                    double gap = xValues[i] - xValues[i - 1];
                    gaps.Add(gap);
                    maxGap = Math.Max(maxGap, gap);
                }
            }

            // since this method will modify the input list. we need to copy one in order to preserve the original order
            double median = MathUtility.QuickMedian(new List<double>(gaps));
            double threshold = SpatialOutlierThreshold * median;

            // illness checking.
            if (maxGap > median * MaxNonEqualDistanceThreshold)
                return true;

            int outlierCount = 0;
            for (int i = 0; i < gaps.Count; i++)
            {
                double gap = gaps[i];
                if (gap > threshold)
                {
                    outlierCount++;

                    // if two spatial outliers are adjacent, then the left point will look like an isolated data point. isolated data points are not addressed in any type of insights so far.
                    if (i != 0 && gaps[i - 1] > threshold)
                    {
                        return true;
                    }

                    // if more than SpatialOutlierRateThreshold of total points are outliers, we think this is a very strong non-equal distance series
                    if (outlierCount > SpatialOutlierRateThreshold * gaps.Count)
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// check if the (almost) equal distance property is hold by considering an additional regular gap.
        /// the motivation is from the stock time series, which has stock records on weekdays, but no records at weekends, regularly.
        /// in order to support this, we should first identify the consistency gap (~2 days), then the left gaps must qualify the equal-distance gap.
        /// [remark]: in order to be regular equal-distance x-values, the segments (with typical gap) must be with equal number of data points (the
        /// first and the last segment can contain fewer data points), and strictly separated by the regular gap. this method does not handle the
        /// noise (a few violation). a suitable place to address the noise condition is a logic layer after data query, and before all time series
        /// analysis, where we can detect these spatial noise points, and eliminate them properly before feeding into time series analysis modules.
        /// </summary>
        /// <param name="xValues">the x-axis values</param>
        /// <param name="typicalGap">the typical gap-distance between two adjacent data points</param>
        /// <param name="regularGap">the regular gap between two segments (e.g., two days corresponds to the weekend in stock dataset)</param>
        /// <param name="gaps">performance: the list of gaps may be used further in other places.</param>
        /// <param name="minGap">performance: caller can directly use this value: the maximum gap along the x values</param>
        /// <param name="maxGap">performance: caller can directly use this value: the minimum gap along the x values</param>
        /// <returns>return true if the x-axis values are equal distance by considering the regular gaps</returns>
        public static bool IsRegularEqualDistance(
            IReadOnlyList<double> xValues,
            out double typicalGap,
            out double regularGap,
            out List<double> gaps,
            out double minGap,
            out double maxGap)
        {
            typicalGap = double.NaN;
            regularGap = double.NaN;
            minGap = double.MaxValue;
            maxGap = double.MinValue;
            gaps = null;

            int length = xValues.Count;

            /* when time series is with length smaller than 9, any regular gaps between equal-distance regions still
             * make the overall time series 'looks' messy; the seasonal component will be weak since the recurrence of
             * seasonal component is too few.
             */
            if (xValues == null || length < 9)
                return false;
            gaps = new List<double>(xValues.Count - 1);
            for (int i = 1; i < length; i++)
            {
                double gap = xValues[i] - xValues[i - 1];
                maxGap = Math.Max(gap, maxGap);
                minGap = Math.Min(gap, minGap);
                gaps.Add(gap);
            }

            // we assume the regular gap must be significantly larger than the typical gap.
            if (maxGap - minGap < minGap)
                return false;

            // records the typical/regular gaps, which is used to estimate a robust typical/regular gap to return.
            List<double> typicalGaps = new List<double>();
            List<double> regularGaps = new List<double>();

            /* if the regular equal-distance property is satisfied, then the middle inter-segment length must be equal, and the
             length of the first or last segment must be equal or smaller than the middle length*/
            int firstSegLength = -1;
            int firstMiddleSegLength = -1;
            int currMiddleSegLength = -1;
            for (int i = 0; i < gaps.Count; i++)
            {
                var currentGap = gaps[i];

                if (currentGap - minGap <= AlmostEqualDistanceThreshold * minGap)
                {
                    // this is a typical gap
                    typicalGaps.Add(currentGap);

                    if (firstSegLength == -1)
                    {
                        // the first segment length is not set yet
                        firstSegLength = 1;
                    }
                    else if (firstMiddleSegLength == -1)
                    {
                        // the first middle segment length is not set yet
                        firstSegLength++;
                    }
                    else if (currMiddleSegLength == -1)
                    {
                        // the first middle segment length is just started
                        firstMiddleSegLength++;
                    }
                    else
                    {
                        // the current middle segment length is just started
                        currMiddleSegLength++;

                        // current segment length is already longer than previous segment length, this is a violation.
                        if (currMiddleSegLength > firstMiddleSegLength)
                            return false;
                    }
                }
                else if (maxGap - currentGap <= AlmostEqualDistanceThreshold * maxGap)
                {
                    // this is a regular gap
                    regularGaps.Add(currentGap);

                    // the first middle segment length is not set yet
                    if (firstMiddleSegLength == -1)
                    {
                        // start counting the length of the first middle segment
                        firstMiddleSegLength = 0;
                    }
                    else if (currMiddleSegLength == -1)
                    {
                        // start counting the length of the next segment
                        currMiddleSegLength = 0;

                        // the first segment cannot be longer than middle ones
                        if (firstSegLength > firstMiddleSegLength)
                            return false;
                    }
                    else
                    {
                        // the length of the middle segments must be identical
                        if (currMiddleSegLength != firstMiddleSegLength)
                            return false;
                        currMiddleSegLength = 0;
                    }
                }
                else
                {
                    // the current gap is neither similar to typical gap nor similar to regular gap, this is a violation.
                    return false;
                }
            }

            // the check of the last segment
            if (currMiddleSegLength > firstMiddleSegLength)
                return false;

            // number of regular gaps indicate the segments, or number of repeating. this should not be too few to indicate reasonable repeating.
            if (regularGaps.Count < BasicParameters.MinRegularGap)
                return false;

            // use the median value to robustly represent the true typical/regular gaps
            typicalGap = MathUtility.QuickMedian(typicalGaps);
            regularGap = MathUtility.QuickMedian(regularGaps);
            return true;
        }

        /// <summary>
        /// determine if a given time series is too noisy or not by considering its normalized residual squares
        /// </summary>
        /// <param name="mrs">normalized residual squares</param>
        /// <returns>true if the normalized residual squares exceeds the given threshold</returns>
        public static bool IsTooNoiseSignal(double mrs)
        {
            return mrs > NoisyLevelThreshold;
        }

        /// <summary>
        /// calculate the angle of a given data point within a sorted residual curve. return true if the angle is valid.
        /// </summary>
        public static bool EvaluateCornerPoint(double x, double y, out double angle, bool inRadians = false)
        {
            angle = -1.0;

            // use outer product to check if the corner point is valid. if it is minus, it means a very rare thing, that this point is actually an anti-corner point
            double outerProduct = (1 - x) * (1 - y) - x * y;
            if (outerProduct <= 0)
                return false;

            double innerProduct = -(x * (1 - x) + y * (1 - y));

            // considering 3 points p1, p2 and p3: (0, 1), (x, y), (1, 0), they form up two vectors v1 = (-x, 1-y), v2 = (1-x, -y)
            // and here we calculate the angle between the two vectors. considering the normalized sorted residual curve, the left-most
            // point is p1, and the right most point is p3, here corner point is defined as p2 which has smallest angle.
            // intuitively, the data points on the left side of corner point are with significant higher residuals than the others,
            // which are most likely to be outliers.
            double cosine = innerProduct / Math.Sqrt((x * x + (1 - y) * (1 - y)) * ((1 - x) * (1 - x) + y * y));
            angle = Math.Acos(cosine);

            // convert to 360-degree unit, which is intuitive for tuning
            if (!inRadians)
                angle = angle * 180 / Math.PI;

            return true;
        }

        /// <summary>
        /// use Gaussian kernel smoothing function to smooth a time series with equal x-axis distance
        /// </summary>
        /// <param name="values">the specified time series</param>
        /// <returns>returns the smoothed time series</returns>
        internal static IReadOnlyList<double> KernelSmooth(IReadOnlyList<double> values)
        {
            if (_smoothingRadius <= 0)
                return values;

            var results = new double[values.Count];
            for (int i = 0; i < values.Count; i++)
                results[i] = KernelSmooth(values, i);
            return results;
        }

        /// <summary>
        /// Gaussian kernel based smoothing. i.e., K(x1, x2) = EXP[-(x-x0)^2/2b^2].
        /// this is a simplified version, which assumes the x-axis values are equal-distance. hence given a fixed point x0, the weight K(x, x0) will look like
        /// 1.0, EXP(-1/2b^2), EXP(-4/2b^2), EXP(-9/2b^2), ... etc. let EXP(-1/2b^2) = lambda, then the weights can be re-formulated as
        /// 1.0, lambda, lambda^4, lambda^9, lambda^16, ... etc.
        /// </summary>
        /// <param name="values">the original one-dimensional time series</param>
        /// <param name="index">the specified position that needs to be smoothed</param>
        /// <returns>return the smoothed value</returns>
        private static double KernelSmooth(IReadOnlyList<double> values, int index)
        {
            int lowIndex = index - _smoothingRadius;
            if (lowIndex < 0)
                lowIndex = 0;
            int upperIndex = index + _smoothingRadius;
            if (upperIndex >= values.Count)
                upperIndex = values.Count - 1;
            double weightedSum = 0;
            for (int i = lowIndex; i <= upperIndex; i++)
            {
                int distance = Math.Abs(i - index);
                double currentWeight = _smoothingWeights[distance];
                weightedSum += values[i] * currentWeight;
            }
            double totalWeight = _cumulativeDenominator[_smoothingRadius + 1 + upperIndex - index] - _cumulativeDenominator[_smoothingRadius - index + lowIndex];
            return weightedSum / totalWeight;
        }

        private static int ReverseComparer(KeyValuePair<double, int> left, KeyValuePair<double, int> right)
        {
            int compare1 = left.Key.CompareTo(right.Key);
            if (compare1 != 0)
                return -compare1;
            return -left.Value.CompareTo(right.Value);
        }
    }

    /// <summary>
    /// this class is used to calculate Combinatorics related mathematics.
    /// </summary>
    public class Combinatorics
    {
        /// <summary>
        /// calculate the logarithm of factorial. the reeason for using logarithm is to control the data value.
        /// or the value can easily exceed the int.MaxValue. see the ref on
        /// http://en.wikipedia.org/wiki/Stirling%27s_approximation
        /// </summary>
        /// <param name="n">the input integer.</param>
        public static double LogFactorial(int n)
        {
            double value = 0.5 * Math.Log(2 * Math.PI * n) + n * Math.Log(n) - n + Math.Log(1 + 1.0 / 12 / n);
            return value;
        }

        /// <summary>
        /// calculate the logarithm of combinations. n!/m!/(n-m)!
        /// </summary>
        /// <param name="n">the base number</param>
        /// <param name="m">the numbers should select out. m is smaller than n</param>
        public static double LogCombination(int n, int m)
        {
            if (n < 0 || m < 0 || m > n)
                throw new Exception("combination calculation input invalid");
            if (n == 0)
                return 0;
            if (m == 0 || m == n) // note that here is LOG!
                return 0;
            return LogFactorial(n) - LogFactorial(m) - LogFactorial(n - m);
        }
    }

    /// <summary>
    /// this class is used for density estimation for spatial data analysis.
    /// currently, it is used to accept/reject if a proper scatter plot can well represent the cross-measure correlation.
    /// if there exists a high density, small region, then the scatter plot will distort the true underlying correlation, we prefer rather not
    /// show it to end user.
    /// </summary>
    public class DensityIdentifier
    {
        /// <summary>
        /// when too few data points, the further density estimation algorithm will not be effective.
        /// </summary>
        private const int MinimumDataPointCount = 5;

        /// <summary>
        /// number of segments on x axis, and y axis.
        /// this value should be relatively large, must NOT be smaller than 4. actually, this constant is related with the high-density ratio threshold.
        /// take these two values (20, 0.3) as an example, suppose all the data points form a linear line, then these data points are occupied by 20
        /// diagonal grids, each one typically with 1/20~5% data points. then 30% points is certainly a much higher ratio.
        /// </summary>
        private const int SegmentCount = 20;

        /// <summary>
        /// when a grid with data points more than this ratio, it is viewed as a high density region.
        /// </summary>
        private const double HighDensityRatio = 0.3;

        /// <summary>
        /// when a gap is larger than typical gap times this value, we think it is a very large gap.
        /// this can be used to avoid some regular insight analysis such as cross measure correlation.
        /// </summary>
        private const double MaxGapThreshold = 8.0;

        /// <summary>
        /// grid-based density estimation to check whether there exists at least one grid with high ratio of data points.
        /// </summary>
        /// <param name="xValues">the x-axis values of the data points</param>
        /// <param name="yValues">the y-axis values of the data points</param>
        /// <returns>false if there does not exist any high density region</returns>
        public static bool HasHighDensityRegions(double[] xValues, double[] yValues)
        {
            if (xValues == null || yValues == null || xValues.Length != yValues.Length || xValues.Length < MinimumDataPointCount)
                return false;
            double xMin = double.MaxValue;
            double xMax = double.MinValue;
            double yMin = double.MaxValue;
            double yMax = double.MinValue;
            foreach (double value in xValues)
            {
                xMin = Math.Min(xMin, value);
                xMax = Math.Max(xMax, value);
            }
            foreach (double value in yValues)
            {
                yMin = Math.Min(yMin, value);
                yMax = Math.Max(yMax, value);
            }

            double xGridSize = (xMax - xMin) / SegmentCount;
            double yGridSize = (yMax - yMin) / SegmentCount;

            // if all the x or y values are identical, then trivial
            if (xGridSize <= double.Epsilon || yGridSize <= double.Epsilon)
                return false;

            // key indicates the location of the grid, value is number of data points within this grid
            Dictionary<long, int> densityPerGrid = new Dictionary<long, int>();

            int count = xValues.Length;
            for (int i = 0; i < count; i++)
            {
                int xGridIndex = (int)((xValues[i] - xMin) / xGridSize);
                int yGridIndex = (int)((yValues[i] - yMin) / yGridSize);
                long hashId = GetHashId(xGridIndex, yGridIndex);
                if (!densityPerGrid.ContainsKey(hashId))
                {
                    densityPerGrid.Add(hashId, 1);
                }
                else
                {
                    densityPerGrid[hashId]++;

                    // if more than 30% data points are within one small grid, which means a high portion of data points located in a small region.
                    if (densityPerGrid[hashId] > count * HighDensityRatio)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// check if there exists a large gap along a set of 2-dimensional data points.
        /// </summary>
        /// <param name="xValues">the x-axis values of the data points</param>
        /// <param name="yValues">the y-axis values of the data points</param>
        /// <returns>true indicates there indeed exists a large gap; false otherwise</returns>
        public static bool HasLargeGap(double[] xValues, double[] yValues)
        {
            if (HasLargeGap(xValues))
                return true;
            if (HasLargeGap(yValues))
                return true;
            return false;
        }

        private static long GetHashId(int xGridIndex, int yGridIndex)
        {
            long xLong = (long)xGridIndex;
            long yLong = (long)yGridIndex;
            return (xLong << 32) + yLong;
        }

        /// <summary>
        /// check if there exists a large gap along a set of one dimensional data points.
        /// </summary>
        /// <param name="values">a list of one dimensional data points</param>
        /// <returns>true indicates there indeed exists a large gap; false otherwise</returns>
        private static bool HasLargeGap(double[] values)
        {
            if (values == null || values.Length <= 3)
                return false;
            List<double> sortedValues = new List<double>(values);
            sortedValues.Sort();

            List<double> gaps = new List<double>(sortedValues.Count - 1);
            double maxGap = double.MinValue;
            for (int i = 1; i < sortedValues.Count; i++)
            {
                var gap = sortedValues[i] - sortedValues[i - 1];

                // when the gap is almost zero, which means they are overlapped, we ignore this case, since it has no effect on chart.
                if (gap < double.Epsilon)
                    continue;
                maxGap = Math.Max(maxGap, gap);
                gaps.Add(gap);
            }
            double typicalGap = MathUtility.QuickMedian(gaps);
            if (typicalGap * MaxGapThreshold < maxGap)
                return true;
            return false;
        }
    }

    /// <summary>
    /// move this from point insight evaluation to numerical library, because in small number analysis for why-analysis,
    /// we need to use these methods to check if extreme points exist
    /// </summary>
    public class CalculateRankSignificance
    {
        public static double OutstandingRankMinusOne(double waitValue, ICollection<double> otherValues, ref bool isSignificant)
        {
            if (waitValue >= 0 || otherValues.Count == 0)
            {
                // when less or equal to 2 points, not so meaningful
                isSignificant = false;
                return 0;
            }

            List<double> minusValues = new List<double>();
            foreach (double value in otherValues)
            {
                if (value < 0)
                {
                    minusValues.Add(-value);
                }
            }
            return OutstandingRankOne(-waitValue, minusValues, ref isSignificant);
        }

        public static double OutstandingRankOne(double waitValue, ICollection<double> otherValues, ref bool isSignificant)
        {
            if (waitValue < 0 || otherValues.Count == 0)
            {
                isSignificant = false;
                return 0;
            }
            List<double> orderedValues = new List<double>();
            foreach (double value in otherValues)
            {
                if (value > 0)
                    orderedValues.Add(value);
            }

            orderedValues.Sort((x, y) => y.CompareTo(x));
            if (orderedValues.Count >= 2)
            {
                // train model
                double alpha = -1;
                double innerProduct = 0;
                double var = 0;
                for (int i = 0; i < orderedValues.Count; i++)
                {
                    double x = i + 2;
                    double y = orderedValues[i];
                    double phy = KernelFunc(x);
                    var += phy * phy;
                    innerProduct += phy * y;
                }
                alpha = innerProduct / var;

                // get error collection
                double meanError = 0;
                double varError = 0;
                for (int i = 0; i < orderedValues.Count; i++)
                {
                    double currentError = Math.Abs(orderedValues[i] - (alpha * KernelFunc(i + 2)));
                    meanError += currentError;
                    varError += currentError * currentError;
                }
                meanError /= orderedValues.Count;
                varError /= orderedValues.Count;
                double stdv = Math.Sqrt(varError);

                // calcualte 3-sigma significance
                double estimatedError = waitValue - (alpha * KernelFunc(1));

                // this is the consideration of the temporal information. the outstanding no. 1 must has higher error then the second
                double secondError = Math.Abs(orderedValues[0] - (alpha * KernelFunc(2)));

                // My proposed solution to detect high-quality outstanding #1 (i.e., it is also not an outstanding top-k insight, for example, not a top-two, or top-three, etc.) is to
                // add such additional checking: v1-α≥k(v2-α/2^β ), Basically, I only check the error of the 1st and 2nd values.
                // Next, I will first show what is a proper k, and then, I will illustrate why checking these two values are enough (i.e., we don’t need to check the left values)
                // In the typical outstanding top-two case,  when v2≫ vi,where i>2, the estimated α* is
                // α*=argmax(α) ⁡∑[i from 2 to ∞] (vi-α/i^β)^2 -> α*=(∑vi/i^β)/(∑i^2β) ≈ (v2/2^β)/(∑1/i^2β ) ≈ 0.294v2, when β=0.7
                // According to the empirical study, under such circumstance, the first value should be at least TWO times larger than the second, user would think this is a valid outstanding no. 1,
                // therefore, we have 2v2-α≥k(v2-α/2^β ),  Put α* and β into, we have k=2.083 as the minimum value.
                // So when we set k≥2.083, when outstanding no.1 insight is output, the first value will be 2 times higher than the second,
                // which is aligned with user’s feeling. Above we only consider the first and second values. Here, let’s consider
                // v2≈v3≫ vi,i>3: because we don’t want to wrongly make an top-three insight as a top-1 insight. according to the empirical study,
                // the first value should be at least 2 times larger than the second, user would think this is a valid outstanding no. 1.
                // According to the calculation, we found that the k=2.083 still valid for this case ((as v3 grows, the error of v1 decreased faster than the error of v2)).
                // It’s not difficult to verify that (by a simple qualitative analysis), in order to avoid wrongly make outstanding top-k as outstanding no.1, this threshold is valid.

                double k = 2.083;

                // based on the intuition, v1 should be at least 5 times larger than v2 when v2 is one magnitude larger than v3. in order to achieve this, k should be set to 5.64 accordingly.
                if (orderedValues[0] / orderedValues[1] > 10)
                {
                    k = 5.64;
                }
                if (estimatedError > k * secondError && estimatedError >= meanError + (1 * stdv))
                {
                    isSignificant = true;
                    return MathUtility.GaussianSignificance(estimatedError, meanError, stdv);
                }
                else
                {
                    isSignificant = false;
                    return 0;
                }
            }
            else
            {
                isSignificant = false;
                return 0;
            }
        }

        /// <summary>
        /// the kernel function used for evaluating significant rank #1
        /// </summary>
        private static double KernelFunc(double x)
        {
            return 1.0 / Math.Pow(x, 0.7);
        }
    }
}
