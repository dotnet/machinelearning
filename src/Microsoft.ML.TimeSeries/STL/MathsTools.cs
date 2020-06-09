using System;

namespace Microsoft.ML.TimeSeries
{
    public class MathUtility
    {
        /// <summary>
        /// calculate the standard sigmoid function
        /// </summary>
        /// <param name="x">the input value</param>
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
