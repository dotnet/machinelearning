using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Microsoft.ML.TimeSeries
{
    public class FftTransform
    {
        /// <summary>
        /// calculate the fast fourier transform.
        /// </summary>
        public static Complex[] Fft(Complex[] array)
        {
            int n = array.Length;
            int newN = Get2Power(n);
            if (newN > n)
            {
                Complex[] newArray = new Complex[newN];
                for (int i = 0; i < newN; i++)
                {
                    if (i < n)
                        newArray[i] = array[i];
                    else
                        newArray[i] = Complex.Zero;
                }
                return RecursiveFft(newArray);
            }
            else
            {
                return RecursiveFft(array);
            }
        }

        /// <summary>
        /// calculate the fast fourier transform.
        /// </summary>
        public static Complex[] RevertFft(Complex[] array)
        {
            int n = array.Length;
            int newN = Get2Power(n);
            if (newN > n)
            {
                Complex[] newArray = new Complex[newN];
                for (int i = 0; i < newN; i++)
                {
                    if (i < n)
                        newArray[i] = array[i];
                    else
                        newArray[i] = Complex.Zero;
                }
                Complex[] result = RevertRecursiveFft(newArray);
                for (int i = 0; i < result.Length; i++)
                    result[i] /= newN;
                return result;
            }
            else
            {
                Complex[] result = RevertRecursiveFft(array);
                for (int i = 0; i < result.Length; i++)
                    result[i] /= n;
                return result;
            }
        }

        /// <summary>
        /// this method is the recursive FftTransform, which use the divide and conquer to achieve nlogn complexity.
        /// the key trick is the selection of n unit roots in the complex space. the length of the input array MUST be 2^k
        /// </summary>
        /// <param name="array">the input polynomial coefficients (or the dual if reverse FftTransform is called)</param>
        /// <returns>return the dual coefficients</returns>
        private static Complex[] RecursiveFft(Complex[] array)
        {
            int n = array.Length;
            if (n == 1)
                return array;
            Complex wn = new Complex(Math.Cos(2 * Math.PI / n), Math.Sin(2 * Math.PI / n));
            Complex w = Complex.One;
            Complex[] array0 = new Complex[n / 2];
            Complex[] array1 = new Complex[n / 2];
            for (int i = 0; i < n / 2; i++)
            {
                array0[i] = array[i * 2];
                array1[i] = array[i * 2 + 1];
            }
            Complex[] y0 = RecursiveFft(array0);
            Complex[] y1 = RecursiveFft(array1);
            Complex[] y = new Complex[n];
            for (int i = 0; i < n / 2; i++)
            {
                y[i] = y0[i] + w * y1[i];
                y[i + n / 2] = y0[i] - w * y1[i];
                w *= wn;
            }
            return y;
        }

        /// <summary>
        /// this method is the revert recursive FftTransform,
        /// </summary>
        /// <param name="array">the input polynomial coefficients (or the dual if reverse FftTransform is called)</param>
        /// <returns>return the dual coefficients</returns>
        private static Complex[] RevertRecursiveFft(Complex[] array)
        {
            int n = array.Length;
            if (n == 1)
                return array;
            Complex wn = new Complex(Math.Cos(2 * Math.PI / n), -Math.Sin(2 * Math.PI / n));
            Complex w = Complex.One;
            Complex[] array0 = new Complex[n / 2];
            Complex[] array1 = new Complex[n / 2];
            for (int i = 0; i < n / 2; i++)
            {
                array0[i] = array[i * 2];
                array1[i] = array[i * 2 + 1];
            }
            Complex[] y0 = RevertRecursiveFft(array0);
            Complex[] y1 = RevertRecursiveFft(array1);
            Complex[] y = new Complex[n];
            for (int i = 0; i < n / 2; i++)
            {
                y[i] = y0[i] + w * y1[i];
                y[i + n / 2] = y0[i] - w * y1[i];
                w *= wn;
            }
            return y;
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
                result = result << 1;
                n = n >> 1;
            }
            if (meet1)
                result = result << 1;
            return result;
        }
    }
}
