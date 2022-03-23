// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.AutoMLService
{
    internal class RandomNumberGenerator
    {
        private readonly Random _random;

        public RandomNumberGenerator()
        {
            this._random = new Random();
        }

        public RandomNumberGenerator(int seed)
        {
            this._random = new Random(seed);
        }

        public int Integer(int high)
        {
            return this._random.Next(high);
        }

        public double Uniform(double low, double high)
        {
            return (this._random.NextDouble() * (high - low)) + low;
        }

        public double Normal(double location, double scale)
        {
            double u = 1 - this.Uniform(0, 1);
            double v = 1 - this.Uniform(0, 1);
            double std = Math.Sqrt(-2.0 * Math.Log(u)) * Math.Sin(2.0 * Math.PI * v);
            return location + (std * scale);
        }

        public double[] Normal(double location, double scale, int size)
        {
            double[] ret = new double[size];
            for (int i = 0; i < size; i++)
            {
                ret[i] = this.Normal(location, scale);
            }

            return ret;
        }

        public int Categorical(double[] possibility)
        {
            double x = this.Uniform(0, 1);
            for (int i = 0; i < possibility.Length; i++)
            {
                x -= possibility[i];
                if (x < 0) { return i; }
            }

            return possibility.Length - 1;
        }

        public int[] Categorical(double[] possibility, int size)
        {
            int[] ret = new int[size];
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = this.Categorical(possibility);
            }

            return ret;
        }
    }
}
