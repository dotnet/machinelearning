// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;

namespace Microsoft.ML.AutoML
{
    internal static class AutoMlUtils
    {
        private const string MLNetMaxThread = "MLNET_MAX_THREAD";

        public static readonly ThreadLocal<Random> Random = new ThreadLocal<Random>(() => new Random());

        /// <summary>
        /// Return number of thread if MLNET_MAX_THREAD is set, otherwise return null.
        /// </summary>
        public static int? GetNumberOfThreadFromEnvrionment()
        {
            var res = Environment.GetEnvironmentVariable(MLNetMaxThread);

            if (int.TryParse(res, out var numberOfThread))
            {
                return numberOfThread;
            }

            return null;
        }

        public static InputOutputColumnPair[] CreateInputOutputColumnPairsFromStrings(string[] inputs, string[] outputs)
        {
            if (inputs.Length != outputs.Length)
            {
                throw new Exception("inputs and outputs count must match");
            }

            var res = new List<InputOutputColumnPair>();
            for (int i = 0; i != inputs.Length; ++i)
            {
                res.Add(new InputOutputColumnPair(outputs[i], inputs[i]));
            }

            return res.ToArray();
        }
    }
}
