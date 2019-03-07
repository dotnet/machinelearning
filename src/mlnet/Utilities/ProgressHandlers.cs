// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Microsoft.ML.CLI.Utilities
{
    internal class ProgressHandlers
    {
        internal class RegressionHandler : IProgress<RunResult<RegressionMetrics>>
        {
            int iterationIndex;
            public RegressionHandler()
            {
                ConsolePrinter.PrintRegressionMetricsHeader();
            }

            public void Report(RunResult<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                ConsolePrinter.PrintRegressionMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics);
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunResult<BinaryClassificationMetrics>>
        {
            int iterationIndex;
            internal BinaryClassificationHandler()
            {
                ConsolePrinter.PrintBinaryClassificationMetricsHeader();
            }

            public void Report(RunResult<BinaryClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                ConsolePrinter.PrintBinaryClassificationMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics);
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunResult<MultiClassClassifierMetrics>>
        {
            int iterationIndex;
            internal MulticlassClassificationHandler()
            {
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader();
            }

            public void Report(RunResult<MultiClassClassifierMetrics> iterationResult)
            {
                iterationIndex++;
                ConsolePrinter.PrintMulticlassClassificationMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics);
            }
        }
    }
}
