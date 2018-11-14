// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Internal.Calibration;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class IniFileUtils
    {
        // This could be done better by having something that actually parses the .ini file and provides more
        //  functionality. For now, we'll just provide the minimum needed. If we went the nicer route, probably would
        //  want a class representing the Ini file, and a separate one for the cal
        // TODO: Consider using built-in .ini file parsing routines, they should work fine.
        // TODO: Should be made more robust to handle things like "Evaluators = " sand such.
        // TODO: Would be nice to place this evaluator above the comments section
        public static string AddEvaluator(string ini, string evaluator)
        {
            int numEvaluators = NumEvaluators(ini);
            if (!ini.Contains("Evaluators=" + numEvaluators))
                throw Contracts.ExceptNotImpl("Need to make the replacing of Evaluators= more robust");
            ini = ini.Replace("Evaluators=" + numEvaluators, "Evaluators=" + (numEvaluators + 1));

            StringBuilder bld = new StringBuilder(ini);
            bld.AppendLine();
            bld.AppendLine();
            bld.AppendLine("[Evaluator:" + (numEvaluators + 1) + "]");
            bld.AppendLine(evaluator);

            return bld.ToString();
        }

        public static int NumEvaluators(string ini)
        {
            // Look for  "Evaluators=101"
            Regex numEvaluators = new Regex("Evaluators=([0-9]+)");
            Match match = numEvaluators.Match(ini);
            Contracts.Check(match.Success, "Unable to retrieve number of evaluators from ini");
            string count = match.Groups[1].Value;
            return int.Parse(count);
        }

        public static string GetCalibratorEvaluatorIni(string originalIni, PlattCalibrator calibrator)
        {
            // Bing-style output as a second evaluator
            // Sigmoid: P(z) = 1/(1+exp(-z)).
            // Calibrator: P(x) = 1/(1+exp(ax+b)), where x is output of model (evaluator 1)
            //  => z = -ax + -b
            StringBuilder newEvaluator = new StringBuilder();
            newEvaluator.AppendLine("EvaluatorType=Aggregator");
            newEvaluator.AppendLine("Type=Sigmoid");
            newEvaluator.AppendLine("Bias=" + -calibrator.ParamB);
            newEvaluator.AppendLine("NumNodes=1");
            newEvaluator.AppendLine("Nodes=E:" + NumEvaluators(originalIni));
            newEvaluator.AppendLine("Weights=" + -calibrator.ParamA);
            return newEvaluator.ToString();
        }
    }
}
