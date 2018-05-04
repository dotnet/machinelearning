// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(SynthConfigRunner), typeof(SynthConfigRunner.Arguments), typeof(SignatureConfigRunner),
    "", "Synth")]

namespace Microsoft.ML.Runtime.Sweeper
{
    /// <summary>
    /// This class gives a simple way of running optimization experiments on synthetic functions, rather than on actual learning problems. 
    /// It was initially created to test the sweeper methods on the Rastrigin function.
    /// </summary>
    public sealed class SynthConfigRunner : ExeConfigRunnerBase
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of threads to use for the sweep (default auto determined by the number of cores)", ShortName = "t")]
            public int? NumThreads;
        }

        private readonly ParallelOptions _parallelOptions;

        public SynthConfigRunner(IHostEnvironment env, Arguments args)
            : base(args, env, "SynthSweepEvaluator")
        {
            Host.CheckUserArg(args.NumThreads == null || args.NumThreads.Value > 0, nameof(args.NumThreads), "Must be positive");
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = args.NumThreads ?? -1 };
            Host.AssertNonEmpty(args.OutputFolderName);
            ProcessFullExePath(args.Exe);
        }

        protected override IEnumerable<IRunResult> RunConfigsCore(ParameterSet[] sweeps, IChannel ch, int min)
        {
            List<IRunResult> results = new List<IRunResult>();
            for (int j = 0; j < sweeps.Length; j++)
            {
                double val = Rastrigin(sweeps[j]);
                results.Add(new RunResult(sweeps[j], val, true));

                // Write results out to files.
                string filePath = string.Format(@"{0}\{1}.out.txt", OutputFolder, min + j);
                string content = string.Format(@"{1}

OVERALL RESULTS
---------------------------------------
ACCURACY:            0.0000 (0.0000)
POS. PRECISION:      0.0000 (0.0000)
POS. RECALL:         0.0000 (0.0000)
NEG. PRECISION:      0.0000 (0.0000)
NEG. RECALL:         0.0000 (0.0000)
LOG-LOSS:            0.0000 (0.0000)
LOG-LOSS REDUCTION:  0.0000 (0.0000)
AUC:                 {0:#,0.0000000} (0.0000)

---------------------------------------
6/23/2016 11:32:57 AM	 Time elapsed(s): 1.000
", val, sweeps[j].ToString());
                var sw = new StreamWriter(filePath);
                sw.Write(content);
                sw.Flush();
                sw.Close();
            }

            return results;
        }

        /// <summary>
        /// Synthetic function used in the optimization literature to test optimization methods. Highly multi-modal,
        /// this functions causes problems for methods that get stuck at local optima (like hill-climbing methods).
        /// This synthetic function takes the place of an actual metric evaluation (hence, a synthetic runner).
        /// </summary>
        /// <param name="ps">The set of parameters to evaluate using the function.</param>
        /// <returns>The numerical evaluation of the parameter values.</returns>
        private double Rastrigin(ParameterSet ps)
        {
            double total = 0;
            foreach (var param in ps)
            {
                double val = float.Parse(param.ValueText);
                total += Math.Pow(val, 2) - 10 * Math.Cos(2 * Math.PI * val);
            }
            return (10 * ps.Count + total) > 0 ? 1.0 / (10 * ps.Count + total) : 1.0;
        }
    }
}
