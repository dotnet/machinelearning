using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoPipeline
{
    internal interface ISweeper:  IEnumerable<ParameterSet>, IEnumerator<ParameterSet>
    {
        /// <summary>
        /// For trainable Sweeper.
        /// </summary>
        /// <param name="input">Output of Sweeper.</param>
        /// <param name="Y">Score from model</param>
        void AddRunHistory(IEnumerable<IRunResult> input, SweeperInput Y);
    }

    internal class SweeperInput
    {
        public double Score { get; set; }
    }

}
