using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.EntryPoints;

namespace Microsoft.ML.AutoML.AutoPipeline.Sweeper
{
    internal interface ISweeper:  IEnumerable<SweeperOutput>, IEnumerator<SweeperOutput> 
    {
        /// <summary>
        /// For trainable Sweeper.
        /// </summary>
        /// <param name="input">Output of Sweeper.</param>
        /// <param name="Y">Score from model</param>
        void Fit(SweeperOutput input, SweeperInput Y);
    }

    internal class SweeperOutput: Dictionary<string, object> { }

    internal class SweeperInput
    {
        public int Score { get; set; }
    }

}
