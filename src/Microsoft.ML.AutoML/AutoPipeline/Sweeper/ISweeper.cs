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

    internal class SweeperOutput: Dictionary<string, object>
    {
        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach(var kv in this)
            {
                sb.Append($"{kv.Key}: {kv.Value.ToString()}");
            }

            return sb.ToString();
        }
    }

    internal class SweeperInput
    {
        public double Score { get; set; }
    }

}
