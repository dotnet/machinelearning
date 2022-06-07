// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;
using TorchSharp.Modules;

namespace Microsoft.ML.TorchSharp.NasBert.Optimizers
{
    /// <summary>
    /// A wrapper of <seealso cref="torch.optim.Optimizer"/> for some extra operations.
    /// </summary>
    internal abstract class BaseOptimizer
    {
        /// <summary>
        /// Create and return an optimizer according to <paramref name="options"/>.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="parameters">The parameters to be optimized by the optimizer.</param>
        public static BaseOptimizer GetOptimizer(TextClassificationTrainer.Options options, IEnumerable<Parameter> parameters)
        {
            return new Adam(options, parameters);
            //var optimizerName = options.Optimizer.ToLower();
            //return optimizerName switch
            //{
            //    "adam" => new Adam(options, parameters),
            //    //"sgd" => new Sgd(options, parameters),
            //    _ => throw new NotSupportedException($"{optimizerName} not supported yet!"),
            //};
        }

        protected TextClassificationTrainer.Options Options { get; set; }
        protected string Name { get; set; }
        protected IEnumerable<Parameter> Parameters { get; set; }
        public torch.optim.Optimizer Optimizer { get; protected set; }
        public double LearningRate => Optimizer.ParamGroups.ToArray()[0].LearningRate;

        protected BaseOptimizer(string name, TextClassificationTrainer.Options options, IEnumerable<Parameter> parameters)
        {
            Name = name;
            Options = options;
            Parameters = parameters.ToArray();
        }

        /// <summary>
        /// Performs a single optimization step.
        /// </summary>
        public void Step()
        {
            if (Options.ClipNorm > 0)
            {
                torch.nn.utils.clip_grad_norm_(Parameters, Options.ClipNorm);
            }
            Optimizer.step();
        }


        /// <summary>
        /// Clears the gradients of all optimized parameters.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "Should match TorchSharp.")]
        public void zero_grad() => Optimizer.zero_grad();

        public double GetGradNorm()
        {
            return Math.Sqrt(Parameters
                .Select(p => p.grad())
                .Where(grad => grad.IsNotNull())      // parameters unused have no gradient
                .Select(grad => grad.square().sum().ToDouble())
                .Sum());
        }

        public bool IsGradNormClipped(double gradNorm)
        {
            return gradNorm > Options.ClipNorm && Options.ClipNorm > 0;
        }

        /// <summary>
        /// Multiplies grads by a constant
        /// </summary>
        /// <param name="c">the constant</param>
        public void MultiplyGrads(double c)
        {
            foreach (var p in Parameters)
            {
                using var grad = p.grad();
                if (grad.IsNotNull())
                {
                    grad.mul_(c);
                }
            }
        }
    }
}
