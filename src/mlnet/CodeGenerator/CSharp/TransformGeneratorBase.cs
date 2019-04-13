// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    /// <summary>
    /// Supports generation of code for trainers (Binary,Multi,Regression)
    /// Ova is an exception though. Need to figure out how to tackle that.
    /// </summary>
    internal abstract class TransformGeneratorBase : ITransformGenerator
    {
        //abstract properties
        internal abstract string MethodName { get; }

        internal virtual string[] Usings => null;

        protected string[] inputColumns;

        protected string[] outputColumns;

        /// <summary>
        /// Generates an instance of TrainerGenerator
        /// </summary>
        /// <param name="node"></param>
        protected TransformGeneratorBase(PipelineNode node)
        {
            Initialize(node);
        }

        private void Initialize(PipelineNode node)
        {
            inputColumns = new string[node.InColumns.Length];
            outputColumns = new string[node.OutColumns.Length];
            int i = 0;
            foreach (var column in node.InColumns)
            {
                inputColumns[i++] = "\"" + column + "\"";
            }
            i = 0;
            foreach (var column in node.OutColumns)
            {
                outputColumns[i++] = "\"" + column + "\"";
            }

        }

        public abstract string GenerateTransformer();

        public string[] GenerateUsings()
        {
            return Usings;
        }
    }
}
