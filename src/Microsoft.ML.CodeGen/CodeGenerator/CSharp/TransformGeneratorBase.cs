// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.Parameter;

namespace Microsoft.ML.CodeGenerator.CSharp
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

        protected string[] InputColumns;

        protected string[] OutputColumns;

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
            InputColumns = new string[node.InColumns.Length];
            OutputColumns = new string[node.OutColumns.Length];
            int i = 0;
            foreach (var column in node.InColumns)
            {
                InputColumns[i++] = "\"" + column + "\"";
            }
            i = 0;
            foreach (var column in node.OutColumns)
            {
                OutputColumns[i++] = "\"" + column + "\"";
            }

        }

        public virtual string GenerateTransformer()
        {
            var parameters = GetType().GetProperties(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                    .Where(prop => prop.GetCustomAttributes(typeof(ParameterAttribute), true).Length > 0)
                    .OrderBy((prop) =>
                    {
                        ParameterAttribute attr = Attribute.GetCustomAttribute(prop, typeof(ParameterAttribute)) as ParameterAttribute;
                        return attr.Position;
                    })
                    .Select(x => (IParameter)x.GetValue(this));
            return $"{MethodName}({string.Join(",", parameters.Select(param => param.ToParameter()))})";
        }

        public string[] GenerateUsings()
        {
            return Usings;
        }
    }
}
