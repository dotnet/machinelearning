// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using Microsoft.ML.AutoML.SourceGenerator.Template;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    [Generator]
    public class EstimatorTypeGenerator : ISourceGenerator
    {
        private const string className = "EstimatorType";
        private const string fullName = Constant.CodeGeneratorNameSpace + "." + className;

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AdditionalFiles.Where(f => f.Path.Contains("code_gen_flag.json")).First() is AdditionalText text)
            {
                var json = text.GetText().ToString();
                var flags = JsonSerializer.Deserialize<Dictionary<string, bool>>(json);
                if (flags.TryGetValue(nameof(EstimatorTypeGenerator), out var res) && res == false)
                {
                    return;
                }
            }

            var trainers = context.AdditionalFiles.Where(f => f.Path.Contains("trainer-estimators.json"))
                                                  .SelectMany(file => Utils.GetEstimatorsFromJson(file.GetText().ToString()).Estimators, (text, estimator) => (estimator.FunctionName, estimator.EstimatorTypes))
                                                  .SelectMany(union => union.EstimatorTypes.Select(t => Utils.CreateEstimatorName(union.FunctionName, t)))
                                                  .ToArray();

            var transformers = context.AdditionalFiles.Where(f => f.Path.Contains("transformer-estimators.json"))
                                                  .SelectMany(file => Utils.GetEstimatorsFromJson(file.GetText().ToString()).Estimators, (text, estimator) => (estimator.FunctionName, estimator.EstimatorTypes))
                                                  .SelectMany(union => union.EstimatorTypes.Select(t => Utils.CreateEstimatorName(union.FunctionName, t)))
                                                  .ToArray();

            var code = new EstimatorType()
            {
                NameSpace = Constant.CodeGeneratorNameSpace,
                ClassName = className,
                TrainerNames = trainers,
                TransformerNames = transformers,
            };

            context.AddSource(className + ".cs", SourceText.From(code.TransformText(), Encoding.UTF8));
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }
    }
}
