// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using Microsoft.ML.AutoML.SourceGenerator.Template;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    [Generator]
    public class SweepableEstimatorFactoryGenerator : ISourceGenerator
    {
        private const string className = "SweepableEstimatorFactory";

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AdditionalFiles.Where(f => f.Path.Contains("code_gen_flag.json")).First() is AdditionalText text)
            {
                var json = text.GetText().ToString();
                var flags = JsonSerializer.Deserialize<Dictionary<string, bool>>(json);
                if (flags.TryGetValue(nameof(SweepableEstimatorFactoryGenerator), out var res) && res == false)
                {
                    return;
                }
            }

            var trainers = context.AdditionalFiles.Where(f => f.Path.Contains("trainer-estimators.json"))
                                                  .SelectMany(file => Utils.GetEstimatorsFromJson(file.GetText().ToString()).Estimators, (text, estimator) => (estimator.FunctionName, estimator.EstimatorTypes, estimator.SearchOption))
                                                  .SelectMany(union => union.EstimatorTypes.Select(t => (Utils.CreateEstimatorName(union.FunctionName, t), Utils.ToTitleCase(union.SearchOption))))
                                                  .ToArray();

            var transformers = context.AdditionalFiles.Where(f => f.Path.Contains("transformer-estimators.json"))
                                                  .SelectMany(file => Utils.GetEstimatorsFromJson(file.GetText().ToString()).Estimators, (text, estimator) => (estimator.FunctionName, estimator.EstimatorTypes, estimator.SearchOption))
                                                  .SelectMany(union => union.EstimatorTypes.Select(t => (Utils.CreateEstimatorName(union.FunctionName, t), Utils.ToTitleCase(union.SearchOption))))
                                                  .ToArray();

            var code = new SweepableEstimatorFactory()
            {
                NameSpace = Constant.CodeGeneratorNameSpace,
                EstimatorNames = trainers.Concat(transformers),
            };

            context.AddSource(className + ".cs", SourceText.From(code.TransformText(), Encoding.UTF8));
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }
    }
}
