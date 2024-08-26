﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using SweepableEstimatorT = Microsoft.ML.AutoML.SourceGenerator.Template.SweepableEstimator_T_;
namespace Microsoft.ML.AutoML.SourceGenerator
{
    [Generator]
    public sealed class SweepableEstimatorGenerator : ISourceGenerator
    {
        private const string SweepableEstimatorAttributeDisplayName = Constant.CodeGeneratorNameSpace + "." + "SweepableEstimatorAttribute";

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AdditionalFiles.Where(f => f.Path.Contains("code_gen_flag.json")).First() is AdditionalText text)
            {
                var json = text.GetText().ToString();
                var flags = JsonSerializer.Deserialize<Dictionary<string, bool>>(json);
                if (flags.TryGetValue(nameof(SweepableEstimatorGenerator), out var res) && res == false)
                {
                    return;
                }
            }

            var estimators = context.AdditionalFiles.Where(f => f.Path.Contains("trainer-estimators.json") || f.Path.Contains("transformer-estimators.json"))
                                                  .SelectMany(file => Utils.GetEstimatorsFromJson(file.GetText().ToString()).Estimators)
                                                  .ToArray();

            var code = estimators.SelectMany(e => e.EstimatorTypes.Select(eType => (e, eType, Utils.CreateEstimatorName(e.FunctionName, eType)))
                                 .Select(x =>
                                 {
                                     if (x.e.SearchOption == null)
                                     {
                                         return
                                         (x.Item3,
                                         new AutoML.SourceGenerator.Template.SweepableEstimator()
                                         {
                                             NameSpace = Constant.CodeGeneratorNameSpace,
                                             UsingStatements = x.e.UsingStatements,
                                             ArgumentsList = x.e.ArgumentsList,
                                             ClassName = x.Item3,
                                             FunctionName = x.e.FunctionName,
                                             NugetDependencies = x.e.NugetDependencies,
                                             Type = x.eType,
                                         }.TransformText());
                                     }
                                     else
                                     {
                                         return
                                         (x.Item3,
                                         new SweepableEstimatorT()
                                         {
                                             NameSpace = Constant.CodeGeneratorNameSpace,
                                             UsingStatements = x.e.UsingStatements,
                                             ArgumentsList = x.e.ArgumentsList,
                                             ClassName = x.Item3,
                                             FunctionName = x.e.FunctionName,
                                             NugetDependencies = x.e.NugetDependencies,
                                             Type = x.eType,
                                             TOption = Utils.ToTitleCase(x.e.SearchOption),
                                         }.TransformText());
                                     }
                                 }));

            foreach (var c in code)
            {
                context.AddSource(c.Item1 + ".cs", SourceText.From(c.Item2, Encoding.UTF8));
            }
        }

        public void Initialize(GeneratorInitializationContext context)
        {
            return;
            //context.RegisterForPostInitialization(i => i.AddSource(nameof(SweepableEstimatorAttribute), SweepableEstimatorAttribute));
        }
    }
}
