// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal static class PipelineExtension
    {
        internal static (string Usings, string TrainerMethod, List<string> PreTrainerTransforms, List<string> PostTrainerTransforms) GenerateTransformsAndTrainers(this Pipeline pipeline)
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();

            // Get pre-trainer transforms
            var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            var preTrainerTransformsAndUsings = GenerateTransformsAndUsings(nodes);

            // Get post trainer transforms
            nodes = pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms
            var postTrainerTransformsAndUsings = GenerateTransformsAndUsings(nodes);

            //Get trainer code and its associated usings.
            (string trainerMethod, string[] trainerUsings) = GenerateTrainerAndUsings(pipeline);
            if (trainerUsings != null)
            {
                usings.AddRange(trainerUsings);
            }

            //Get transforms code and its associated (unique) usings.
            var preTrainerTransforms = preTrainerTransformsAndUsings?.Select(t => t.Item1).ToList();
            var postTrainerTransforms = postTrainerTransformsAndUsings?.Select(t => t.Item1).ToList();
            usings.AddRange(preTrainerTransformsAndUsings.Where(t => t.Item2 != null).SelectMany(t => t.Item2));
            usings.AddRange(postTrainerTransformsAndUsings.Where(t => t.Item2 != null).SelectMany(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            return (usingsBuilder.ToString(), trainerMethod, preTrainerTransforms, postTrainerTransforms);
        }

        internal static IList<(string, string[])> GenerateTransformsAndUsings(IEnumerable<PipelineNode> nodes)
        {
            //var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            //var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
            var results = new List<(string, string[])>();
            foreach (var node in nodes)
            {
                ITransformGenerator generator = TransformGeneratorFactory.GetInstance(node);
                results.Add((generator.GenerateTransformer(), generator.GenerateUsings()));
            }

            return results;
        }

        internal static (string, string[]) GenerateTrainerAndUsings(Pipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));
            try
            {
                var node = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer).First();
                ITrainerGenerator generator = TrainerGeneratorFactory.GetInstance(node);
                var trainerString = generator.GenerateTrainer();
                var trainerUsings = generator.GenerateUsings();
                return (trainerString, trainerUsings);
            }
            catch (Exception)
            {
                return (string.Empty, new string[0]);
            }
        }
    }
}
