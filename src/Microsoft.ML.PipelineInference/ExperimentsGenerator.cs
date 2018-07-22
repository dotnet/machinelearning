// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public sealed class ExperimentsGenerator
    {
        /// <summary>
        /// Holds the sweep and the trainer for sweep commands.
        /// </summary>
        public sealed class Sweep
        {
            public readonly Pattern Pattern;
            // for one pattern there will be one learner, but several combinations.
            public readonly TrainerSweeper TrainerSweeper;

            internal Sweep(Pattern pattern, TrainerSweeper trainerSweeper)
            {
                Pattern = pattern;
                TrainerSweeper = trainerSweeper;
            }
        }

        /// <summary>
        /// Pattern of the Sweep command.
        /// </summary>
        public sealed class Pattern
        {
            public readonly TransformInference.SuggestedTransform[] Transforms;
            public readonly RecipeInference.SuggestedRecipe.SuggestedLearner Learner;
            public readonly string Loader;

            internal Pattern(TransformInference.SuggestedTransform[] transforms,
                RecipeInference.SuggestedRecipe.SuggestedLearner learner,
                string loader)
            {
                Transforms = transforms;
                Learner = learner;
                Loader = loader;
            }

            public string ToStringRep(string mode, string trainData, string testData, string dout = null)
            {
                StringBuilder command = new StringBuilder($"{mode}");
                command.Append(Loader);

                foreach (var transform in Transforms)
                    command.Append($" xf={transform.Transform}");

                command.Append($" tr={Learner.LoadableClassInfo.LoadNames[0]}{{{Learner.Settings}}}");
                command.Append($" data={{{trainData}}}");

                if (mode.Equals(TrainTestCommand.LoadName) && testData != null)
                    command.Append($" testFile={{{testData}}}");

                if (dout != null)
                    command.Append($" dout={{{dout}}}");

                return command.ToString();
            }
        }

        /// <summary>
        /// Sweep parameters for the trainer.
        /// </summary>
        public sealed class TrainerSweeper
        {
            public List<string> Parameters;

            internal TrainerSweeper()
            {
                Parameters = new List<string>();
            }

            public string ToStringRep(string sweeperType)
            {
                if (Parameters.Count > 0)
                {
                    StringBuilder p = new StringBuilder();
                    Parameters.ForEach(par => p.Append(par));

                    return $" sweeper={sweeperType}{{{p}}}";
                }

                return null;
            }
        }

        public static List<Sweep> GenerateCandidates(IHostEnvironment env, string dataFile, string schemaDefinitionFile)
        {
            var patterns = new List<Sweep>();
            string loaderSettings = "";
            Type predictorType;
            TransformInference.InferenceResult inferenceResult;

            // Get the initial recipes for this data.
            RecipeInference.SuggestedRecipe[] recipes = RecipeInference.InferRecipesFromData(env, dataFile, schemaDefinitionFile, out predictorType, out loaderSettings, out inferenceResult);

            //get all the trainers for this task, and generate the initial set of candidates.
            // Exclude the hidden learners, and the metalinear learners. 
            var trainers = ComponentCatalog.GetAllDerivedClasses(typeof(ITrainer), predictorType).Where(cls => !cls.IsHidden);

            var loaderSubComponent = new SubComponent("TextLoader", loaderSettings);
            string loader = $" loader={loaderSubComponent}";

            // REVIEW: there are more learners than recipes atm. 
            // Flip looping through recipes, than through learners if the cardinality changes.  
            foreach (ComponentCatalog.LoadableClassInfo cl in trainers)
            {
                string learnerSettings;
                TrainerSweeper trainerSweeper = new TrainerSweeper();
                trainerSweeper.Parameters.AddRange(RecipeInference.GetLearnerSettingsAndSweepParams(env, cl, out learnerSettings));

                foreach (var recipe in recipes)
                {

                    RecipeInference.SuggestedRecipe.SuggestedLearner learner = new RecipeInference.SuggestedRecipe.SuggestedLearner
                    {
                        LoadableClassInfo = cl,
                        Settings = learnerSettings
                    };

                    Pattern pattern = new Pattern(recipe.Transforms, learner, loader);
                    Sweep sweep = new Sweep(pattern, trainerSweeper);
                    patterns.Add(sweep);
                }
            }

            return patterns;
        }
    }
}
