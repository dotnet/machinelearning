// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.AutoML
{
    internal static class RecipeInference
    {
        /// <summary>
        /// Given a predictor type, return a set of all permissible trainers (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static IEnumerable<SuggestedTrainer> AllowedTrainers(MLContext mlContext, TaskKind task,
            ColumnInformation columnInfo, IEnumerable<TrainerName> trainerAllowList)
        {
            var trainerExtensions = TrainerExtensionCatalog.GetTrainers(task, trainerAllowList, columnInfo);

            var trainers = new List<SuggestedTrainer>();
            foreach (var trainerExtension in trainerExtensions)
            {
                var learner = new SuggestedTrainer(mlContext, trainerExtension, columnInfo);
                trainers.Add(learner);
            }
            return trainers.ToArray();
        }
    }
}
