// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class SuggestedPipelineBuilder
    {
        public static SuggestedPipeline Build(MLContext context,
            ICollection<SuggestedTransform> transforms,
            ICollection<SuggestedTransform> transformsPostTrainer,
            SuggestedTrainer trainer,
            CacheBeforeTrainer cacheBeforeTrainerSettings)
        {
            var trainerInfo = trainer.BuildTrainer().Info;
            AddNormalizationTransforms(context, trainerInfo, transforms);
            var cacheBeforeTrainer = ShouldCacheBeforeTrainer(trainerInfo, cacheBeforeTrainerSettings);
            return new SuggestedPipeline(transforms, transformsPostTrainer, trainer, context, cacheBeforeTrainer);
        }

        private static void AddNormalizationTransforms(MLContext context,
            TrainerInfo trainerInfo,
            ICollection<SuggestedTransform> transforms)
        {
            // Only add normalization if trainer needs it
            if (!trainerInfo.NeedNormalization)
            {
                return;
            }

            var transform = NormalizingExtension.CreateSuggestedTransform(context, DefaultColumnNames.Features, DefaultColumnNames.Features);
            transforms.Add(transform);
        }

        private static bool ShouldCacheBeforeTrainer(TrainerInfo trainerInfo, CacheBeforeTrainer cacheBeforeTrainerSettings)
        {
            return cacheBeforeTrainerSettings == CacheBeforeTrainer.On || (cacheBeforeTrainerSettings == CacheBeforeTrainer.Auto && trainerInfo.WantCaching);
        }
    }
}
