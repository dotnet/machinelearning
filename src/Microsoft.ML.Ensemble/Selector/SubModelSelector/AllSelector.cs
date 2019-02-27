﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(AllSelector), null, typeof(SignatureEnsembleSubModelSelector), AllSelector.UserName, AllSelector.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class AllSelector : BaseSubModelSelector<Single>, IBinarySubModelSelector, IRegressionSubModelSelector
    {
        public const string UserName = "All Selector";
        public const string LoadName = "AllSelector";

        public override Single ValidationDatasetProportion => 0;

        protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public AllSelector(IHostEnvironment env)
            : base(env, LoadName)
        {
        }
    }
}
