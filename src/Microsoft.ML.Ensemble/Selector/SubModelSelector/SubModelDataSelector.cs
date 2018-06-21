// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Internallearn;

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public abstract class SubModelDataSelector<TOutput> : BaseSubModelSelector<TOutput>
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "lp", SortOrder = 50,
                HelpText = "The proportion of best base learners to be selected. The range is 0.0-1.0")]
            [TGUI(Label = "Learners Selection Proportion")]
            public Single LearnersSelectionProportion = 0.5f;

            [Argument(ArgumentType.AtMostOnce, ShortName = "vp", SortOrder = 50,
                HelpText = "The proportion of instances to be selected to test the individual base learner. If it is 0, it uses training set")]
            [TGUI(Label = "Validation Dataset Proportion")]
            public Single ValidationDatasetProportion = 0.3f;
        }

        public Single LearnersSelectionProportion { get; }

        public override Single ValidationDatasetProportion { get; }

        protected SubModelDataSelector(ArgumentsBase args, IHostEnvironment env, string name)
            : base(env, name)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckParam(0 <= args.ValidationDatasetProportion && args.ValidationDatasetProportion < 1,
                nameof(args.ValidationDatasetProportion),
                "Should be greater than or equal to 0 and less than 1");
            Host.CheckParam(0 < args.LearnersSelectionProportion && args.LearnersSelectionProportion < 1,
                nameof(args.LearnersSelectionProportion),
                "Should be greater than 0 and less than 1");
            LearnersSelectionProportion = args.LearnersSelectionProportion;
            ValidationDatasetProportion = args.ValidationDatasetProportion;
        }
    }
}
