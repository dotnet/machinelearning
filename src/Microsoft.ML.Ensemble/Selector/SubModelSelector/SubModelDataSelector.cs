using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Internallearn;
using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public abstract class SubModelDataSelector<TOutput> : BaseSubModelSelector<TOutput>
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The proportion of best base learners to be selected. The range is 0.0-1.0", ShortName = "lp", SortOrder = 50)]
            [TGUI(Label = "Learners Selection Proportion")]
            public Single LearnersSelectionProportion = 0.5f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The proportion of instances to be selected to test the individual base learner. If it is 0, it uses training set", ShortName = "vp", SortOrder = 50)]
            [TGUI(Label = "Validation Dataset Proportion")]
            public Single ValidationDatasetProportion = 0.3f;
        }

        private readonly Single _learnersSelectionProportion;
        private readonly Single _validationDatasetProportion;

        public Single LearnersSelectionProportion { get { return _learnersSelectionProportion; } }

        public override Single ValidationDatasetProportion { get { return _validationDatasetProportion; } }

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
            _learnersSelectionProportion = args.LearnersSelectionProportion;
            _validationDatasetProportion = args.ValidationDatasetProportion;
        }
    }
}
