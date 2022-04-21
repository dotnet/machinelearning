﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal interface ITuner
    {
        Parameter Propose(TrialSettings settings);

        void Update(TrialResult result);
    }
}
