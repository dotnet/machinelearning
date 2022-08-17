// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML
{
    internal interface IHyperParameterProposer
    {
        TrialSettings Propose(TrialSettings settings);

        void Update(TrialSettings parameter, TrialResult result);
    }
}
