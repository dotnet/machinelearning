// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal interface ITrialSettingsProposer
    {
        TrialSettings Propose(TrialSettings settings);

        void Update(TrialSettings parameter, TrialResult result);
    }

    internal interface ISavableProposer : ITrialSettingsProposer
    {
        void SaveStatusToFile(string fileName);

        void LoadStatusFromFile(string fileName);
    }
}
