// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.DotNet.Interactive;

namespace Microsoft.ML.AutoML.Interactive
{
    // this monitor redirects output result to context.log
    public class NotebookMonitor : IMonitor
    {
        private DisplayedValue _valueToUpdate = null;

        public TrialResult BestTrial { get; set; }
        public TrialResult MostRecentTrial { get; set; }
        public TrialSettings ActiveTrial { get; set; }
        public List<TrialResult> CompletedTrials { get; set; }

        public NotebookMonitor()
        {
            this.CompletedTrials = new List<TrialResult>();
        }

        public void ReportBestTrial(TrialResult result)
        {
            this.BestTrial = result;
            Update();
        }

        public void ReportCompletedTrial(TrialResult result)
        {
            // Todo: Make HTML/Pretty
            // mostRecentTrial.Update($"Completed Trial - Id: {result.TrialSettings.TrialId} - Metric: {result.Metric} - Pipeline: {result.TrialSettings.Pipeline} - Duration: {result.DurationInMilliseconds}");
            this.MostRecentTrial = result;
            this.CompletedTrials.Add(result);
            Update();
        }

        public void ReportFailTrial(TrialResult result)
        {
            // Todo: Make HTML/Pretty
            // $"Update Failed Trial - Id: {result.TrialSettings.TrialId} - Metric: {result.Metric} - Pipeline: {result.TrialSettings.Pipeline}".Display();
            Update();
        }

        public void ReportRunningTrial(TrialSettings setting)
        {
            this.ActiveTrial = setting;
            Update();
        }

        public void Update()
        {
            if (this._valueToUpdate != null)
            {
                this._valueToUpdate.Update(this);
            }
        }

        public void SetUpdate(DisplayedValue valueToUpdate)
        {
            this._valueToUpdate = valueToUpdate;
        }
    }

}
