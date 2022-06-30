using Microsoft.DotNet.Interactive;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using System;
using System.Threading.Tasks;
using System.Text.Json;


namespace Microsoft.ML.AutoML
{
    public class NotebookMonitor : IMonitor
    {
        private DisplayedValue? ValueToUpdate;
        private DateTime _lastUpdate = DateTime.MinValue;

        public TrialResult? BestTrial { get; set; }
        public TrialResult? MostRecentTrial { get; set; }
        public TrialSettings? ActiveTrial { get; set; }
        public List<TrialResult> CompletedTrials { get; set; }
        public DataFrame DataFrame { get; set; }

        public NotebookMonitor()
        {
            this.CompletedTrials = new List<TrialResult>();
            this.DataFrame = new DataFrame(new PrimitiveDataFrameColumn<int>("Trial"), new PrimitiveDataFrameColumn<float>("Metric"), new StringDataFrameColumn("Trainer"), new StringDataFrameColumn("Parameters"));
        }

        public void ReportBestTrial(TrialResult result)
        {
            this.BestTrial = result;
            Update();
        }

        public void ReportCompletedTrial(TrialResult result)
        {
            this.MostRecentTrial = result;
            this.CompletedTrials.Add(result);

            var activeRunParam = JsonSerializer.Serialize(result.TrialSettings.Parameter, new JsonSerializerOptions() { WriteIndented = false, });

            this.DataFrame.Append(new List<KeyValuePair<string, object>>()
            {
                new KeyValuePair<string, object>("Trial",result.TrialSettings.TrialId),
                new KeyValuePair<string, object>("Metric", result.Metric),
                new KeyValuePair<string, object>("Trainer",result.TrialSettings.Pipeline.ToString().Replace("Unknown=>","")),
                new KeyValuePair<string, object>("Parameters",activeRunParam),
            }, true);
            Update();
        }

        public void ReportFailTrial(TrialResult result)
        {
            // TODO figure out what to do with failed trials.
            Update();
        }

        public void ReportRunningTrial(TrialSettings setting)
        {
            this.ActiveTrial = setting;
            Update();
        }

        private bool updatePending = false;
        public void Update()
        {
            Task.Run(async () =>
            {
                if (updatePending == true)
                {
                    // Keep waiting
                }
                else
                {
                    int timeRemaining = 5000 - (int)(DateTime.Now.Millisecond - this._lastUpdate.Millisecond);
                    updatePending = true;
                    if (timeRemaining > 0)
                    {
                        await Task.Delay(timeRemaining);
                    }
                    if (this.ValueToUpdate != null)
                    {
                        updatePending = false;
                        this._lastUpdate = DateTime.Now;
                        this.ValueToUpdate.Update(this);
                    }
                }
            });

        }

        public void SetUpdate(DisplayedValue valueToUpdate)
        {
            this.ValueToUpdate = valueToUpdate;
            Update();
        }
    }
}
