// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.AspNetCore.Html;
using Microsoft.Data.Analysis;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.Commands;
using Microsoft.DotNet.Interactive.Formatting;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Plotly.NET.CSharp;
using static Microsoft.DotNet.Interactive.Formatting.PocketViewTags;


namespace Microsoft.ML.AutoML
{
    public class AutoMLMonitorKernelExtension : IKernelExtension
    {
        public async Task OnLoadAsync(Kernel kernel)
        {
            Formatter.Register<NotebookMonitor>((monitor, writer) =>
            {
                WriteSummary(monitor, writer);
                WriteChart(monitor, writer);
                WriteTable(monitor, writer);
            }, "text/html");

            if (Kernel.Root?.FindKernel("csharp") is { } csKernel)
            {
                await LoadExtensionApiAsync(csKernel);
            }
        }

        private static async Task LoadExtensionApiAsync(Kernel cSharpKernel)
        {
            await cSharpKernel.SendAsync(new SubmitCode($@"#r ""{typeof(AutoMLMonitorKernelExtension).Assembly.Location}""
using {typeof(NotebookMonitor).Namespace};"));
        }

        private static void WriteSummary(NotebookMonitor monitor, TextWriter writer)
        {

            var summary = new List<IHtmlContent>();

            if (monitor.BestTrial != null)
            {
                var bestTrialParam = JsonSerializer.Serialize(monitor.BestTrial.TrialSettings.Parameter, new JsonSerializerOptions() { WriteIndented = true, });
                summary.Add(h3("Best Trial"));
                summary.Add(p($"Id: {monitor.BestTrial.TrialSettings.TrialId}"));
                summary.Add(p($"Trainer: {monitor.BestTrial.TrialSettings.Pipeline}".Replace("Unknown=>", "")));
                summary.Add(p($"Parameters: {bestTrialParam}"));
            }
            if (monitor.ActiveTrial != null)
            {

                var activeTrialParam = JsonSerializer.Serialize(monitor.ActiveTrial.Parameter, new JsonSerializerOptions() { WriteIndented = true, });

                summary.Add(h3("Active Trial"));
                summary.Add(p($"Id: {monitor.ActiveTrial.TrialId}"));
                summary.Add(p($"Trainer: {monitor.ActiveTrial.Pipeline}".Replace("Unknown=>", "")));
                summary.Add(p($"Parameters: {activeTrialParam}"));
            }

            writer.Write(div(summary));
        }

        private static void WriteChart(NotebookMonitor monitor, TextWriter writer)
        {
            var x = monitor.CompletedTrials.Select(x => x.TrialSettings.TrialId);
            var y = monitor.CompletedTrials.Select(x => x.Metric);

            var chart = Chart.Point<int, double, string>(x, y, "Plot Metrics over Trials.")
            .WithTraceInfo(ShowLegend: false)
            .WithXAxisStyle<double, double, string>(TitleText: "Trial", ShowGrid: false)
            .WithYAxisStyle<double, double, string>(TitleText: "Metric", ShowGrid: false);

            var chartHeader = new List<IHtmlContent>();
            chartHeader.Add(h3("Plot Metrics over Trials"));
            writer.Write(div(chartHeader));


            Formatter.GetPreferredFormatterFor(typeof(Plotly.NET.GenericChart.GenericChart), "text/html").Format(chart, writer);

            // Works around issue with earlier versions of Plotly.NET - https://github.com/plotly/Plotly.NET/pull/305
            if (writer.ToString().EndsWith("</div    \r\n"))
            {
                writer.Write(">");
            }
        }

        private static void WriteTable(NotebookMonitor notebookMonitor, TextWriter writer)
        {
            var tableHeader = new List<IHtmlContent>();
            tableHeader.Add(h3("All Trials Table"));
            writer.Write(div(tableHeader));
            Formatter.GetPreferredFormatterFor(typeof(DataFrame), "text/html").Format(notebookMonitor.TrialData, writer);
        }
    }
}
