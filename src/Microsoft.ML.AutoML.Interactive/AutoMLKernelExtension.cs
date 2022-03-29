// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.Formatting;
using System;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using XPlot.Plotly;

namespace Microsoft.ML.AutoML.Interactive
{
    public class AutoMLKernelExtension : IKernelExtension
    {
#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
        public async Task OnLoadAsync(Kernel kernel)
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
        {
            Formatter.Register<NotebookMonitor>((monitor, writer) =>
            {

                var chart = Chart.Plot(
                            new Scatter()
                            {
                                x = monitor.CompletedTrials.Select(x => x.TrialSettings.TrialId),
                                y = monitor.CompletedTrials.Select(x => x.Metric),
                                mode = "markers",
                            }
                        );

                var layout = new Layout.Layout() { title = $"Plot metrics over trials." };
                chart.WithLayout(layout);
                chart.Width = 500;
                chart.Height = 500;
                chart.WithXTitle("Trial");
                chart.WithYTitle("Metric");
                chart.WithLegend(false);


                var scriptJs = chart.GetInlineJS().Replace("<script>", String.Empty).Replace("</script>", String.Empty);



                var trialHtml = String.Join(" ", monitor.CompletedTrials.Select(x => $@"
		<tr>
            <td>{x.TrialSettings.TrialId}</td>
            <td>{x.Metric}</td>
            <td>{x.TrialSettings.Pipeline}</td>
        </tr>
	"));



                var bestRun = monitor.BestTrial == null ? "" :
                $@"
	<h3>Best Run</h3>
	<p>
		Trial: {monitor.BestTrial.TrialSettings.TrialId} <br>
	</p>
	";

                var activeRunParam = monitor.ActiveTrial == null ? "" : JsonSerializer.Serialize(monitor.ActiveTrial.Parameter, new JsonSerializerOptions() { WriteIndented = true, });
                var activeRun = monitor.ActiveTrial == null ? "" :
                $@"
	<h3>Active Run</h3>
	<p>
		Trial: {monitor.ActiveTrial.TrialId} <br>
		Pipeline: {monitor.ActiveTrial.Pipeline}<br>
		Parameter: {activeRunParam}
	</p>
	";


                writer.Write($@"
<div>
	{bestRun}
	{activeRun}
</div>
<div style=""width: {chart.Width}px; height: {chart.Height}px;"" id=""{chart.Id}"">
</div>
<div>
	<table id=""example"" class=""display"" style=""width:100%"">
        <thead>
            <tr>
                <th>Trial</th>
                <th>Metric</th>
                <th>Pipeline</th>
            </tr>
        </thead>
        <tbody>

		{trialHtml}
            
		<tbody>
	</table>
</div>
<script type=""text/javascript"">
var renderPlotly = function() {{
    var xplotRequire = require.config({{context:'xplot-3.0.1',paths:{{plotly:'https://cdn.plot.ly/plotly-1.49.2.min'}}}}) || require;
    xplotRequire(['plotly'], function(Plotly) {{ 

{scriptJs}
        
}});
}};
// ensure `require` is available globally
if ((typeof(require) !==  typeof(Function)) || (typeof(require.config) !== typeof(Function))) {{
    let require_script = document.createElement('script');
    require_script.setAttribute('src', 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js');
    require_script.setAttribute('type', 'text/javascript');
    require_script.onload = function() {{
        renderPlotly();
    }};

    document.getElementsByTagName('head')[0].appendChild(require_script);
}}
else {{
    renderPlotly();
}}
</script>
");
            }, "text/html");
        }
    }
}
