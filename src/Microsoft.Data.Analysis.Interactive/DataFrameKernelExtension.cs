// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Html;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.Formatting;
using static Microsoft.DotNet.Interactive.Formatting.PocketViewTags;

namespace Microsoft.Data.Analysis.Interactive
{
    public class DataFrameKernelExtension : IKernelExtension
    {
        public Task OnLoadAsync(Kernel kernel)
        {
            RegisterDataFrame();

            return Task.CompletedTask;
        }

        public static void RegisterDataFrame()
        {
            Formatter<DataFrame>.Register((df, writer) =>
            {
                const int MAX = 10000;
                const int SIZE = 10;

                var uniqueId = DateTime.Now.Ticks;

                var header = new List<IHtmlContent>
                {
                    th(i("index"))
                };
                header.AddRange(df.Columns.Select(c => (IHtmlContent)th(c.Name)));

                if (df.Rows.Count > SIZE)
                {
                    var maxMessage = df.Rows.Count > MAX ? $" (showing a max of {MAX} rows)" : string.Empty;
                    var title = h3[style: "text-align: center;"]($"DataFrame - {df.Rows.Count} rows {maxMessage}");

                    // table body
                    var maxRows = Math.Min(MAX, df.Rows.Count);
                    var rows = new List<List<IHtmlContent>>();
                    for (var index = 0; index < maxRows; index++)
                    {
                        var cells = new List<IHtmlContent>
                        {
                            td(i((index)))
                        };
                        foreach (var obj in df.Rows[index])
                        {
                            cells.Add(td(obj));
                        }
                        rows.Add(cells);
                    }

                    //navigator      
                    var footer = new List<IHtmlContent>();
                    BuildHideRowsScript(uniqueId);

                    var paginateScriptFirst = BuildHideRowsScript(uniqueId) + GotoPageIndex(uniqueId, 0) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptFirst]("⏮"));

                    var paginateScriptPrevTen = BuildHideRowsScript(uniqueId) + UpdatePageIndex(uniqueId, -10, (maxRows - 1) / SIZE) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptPrevTen]("⏪"));

                    var paginateScriptPrev = BuildHideRowsScript(uniqueId) + UpdatePageIndex(uniqueId, -1, (maxRows - 1) / SIZE) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptPrev]("◀️"));

                    footer.Add(b[style: "margin: 2px;"]("Page"));
                    footer.Add(b[id: $"page_{uniqueId}", style: "margin: 2px;"]("1"));

                    var paginateScriptNext = BuildHideRowsScript(uniqueId) + UpdatePageIndex(uniqueId, 1, (maxRows - 1) / SIZE) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptNext]("▶️"));

                    var paginateScriptNextTen = BuildHideRowsScript(uniqueId) + UpdatePageIndex(uniqueId, 10, (maxRows - 1) / SIZE) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptNextTen]("⏩"));

                    var paginateScriptLast = BuildHideRowsScript(uniqueId) + GotoPageIndex(uniqueId, (maxRows - 1) / SIZE) + BuildPageScript(uniqueId, SIZE);
                    footer.Add(button[style: "margin: 2px;", onclick: paginateScriptLast]("⏭️"));

                    //table
                    var t = table[id: $"table_{uniqueId}"](
                        caption(title),
                        thead(tr(header)),
                        tbody(rows.Select(r => tr[style: "display: none"](r))),
                        tfoot(tr(td[colspan: df.Columns.Count + 1, style: "text-align: center;"](footer)))
                    );
                    writer.Write(t);

                    //show first page
                    writer.Write($"<script>{BuildPageScript(uniqueId, SIZE)}</script>");
                }
                else
                {
                    var rows = new List<List<IHtmlContent>>();
                    for (var index = 0; index < df.Rows.Count; index++)
                    {
                        var cells = new List<IHtmlContent>
                        {
                            td(i((index)))
                        };
                        foreach (var obj in df.Rows[index])
                        {
                            cells.Add(td(obj));
                        }
                        rows.Add(cells);
                    }

                    //table
                    var t = table[id: $"table_{uniqueId}"](
                        thead(tr(header)),
                        tbody(rows.Select(r => tr(r)))
                    );
                    writer.Write(t);
                }
            }, "text/html");
        }

        private static string BuildHideRowsScript(long uniqueId)
        {
            var script = $"var allRows = document.querySelectorAll('#table_{uniqueId} tbody tr:nth-child(n)'); ";
            script += "for (let i = 0; i < allRows.length; i++) { allRows[i].style.display='none'; } ";
            return script;
        }

        private static string BuildPageScript(long uniqueId, int size)
        {
            var script = $"var page = parseInt(document.querySelector('#page_{uniqueId}').innerHTML) - 1; ";
            script += $"var pageRows = document.querySelectorAll(`#table_{uniqueId} tbody tr:nth-child(n + ${{page * {size} + 1 }})`); ";
            script += $"for (let j = 0; j < {size}; j++) {{ pageRows[j].style.display='table-row'; }} ";
            return script;
        }

        private static string GotoPageIndex(long uniqueId, long page)
        {
            var script = $"document.querySelector('#page_{uniqueId}').innerHTML = {page + 1}; ";
            return script;
        }

        private static string UpdatePageIndex(long uniqueId, int step, long maxPage)
        {
            var script = $"var page = parseInt(document.querySelector('#page_{uniqueId}').innerHTML) - 1; ";
            script += $"page = parseInt(page) + parseInt({step}); ";
            script += $"page = page < 0 ? 0 : page; ";
            script += $"page = page > {maxPage} ? {maxPage} : page; ";
            script += $"document.querySelector('#page_{uniqueId}').innerHTML = page + 1; ";
            return script;
        }
    }
}
