// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed partial class CmdIndenterTests : BaseTestBaseline
    {
        public CmdIndenterTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Cmd Parsing")]
        public void TestCmdIndenter()
        {
            Run();
        }

        private string GetResText(string resName)
        {
            var stream = typeof(CmdIndenterTests).Assembly.GetManifestResourceStream(resName);
            if (stream == null)
                return string.Format("<couldn't read {0}>", resName);

            using (var reader = new StreamReader(stream))
            {
                return reader.ReadToEnd();
            }
        }

        internal void Run()
        {
            string text = GetResText("Microsoft.ML.Runtime.RunTests.CmdLine.IndenterTestInput.txt");

            string outName = "CmdIndenterOutput.txt";
            string outPath = DeleteOutputPath("CmdLine", outName);

            using (var writer = File.CreateText(outPath))
            {
                var wrt = IndentingTextWriter.Wrap(writer);

                // Individual scripts are separated by $
                int count = 0;
                int ichLim = 0;
                int lineLim = 1;
                while (ichLim < text.Length)
                {
                    int ichMin = ichLim;
                    int lineMin = lineLim;

                    while (ichLim < text.Length && text[ichLim] != '$')
                    {
                        if (text[ichLim] == '\n')
                            lineLim++;
                        ichLim++;
                    }

                    while (ichMin < ichLim && char.IsWhiteSpace(text[ichMin]))
                    {
                        if (text[ichMin] == '\n')
                            lineMin++;
                        ichMin++;
                    }

                    if (ichMin < ichLim)
                    {
                        // Process the script.
                        count++;
                        string scriptName = string.Format("Script {0}, lines {1} to {2}", count, lineMin, lineLim);
                        wrt.WriteLine("===== Start {0} =====", scriptName);
                        string input = text.Substring(ichMin, ichLim - ichMin);
                        try
                        {
                            wrt.WriteLine(CmdIndenter.GetIndentedCommandLine(input));
                        }
                        catch (Exception ex)
                        {
                            if (!ex.IsMarked())
                                wrt.WriteLine("Unknown Exception!");
                            wrt.WriteLine("Exception: {0}", ex.Message);
                        }

                        wrt.WriteLine("===== End {0} =====", scriptName);
                    }

                    // Skip the $
                    ichLim++;
                }
            }

            CheckEquality("CmdLine", outName);

            Done();
        }
    }
}
