// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TextFileSampleTests : BaseTestClass
    {
        public TextFileSampleTests(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletesCanParseLargeRandomStream(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(CanParseLargeRandomStream);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("CanParseLargeRandomStream test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void CanParseLargeRandomStream()
        {
            using (var stream = new MemoryStream())
            {
                const int numRows = 100000;
                const int rowSize = 100;

                var eol = Encoding.UTF8.GetBytes("\r\n");

                for (var i = 0; i < numRows; i++)
                {
                    var row = new byte[rowSize];
                    AutoMlUtils.Random.Value.NextBytes(row);

                    // ensure byte array has no 0s, so text file sampler doesn't
                    // think file is encoded with UTF-16 or UTF-32 without a BOM
                    for (var k = 0; k < row.Length; k++)
                    {
                        if(row[k] == 0)
                        {
                            row[k] = 1;
                        }
                    }
                    stream.Write(row, 0, rowSize);
                    stream.Write(eol, 0, eol.Length);
                }

                stream.Seek(0, SeekOrigin.Begin);

                var sample = TextFileSample.CreateFromFullStream(stream);
                Assert.NotNull(sample);
                Assert.True(sample.FullFileSize > 0);
            }
        }
    }
}
