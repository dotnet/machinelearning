// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public class TestHosts
    {
        /// <summary>
        /// Tests that MLContext's Log event intercepts messages properly.
        /// </summary>
        [Fact]
        public void LogEventProcessesMessages()
        {
            var messages = new List<string>();

            var env = new MLContext();
            env.Log += (sender, e) => messages.Add(e.Message);

            // create a dummy text reader to trigger log messages
            env.Data.CreateTextLoader(new TextLoader.Options { Columns = new[] { new TextLoader.Column("TestColumn", DataKind.Single, 0) } });

            Assert.True(messages.Count > 0);
        }
    }
}
