// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class MLContextManagerTests : BaseTestClass
    {
        public MLContextManagerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void DefaultMLContextManager_should_create_child_context_base_on_main_context()
        {
            var mainContext = new MLContext(10);
            mainContext.GpuDeviceId = 10;
            mainContext.FallbackToCpu = false;
            mainContext.TempFilePath = "temp";

            var contextManager = new DefaultMLContextManager(mainContext);

            var childContext = contextManager.CreateMLContext();
            Assert.Equal(10, childContext.GpuDeviceId);
            Assert.False(childContext.FallbackToCpu);
            Assert.Equal("temp", childContext.TempFilePath);

            Assert.Equal(10, ((IHostEnvironmentInternal)childContext.Model.GetEnvironment()).Seed);
        }

        [Fact]
        public void DefaultMLContextManager_main_context_should_replay_log_from_child_context()
        {
            var mainContext = new MLContext(10);
            var contextManager = new DefaultMLContextManager(mainContext);
            var childContext = contextManager.CreateMLContext();

            var channel = ((IChannelProvider)childContext).Start("childContext");

            var messages = new List<LoggingEventArgs>();
            mainContext.Log += (o, e) =>
            {
                messages.Add(e);
            };

            channel.Trace("trace");
            channel.Warning("warning");
            channel.Info("info");
            channel.Error("error");

            Assert.Equal(4, messages.Count());
        }
    }
}
