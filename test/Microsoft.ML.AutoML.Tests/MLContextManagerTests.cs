// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
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
            childContext.GpuDeviceId.Should().Be(10);
            childContext.FallbackToCpu.Should().BeFalse();
            childContext.TempFilePath.Should().Be("temp");

            ((IHostEnvironmentInternal)childContext.Model.GetEnvironment()).Seed.Should().Be(10);
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

            messages.Count().Should().Be(4);
        }
    }
}
