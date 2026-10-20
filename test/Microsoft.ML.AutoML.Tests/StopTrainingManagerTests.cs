// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class StopTrainingManagerTests : BaseTestClass
    {
        public StopTrainingManagerTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void CancellationTokenStopTrainingManager_isStopTrainingRequested_test()
        {
            var cts = new CancellationTokenSource();
            var manager = new CancellationTokenStopTrainingManager(cts.Token, null);

            var isOnStopTrainingGetInvoked = false;
            manager.OnStopTraining += (o, e) =>
            {
                isOnStopTrainingGetInvoked = true;
            };

            Assert.False(manager.IsStopTrainingRequested());

            cts.Cancel();

            Assert.True(manager.IsStopTrainingRequested());
            Assert.True(isOnStopTrainingGetInvoked);
        }

        [Fact]
        public async Task TimeoutTrainingStopManager_isStopTrainingRequested_test()
        {
            var manager = new TimeoutTrainingStopManager(TimeSpan.FromSeconds(1), null);

            var isOnStopTrainingGetInvoked = false;
            manager.OnStopTraining += (o, e) =>
            {
                isOnStopTrainingGetInvoked = true;
            };

            var waitForCompleted = Task.Run(async () =>
            {
                while (!isOnStopTrainingGetInvoked)
                {
                    await Task.Delay(1000);
                }
            });

            await waitForCompleted;
            Assert.True(manager.IsStopTrainingRequested());
            Assert.True(isOnStopTrainingGetInvoked);
        }

        [Fact]
        public async Task AggregateTrainingStopManager_isStopTrainingRequested_test()
        {
            var cts = new CancellationTokenSource();
            var timeoutManager = new TimeoutTrainingStopManager(TimeSpan.FromSeconds(1), null);
            var cancellationManager = new CancellationTokenStopTrainingManager(cts.Token, null);
            var aggregationManager = new AggregateTrainingStopManager(null, timeoutManager, cancellationManager);

            var isOnStopTrainingGetInvoked = false;
            aggregationManager.OnStopTraining += (o, e) =>
            {
                isOnStopTrainingGetInvoked = true;
            };

            var waitForCompleted = Task.Run(async () =>
            {
                while (!isOnStopTrainingGetInvoked)
                {
                    await Task.Delay(1000);
                }
            });

            await waitForCompleted;
            Assert.True(aggregationManager.IsStopTrainingRequested());
            Assert.True(timeoutManager.IsStopTrainingRequested());
            Assert.False(cancellationManager.IsStopTrainingRequested());
            Assert.True(isOnStopTrainingGetInvoked);

            aggregationManager = new AggregateTrainingStopManager(null, new TimeoutTrainingStopManager(TimeSpan.FromSeconds(100), null), cancellationManager);
            isOnStopTrainingGetInvoked = false;
            aggregationManager.OnStopTraining += (o, e) =>
            {
                isOnStopTrainingGetInvoked = true;
            };
            cts.Cancel();

            await waitForCompleted;
            Assert.True(aggregationManager.IsStopTrainingRequested());
            Assert.True(cancellationManager.IsStopTrainingRequested());
            Assert.True(isOnStopTrainingGetInvoked);
        }
    }
}
