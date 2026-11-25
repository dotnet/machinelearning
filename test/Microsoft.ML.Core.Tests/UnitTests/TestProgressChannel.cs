// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests;

public sealed class TestProgressChannel : BaseTestClass
{
    public TestProgressChannel(ITestOutputHelper output) : base(output)
    {
    }

    [Fact]
    public void SiblingChannelsKeepIndependentProgressAndNestingOrder()
    {
        var env = new MLContext(1);
        var tracker = new ProgressReporting.ProgressTracker(env);
        using var root = new ProgressReporting.ProgressChannel(env, tracker, "Root");
        using var first = root.StartProgressChannel("First");
        first.SetHeader(new ProgressHeader(new[] { "First" }, unitNames: null), entry => entry.SetMetric(0, 1));
        using var nested = first.StartProgressChannel("Nested");
        nested.SetHeader(new ProgressHeader(new[] { "Nested" }, unitNames: null), entry => entry.SetMetric(0, 3));
        using var second = root.StartProgressChannel("Second");
        second.SetHeader(new ProgressHeader(new[] { "Second" }, unitNames: null), entry => entry.SetMetric(0, 2));

        var progress = root.GetProgress();
        Assert.Equal(3, progress.Header.MetricNames.Count);
        Assert.Contains("First", progress.Header.MetricNames);
        Assert.Contains("Second", progress.Header.MetricNames);
        Assert.Equal("Nested", progress.Header.MetricNames.Last());
        Assert.Equal(new double?[] { 1, 2, 3 }, progress.Metrics.OrderBy(value => value));

        second.Dispose();
        progress = root.GetProgress();
        Assert.Equal(new[] { "First", "Nested" }, progress.Header.MetricNames);
        Assert.Equal(new double?[] { 1, 3 }, progress.Metrics);

        nested.Dispose();
        progress = root.GetProgress();
        Assert.Equal("First", Assert.Single(progress.Header.MetricNames));
        Assert.Equal(1, Assert.Single(progress.Metrics));
    }
}
