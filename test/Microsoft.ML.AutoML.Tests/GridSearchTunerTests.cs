// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Tests
{
    public class GridSearchTunerTests : BaseTestClass
    {
        public GridSearchTunerTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void GridSearchTuner_should_search_entire_naive_search_space()
        {
            var defaultStep = 10;
            var searchSpace = new SearchSpace<NaiveSearchSpace>();
            var tuner = new GridSearchTuner(searchSpace, defaultStep);
            var parameters = new List<Parameter>();
            var steps = 1;
            foreach (var step in searchSpace.Step)
            {
                steps *= step ?? defaultStep;
            }

            foreach (var i in Enumerable.Range(0, steps))
            {
                var settings = new TrialSettings();
                parameters.Add(tuner.Propose(settings));
            }

            parameters.Distinct().Count().Should().Be(steps);
        }

        private class NaiveSearchSpace
        {
            [Range((int)-10, 10, 0, false)]
            public int IntProperty { get; set; }

            [Range(-10f, 10, 0, false)]
            public float SingleProperty { get; set; }

            [Choice("a", "b", "c")]
            public string TextProperty { get; set; }

            [BooleanChoice]
            public bool BooleanProperty { get; set; }
        }
    }
}
