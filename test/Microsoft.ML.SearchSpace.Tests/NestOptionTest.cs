// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.ML.SearchSpace.Option;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class NestOptionTest : TestBase
    {
        public NestOptionTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void NestOption_sampling_from_uniform_space_test()
        {
            var nestOption = new NestOption();
            nestOption.Add("choice", new ChoiceOption("a", "b", "c"));
            nestOption.Add("int", new UniformIntOption(0, 1));
            var anotherNestOption = new NestOption();
            anotherNestOption["choice"] = new ChoiceOption("d", "e");
            anotherNestOption["int"] = new UniformIntOption(2, 3);
            nestOption["nestOption"] = anotherNestOption;

            nestOption.FeatureSpaceDim.Should().Be(4);
            var parameter = nestOption.SampleFromFeatureSpace(new double[] { 0, 0, 0, 0 });
            parameter["nestOption"]["choice"].AsType<string>().Should().Be("d");
            parameter["nestOption"]["int"].AsType<int>().Should().Be(2);
            parameter["choice"].AsType<string>().Should().Be("a");
            parameter["int"].AsType<int>().Should().Be(0);

            parameter = nestOption.SampleFromFeatureSpace(new double[] { 1, 1, 1, 1 });
            parameter["nestOption"]["choice"].AsType<string>().Should().Be("e");
            parameter["nestOption"]["int"].AsType<int>().Should().Be(3);
            parameter["choice"].AsType<string>().Should().Be("c");
            parameter["int"].AsType<int>().Should().Be(1);
        }

        [Fact]
        public void NestOption_mapping_to_uniform_space_test()
        {
            var nestOption = new NestOption();
            nestOption.Add("choice", new ChoiceOption("a", "b", "c"));
            nestOption.Add("int", new UniformIntOption(0, 1));

            var parameter = Parameter.CreateNestedParameter();
            parameter["choice"] = Parameter.FromString("a");
            parameter["int"] = Parameter.FromInt(0);
            nestOption.MappingToFeatureSpace(parameter).Should().Equal(0, 0);
        }

        [Fact]
        public void NestOption_mapping_order_test()
        {
            // each dimension in uniform space should be mapping to the options under nest option in a certain (key ascending) order.
            var nestOption = new NestOption();
            nestOption["a"] = new UniformIntOption(0, 1);
            nestOption["b"] = new UniformIntOption(1, 2);
            nestOption["c"] = new UniformIntOption(2, 3);

            // changing of the first dimension should be reflected in option "a"
            var parameter = nestOption.SampleFromFeatureSpace(new double[] { 0, 0.5, 0.5 });
            parameter["a"].AsType<int>().Should().Be(0);
            parameter = nestOption.SampleFromFeatureSpace(new double[] { 1, 0.5, 0.5 });
            parameter["a"].AsType<int>().Should().Be(1);

            nestOption.Remove("a");

            // the first dimension should be option "b"
            parameter = nestOption.SampleFromFeatureSpace(new double[] { 0, 0.5 });
            parameter["b"].AsType<int>().Should().Be(1);
            parameter = nestOption.SampleFromFeatureSpace(new double[] { 1, 0.5 });
            parameter["b"].AsType<int>().Should().Be(2);
        }
    }
}
