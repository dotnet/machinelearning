// <copyright file="ChoiceOptionTest.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;
using Xunit;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tests
{
    public class ChoiceOptionTest
    {
        [Fact]
        public void Choice_option_sampling_from_uniform_space_test()
        {
            var option = new ChoiceOption("a", "b", "c");
            option.SampleFromFeatureSpace(new[] { 0.0 }).AsType<string>().Should().Be("a");
            option.SampleFromFeatureSpace(new[] { 0.33 }).AsType<string>().Should().Be("a");
            option.SampleFromFeatureSpace(new[] { 0.34 }).AsType<string>().Should().Be("b");
            option.SampleFromFeatureSpace(new[] { 0.66 }).AsType<string>().Should().Be("b");
            option.SampleFromFeatureSpace(new[] { 0.67 }).AsType<string>().Should().Be("c");
            option.SampleFromFeatureSpace(new[] { 0.97 }).AsType<string>().Should().Be("c");
        }

        [Fact]
        public void Choice_option_with_one_value_sampling_from_uniform_space_test()
        {
            var option = new ChoiceOption("a");
            option.SampleFromFeatureSpace(new[] { 0.0 }).AsType<string>().Should().Be("a");
            option.SampleFromFeatureSpace(new[] { 0.99 }).AsType<string>().Should().Be("a");
        }

        [Fact]
        public void Choice_option_mapping_to_uniform_space_test()
        {
            var option = new ChoiceOption("a", "b", "c");
            option.MappingToFeatureSpace(new Parameter("a"))[0].Should().BeApproximately(0, 1e-5);
            option.MappingToFeatureSpace(new Parameter("b"))[0].Should().BeApproximately(0.333333, 1e-5);
            option.MappingToFeatureSpace(new Parameter("c"))[0].Should().BeApproximately(0.666666, 1e-5);
        }

        [Fact]
        public void Choice_option_with_one_value_mapping_to_uniform_space_test()
        {
            var option = new ChoiceOption("a");
            option.MappingToFeatureSpace(new Parameter("a"))[0].Should().BeApproximately(0, 1e-5);
        }
    }
}
