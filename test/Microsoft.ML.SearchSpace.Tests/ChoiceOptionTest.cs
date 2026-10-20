// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.SearchSpace.Option;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class ChoiceOptionTest : TestBase
    {
        public ChoiceOptionTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void Choice_option_sampling_from_uniform_space_test()
        {
            var option = new ChoiceOption("a", "b", "c");
            Assert.Equal("a", option.SampleFromFeatureSpace(new[] { 0.0 }).AsType<string>());
            Assert.Equal("a", option.SampleFromFeatureSpace(new[] { 0.33 }).AsType<string>());
            Assert.Equal("b", option.SampleFromFeatureSpace(new[] { 0.34 }).AsType<string>());
            Assert.Equal("b", option.SampleFromFeatureSpace(new[] { 0.66 }).AsType<string>());
            Assert.Equal("c", option.SampleFromFeatureSpace(new[] { 0.67 }).AsType<string>());
            Assert.Equal("c", option.SampleFromFeatureSpace(new[] { 0.97 }).AsType<string>());
            Assert.Equal("c", option.SampleFromFeatureSpace(new[] { 0.999999999 }).AsType<string>());
            Assert.Equal("c", option.SampleFromFeatureSpace(new[] { 1.0 }).AsType<string>());
        }

        [Fact]
        public void Choice_option_with_one_value_sampling_from_uniform_space_test()
        {
            var option = new ChoiceOption("a");
            Assert.Equal("a", option.SampleFromFeatureSpace(new[] { 0.0 }).AsType<string>());
            Assert.Equal("a", option.SampleFromFeatureSpace(new[] { 0.99 }).AsType<string>());
        }

        [Fact]
        public void Choice_option_mapping_to_uniform_space_test()
        {
            var option = new ChoiceOption("a", "b", "c");
            Assert.Equal(0, option.MappingToFeatureSpace(Parameter.FromString("a"))[0], tolerance: 1e-5);
            Assert.Equal(0.333333, option.MappingToFeatureSpace(Parameter.FromString("b"))[0], tolerance: 1e-5);
            Assert.Equal(0.666666, option.MappingToFeatureSpace(Parameter.FromString("c"))[0], tolerance: 1e-5);

            option = new ChoiceOption("a", "b", "c", "d");
            var parameter = option.SampleFromFeatureSpace(new[] { 0.5 });
            Assert.Equal("c", parameter.AsType<string>());
            Assert.Equal(new[] { 0.5 }, option.MappingToFeatureSpace(parameter));
        }

        [Fact]
        public void Choice_option_dimension_should_be_0_if_contains_only_one_value()
        {
            var option = new ChoiceOption("b");
            Assert.Equal(0, option.FeatureSpaceDim);
            Assert.Empty(option.Default);
            Assert.Equal("b", option.SampleFromFeatureSpace(new double[0]).AsType<string>());
            Assert.Empty(option.MappingToFeatureSpace(Parameter.FromString("b")));
        }

        [Fact]
        public void Choice_int_option_test()
        {
            var searchSpace = new SearchSpace();
            searchSpace["choice"] = new ChoiceOption(1, 2, 3);
            var parameter = searchSpace.SampleFromFeatureSpace(new[] { 0.5 });
            Assert.Equal(2, parameter["choice"].AsType<int>());
        }
    }
}
