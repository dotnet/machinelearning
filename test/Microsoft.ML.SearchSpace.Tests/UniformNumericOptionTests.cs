// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using FluentAssertions;
using Microsoft.ML.SearchSpace.Option;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class UniformNumericOptionTests : TestBase
    {
        public UniformNumericOptionTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void Uniform_integer_option_sampling_from_uniform_space_test()
        {
            var option = new UniformIntOption(0, 100);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => i * 0.1);
            var sampleOutputs = sampleInputs.Select(i => option.SampleFromFeatureSpace(new[] { i }));

            sampleOutputs.Select(x => x.AsType<int>()).Should().Equal(0, 10, 20, 30, 40, 50, 60, 70, 80, 90);
        }

        [Fact]
        public void Uniform_log_integer_option_sampling_from_uniform_space_test()
        {
            var option = new UniformIntOption(1, 1024, true);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => i * 0.1);
            var sampleOutputs = sampleInputs.Select(i => option.SampleFromFeatureSpace(new[] { i }));

            sampleOutputs.Select(x => x.AsType<int>()).Should().Equal(1, 2, 4, 8, 16, 32, 64, 128, 256, 512);
        }

        [Fact]
        public void Uniform_integer_option_mapping_to_uniform_space_test()
        {
            var option = new UniformIntOption(0, 100);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => Parameter.FromInt(i * 10));
            var sampleOutputs = sampleInputs.Select(i => option.MappingToFeatureSpace(i)[0]);

            sampleOutputs.Should().Equal(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        }

        [Fact]
        public void Uniform_integer_option_mapping_to_uniform_space_test_2()
        {
            var option = new UniformIntOption(20, 1024, defaultValue: 20, logBase: true);
            option.SampleFromFeatureSpace(new double[] { 0.0 }).AsType<int>().Should().Be(20);
        }

        [Fact]
        public void Uniform_log_integer_option_mapping_to_uniform_space_test()
        {
            var option = new UniformIntOption(1, 1024, true);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => Parameter.FromInt(Convert.ToInt32(Math.Pow(2, i))));
            var sampleOutputs = sampleInputs.Select(i => option.MappingToFeatureSpace(i)[0]).ToArray();

            foreach (var i in Enumerable.Range(0, 10))
            {
                sampleOutputs[i].Should().BeApproximately(0.1 * i, 0.0001);
            }
        }

        [Fact]
        public void Uniform_double_option_sampling_from_uniform_space_test()
        {
            var option = new UniformDoubleOption(0, 100);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => i * 0.1);
            var sampleOutputs = sampleInputs.Select(i => option.SampleFromFeatureSpace(new[] { i }));

            sampleOutputs.Select((x, i) => (x.AsType<double>(), i * 10))
                         .All((x) => Math.Abs(x.Item1 - x.Item2) < 1e-5)
                         .Should().BeTrue();
        }

        [Fact]
        public void Uniform_log_double_option_sampling_from_uniform_space_test()
        {
            var option = new UniformDoubleOption(1, 1024, true);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => i * 0.1);
            var sampleOutputs = sampleInputs.Select(i => option.SampleFromFeatureSpace(new[] { i }));

            sampleOutputs.Select((x, i) => (x.AsType<double>(), Math.Pow(2, i)))
                         .All((x) => Math.Abs(x.Item1 - x.Item2) < 1e-5)
                         .Should().BeTrue();
        }

        [Fact]
        public void Uniform_double_option_mapping_to_uniform_space_test()
        {
            var option = new UniformDoubleOption(0, 100);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => Parameter.FromDouble(i * 10.0));
            var sampleOutputs = sampleInputs.Select(i => option.MappingToFeatureSpace(i)[0]);

            sampleOutputs.Should().Equal(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        }

        [Fact]
        public void Uniform_log_double_option_mapping_to_uniform_space_test()
        {
            var option = new UniformDoubleOption(1, 1024, true);

            var sampleInputs = Enumerable.Range(0, 10).Select(i => Parameter.FromDouble(Math.Pow(2, i)));
            var sampleOutputs = sampleInputs.Select(i => option.MappingToFeatureSpace(i)).ToArray();

            foreach (var i in Enumerable.Range(0, 10))
            {
                sampleOutputs[i][0].Should().BeApproximately(0.1 * i, 1e-5);
            }
        }

        [Fact]
        public void Uniform_log_double_option_round_up_test()
        {
            var option = new UniformDoubleOption(2e-10, 1, defaultValue: 2e-10, logBase: true);
            option.Default.Should().Equal(0);
            option.SampleFromFeatureSpace(option.Default).AsType<double>().Should().Be(2e-10);
            option.SampleFromFeatureSpace(new[] { 1.0 }).AsType<double>().Should().Be(1.0);
        }
    }
}
