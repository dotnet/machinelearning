using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.ML.AutoPipeline;
using Microsoft.ML.Sweeper;
using Xunit;

namespace Microsoft.ML.AutoPipeline.Test
{
    public class OptionBuilderTest
    {
        [Fact]
        public void OptionBuilder_should_create_default_option()
        {
            var builder = new TestOptionBuilder();
            var option = builder.CreateDefaultOption();
            option.LongOption.Should().Equals(10);
            option.FloatOption.Should().Equals(1f);
            option.StringOption.Should().Equals("str");
        }


        [Fact]
        public void OptionBuilder_should_build_optoin_from_parameter_set()
        {
            var builder = new TestOptionBuilder();
            var input = new List<Microsoft.ML.IParameterValue>()
            {
                new Microsoft.ML.Sweeper.LongParameterValue("LongOption", 2),
                new Microsoft.ML.Sweeper.FloatParameterValue("FloatOption", 2f),
                new Microsoft.ML.Sweeper.StringParameterValue("StringOption", "2"),
            };

            var paramSet = new Microsoft.ML.ParameterSet(input);

            var option = builder.BuildOption(paramSet);
            option.LongOption.Should().Equals(2);
            option.FloatOption.Should().Equals(2f);
            option.StringOption.Should().Equals("2");
        }

        [Fact]
        public void OptionBuilder_should_work_with_random_sweeper()
        {
            var context = new MLContext();
            var builder = new TestOptionBuilder();
            var maximum = 10;
            var randomSweeper = new RandomSweeper(context, builder.ValueGenerators, maximum);

            foreach (var sweeperOutput in randomSweeper)
            {
                maximum -= 1;
                var option = builder.BuildOption(sweeperOutput);
                option.LongOption
                      .Should()
                      .BeLessOrEqualTo(100)
                      .And
                      .BeGreaterOrEqualTo(0);

                option.FloatOption
                      .Should()
                      .BeLessOrEqualTo(100f)
                      .And
                      .BeGreaterOrEqualTo(0f);

                option.StringOption
                      .Should()
                      .BeOneOf(new string[] { "str1", "str2", "str3", "str4" });

                maximum.Should().BeGreaterThan(-2);
            }
        }


        private class TestOption
        {
            public long LongOption = 1;

            public float FloatOption = 1f;

            public string StringOption = string.Empty;
        }

        private class TestOptionBuilder : OptionBuilder<TestOption>
        {
            [Parameter("LongOption", 0, 100)]
            public long LongOption = 2;

            [Parameter("FloatOption", 0f, 100f)]
            public float FloatOption;

            [Parameter("StringOption", new string[] { "str1", "str2", "str3", "str4" })]
            public string StringOption = "str";
        }
    }


}
