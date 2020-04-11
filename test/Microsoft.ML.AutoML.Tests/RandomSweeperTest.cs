using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.ML.AutoML.AutoPipeline.Sweeper;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.AutoPipeline.Test
{
    public class RandomSweeperTest : BaseTestClass
    {
        public RandomSweeperTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void RandomSweeper_should_be_enumable()
        {
            var param = new Dictionary<string, ParameterAttribute>();
            param.Add("key1", new ParameterAttribute(0, 10));
            param.Add("key2", new ParameterAttribute(0f, 10f));
            param.Add("key3", new ParameterAttribute(new string[] { "str1", "str2", "str3" }));

            var randomSweeper = new RandomSweeper(param, 10);
            var attempts = 0;
            foreach(var option in randomSweeper)
            {
                attempts += 1;
                option.Keys.Should().Contain("key1");
                option.Keys.Should().Contain("key2");
                option.Keys.Should().Contain("key3");

                (option["key1"] as int?)?
                    .Should()
                    .BeLessOrEqualTo(10)
                    .And
                    .BeGreaterOrEqualTo(0);
                (option["key2"] as float?)?
                    .Should()
                    .BeLessOrEqualTo(10f)
                    .And
                    .BeGreaterOrEqualTo(0f);
                (option["key3"] as string)
                    .Should()
                    .BeOneOf(new string[] { "str1", "str2", "str3" });
            }

            attempts.Should().Be(11);
        }
    }
}

