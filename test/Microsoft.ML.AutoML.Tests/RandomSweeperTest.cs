using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.ML.AutoPipeline;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoPipeline.Test
{
    public class RandomSweeperTest : BaseTestClass
    {
        public RandomSweeperTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void RandomSweeper_should_be_enumable()
        {
            var param = new List<ParameterAttribute>();
            param.Add(new ParameterAttribute("key1", 0, 10));
            param.Add(new ParameterAttribute("key2", 0f, 10f));
            param.Add(new ParameterAttribute("key3", new string[] { "str1", "str2", "str3" }));
            var context = new MLContext();

            var randomSweeper = new RandomSweeper(context, param.Select(x => x.ValueGenerator).ToArray(), 10);
            var attempts = 0;
            foreach(var option in randomSweeper)
            {
                attempts += 1;
                
                (option["key1"].RawValue as int?)?
                    .Should()
                    .BeLessOrEqualTo(10)
                    .And
                    .BeGreaterOrEqualTo(0);
                (option["key2"].RawValue as float?)?
                    .Should()
                    .BeLessOrEqualTo(10f)
                    .And
                    .BeGreaterOrEqualTo(0f);
                (option["key3"].RawValue as string)
                    .Should()
                    .BeOneOf(new string[] { "str1", "str2", "str3" });
            }

            attempts.Should().Be(10);
        }
    }
}

