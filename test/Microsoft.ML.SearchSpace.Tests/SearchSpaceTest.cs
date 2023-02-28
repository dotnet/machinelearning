// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers.Text;
using System.Buffers;
using System.Text.Json;
using System.Text.Json.Serialization;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML.SearchSpace.Option;
using Microsoft.ML.SearchSpace.Tuner;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class SearchSpaceTest : TestBase
    {
        private readonly JsonSerializerOptions _settings = new JsonSerializerOptions()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            NumberHandling = JsonNumberHandling.Strict,
        };

        public SearchSpaceTest(ITestOutputHelper output)
            : base(output)
        {
            _settings.Converters.Add(new DoubleConverter());
            _settings.Converters.Add(new SingleConverter());
        }

        [Fact]
        public void SearchSpace_sample_from_feature_space_test()
        {
            var ss = new SearchSpace<BasicSearchSpace>();
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0 });

            param.ChoiceStr.Should().Be("a");
            param.UniformDouble.Should().Be(-1000);
            param.UniformFloat.Should().Be(-1000);
            param.UniformInt.Should().Be(-1000);
            param.ChoiceInt.Should().Be(1);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
            param.ChoiceStr.Should().Be("c");
            param.UniformDouble.Should().Be(0);
            param.UniformFloat.Should().Be(0);
            param.ChoiceInt.Should().Be(3);
            param.UniformInt.Should().Be(0);
        }

        [Fact]
        public void SearchSpace_mapping_to_feature_space_test()
        {
            var ss = new SearchSpace<BasicSearchSpace>();
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0 });
            var features = ss.MappingToFeatureSpace(param);
            features.Should().Equal(0, 0, 0, 0, 0, 0, 0);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5 });
            features = ss.MappingToFeatureSpace(param);
            features.Should().Equal(0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5);
        }

        [Fact]
        public void Nest_search_space_mapping_to_feature_space_test()
        {
            var ss = new SearchSpace<NestSearchSpace>();
            ss.FeatureSpaceDim.Should().Be(9);
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0, 0, 0 });
            var features = ss.MappingToFeatureSpace(param);
            features.Should().Equal(0, 0, 0, 0, 0, 0, 0, 0, 0);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5 });
            features = ss.MappingToFeatureSpace(param);
            features.Should().Equal(0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5);
        }

        [Fact]
        public void Nest_searchSpace_sample_from_feature_space_test()
        {
            var option = new NestSearchSpace()
            {
                BasicSS = new BasicSearchSpace()
                {
                    DefaultSearchSpace = new DefaultSearchSpace()
                    {
                        Strings = new[] { "B", "C", "D" },
                    },
                },
            };
            var ss = new SearchSpace<NestSearchSpace>(option);

            ss.FeatureSpaceDim.Should().Be(9);
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0, 0, 0 });

            param.UniformDouble.Should().Be(-1000);
            param.UniformFloat.Should().Be(-1000);
            param.BasicSS.UniformInt.Should().Be(-1000);
            param.BasicSS.UniformDouble.Should().Be(-1000);
            param.BasicSS.UniformFloat.Should().Be(-1000);
            param.BasicSS.ChoiceStr.Should().Be("a");
            param.BasicSS.DefaultSearchSpace.Strings.Should().BeEquivalentTo("B", "C", "D");
            param.BasicSS.ChoiceBoolean.Should().BeTrue();
            param.BasicSS.JTokenType.Should().Be(JsonTokenType.None);
            param.BasicSS.ChoiceInt.Should().Be(1);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });

            param.UniformDouble.Should().Be(0);
            param.UniformFloat.Should().Be(0);
            param.BasicSS.UniformInt.Should().Be(0);
            param.BasicSS.UniformDouble.Should().Be(0);
            param.BasicSS.UniformFloat.Should().Be(0);
            param.BasicSS.ChoiceStr.Should().Be("c");
            param.BasicSS.DefaultSearchSpace.Strings.Should().BeEquivalentTo("B", "C", "D");
            param.BasicSS.ChoiceInt.Should().Be(3);
            param.BasicSS.ChoiceBoolean.Should().BeFalse();
            param.BasicSS.JTokenType.Should().Be(JsonTokenType.StartArray);
        }

        [Fact]
        public void Search_space_add_option_test()
        {
            var ss = new SearchSpace();
            ss.FeatureSpaceDim.Should().Be(0);

            ss.Add("A", new UniformIntOption(-1000, 1000));
            ss.FeatureSpaceDim.Should().Be(1);

            var param = ss.SampleFromFeatureSpace(new[] { 0.5 });
            param["A"].AsType<int>().Should().Be(0);
        }

        [Fact]
        public void Search_space_remove_option_test()
        {
            var option = new BasicSearchSpace();
            var ss = new SearchSpace<BasicSearchSpace>(option);
            ss.FeatureSpaceDim.Should().Be(7);

            ss.Remove("UniformInt").Should().BeTrue();
            ss.FeatureSpaceDim.Should().Be(6);
            ss.Keys.Should().BeEquivalentTo("ChoiceStr", "UniformDouble", "UniformFloat", "ChoiceBoolean", "JTokenType", "ChoiceInt");

            var parameter = ss.SampleFromFeatureSpace(new double[] { 0, 0, 0, 0, 0, 0 });

            parameter.DefaultSearchSpace.Strings.Should().BeEquivalentTo("A", "B", "C");
            parameter.DefaultSearchSpace.String.Should().BeNullOrEmpty();
            parameter.ChoiceStr.Should().Be("a");
            parameter.ChoiceBoolean.Should().BeTrue();
            parameter.JTokenType.Should().Be(JsonTokenType.None);
            parameter.ChoiceInt.Should().Be(1);
        }

        [Fact]
        public void Search_space_default_value_test()
        {
            var ss = new SearchSpace<NestSearchSpace>();
            var defaultTuner = new DefaultValueTuner(ss);
            var param = defaultTuner.Propose().AsType<NestSearchSpace>();

            param.UniformDouble.Should().Be(0);
            param.UniformFloat.Should().Be(0);
            param.BasicSS.UniformInt.Should().Be(0);
            param.BasicSS.UniformDouble.Should().Be(0);
            param.BasicSS.UniformFloat.Should().Be(0);
            param.BasicSS.ChoiceStr.Should().Be("a");
            param.BasicSS.ChoiceBoolean.Should().BeTrue();
            param.BasicSS.JTokenType.Should().Be(JsonTokenType.Null);
            param.BasicSS.ChoiceInt.Should().Be(1);
        }

        [Fact]
        public void Search_space_default_search_space_test()
        {
            var defaultSearchSpace = new DefaultSearchSpace()
            {
                String = "String",
                Int = 10,
                Bool = true,
                JTokenType = JsonTokenType.Null,
            };

            var ss = new SearchSpace<DefaultSearchSpace>(defaultSearchSpace);
            var param = ss.SampleFromFeatureSpace(new double[0]);

            param.Int.Should().Be(10);
            param.Float.Should().Be(0f);
            param.Double.Should().Be(0);
            param.Bool.Should().BeTrue();
            param.String.Should().Be("String");
            param.Strings.Should().BeEquivalentTo("A", "B", "C");
            param.JTokenType.Should().Be(JsonTokenType.Null);
            param.NullString.Should().BeNull();
            ss.FeatureSpaceDim.Should().Be(0);
            ss.MappingToFeatureSpace(param).Should().HaveCount(0);
        }

        [Fact]
        public void Search_space_hash_code_test()
        {
            var ss = new SearchSpace<BasicSearchSpace>();
            ss.GetHashCode().Should().Be(2005165306);

            ss.Remove("UniformInt");
            ss.GetHashCode().Should().Be(125205970);
        }

        [Fact]
        public void SearchSpace_sampling_from_uniform_space_test()
        {
            var searchSpace = new Option.SearchSpace();
            searchSpace.Add("choice", new ChoiceOption("a", "b", "c"));
            searchSpace.Add("int", new UniformIntOption(0, 1));
            var anotherNestOption = new Option.SearchSpace();
            anotherNestOption["choice"] = new ChoiceOption("d", "e");
            anotherNestOption["int"] = new UniformIntOption(2, 3);
            searchSpace["nestOption"] = anotherNestOption;

            searchSpace.FeatureSpaceDim.Should().Be(4);
            var parameter = searchSpace.SampleFromFeatureSpace(new double[] { 0, 0, 0, 0 });
            parameter["nestOption"]["choice"].AsType<string>().Should().Be("d");
            parameter["nestOption"]["int"].AsType<int>().Should().Be(2);
            parameter["choice"].AsType<string>().Should().Be("a");
            parameter["int"].AsType<int>().Should().Be(0);

            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 1, 1, 1 });
            parameter["nestOption"]["choice"].AsType<string>().Should().Be("e");
            parameter["nestOption"]["int"].AsType<int>().Should().Be(3);
            parameter["choice"].AsType<string>().Should().Be("c");
            parameter["int"].AsType<int>().Should().Be(1);
        }

        [Fact]
        public void SearchSpace_mapping_to_uniform_space_test()
        {
            var searchSpace = new SearchSpace();
            searchSpace.Add("choice", new ChoiceOption("a", "b", "c"));
            searchSpace.Add("int", new UniformIntOption(0, 1));

            var parameter = Parameter.CreateNestedParameter();
            parameter["choice"] = Parameter.FromString("a");
            parameter["int"] = Parameter.FromInt(0);
            searchSpace.MappingToFeatureSpace(parameter).Should().Equal(0, 0);
        }

        [Fact]
        public void SearchSpace_mapping_order_test()
        {
            // each dimension in uniform space should be mapping to the options under nest option in a certain (key ascending) order.
            var searchSpace = new SearchSpace();
            searchSpace["a"] = new UniformIntOption(0, 1);
            searchSpace["b"] = new UniformIntOption(1, 2);
            searchSpace["c"] = new UniformIntOption(2, 3);

            // changing of the first dimension should be reflected in option "a"
            var parameter = searchSpace.SampleFromFeatureSpace(new double[] { 0, 0.5, 0.5 });
            parameter["a"].AsType<int>().Should().Be(0);
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 0.5, 0.5 });
            parameter["a"].AsType<int>().Should().Be(1);

            searchSpace.Remove("a");

            // the first dimension should be option "b"
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 0, 0.5 });
            parameter["b"].AsType<int>().Should().Be(1);
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 0.5 });
            parameter["b"].AsType<int>().Should().Be(2);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void Trainer_default_search_space_test()
        {
            CreateAndVerifyDefaultSearchSpace<SgdNonCalibratedTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SgdCalibratedTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SdcaLogisticRegressionBinaryTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SdcaMaximumEntropyMulticlassTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SdcaNonCalibratedBinaryTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SdcaNonCalibratedMulticlassTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<SdcaRegressionTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<AveragedPerceptronTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<OnlineGradientDescentTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<LbfgsLogisticRegressionBinaryTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<LbfgsMaximumEntropyMulticlassTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<LbfgsPoissonRegressionTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<LinearSvmTrainer.Options>();
            CreateAndVerifyDefaultSearchSpace<LdSvmTrainer.Options>();
        }

        private void CreateAndVerifyDefaultSearchSpace<TOption>()
            where TOption : class, new()
        {
            var ss = new SearchSpace<TOption>();
            var json = JsonSerializer.Serialize(ss, _settings);
            NamerFactory.AdditionalInformation = typeof(TOption).FullName;
            Approvals.Verify(json);
        }

        private class DefaultSearchSpace
        {
            public int Int { get; set; }

            public float Float { get; set; }

            public double Double { get; set; }

            public bool Bool { get; set; }

            public string String { get; set; }

            public string[] Strings { get; set; } = new[] { "A", "B", "C" };

            public JsonTokenType JTokenType { get; set; }

            public string NullString { get; set; }
        }

        private class BasicSearchSpace
        {
            [Range(-1000, 1000, init: 0)]
            public int UniformInt { get; set; }

            [Choice("a", "b", "c", "d")]
            public string ChoiceStr { get; set; }

            [Choice(1, 2, 3, 4)]
            public int ChoiceInt { get; set; }

            [Range(-1000.0, 1000, init: 0)]
            public double UniformDouble { get; set; }

            [Range(-1000.0f, 1000, init: 0)]
            public float UniformFloat { get; set; }

            [BooleanChoice(true)]
            public bool ChoiceBoolean { get; set; }

            [Choice(new object[] { JsonTokenType.None, JsonTokenType.EndObject, JsonTokenType.StartArray, JsonTokenType.Null }, defaultValue: JsonTokenType.Null)]
            public JsonTokenType JTokenType { get; set; }

            public DefaultSearchSpace DefaultSearchSpace { get; set; } = new DefaultSearchSpace();
        }

        private class NestSearchSpace
        {
            [NestOption]
            public BasicSearchSpace BasicSS { get; set; }

            [Range(-1000.0, 1000, init: 0)]
            public double UniformDouble { get; set; }

            [Range(-1000.0f, 1000, init: 0)]
            public float UniformFloat { get; set; }
        }

        class DoubleConverter : JsonConverter<double>
        {
            public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
                => Convert.ToDouble(reader.GetDecimal());

            public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
            {
                writer.WriteNumberValue(Math.Round(Convert.ToDecimal(value), 6));
            }
        }

        class SingleConverter : JsonConverter<float>
        {
            public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
                => Convert.ToSingle(reader.GetDecimal());

            public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
            {
                writer.WriteNumberValue(Convert.ToDecimal(value));
            }
        }
    }
}
