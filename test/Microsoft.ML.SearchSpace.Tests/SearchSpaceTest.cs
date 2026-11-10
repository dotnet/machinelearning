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

            Assert.Equal("a", param.ChoiceStr);
            Assert.Equal(-1000.0, param.UniformDouble);
            Assert.Equal(-1000f, param.UniformFloat);
            Assert.Equal(-1000, param.UniformInt);
            Assert.Equal(1, param.ChoiceInt);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
            Assert.Equal("c", param.ChoiceStr);
            Assert.Equal(0.0, param.UniformDouble);
            Assert.Equal(0f, param.UniformFloat);
            Assert.Equal(3, param.ChoiceInt);
            Assert.Equal(0, param.UniformInt);
        }

        [Fact]
        public void SearchSpace_mapping_to_feature_space_test()
        {
            var ss = new SearchSpace<BasicSearchSpace>();
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0 });
            var features = ss.MappingToFeatureSpace(param);
            Assert.Equal(new double[] { 0, 0, 0, 0, 0, 0, 0 }, features);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5 });
            features = ss.MappingToFeatureSpace(param);
            Assert.Equal(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5 }, features);
        }

        [Fact]
        public void Nest_search_space_mapping_to_feature_space_test()
        {
            var ss = new SearchSpace<NestSearchSpace>();
            Assert.Equal(9, ss.FeatureSpaceDim);
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0, 0, 0 });
            var features = ss.MappingToFeatureSpace(param);
            Assert.Equal(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, features);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5 });
            features = ss.MappingToFeatureSpace(param);
            Assert.Equal(new[] { 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5 }, features);
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

            Assert.Equal(9, ss.FeatureSpaceDim);
            var param = ss.SampleFromFeatureSpace(new[] { 0.0, 0, 0, 0, 0, 0, 0, 0, 0 });

            Assert.Equal(-1000.0, param.UniformDouble);
            Assert.Equal(-1000f, param.UniformFloat);
            Assert.Equal(-1000, param.BasicSS.UniformInt);
            Assert.Equal(-1000.0, param.BasicSS.UniformDouble);
            Assert.Equal(-1000f, param.BasicSS.UniformFloat);
            Assert.Equal("a", param.BasicSS.ChoiceStr);
            Assert.Equivalent(new[] { "B", "C", "D" }, param.BasicSS.DefaultSearchSpace.Strings, strict: true);
            Assert.True(param.BasicSS.ChoiceBoolean);
            Assert.Equal(JsonTokenType.None, param.BasicSS.JTokenType);
            Assert.Equal(1, param.BasicSS.ChoiceInt);

            param = ss.SampleFromFeatureSpace(new[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });

            Assert.Equal(0.0, param.UniformDouble);
            Assert.Equal(0f, param.UniformFloat);
            Assert.Equal(0, param.BasicSS.UniformInt);
            Assert.Equal(0.0, param.BasicSS.UniformDouble);
            Assert.Equal(0f, param.BasicSS.UniformFloat);
            Assert.Equal("c", param.BasicSS.ChoiceStr);
            Assert.Equivalent(new[] { "B", "C", "D" }, param.BasicSS.DefaultSearchSpace.Strings, strict: true);
            Assert.Equal(3, param.BasicSS.ChoiceInt);
            Assert.False(param.BasicSS.ChoiceBoolean);
            Assert.Equal(JsonTokenType.StartArray, param.BasicSS.JTokenType);
        }

        [Fact]
        public void Search_space_add_option_test()
        {
            var ss = new SearchSpace();
            Assert.Equal(0, ss.FeatureSpaceDim);

            ss.Add("A", new UniformIntOption(-1000, 1000));
            Assert.Equal(1, ss.FeatureSpaceDim);

            var param = ss.SampleFromFeatureSpace(new[] { 0.5 });
            Assert.Equal(0, param["A"].AsType<int>());
        }

        [Fact]
        public void Search_space_remove_option_test()
        {
            var option = new BasicSearchSpace();
            var ss = new SearchSpace<BasicSearchSpace>(option);
            Assert.Equal(7, ss.FeatureSpaceDim);

            Assert.True(ss.Remove("UniformInt"));
            Assert.Equal(6, ss.FeatureSpaceDim);
            Assert.Equivalent(new[] { "ChoiceStr", "UniformDouble", "UniformFloat", "ChoiceBoolean", "JTokenType", "ChoiceInt" }, ss.Keys, strict: true);

            var parameter = ss.SampleFromFeatureSpace(new double[] { 0, 0, 0, 0, 0, 0 });

            Assert.Equivalent(new[] { "A", "B", "C" }, parameter.DefaultSearchSpace.Strings, strict: true);
            Assert.True(string.IsNullOrEmpty(parameter.DefaultSearchSpace.String));
            Assert.Equal("a", parameter.ChoiceStr);
            Assert.True(parameter.ChoiceBoolean);
            Assert.Equal(JsonTokenType.None, parameter.JTokenType);
            Assert.Equal(1, parameter.ChoiceInt);
        }

        [Fact]
        public void Search_space_default_value_test()
        {
            var ss = new SearchSpace<NestSearchSpace>();
            var defaultTuner = new DefaultValueTuner(ss);
            var param = defaultTuner.Propose().AsType<NestSearchSpace>();

            Assert.Equal(0.0, param.UniformDouble);
            Assert.Equal(0f, param.UniformFloat);
            Assert.Equal(0, param.BasicSS.UniformInt);
            Assert.Equal(0.0, param.BasicSS.UniformDouble);
            Assert.Equal(0f, param.BasicSS.UniformFloat);
            Assert.Equal("a", param.BasicSS.ChoiceStr);
            Assert.True(param.BasicSS.ChoiceBoolean);
            Assert.Equal(JsonTokenType.Null, param.BasicSS.JTokenType);
            Assert.Equal(1, param.BasicSS.ChoiceInt);
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

            Assert.Equal(10, param.Int);
            Assert.Equal(0f, param.Float);
            Assert.Equal(0.0, param.Double);
            Assert.True(param.Bool);
            Assert.Equal("String", param.String);
            Assert.Equivalent(new[] { "A", "B", "C" }, param.Strings, strict: true);
            Assert.Equal(JsonTokenType.Null, param.JTokenType);
            Assert.Null(param.NullString);
            Assert.Equal(0, ss.FeatureSpaceDim);
            Assert.Empty(ss.MappingToFeatureSpace(param));
        }

        [Fact]
        public void Search_space_hash_code_test()
        {
            var ss = new SearchSpace<BasicSearchSpace>();
            Assert.Equal(2005165306, ss.GetHashCode());

            ss.Remove("UniformInt");
            Assert.Equal(125205970, ss.GetHashCode());
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

            Assert.Equal(4, searchSpace.FeatureSpaceDim);
            var parameter = searchSpace.SampleFromFeatureSpace(new double[] { 0, 0, 0, 0 });
            Assert.Equal("d", parameter["nestOption"]["choice"].AsType<string>());
            Assert.Equal(2, parameter["nestOption"]["int"].AsType<int>());
            Assert.Equal("a", parameter["choice"].AsType<string>());
            Assert.Equal(0, parameter["int"].AsType<int>());

            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 1, 1, 1 });
            Assert.Equal("e", parameter["nestOption"]["choice"].AsType<string>());
            Assert.Equal(3, parameter["nestOption"]["int"].AsType<int>());
            Assert.Equal("c", parameter["choice"].AsType<string>());
            Assert.Equal(1, parameter["int"].AsType<int>());
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
            Assert.Equal(new double[] { 0, 0 }, searchSpace.MappingToFeatureSpace(parameter));
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
            Assert.Equal(0, parameter["a"].AsType<int>());
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 0.5, 0.5 });
            Assert.Equal(1, parameter["a"].AsType<int>());

            searchSpace.Remove("a");

            // the first dimension should be option "b"
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 0, 0.5 });
            Assert.Equal(1, parameter["b"].AsType<int>());
            parameter = searchSpace.SampleFromFeatureSpace(new double[] { 1, 0.5 });
            Assert.Equal(2, parameter["b"].AsType<int>());
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
