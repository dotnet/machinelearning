// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using Microsoft.ML.SearchSpace.Option;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    public class ParameterTest : TestBase
    {
        private readonly JsonSerializerOptions _settings = new JsonSerializerOptions()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };

        public ParameterTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void Array_parameter_serialize_test()
        {
            var array = new[] { "A", "B", "C" };
            var parameter = Parameter.FromIEnumerable(array);

            var json = JsonSerializer.Serialize(parameter, _settings);
            Approvals.Verify(json);

            parameter = JsonSerializer.Deserialize<Parameter>(json);
            Assert.Equal(new[] { "A", "B", "C" }, parameter.AsType<string[]>());
        }

        [Fact]
        public void Parameter_with_single_value_test()
        {
            var parameter = Parameter.FromInt(3);

            // integer
            Assert.Equal(3, parameter.AsType<int>());

            // double
            parameter = Parameter.FromDouble(3.1415926);
            Assert.Equal(3.1415926, parameter.AsType<double>());

            // float
            parameter = Parameter.FromFloat(3.1415926F);
            Assert.Equal(3.1415926F, parameter.AsType<float>());

            // string
            parameter = Parameter.FromString("abc");
            Assert.Equal("abc", parameter.AsType<string>());

            // bool
            parameter = Parameter.FromBool(false);
            Assert.False(parameter.AsType<bool>());

            // string[]
            parameter = Parameter.FromIEnumerable(new[] { "a", "b", "c" });
            Assert.Equivalent(new[] { "a", "b", "c" }, parameter.AsType<string[]>(), strict: true);

            // enum
            parameter = Parameter.FromEnum(JsonTokenType.None);
            Assert.Equal(JsonTokenType.None, parameter.AsType<JsonTokenType>());

            // long
            parameter = Parameter.FromLong(long.MaxValue);
            Assert.Equal(long.MaxValue, parameter.AsType<long>());
        }

        [Fact]
        public void Nested_parameter_test()
        {
            var b = new B();
            var parameter = Parameter.FromObject(b);

            Assert.Equal(0, parameter["Int"].AsType<int>());
            Assert.Equal(1f, parameter["Float"].AsType<float>());
            Assert.Equal(2.0, parameter["Double"].AsType<double>());
            Assert.False(parameter["Bool"].AsType<bool>());
            Assert.Equal("String", parameter["String"].AsType<string>());
            Assert.Equivalent(new[] { "A", "B", "C" }, parameter["Strings"].AsType<string[]>(), strict: true);
            Assert.Equal(JsonTokenType.Null, parameter["JTokenType"].AsType<JsonTokenType>());
            Assert.Equivalent(new[] { "A", "B", "C" }, parameter["A"].AsType<A>().Strings, strict: true);
            Assert.Equivalent(new[] { "A", "B", "C" }, parameter.AsType<B>().Strings, strict: true);
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void Nested_parameter_serialize_test()
        {
            var b = new B();
            b.String = null;
            var parameter = Parameter.FromObject(b);
            var json = JsonSerializer.Serialize(parameter, _settings);
            Approvals.Verify(json);

            parameter = JsonSerializer.Deserialize<Parameter>(json);
            Assert.Equal(0, parameter["Int"].AsType<int>());
            Assert.Equal(1f, parameter["Float"].AsType<float>());
            Assert.Equal(2.0, parameter["Double"].AsType<double>());
            Assert.False(parameter["Bool"].AsType<bool>());
            Assert.Equivalent(new[] { "A", "B", "C" }, parameter["Strings"].AsType<string[]>(), strict: true);
            Assert.Equal(JsonTokenType.Null, parameter["JTokenType"].AsType<JsonTokenType>());
            json = JsonSerializer.Serialize(parameter, _settings);
            Approvals.Verify(json);
        }

        [Fact]
        public void Parameter_AsType_should_be_culture_invariant()
        {
            var originalCuture = Thread.CurrentThread.CurrentCulture;
            var culture = new CultureInfo("ru", false);
            Thread.CurrentThread.CurrentCulture = culture;
            var ss = new SearchSpace();
            ss.Add("_SampleSize", new UniformDoubleOption(10000, 20000));
            var parameter = ss.SampleFromFeatureSpace(new[] { 0.5 });
            Assert.Equal(15000.0, parameter["_SampleSize"].AsType<double>());
            Thread.CurrentThread.CurrentCulture = originalCuture;
        }

        [Fact]
        public void Parameter_equatable_test()
        {
            var b = new B()
            {
                String = "StringA",
                Strings = new[] { "a" },
            };

            var paramB1 = Parameter.FromObject(b);
            var paramB2 = Parameter.FromObject(b);

            Assert.True(paramB1.Equals(paramB2));

            b.Bool = true;
            paramB2 = Parameter.FromObject(b);

            Assert.False(paramB1.Equals(paramB2));
            Assert.False(paramB1.Equals(null));
        }

        private class A
        {
            public int Int { get; set; } = 0;

            public float Float { get; set; } = 1f;

            public double Double { get; set; } = 2;

            public bool Bool { get; set; } = false;

            public string String { get; set; } = "String";

            public string[] Strings { get; set; } = new[] { "A", "B", "C" };

            public JsonTokenType JTokenType { get; set; } = JsonTokenType.Null;
        }

        private class B : A
        {
            public A A { get; set; } = new A();
        }
    }
}
