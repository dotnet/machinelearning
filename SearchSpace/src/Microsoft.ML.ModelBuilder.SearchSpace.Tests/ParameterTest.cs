// <copyright file="ParameterTest.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tests
{
    public class ParameterTest : TestBase
    {
        public ParameterTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void Parameter_with_single_value_test()
        {
            var parameter = new Parameter(3);

            // integer
            parameter.AsType<int>().Should().Be(3);

            // double
            parameter = new Parameter(3.1415926);
            parameter.AsType<double>().Should().Be(3.1415926);

            // float
            parameter = new Parameter(3.1415926F);
            parameter.AsType<float>().Should().Be(3.1415926F);

            // string
            parameter = new Parameter("abc");
            parameter.AsType<string>().Should().Be("abc");

            // bool
            parameter = new Parameter(false);
            parameter.AsType<bool>().Should().BeFalse();

            // string[]
            parameter = new Parameter(new[] { "a", "b", "c" });
            parameter.AsType<string[]>().Should().BeEquivalentTo("a", "b", "c");

            // enum
            parameter = new Parameter(JTokenType.Array);
            parameter.AsType<JTokenType>().Should().Be(JTokenType.Array);
        }

        [Fact]
        public void Nested_parameter_test()
        {
            var b = new B();
            var parameter = new Parameter(b);

            parameter["Int"].AsType<int>().Should().Be(0);
            parameter["Float"].AsType<float>().Should().Be(1f);
            parameter["Double"].AsType<double>().Should().Be(2);
            parameter["Bool"].AsType<bool>().Should().BeFalse();
            parameter["String"].AsType<string>().Should().Be("String");
            parameter["Strings"].AsType<string[]>().Should().BeEquivalentTo("A", "B", "C");
            parameter["JTokenType"].AsType<JTokenType>().Should().Be(JTokenType.Null);
            parameter["A"].AsType<A>().Strings.Should().BeEquivalentTo("A", "B", "C");
            parameter.AsType<B>().Strings.Should().BeEquivalentTo("A", "B", "C");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void Nested_parameter_serialize_test()
        {
            var b = new B();
            b.String = null;
            var parameter = new Parameter(b);
            var json = JsonConvert.SerializeObject(parameter);
            Approvals.Verify(json);

            parameter = JsonConvert.DeserializeObject<Parameter>(json);
            parameter["Int"].AsType<int>().Should().Be(0);
            parameter["Float"].AsType<float>().Should().Be(1f);
            parameter["Double"].AsType<double>().Should().Be(2);
            parameter["Bool"].AsType<bool>().Should().BeFalse();
            parameter["Strings"].AsType<string[]>().Should().BeEquivalentTo("A", "B", "C");
            parameter["JTokenType"].AsType<JTokenType>().Should().Be(JTokenType.Null);
            json = JsonConvert.SerializeObject(parameter);
            Approvals.Verify(json);
        }

        private class A
        {
            public int Int { get; set; } = 0;

            public float Float { get; set; } = 1f;

            public double Double { get; set; } = 2;

            public bool Bool { get; set; } = false;

            public string String { get; set; } = "String";

            public string[] Strings { get; set; } = new[] { "A", "B", "C" };

            public JTokenType JTokenType { get; set; } = JTokenType.Null;
        }

        private class B : A
        {
            public A A { get; set; } = new A();
        }
    }
}
