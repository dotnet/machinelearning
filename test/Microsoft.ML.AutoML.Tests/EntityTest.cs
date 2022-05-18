// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class EntityTest : BaseTestClass
    {
        public EntityTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void Entity_operator_overload_test()
        {
            var a = new StringEntity("a");
            var b = new StringEntity("b");
            var c = new StringEntity("c");

            (a + b + c).ToString().Should().Be("a + b + c");
            (a * b * c).ToString().Should().Be("a * b * c");
            (a + b * c).ToString().Should().Be("a + b * c");
            ((a + b) * c).ToString().Should().Be("(a + b) * c");
        }

        [Fact]
        public void Entity_symplify_test()
        {
            var a = new StringEntity("a");
            var b = new StringEntity("b");
            var c = new StringEntity("c");
            var d = new StringEntity("d");
            var e = new StringEntity("e");

            (a + b + c + d + e).Simplify().ToString().Should().Be("a + b + c + d + e");
            ((a + b) + (c + d) + e).Simplify().ToString().Should().Be("a + b + c + d + e");
            (a * b * c * d * e).Simplify().ToString().Should().Be("a * b * c * d * e");
            ((a * b) * c * (d * e)).Simplify().ToString().Should().Be("a * b * c * d * e");
            (a * (b + c + d * e)).Simplify().ToString().Should().Be("a * b + a * c + a * d * e");
            (a * (b + c * d)).Simplify().ToString().Should().Be("a * b + a * c * d");
            ((a + b) * (c + d)).Simplify().ToString().Should().Be("a * c + a * d + b * c + b * d");
            (a * (b + c) * (d + e)).Simplify().ToString().Should().Be("a * b * d + a * b * e + a * c * d + a * c * e");
            ((a + b + c + d * e) * e).Simplify().ToString().Should().Be("a * e + b * e + c * e + d * e * e");
            ((a + b + c) * (b + c + d) * (d + e)).Simplify().ToString().Should().Be("a * b * d + a * c * d + b * b * d + b * c * d + a * d * d + b * d * d + c * b * d + c * c * d + a * b * e + a * c * e + b * b * e + b * c * e + a * d * e + b * d * e + c * b * e + c * c * e + c * d * d + c * d * e");
        }

        [Fact]
        public void Entity_to_terms_test()
        {
            var a = new StringEntity("a");
            var b = new StringEntity("b");
            var c = new StringEntity("c");
            var d = new StringEntity("d");
            var e = new StringEntity("e");
            a.ToTerms()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a");
            (a * b * c * d * e).ToTerms()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a * b * c * d * e");
            (a + b + c + d + e).ToTerms()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a", "b", "c", "d", "e");

            (a + b + c + d * e).ToTerms()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a", "b", "c", "d * e");
        }

        [Fact]
        public void Entity_to_value_test()
        {
            var a = new StringEntity("a");
            var b = new StringEntity("b");
            var c = new StringEntity("c");
            var d = new StringEntity("d");
            var e = new StringEntity("e");
            a.ValueEntities()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a");
            (a * b * c * d * e).ValueEntities()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a", "b", "c", "d", "e");
            (a + b + c + d + e).ValueEntities()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a", "b", "c", "d", "e");

            (a + b + c + d * e).ValueEntities()
             .Select(x => x.ToString())
             .Should()
             .BeEquivalentTo("a", "b", "c", "d", "e");
        }

        [Fact]
        public void Entity_create_from_expression()
        {
            var entity = Entity.FromExpression("a+b+c");
            entity.ToString().Should().Be("a + b + c");

            entity = Entity.FromExpression("abc");
            entity.ToString().Should().Be("abc");

            entity = Entity.FromExpression("(a+b)*(c+d)");
            entity.ToString().Should().Be("(a + b) * (c + d)");
        }
    }
}
