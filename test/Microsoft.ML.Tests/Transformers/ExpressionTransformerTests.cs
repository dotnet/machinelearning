// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class ExpressionTransformerTests : TestDataPipeBase
    {
        public ExpressionTransformerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestExpressionTransformer()
        {
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");

            var loader = new TextLoader(ML, new TextLoader.Options
            {
                Columns = new[]{
                    new TextLoader.Column("Float", DataKind.Single, 9),
                    new TextLoader.Column("FloatVector", DataKind.Single, 9, 14),
                    new TextLoader.Column("Double", DataKind.Double, 9),
                    new TextLoader.Column("DoubleVector", DataKind.Double, 9, 14),
                    new TextLoader.Column("Int", DataKind.Int32, 9),
                    new TextLoader.Column("IntVector", DataKind.Int32, 9, 14),
                    new TextLoader.Column("Text", DataKind.String, 1),
                    new TextLoader.Column("TextVector", DataKind.String, 2, 8),
                },
                Separator = "\t",
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var expr = ML.Transforms.Expression("Expr1", "x=>x/2", "Double").
                Append(ML.Transforms.Expression("Expr2", "(x,y)=>(x+y)/3", "Float", "FloatVector")).
                Append(ML.Transforms.Expression("Expr3", "(x,y)=>x*y", "Float", "Int")).
                Append(ML.Transforms.Expression("Expr4", "(x,y,z)=>abs(x-y)*z", "Float", "FloatVector", "Double")).
                Append(ML.Transforms.Expression("Expr5", "x=>len(concat(upper(x),lower(x)))", "Text")).
                Append(ML.Transforms.Expression("Expr6", "(x,y)=>right(x,y)", "TextVector", "Int"));

            TestEstimatorCore(expr, loader.Load(dataPath));

            var transformed = expr.Fit(loader.Load(dataPath)).Transform(loader.Load(dataPath));
            Assert.True(transformed.Schema["Expr1"].Type == NumberDataViewType.Double);
            Assert.Equal(6, transformed.Schema["Expr2"].Type.GetValueCount());
            Assert.True(transformed.Schema["Expr2"].Type.GetItemType() == NumberDataViewType.Single);
            Assert.True(transformed.Schema["Expr3"].Type == NumberDataViewType.Single);
            Assert.True(transformed.Schema["Expr4"].Type.GetItemType() == NumberDataViewType.Double);
            Assert.Equal(6, transformed.Schema["Expr4"].Type.GetValueCount());
            Assert.True(transformed.Schema["Expr5"].Type == NumberDataViewType.Int32);
            Assert.True(transformed.Schema["Expr6"].Type.GetItemType() == TextDataViewType.Instance);
            Assert.Equal(7, transformed.Schema["Expr6"].Type.GetValueCount());
        }
    }
}
