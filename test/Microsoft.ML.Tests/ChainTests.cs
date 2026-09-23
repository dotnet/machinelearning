// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class ChainTests : TestDataPipeBase
    {
        public ChainTests(ITestOutputHelper helper) : base(helper)
        {
        }

        private class MyData
        {
            public float Feature { get; set; }
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void EstimatorChainAcceptsNullOrEmptyEstimators(bool useNull)
        {
            var constructor = typeof(EstimatorChain<ITransformer>).GetConstructor(
                BindingFlags.Instance | BindingFlags.NonPublic, null,
                new[] { typeof(IHostEnvironment), typeof(IEstimator<ITransformer>[]), typeof(TransformerScope[]), typeof(bool[]) }, null);
            Assert.NotNull(constructor);

            var chain = (EstimatorChain<ITransformer>)constructor.Invoke(new object[]
            {
                null,
                useNull ? null : Array.Empty<IEstimator<ITransformer>>(),
                useNull ? null : Array.Empty<TransformerScope>(),
                useNull ? null : Array.Empty<bool>()
            });

            Assert.Null(chain.LastEstimator);
            var data = ML.Data.LoadFromEnumerable(new[] { new MyData() });
            Assert.Same(data, chain.Fit(data).Transform(data));

            var estimator = ML.Transforms.CopyColumns("F1", "Feature");
            Assert.Same(estimator, chain.Append(estimator).LastEstimator);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TransformerChainAcceptsNullOrEmptyTransformers(bool useNull)
        {
            var chain = new TransformerChain<ITransformer>(
                useNull ? null : Array.Empty<ITransformer>(),
                useNull ? null : Array.Empty<TransformerScope>());

            Assert.Null(chain.LastTransformer);
            Assert.Empty(chain);
            var data = ML.Data.LoadFromEnumerable(new[] { new MyData() });
            Assert.Same(data, chain.Transform(data));
            Assert.Same(data.Schema, chain.GetOutputSchema(data.Schema));
        }

        [Fact]
        public void TransformerChainEnumeratesTransformersOnce()
        {
            var data = ML.Data.LoadFromEnumerable(new[] { new MyData() });
            var first = ML.Transforms.CopyColumns("F1", "Feature").Fit(data);
            var last = ML.Transforms.CopyColumns("F2", "F1").Fit(first.Transform(data));
            int enumerationCount = 0;

            IEnumerable<ITransformer> GetTransformers()
            {
                enumerationCount++;
                yield return first;
                yield return last;
            }

            var chain = new TransformerChain<ColumnCopyingTransformer>(
                GetTransformers(), new[] { TransformerScope.Everything, TransformerScope.Everything });

            Assert.Equal(1, enumerationCount);
            Assert.Equal(new ITransformer[] { first, last }, chain);
            Assert.Same(last, chain.LastTransformer);
            Assert.Equal(typeof(float), chain.Transform(data).Schema["F2"].Type.RawType);
        }
    }
}
