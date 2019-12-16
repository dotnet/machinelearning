// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    [TlcModule.ComponentKind("CountTableBuilder")]
    internal interface ICountTableBuilderFactory : IComponentFactory<CountTableBuilderBase>
    {
    }

    public abstract class CountTableBuilderBase
    {
        private protected CountTableBuilderBase()
        {
        }

        internal abstract InternalCountTableBuilderBase GetInternalBuilder(long labelCardinality);

        public static CountTableBuilderBase CreateCMCountTableBuilder(int depth = 4, int width = 1 << 23)
            => new CMCountTableBuilder(depth, width);

        public static CountTableBuilderBase CreateDictionaryCountTableBuilder(float garbageThreshold = 0)
            => new DictCountTableBuilder(garbageThreshold);
    }

    internal abstract class InternalCountTableBuilderBase
    {
        protected readonly int LabelCardinality;
        protected readonly double[] PriorCounts;

        protected InternalCountTableBuilderBase(long labelCardinality)
        {
            LabelCardinality = (int)labelCardinality;
            Contracts.CheckParam(LabelCardinality == labelCardinality, nameof(labelCardinality), "Label cardinality must be less than int.MaxValue");

            PriorCounts = new double[LabelCardinality];
        }

        internal void Increment(long key, long labelKey)
        {
            Contracts.Check(0 <= labelKey && labelKey < LabelCardinality);
            PriorCounts[labelKey]++;

            IncrementCore(key, labelKey);
        }

        protected abstract void IncrementCore(long key, long labelKey);

        internal abstract CountTableBase CreateCountTable();
    }
}