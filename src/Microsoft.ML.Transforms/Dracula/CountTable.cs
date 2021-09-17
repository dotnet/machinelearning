// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal interface ICountTable
    {
        /// <summary>
        /// Populate the <paramref name="counts"/> array with the counts for the input key
        /// </summary>
        void GetCounts(long key, Span<float> counts);

        /// <summary>
        /// Garbage threshold the table is using
        /// </summary>
        float GarbageThreshold { get; }

        IReadOnlyCollection<float> GarbageCounts { get; }
        ReadOnlySpan<double> PriorFrequencies { get; }
    }

    /// <summary>
    /// Signature for CountTableBuilder.
    /// </summary>
    internal delegate void SignatureCountTableBuilder();

    internal abstract class CountTableBase : ICountTable, ICanSaveModel
    {
        public const int LabelCardinalityLim = 100;

        public readonly int LabelCardinality; // number of values the label can assume
        private readonly double[] _priorFrequencies;

        public float GarbageThreshold { get; private set; } // garbage bin threshold
        private readonly float[] _garbageCounts; // counts of garbage labels. Size = labelCardinality
        public IReadOnlyCollection<float> GarbageCounts => _garbageCounts;

        public ReadOnlySpan<double> PriorFrequencies => _priorFrequencies;

        protected CountTableBase(int labelCardinality, float[] priorCounts, float garbageThreshold, float[] garbageCounts)
        {
            Contracts.Check(0 < labelCardinality && labelCardinality < LabelCardinalityLim, "Label cardinality out of bounds");
            Contracts.CheckValue(priorCounts, nameof(priorCounts));
            Contracts.Check(priorCounts.All(x => x >= 0));
            Contracts.Check(priorCounts.Length == labelCardinality);
            Contracts.Check(garbageThreshold >= 0, "Garbage threshold must be non-negative");

            if (garbageThreshold > 0)
            {
                Contracts.CheckValue(garbageCounts, nameof(garbageCounts));
                Contracts.Check(garbageCounts.Length == labelCardinality);
                Contracts.Check(garbageCounts.All(x => x >= 0));
            }

            LabelCardinality = labelCardinality;
            _garbageCounts = garbageCounts;
            GarbageThreshold = garbageThreshold;

            var priorSum = priorCounts.Sum();
            _priorFrequencies = new double[priorCounts.Length];
            if (priorSum > 0)
            {
                for (int i = 0; i < priorCounts.Length; i++)
                    _priorFrequencies[i] = priorCounts[i] / priorSum;
            }
            else
            {
                // if there is no prior computed, defer to 1/N
                var d = 1.0 / LabelCardinality;
                for (int i = 0; i < LabelCardinality; i++)
                    _priorFrequencies[i] = d;
            }
        }

        protected CountTableBase(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            env.AssertNonWhiteSpace(name);
            env.AssertValue(ctx);

            // *** Binary format ***
            // int: label cardinality
            // double[]: prior frequencies
            // float: garbage threshold
            // float[]: garbage counts

            LabelCardinality = ctx.Reader.ReadInt32();
            env.CheckDecode(0 < LabelCardinality && LabelCardinality < LabelCardinalityLim);

            _priorFrequencies = ctx.Reader.ReadDoubleArray();
            env.CheckDecode(Utils.Size(_priorFrequencies) == LabelCardinality);
            env.CheckDecode(_priorFrequencies.All(x => x >= 0));

            GarbageThreshold = ctx.Reader.ReadSingle();
            env.CheckDecode(GarbageThreshold >= 0);

            _garbageCounts = ctx.Reader.ReadSingleArray();
            if (GarbageThreshold == 0)
                env.CheckDecode(Utils.Size(_garbageCounts) == 0);
            else
            {
                env.CheckDecode(Utils.Size(_garbageCounts) == LabelCardinality);
                env.CheckDecode(_garbageCounts.All(x => x >= 0));
            }
        }

        public abstract void GetCounts(long key, Span<float> counts);

        public virtual void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: label cardinality
            // Single[]: prior counts
            // Single: garbage threshold
            // Single[]: garbage counts

            Contracts.Assert(0 < LabelCardinality && LabelCardinality < LabelCardinalityLim);
            ctx.Writer.Write(LabelCardinality);

            Contracts.Assert(Utils.Size(_priorFrequencies) == LabelCardinality);
            Contracts.Assert(_priorFrequencies.All(x => x >= 0));
            ctx.Writer.WriteDoubleArray(_priorFrequencies);

            Contracts.Assert(GarbageThreshold >= 0);
            ctx.Writer.Write(GarbageThreshold);

            if (GarbageThreshold == 0)
                Contracts.Assert(Utils.Size(_garbageCounts) == 0);
            else
            {
                Contracts.Assert(Utils.Size(_garbageCounts) == LabelCardinality);
                Contracts.Assert(_garbageCounts.All(x => x >= 0));
            }

            ctx.Writer.WriteSingleArray(_garbageCounts);
        }

        public abstract InternalCountTableBuilderBase ToBuilder(long labelCardinality);
    }
}
