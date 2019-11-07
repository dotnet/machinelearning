// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    public interface ICountTable
    {
        /// <summary>
        /// Populate the <paramref name="counts"/> array with the counts for the input key
        /// </summary>
        void GetCounts(long key, Span<float> counts);

        ///// <summary>
        ///// Populates the <paramref name="counts"/> array with the raw counts for the specified hash id and hash value.
        ///// </summary>
        //void GetRawCounts(RawCountKey key, float[] counts);

        ///// <summary>
        ///// Gets the raw count keys from the count table.
        ///// </summary>
        //IEnumerable<RawCountKey> AllRawCountKeys();

        /// <summary>
        /// Garbage threshold the table is using
        /// </summary>
        float GarbageThreshold { get; }

        /// <summary>
        /// Populate the <paramref name="priorCounts"/> and <paramref name="garbageCounts"/> with
        /// respective priors. If the <see cref="GarbageThreshold"/> is zero, the second argument is not
        /// affected, and can be null
        /// </summary>
        void GetPriors(float[] priorCounts, float[] garbageCounts);
    }

    /// <summary>
    /// Signature for CountTableBuilder.
    /// </summary>
    public delegate void SignatureCountTableBuilder();

    /// <summary>
    /// The key for a single entry in the raw count file.
    /// </summary>
    public struct RawCountKey
    {
        public readonly int HashId;
        public readonly long HashValue;

        public RawCountKey(int hashId, long hashValue)
        {
            HashId = hashId;
            HashValue = hashValue;
        }
    }

    internal abstract class CountTableBase : ICountTable, ICanSaveModel
    {
        public const int LabelCardinalityLim = 100;

        protected readonly IHost Host;
        protected readonly int LabelCardinality; // number of values the label can assume
        protected readonly float[] PriorCounts; // overall counts of labels. Size = labelCardinality

        public float GarbageThreshold { get; private set; } // garbage bin threshold
        protected readonly float[] GarbageCounts; // counts of garbage labels. Size = labelCardinality

        protected CountTableBase(IHostEnvironment env, string name, int labelCardinality, float[] priorCounts, float garbageThreshold, float[] garbageCounts)
        {
            Contracts.AssertValue(env, "env");
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.Check(0 < labelCardinality && labelCardinality < LabelCardinalityLim, "Label cardinality out of bounds");
            Host.CheckValue(priorCounts, nameof(priorCounts));
            Host.Check(priorCounts.All(x => x >= 0));
            Host.Check(priorCounts.Length == labelCardinality);
            Host.Check(garbageThreshold >= 0, "Garbage threshold must be non-negative");

            if (garbageThreshold > 0)
            {
                Host.CheckValue(garbageCounts, nameof(garbageCounts));
                Host.Check(garbageCounts.Length == labelCardinality);
                Host.Check(garbageCounts.All(x => x >= 0));
            }

            LabelCardinality = labelCardinality;
            PriorCounts = priorCounts;
            GarbageCounts = garbageCounts;
            GarbageThreshold = garbageThreshold;
        }

        protected CountTableBase(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env, "env");
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: label cardinality
            // Single[]: prior counts
            // Single: garbage threshold
            // Single[]: garbage counts

            LabelCardinality = ctx.Reader.ReadInt32();
            Host.CheckDecode(0 < LabelCardinality && LabelCardinality < LabelCardinalityLim);

            PriorCounts = ctx.Reader.ReadSingleArray();
            Host.CheckDecode(Utils.Size(PriorCounts) == LabelCardinality);
            Host.CheckDecode(PriorCounts.All(x => x >= 0));

            GarbageThreshold = ctx.Reader.ReadSingle();
            Host.CheckDecode(GarbageThreshold >= 0);

            GarbageCounts = ctx.Reader.ReadSingleArray();
            if (GarbageThreshold == 0)
                Host.CheckDecode(Utils.Size(GarbageCounts) == 0);
            else
            {
                Host.CheckDecode(Utils.Size(GarbageCounts) == LabelCardinality);
                Host.CheckDecode(GarbageCounts.All(x => x >= 0));
            }
        }

        public abstract void GetCounts(long key, Span<float> counts);

        public void GetPriors(float[] priorCounts, float[] garbageCounts)
        {
            Host.CheckValue(priorCounts, nameof(priorCounts));
            Host.Check(priorCounts.Length == LabelCardinality);
            if (GarbageThreshold > 0)
            {
                Host.CheckValue(garbageCounts, nameof(garbageCounts));
                Host.Check(garbageCounts.Length == LabelCardinality);
            }

            Array.Copy(PriorCounts, priorCounts, LabelCardinality);
            if (GarbageThreshold > 0)
                Array.Copy(GarbageCounts, garbageCounts, LabelCardinality);
        }

        public virtual void Save(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: label cardinality
            // Single[]: prior counts
            // Single: garbage threshold
            // Single[]: garbage counts

            Host.Assert(0 < LabelCardinality && LabelCardinality < LabelCardinalityLim);
            ctx.Writer.Write(LabelCardinality);

            Host.Assert(Utils.Size(PriorCounts) == LabelCardinality);
            Host.Assert(PriorCounts.All(x => x >= 0));
            ctx.Writer.WriteSingleArray(PriorCounts);

            Host.Assert(GarbageThreshold >= 0);
            ctx.Writer.Write(GarbageThreshold);

            if (GarbageThreshold == 0)
                Host.Assert(Utils.Size(GarbageCounts) == 0);
            else
            {
                Host.Assert(Utils.Size(GarbageCounts) == LabelCardinality);
                Host.Assert(GarbageCounts.All(x => x >= 0));
            }

            ctx.Writer.WriteSingleArray(GarbageCounts);
        }
    }
}