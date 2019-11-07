// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    // REVIEW petelu: Right now there's not enough common functionality across existing count table builders
    // to warrant creating a common base class. It might be later.
    //public interface ICountTableBuilder
    //{
    //    /// <summary>
    //    /// Increment the count for a given key and label
    //    /// </summary>
    //    /// <param name="key">The key to increment count for</param>
    //    /// <param name="labelKey">The label to increment count for</param>
    //    /// <param name="amount">The amount of increment</param>
    //    /// <returns>Previous amount associated with this label &amp; count</returns>
    //    Double Increment(long key, long labelKey, Double amount);

    //    /// <summary>
    //    /// Inserts or updates the raw counts with the specified key (hash ID and hash value). This method is used to import the counts from the text file
    //    /// </summary>
    //    void InsertOrUpdateRawCounts(int hashId, long hashValue, Single[] counts);

    //    /// <summary>
    //    /// Finalizes training and generates the count table
    //    /// </summary>
    //    ICountTable CreateCountTable();

    //    void Reset();
    //}

    public abstract class CountTableBuilderBase
    {
        private protected CountTableBuilderBase()
        {
        }

        internal abstract CountTableBuilderHelperBase GetBuilderHelper(long labelCardinality);
    }

    internal abstract class CountTableBuilderHelperBase
    {
        protected readonly IHost Host;
        protected readonly int LabelCardinality;
        protected readonly double[] PriorCounts;

        protected CountTableBuilderHelperBase(IHostEnvironment env, string name, long labelCardinality)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            env.CheckParam(labelCardinality > 0, nameof(labelCardinality));

            Host = env.Register(name);
            LabelCardinality = (int)labelCardinality;
            Host.CheckParam(LabelCardinality == labelCardinality, nameof(labelCardinality), "Label cardinality must be less than int.MaxValue");

            PriorCounts = new double[LabelCardinality];
        }

        internal abstract double Increment(long key, long labelKey, double amount);

        internal abstract void InsertOrUpdateRawCounts(int hashId, long hashValue, float[] counts);

        internal abstract ICountTable CreateCountTable();
    }
}