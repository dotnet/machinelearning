// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics.Contracts;
using System.Linq;

namespace Microsoft.ML.SearchSpace;

/// <summary>
/// Choice attribute
/// </summary>
[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
public sealed class ChoiceAttribute : Attribute
{
    /// <summary>
    /// Create a <see cref="ChoiceAttribute"/> with <paramref name="candidates"/>.
    /// </summary>
    public ChoiceAttribute(params object[] candidates)
    {
        var candidatesType = candidates.Select(o => o.GetType()).Distinct();
        Contract.Assert(candidatesType.Count() == 1, "multiple candidates type detected");
        this.Candidates = candidates;
        this.DefaultValue = null;
    }

    /// <summary>
    /// Create a <see cref="ChoiceAttribute"/> with <paramref name="candidates"/> and <paramref name="defaultValue"/>.
    /// </summary>
    public ChoiceAttribute(object[] candidates, object defaultValue)
    {
        var candidatesType = candidates.Select(o => o.GetType()).Distinct();
        Contract.Assert(candidatesType.Count() == 1, "multiple candidates type detected");
        Contract.Assert(candidatesType.First() == defaultValue.GetType(), "candidates type doesn't match with defaultValue type");

        this.Candidates = candidates;
        this.DefaultValue = defaultValue;
    }

    /// <summary>
    /// Get the candidates of this option.
    /// </summary>
    public object[] Candidates { get; }

    /// <summary>
    /// Get the default value of this option.
    /// </summary>
    public object DefaultValue { get; }
}
