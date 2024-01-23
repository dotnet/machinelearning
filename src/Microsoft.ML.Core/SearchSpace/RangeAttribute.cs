// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.SearchSpace;

/// <summary>
/// Range attribute
/// </summary>
[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
public sealed class RangeAttribute : Attribute
{
    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(double min, double max, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = null;
        this.LogBase = logBase;
    }

    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(double min, double max, double init, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = init;
        this.LogBase = logBase;
    }

    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(int min, int max, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = null;
        this.LogBase = logBase;
    }

    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(int min, int max, int init, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = init;
        this.LogBase = logBase;
    }

    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(float min, float max, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = null;
        this.LogBase = logBase;
    }

    /// <summary>
    /// Create a <see cref="RangeAttribute"/>
    /// </summary>
    public RangeAttribute(float min, float max, float init, bool logBase = false)
    {
        this.Min = min;
        this.Max = max;
        this.Init = init;
        this.LogBase = logBase;
    }

    public object Min { get; }

    public object Max { get; }

    public object Init { get; }

    public bool LogBase { get; }
}
