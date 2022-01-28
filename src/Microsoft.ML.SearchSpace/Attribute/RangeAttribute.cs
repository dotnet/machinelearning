// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public class RangeAttribute : Attribute
    {
        public RangeAttribute(double min, double max, double? init = null, bool logBase = false)
        {
            this.Option = new UniformDoubleOption(min, max, logBase, init);
        }

        public RangeAttribute(int min, int max, int? init = null, bool logBase = false)
        {
            this.Option = new UniformIntOption(min, max, logBase, init);
        }

        public RangeAttribute(float min, float max, float? init = null, bool logBase = false)
        {
            this.Option = new UniformSingleOption(min, max, logBase, init);
        }

        public OptionBase Option { get; }
    }
}
