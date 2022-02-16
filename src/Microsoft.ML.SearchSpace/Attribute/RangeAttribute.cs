// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public sealed class RangeAttribute : Attribute
    {
        public RangeAttribute(double min, double max, bool logBase = false)
        {
            this.Option = new UniformDoubleOption(min, max, logBase);
        }

        public RangeAttribute(double min, double max, double init, bool logBase = false)
        {
            this.Option = new UniformDoubleOption(min, max, logBase, init);
        }

        public RangeAttribute(int min, int max, bool logBase = false)
        {
            this.Option = new UniformIntOption(min, max, logBase);
        }

        public RangeAttribute(int min, int max, int init, bool logBase = false)
        {
            this.Option = new UniformIntOption(min, max, logBase, init);
        }

        public RangeAttribute(float min, float max, bool logBase = false)
        {
            this.Option = new UniformSingleOption(min, max, logBase);
        }

        public RangeAttribute(float min, float max, float init, bool logBase = false)
        {
            this.Option = new UniformSingleOption(min, max, logBase, init);
        }

        public OptionBase Option { get; }
    }
}
