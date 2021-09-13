// <copyright file="RangeAttribute.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public class RangeAttribute : Attribute
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
