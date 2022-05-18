// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// attribution class for <see cref="UniformDoubleOption"/>, <see cref="UniformSingleOption"/> and <see cref="UniformIntOption"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public sealed class RangeAttribute : Attribute
    {
        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformDoubleOption"/>.
        /// </summary>
        public RangeAttribute(double min, double max, bool logBase = false)
        {
            Option = new UniformDoubleOption(min, max, logBase);
        }

        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformDoubleOption"/>.
        /// </summary>
        public RangeAttribute(double min, double max, double init, bool logBase = false)
        {
            Option = new UniformDoubleOption(min, max, logBase, init);
        }

        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformIntOption"/>.
        /// </summary>
        public RangeAttribute(int min, int max, bool logBase = false)
        {
            Option = new UniformIntOption(min, max, logBase);
        }

        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformIntOption"/>.
        /// </summary>
        public RangeAttribute(int min, int max, int init, bool logBase = false)
        {
            Option = new UniformIntOption(min, max, logBase, init);
        }

        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformSingleOption"/>.
        /// </summary>
        public RangeAttribute(float min, float max, bool logBase = false)
        {
            Option = new UniformSingleOption(min, max, logBase);
        }

        /// <summary>
        /// Create a <see cref="RangeAttribute"/> for <see cref="UniformSingleOption"/>.
        /// </summary>
        public RangeAttribute(float min, float max, float init, bool logBase = false)
        {
            Option = new UniformSingleOption(min, max, logBase, init);
        }

        internal OptionBase Option { get; }
    }
}
