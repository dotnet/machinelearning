// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// Boolean choice attribute
    /// </summary>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public sealed class BooleanChoiceAttribute : Attribute
    {
        /// <summary>
        /// Create a <see cref="BooleanChoiceAttribute"/>.
        /// </summary>
        public BooleanChoiceAttribute()
        {
            Option = new ChoiceOption(true, false);
        }

        /// <summary>
        /// Create a <see cref="BooleanChoiceAttribute"/> with default value.
        /// </summary>
        /// <param name="defaultValue">default value for this option.</param>
        public BooleanChoiceAttribute(bool defaultValue)
        {
            Option = new ChoiceOption(new object[] { true, false }, defaultChoice: defaultValue);
        }

        internal ChoiceOption Option { get; }
    }
}
