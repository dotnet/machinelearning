// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// attribution class for <see cref="ChoiceOption"/>. The property or field it applys to will be treated as <see cref="ChoiceOption"/> in <see cref="SearchSpace{T}"/>.
    /// <seealso cref="SearchSpace{T}"/>
    /// <seealso cref="SearchSpace"/>
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
            Contracts.Assert(candidatesType.Count() == 1, "multiple candidates type detected");

            Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray());
        }

        /// <summary>
        /// Create a <see cref="ChoiceAttribute"/> with <paramref name="candidates"/> and <paramref name="defaultValue"/>.
        /// </summary>
        public ChoiceAttribute(object[] candidates, object defaultValue)
        {
            var candidatesType = candidates.Select(o => o.GetType()).Distinct();
            Contracts.Assert(candidatesType.Count() == 1, "multiple candidates type detected");
            Contracts.Assert(candidatesType.First() == defaultValue.GetType(), "candidates type doesn't match with defaultValue type");

            Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray(), Convert.ToString(defaultValue, CultureInfo.InvariantCulture));
        }

        internal ChoiceOption Option { get; }
    }
}
