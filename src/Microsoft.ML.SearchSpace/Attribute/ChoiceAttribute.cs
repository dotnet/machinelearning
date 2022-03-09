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
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    internal sealed class ChoiceAttribute : Attribute
    {
        public ChoiceAttribute(params object[] candidates)
        {
            var candidatesType = candidates.Select(o => o.GetType()).Distinct();
            Contracts.Assert(candidatesType.Count() == 1, "multiple candidates type detected");

            this.Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray());
        }

        public ChoiceAttribute(object[] candidates, object defaultValue)
        {
            var candidatesType = candidates.Select(o => o.GetType()).Distinct();
            Contracts.Assert(candidatesType.Count() == 1, "multiple candidates type detected");
            Contracts.Assert(candidatesType.First() == defaultValue.GetType(), "candidates type doesn't match with defaultValue type");

            this.Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray(), Convert.ToString(defaultValue, CultureInfo.InvariantCulture));
        }

        public ChoiceOption Option { get; }
    }
}
