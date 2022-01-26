// <copyright file="ChoiceAttribute.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public class ChoiceAttribute : Attribute
    {
        public ChoiceAttribute(params object[] candidates)
        {
            this.Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray());
        }

        public ChoiceAttribute(object[] candidates, object defaultValue)
        {
            this.Option = new ChoiceOption(candidates.Select(c => Convert.ToString(c, CultureInfo.InvariantCulture)).ToArray(), Convert.ToString(defaultValue, CultureInfo.InvariantCulture));
        }

        public ChoiceOption Option { get; }
    }
}
