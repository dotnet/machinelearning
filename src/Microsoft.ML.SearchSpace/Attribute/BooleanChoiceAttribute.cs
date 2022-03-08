// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    internal sealed class BooleanChoiceAttribute : Attribute
    {
        public BooleanChoiceAttribute()
        {
            this.Option = new ChoiceOption(true, false);
        }

        public BooleanChoiceAttribute(bool defaultValue)
        {
            this.Option = new ChoiceOption(new object[] { true, false }, defaultChoice: defaultValue);
        }

        public ChoiceOption Option { get; }
    }
}
