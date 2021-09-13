// <copyright file="ChoiceOption.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

#nullable enable

namespace Microsoft.ML.ModelBuilder.SearchSpace.Option
{
    public class ChoiceOption : OptionBase
    {
        private UniformIntOption option;

        public ChoiceOption(params string[] choices)
        {
            Contract.Requires(choices.Length > 0 && choices.Length < 1074, $"the length of choices must be (0, 1074)");
            var distinctChoices = choices.Distinct();
            Contract.Requires(distinctChoices.Count() == choices.Length, $"choices must not contain repeated values");

            this.Choices = distinctChoices.OrderBy(x => x).ToArray();
            this.option = new UniformIntOption(0, this.Choices.Length);
            this.Default = Enumerable.Repeat(0.0, this.FeatureSpaceDim).ToArray();
        }

        public ChoiceOption(string[] choices, string? defaultChoice)
            : this(choices)
        {
            if (defaultChoice != null)
            {
                this.Default = this.MappingToFeatureSpace(new Parameter(defaultChoice));
            }
        }

        public string[] Choices { get; }

        public override int FeatureSpaceDim => 1;

        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var value = param.AsType<string>();
            var x = Array.BinarySearch(this.Choices, value);
            Contract.Requires(x != -1, $"{value} not contains");

            return this.option.MappingToFeatureSpace(new Parameter(x));
        }

        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            Contract.Requires(values.Length == 1, "values length must be 1");
            var param = this.option.SampleFromFeatureSpace(values);
            return new Parameter(this.Choices[param.AsType<int>()]);
        }
    }
}
