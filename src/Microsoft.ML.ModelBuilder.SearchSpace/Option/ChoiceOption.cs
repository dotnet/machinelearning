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

        public ChoiceOption(params object[] choices)
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
                this.Default = this.MappingToFeatureSpace(Parameter.FromString(defaultChoice));
            }
        }

        public object[] Choices { get; }

        public override int FeatureSpaceDim => this.Choices.Length == 1 ? 0 : 1;

        public override int?[] Step => new int?[] { this.Choices.Length };

        public override double[] MappingToFeatureSpace(IParameter param)
        {
            if(this.FeatureSpaceDim == 0)
            {
                return new double[0];
            }

            var value = param.AsType<string>();
            var x = Array.BinarySearch(this.Choices, value);
            Contract.Requires(x != -1, $"{value} not contains");

            return this.option.MappingToFeatureSpace(Parameter.FromInt(x));
        }

        public override IParameter SampleFromFeatureSpace(double[] values)
        {
            Contract.Requires(values.Length >= 0, "values length must be greater than 0");
            if (values.Length == 0)
            {
                return Parameter.FromObject(this.Choices[0]);
            }

            var param = this.option.SampleFromFeatureSpace(values);
            return Parameter.FromObject(this.Choices[param.AsType<int>()]);
        }
    }
}
