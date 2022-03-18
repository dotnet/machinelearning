// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;

#nullable enable

namespace Microsoft.ML.SearchSpace.Option
{
    /// <summary>
    /// This class represent option for discrete value, such as string, enum, etc..
    /// </summary>
    public sealed class ChoiceOption : OptionBase
    {
        private readonly UniformIntOption _option;

        /// <summary>
        /// Create <see cref="ChoiceOption"/> with <paramref name="choices"/>
        /// </summary>
        public ChoiceOption(params object[] choices)
        {
            Contracts.Check(choices.Length > 0 && choices.Length < 1074, "the length of choices must be (0, 1074)");
            var distinctChoices = choices.Distinct();
            Contracts.Check(distinctChoices.Count() == choices.Length, "choices must not contain repeated values");

            this.Choices = distinctChoices.OrderBy(x => x).ToArray();
            this._option = new UniformIntOption(0, this.Choices.Length);
            this.Default = Enumerable.Repeat(0.0, this.FeatureSpaceDim).ToArray();
        }

        /// <summary>
        /// Create <see cref="ChoiceOption"/> with <paramref name="choices"/> and <paramref name="defaultChoice"/>.
        /// </summary>
        public ChoiceOption(object[] choices, object? defaultChoice)
            : this(choices)
        {
            if (defaultChoice != null)
            {
                this.Default = this.MappingToFeatureSpace(Parameter.FromObject(defaultChoice));
            }
        }

        /// <summary>
        /// Get all choices.
        /// </summary>
        public object[] Choices { get; }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int FeatureSpaceDim => this.Choices.Length == 1 ? 0 : 1;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int?[] Step => new int?[] { this.Choices.Length };

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] MappingToFeatureSpace(Parameter param)
        {
            if (this.FeatureSpaceDim == 0)
            {
                return new double[0];
            }

            var value = param.AsType<object>();
            var x = Array.BinarySearch(this.Choices, value);
            Contracts.Check(x != -1, $"{value} not contains");

            return this._option.MappingToFeatureSpace(Parameter.FromInt(x));
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            Contracts.Check(values.Length >= 0, "values length must be greater than 0");
            if (values.Length == 0)
            {
                return Parameter.FromObject(this.Choices[0]);
            }

            var param = this._option.SampleFromFeatureSpace(values);
            return Parameter.FromObject(this.Choices[param.AsType<int>()]);
        }
    }
}
