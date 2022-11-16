// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics.Contracts;
using System.Linq;

#nullable enable

namespace Microsoft.ML.SearchSpace.Option
{
    /// <summary>
    /// This class represent option for discrete value, such as string, enum, etc..
    /// </summary>
    public sealed class ChoiceOption : OptionBase
    {
        private readonly UniformSingleOption _option;

        /// <summary>
        /// Create <see cref="ChoiceOption"/> with <paramref name="choices"/>
        /// </summary>
        public ChoiceOption(params object[] choices)
        {
            Contract.Assert(choices.Length > 0 && choices.Length < 1074, "the length of choices must be (0, 1074)");
            var distinctChoices = choices.Distinct();
            Contract.Assert(distinctChoices.Count() == choices.Length, "choices must not contain repeated values");

            Choices = distinctChoices.Select(o => Parameter.FromObject(o)).ToArray();
            _option = new UniformSingleOption(0, Choices.Length);
            Default = Enumerable.Repeat(0.0, FeatureSpaceDim).ToArray();
        }

        /// <summary>
        /// Create <see cref="ChoiceOption"/> with <paramref name="choices"/> and <paramref name="defaultChoice"/>.
        /// </summary>
        public ChoiceOption(object[] choices, object? defaultChoice)
            : this(choices)
        {
            if (defaultChoice != null)
            {
                Default = MappingToFeatureSpace(Parameter.FromObject(defaultChoice));
            }
        }

        /// <summary>
        /// Get all choices.
        /// </summary>
        public Parameter[] Choices { get; }

        /// <inheritdoc/>
        public override int FeatureSpaceDim => Choices.Length == 1 ? 0 : 1;

        /// <inheritdoc/>
        public override int?[] Step => new int?[] { Choices.Length };

        /// <inheritdoc/>
        public override double[] MappingToFeatureSpace(Parameter param)
        {
            if (FeatureSpaceDim == 0)
            {
                return new double[0];
            }

            var x = Array.IndexOf(Choices, param);
            Contract.Assert(x >= 0, $"{param} not contains");

            return _option.MappingToFeatureSpace(Parameter.FromInt(x));
        }

        /// <inheritdoc/>
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            Contract.Assert(values.Length >= 0, "values length must be greater than 0");
            if (values.Length == 0)
            {
                return Choices[0];
            }

            var param = _option.SampleFromFeatureSpace(values);
            var value = param.AsType<float>();
            var idx = Convert.ToInt32(Math.Floor(value));

            // idx will be equal to choices.length if value is [1]
            // so we need to handle special case here.
            if (idx >= Choices.Length)
            {
                idx = Choices.Length - 1;
            }
            return Choices[idx];
        }
    }
}
