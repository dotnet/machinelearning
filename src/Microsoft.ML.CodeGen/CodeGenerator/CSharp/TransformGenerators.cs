// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.Parameter;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal class Normalizer : TransformGeneratorBase
    {
        [Parameter(1)]
        private NameParameter Input { get; set; }

        [Parameter(0)]
        private NameParameter Output { get; set; }

        public Normalizer(PipelineNode node) : base(node)
        {
            Input = new NameParameter()
            {
                ParameterName = "inputColumn",
                ParameterValue = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"",
            };

            Output = new NameParameter()
            {
                ParameterName = "outputColumn",
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };
        }

        internal override string MethodName => "NormalizeMinMax";
    }

    internal class OneHotEncoding : TransformGeneratorBase
    {
        [Parameter(0)]
        private InputOutputColumnPairParameter InputOutputColumnPair { get; set; }

        public OneHotEncoding(PipelineNode node) : base(node)
        {
            InputOutputColumnPair = new InputOutputColumnPairParameter()
            {
                OutputColumns = OutputColumns,
                InputColumns = InputColumns,
            };
        }

        internal override string MethodName => "Categorical.OneHotEncoding";
    }

    internal class ColumnConcat : TransformGeneratorBase
    {
        [Parameter(0)]
        private NameParameter OutputColumnName { get; set; }

        [Parameter(1)]
        private NameArrayParameter InputColumnNames { get; set; }

        public ColumnConcat(PipelineNode node) : base(node)
        {
            OutputColumnName = new NameParameter()
            {
                ParameterName = "OutputColumnName",
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };

            InputColumnNames = new NameArrayParameter()
            {
                ParameterName = "InputColumnNames",
                ArrayParameterValue = InputColumns,
            };
        }

        internal override string MethodName => "Concatenate";
    }

    internal class ColumnCopying : TransformGeneratorBase
    {
        [Parameter(0)]
        private NameParameter OutputColumnName { get; set; }

        [Parameter(1)]
        private NameParameter InputColumnName { get; set; }

        public ColumnCopying(PipelineNode node) : base(node)
        {
            OutputColumnName = new NameParameter()
            {
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };

            InputColumnName = new NameParameter()
            {
                ParameterValue = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"",
            };
        }

        internal override string MethodName => "CopyColumns";
    }

    internal class KeyToValueMapping : TransformGeneratorBase
    {
        [Parameter(0)]
        private NameParameter OutputColumnName { get; set; }

        [Parameter(1)]
        private NameParameter InputColumnName { get; set; }

        public KeyToValueMapping(PipelineNode node) : base(node)
        {
            OutputColumnName = new NameParameter()
            {
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };

            InputColumnName = new NameParameter()
            {
                ParameterValue = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"",
            };
        }

        internal override string MethodName => "Conversion.MapKeyToValue";
    }

    internal class MissingValueIndicator : TransformGeneratorBase
    {
        [Parameter(0)]
        private InputOutputColumnPairParameter InputOutputColumnPair { get; set; }

        public MissingValueIndicator(PipelineNode node) : base(node)
        {
            InputOutputColumnPair = new InputOutputColumnPairParameter()
            {
                InputColumns = InputColumns,
                OutputColumns = OutputColumns,
            };
        }

        internal override string MethodName => "IndicateMissingValues";
    }

    internal class MissingValueReplacer : TransformGeneratorBase
    {
        [Parameter(0)]
        private InputOutputColumnPairParameter InputOutputColumnPair { get; set; }

        public MissingValueReplacer(PipelineNode node) : base(node)
        {
            InputOutputColumnPair = new InputOutputColumnPairParameter()
            {
                InputColumns = InputColumns,
                OutputColumns = OutputColumns,
            };
        }

        internal override string MethodName => "ReplaceMissingValues";
    }

    internal class OneHotHashEncoding : TransformGeneratorBase
    {
        [Parameter(0)]
        private InputOutputColumnPairParameter InputOutputColumnPair { get; set; }

        public OneHotHashEncoding(PipelineNode node) : base(node)
        {
            InputOutputColumnPair = new InputOutputColumnPairParameter()
            {
                InputColumns = InputColumns,
                OutputColumns = OutputColumns,
            };
        }

        internal override string MethodName => "Categorical.OneHotHashEncoding";
    }

    internal class TextFeaturizing : TransformGeneratorBase
    {
        [Parameter(0)]
        private NameParameter OutputColumnName { get; set; }

        [Parameter(1)]
        private NameParameter InputColumnName { get; set; }

        public TextFeaturizing(PipelineNode node) : base(node)
        {
            OutputColumnName = new NameParameter()
            {
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };

            InputColumnName = new NameParameter()
            {
                ParameterValue = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"",
            };
        }

        internal override string MethodName => "Text.FeaturizeText";
    }

    internal class TypeConverting : TransformGeneratorBase
    {
        [Parameter(0)]
        private InputOutputColumnPairParameter InputOutputColumnPair { get; set; }

        public TypeConverting(PipelineNode node) : base(node)
        {
            InputOutputColumnPair = new InputOutputColumnPairParameter()
            {
                InputColumns = InputColumns,
                OutputColumns = OutputColumns,
            };
        }

        internal override string MethodName => "Conversion.ConvertType";
    }

    internal class ValueToKeyMapping : TransformGeneratorBase
    {
        [Parameter(0)]
        private NameParameter OutputColumnName { get; set; }

        [Parameter(1)]
        private NameParameter InputColumnName { get; set; }

        public ValueToKeyMapping(PipelineNode node) : base(node)
        {
            OutputColumnName = new NameParameter()
            {
                ParameterValue = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null"),
            };

            InputColumnName = new NameParameter()
            {
                ParameterValue = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"",
            };
        }

        internal override string MethodName => "Conversion.MapValueToKey";
    }
}
