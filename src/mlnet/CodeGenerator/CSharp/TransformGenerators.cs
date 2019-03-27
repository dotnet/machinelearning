// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class Normalizer : TransformGeneratorBase
    {
        public Normalizer(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Normalize";

        internal override string[] Usings => null;

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class OneHotEncoding : TransformGeneratorBase
    {
        public OneHotEncoding(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Categorical.OneHotEncoding";

        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        private string ArgumentsName = "OneHotEncodingEstimator.ColumnOptions";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < inputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(outputColumns[i]);
                sb.Append(",");
                sb.Append(inputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,

            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class ColumnConcat : TransformGeneratorBase
    {
        public ColumnConcat(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Concatenate";

        internal override string[] Usings => null;

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append("new []{");
            foreach (var col in inputColumns)
            {
                sb.Append(col);
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1);
            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class ColumnCopying : TransformGeneratorBase
    {
        public ColumnCopying(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "CopyColumns";

        internal override string[] Usings => null;

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class KeyToValueMapping : TransformGeneratorBase
    {
        public KeyToValueMapping(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Conversion.MapKeyToValue";

        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class MissingValueIndicator : TransformGeneratorBase
    {
        public MissingValueIndicator(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "IndicateMissingValues";

        internal override string[] Usings => null;

        private string ArgumentsName = "ColumnOptions";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < inputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(outputColumns[i]);
                sb.Append(",");
                sb.Append(inputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,
            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class MissingValueReplacer : TransformGeneratorBase
    {
        public MissingValueReplacer(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "ReplaceMissingValues";

        private string ArgumentsName = "MissingValueReplacingEstimator.ColumnOptions";
        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < inputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(outputColumns[i]);
                sb.Append(",");
                sb.Append(inputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,

            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class OneHotHashEncoding : TransformGeneratorBase
    {
        public OneHotHashEncoding(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Categorical.OneHotHashEncoding";

        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        private string ArgumentsName = "OneHotHashEncodingEstimator.ColumnOptions";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < inputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(outputColumns[i]);
                sb.Append(",");
                sb.Append(inputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,

            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class TextFeaturizing : TransformGeneratorBase
    {
        public TextFeaturizing(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Text.FeaturizeText";

        internal override string[] Usings => null;

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class TypeConverting : TransformGeneratorBase
    {
        public TypeConverting(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Conversion.ConvertType";

        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        private string ArgumentsName = "TypeConvertingEstimator.ColumnOptions";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < inputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(outputColumns[i]);
                sb.Append(",");
                sb.Append("DataKind.Single");
                sb.Append(",");
                sb.Append(inputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,

            sb.Append("}");
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class ValueToKeyMapping : TransformGeneratorBase
    {
        public ValueToKeyMapping(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Conversion.MapValueToKey";

        internal override string[] Usings => new string[] { "using Microsoft.ML.Transforms;\r\n" };

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = inputColumns.Count() > 0 ? inputColumns[0] : "\"Features\"";
            string outputColumn = outputColumns.Count() > 0 ? outputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

}
