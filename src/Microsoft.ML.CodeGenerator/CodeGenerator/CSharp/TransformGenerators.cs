// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal class Normalizer : TransformGeneratorBase
    {
        public Normalizer(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "NormalizeMinMax";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
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

        private const string ArgumentsName = "InputOutputColumnPair";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
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

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append("new []{");
            foreach (var col in InputColumns)
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

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
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

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class Hashing : TransformGeneratorBase
    {
        public Hashing(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "Conversion.Hash";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
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

        private const string ArgumentsName = "InputOutputColumnPair";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
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

        private const string ArgumentsName = "InputOutputColumnPair";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
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

        private const string ArgumentsName = "InputOutputColumnPair";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
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

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
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

        private const string ArgumentsName = "InputOutputColumnPair";

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(ArgumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
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

        public override string GenerateTransformer()
        {
            StringBuilder sb = new StringBuilder();
            string inputColumn = InputColumns.Count() > 0 ? InputColumns[0] : "\"Features\"";
            string outputColumn = OutputColumns.Count() > 0 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} are null");
            sb.Append(MethodName);
            sb.Append("(");
            sb.Append(outputColumn);
            sb.Append(",");
            sb.Append(inputColumn);
            sb.Append(")");
            return sb.ToString();
        }
    }

    internal class ImageLoadingRawBytes : TransformGeneratorBase
    {
        public ImageLoadingRawBytes(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "LoadRawImageBytes";

        public override string GenerateTransformer()
        {
            string inputColumn = InputColumns.Count() == 1 ? InputColumns[0] : throw new Exception($"input columns for the suggested transform: {MethodName} is not exactly one.");
            string outputColumn = OutputColumns.Count() == 1 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} it not exactly one.");

            // example: Transforms.LoadImages(output, inputfolder, input)
            return $"{MethodName}({outputColumn}, {@"null"}, {inputColumn})";
        }
    }

    internal class ImageLoading : TransformGeneratorBase
    {
        public ImageLoading(PipelineNode node) : base(node)
        {
        }

        internal override string MethodName => "LoadImages";

        public override string GenerateTransformer()
        {
            string inputColumn = InputColumns.Count() == 1 ? InputColumns[0] : throw new Exception($"input columns for the suggested transform: {MethodName} is not exactly one.");
            string outputColumn = OutputColumns.Count() == 1 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} it not exactly one.");

            // example: Transforms.LoadImages(output, inputfolder, input)
            return $"{MethodName}({outputColumn}, {@"null"}, {inputColumn})";
        }
    }

    internal class ImageResizing : TransformGeneratorBase
    {
        public ImageResizing(PipelineNode node) : base(node) { }
        internal override string MethodName => "ResizeImages";

        public override string GenerateTransformer()
        {
            return @"ResizeImages(""ImageSource_featurized"", 224, 224, ""ImageSource_featurized"")";
        }
    }

    internal class ObjectDetectionImageResizing : TransformGeneratorBase
    {
        public ObjectDetectionImageResizing(PipelineNode node) : base(node) { }
        internal override string MethodName => "ResizeImages";

        public override string GenerateTransformer()
        {
            return @"ResizeImages(outputColumnName: ""ImageSource_featurized"", imageWidth: 800, imageHeight: 600, inputColumnName: ""ImageSource_featurized"")";
        }
    }

    internal class PixelExtract : TransformGeneratorBase
    {
        public PixelExtract(PipelineNode node) : base(node) { }
        internal override string MethodName => "ExtractPixels";

        public override string GenerateTransformer()
        {
            string inputColumn = InputColumns.Count() == 1 ? InputColumns[0] : throw new Exception($"input columns for the suggested transform: {MethodName} is not exactly one.");
            string outputColumn = OutputColumns.Count() == 1 ? OutputColumns[0] : throw new Exception($"output columns for the suggested transform: {MethodName} it not exactly one.");
            return $"ExtractPixels({outputColumn}, {inputColumn})";
        }
    }

    internal class ApplyOnnxModel : TransformGeneratorBase
    {
        public ApplyOnnxModel(PipelineNode node) : base(node) { }
        internal override string MethodName => "ApplyOnnxModel";

        public override string GenerateTransformer()
        {
            // TODO ONNX_MODEL is fixed in this transformer, maybe update it to accept a real onnx model path.
            return $"ApplyOnnxModel(modelFile: ONNX_MODEL)";
        }
    }
}
