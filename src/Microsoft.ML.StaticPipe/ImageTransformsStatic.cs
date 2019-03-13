// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Image;

namespace Microsoft.ML.StaticPipe
{
    public static class ImageLoadingStaticExtensions
    {
        internal sealed class OutPipelineColumn : Custom<UnknownSizeBitmap>
        {
            private readonly Scalar<string> _input;

            public OutPipelineColumn(Scalar<string> path, string relativeTo)
                : base(new Reconciler(relativeTo), path)
            {
                Contracts.AssertValue(path);
                _input = path;
            }

            /// <summary>
            /// Reconciler to an <see cref="ImageLoadingEstimator"/> for the <see cref="PipelineColumn"/>.
            /// </summary>
            /// <remarks>
            /// We must create a new reconciler per call, because the relative path of <see cref="ImageLoadingTransformer.Options.ImageFolder"/>
            /// is considered a transform-wide option, as it is not specified in <see cref="ImageLoadingTransformer.Column"/>. However, we still
            /// implement <see cref="IEquatable{T}"/> so the analyzer can still equate two of these things if they happen to share the same
            /// path, so we can be a bit more efficient with respect to our estimator declarations.
            /// </remarks>
            /// <see cref="ImageStaticPipe.LoadAsImage(Scalar{string}, string)"/>
            private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
            {
                private readonly string _relTo;

                public Reconciler(string relativeTo)
                {
                    Contracts.AssertValueOrNull(relativeTo);
                    _relTo = relativeTo;
                }

                public bool Equals(Reconciler other)
                    => other != null && other._relTo == _relTo;

                public override bool Equals(object obj)
                    => obj is Reconciler other && Equals(other);

                public override int GetHashCode()
                    => _relTo?.GetHashCode() ?? 0;

                public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                    PipelineColumn[] toOutput,
                    IReadOnlyDictionary<PipelineColumn, string> inputNames,
                    IReadOnlyDictionary<PipelineColumn, string> outputNames,
                    IReadOnlyCollection<string> usedNames)
                {
                    var cols = new (string outputColumnName, string inputColumnName)[toOutput.Length];
                    for (int i = 0; i < toOutput.Length; ++i)
                    {
                        var outCol = (OutPipelineColumn)toOutput[i];
                        cols[i] = (outputNames[outCol], inputNames[outCol._input]);
                    }
                    return new ImageLoadingEstimator(env, _relTo, cols);
                }
            }
        }
    }

    public static class ImageGreyScalingStaticExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        internal sealed class OutPipelineColumn<T> : Custom<T>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutPipelineColumn(Custom<T> input)
                : base(Reconciler.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(typeof(T) == typeof(Bitmap) || typeof(T) == typeof(UnknownSizeBitmap));
                Input = input;
            }
        }

        /// <summary>
        /// Reconciler to an <see cref="ImageGrayscalingEstimator"/> for the <see cref="PipelineColumn"/>.
        /// </summary>
        /// <remarks>Because we want to use the same reconciler for </remarks>
        /// <see cref="ImageStaticPipe.AsGrayscale(Custom{Bitmap})"/>
        /// <see cref="ImageStaticPipe.AsGrayscale(Custom{UnknownSizeBitmap})"/>
        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var cols = new (string outputColumnName, string inputColumnName)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (IColInput)toOutput[i];
                    cols[i] = (outputNames[toOutput[i]], inputNames[outCol.Input]);
                }
                return new ImageGrayscalingEstimator(env, cols);
            }
        }
    }

    public static class ImageResizingStaticExtensions
    {
        internal sealed class OutPipelineColumn : Custom<Bitmap>
        {
            private readonly PipelineColumn _input;
            private readonly int _width;
            private readonly int _height;
            private readonly ImageResizingEstimator.ResizingKind _resizing;
            private readonly ImageResizingEstimator.Anchor _cropAnchor;

            public OutPipelineColumn(PipelineColumn input, int width, int height,
            ImageResizingEstimator.ResizingKind resizing, ImageResizingEstimator.Anchor cropAnchor)
                : base(Reconciler.Inst, input)
            {
                Contracts.AssertValue(input);
                _input = input;
                _width = width;
                _height = height;
                _resizing = resizing;
                _cropAnchor = cropAnchor;
            }

            private ImageResizingEstimator.ColumnOptions MakeColumnOptions(string outputColumnName, string inputColumnName)
                => new ImageResizingEstimator.ColumnOptions(outputColumnName, _width, _height, inputColumnName, _resizing, _cropAnchor);

            /// <summary>
            /// Reconciler to an <see cref="ImageResizingTransformer"/> for the <see cref="PipelineColumn"/>.
            /// </summary>
            /// <seealso cref="ImageStaticPipe.Resize(Custom{Bitmap}, int, int, ImageResizingEstimator.ResizingKind, ImageResizingEstimator.Anchor)"/>
            /// <seealso cref="ImageStaticPipe.Resize(Custom{UnknownSizeBitmap}, int, int, ImageResizingEstimator.ResizingKind, ImageResizingEstimator.Anchor)"/>
            private sealed class Reconciler : EstimatorReconciler
            {
                public static Reconciler Inst = new Reconciler();

                private Reconciler()
                {
                }

                public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                    PipelineColumn[] toOutput,
                    IReadOnlyDictionary<PipelineColumn, string> inputNames,
                    IReadOnlyDictionary<PipelineColumn, string> outputNames,
                    IReadOnlyCollection<string> usedNames)
                {
                    var cols = new ImageResizingEstimator.ColumnOptions[toOutput.Length];
                    for (int i = 0; i < toOutput.Length; ++i)
                    {
                        var outCol = (OutPipelineColumn)toOutput[i];
                        cols[i] = outCol.MakeColumnOptions(outputNames[outCol], inputNames[outCol._input]);
                    }
                    return new ImageResizingEstimator(env, cols);
                }
            }
        }
    }

    public static class ImagePixelExtractingStaticExtensions
    {
        private interface IColInput
        {
            Custom<Bitmap> Input { get; }

            ImagePixelExtractingEstimator.ColumnOptions MakeColumnOptions(string outputColumnName, string inputColumnName);
        }

        internal sealed class OutPipelineColumn<T> : Vector<T>, IColInput
        {
            public Custom<Bitmap> Input { get; }
            private static readonly ImagePixelExtractingTransformer.Options _defaultArgs = new ImagePixelExtractingTransformer.Options();
            private readonly ImagePixelExtractingTransformer.Column _colParam;

            public OutPipelineColumn(Custom<Bitmap> input, ImagePixelExtractingTransformer.Column col)
                : base(Reconciler.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(byte));
                Input = input;
                _colParam = col;
            }

            public ImagePixelExtractingEstimator.ColumnOptions MakeColumnOptions(string outputColumnName, string inputColumnName)
            {
                // In principle, the analyzer should only call the the reconciler once for these columns.
                Contracts.Assert(_colParam.Source == null);
                Contracts.Assert(_colParam.Name == null);

                _colParam.Name = outputColumnName;
                _colParam.Source = inputColumnName;
                return new ImagePixelExtractingEstimator.ColumnOptions(_colParam, _defaultArgs);
            }
        }

        /// <summary>
        /// Reconciler to an <see cref="ImagePixelExtractingEstimator"/> for the <see cref="PipelineColumn"/>.
        /// </summary>
        /// <remarks>Because we want to use the same reconciler for </remarks>
        /// <see cref="ImageStaticPipe.ExtractPixels(Custom{Bitmap}, bool, bool, bool, bool, ImagePixelExtractingEstimator.ColorsOrder, bool, float, float)"/>
        /// <see cref="ImageStaticPipe.ExtractPixelsAsBytes(Custom{Bitmap}, bool, bool, bool, bool,ImagePixelExtractingEstimator.ColorsOrder, bool)"/>
        private sealed class Reconciler : EstimatorReconciler
        {
            /// <summary>
            /// Because there are no global settings that cannot be overridden, we can always just use the same reconciler.
            /// </summary>
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var cols = new ImagePixelExtractingEstimator.ColumnOptions[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (IColInput)toOutput[i];
                    cols[i] = outCol.MakeColumnOptions(outputNames[toOutput[i]], inputNames[outCol.Input]);
                }
                return new ImagePixelExtractingEstimator(env, cols);
            }
        }
    }
}
