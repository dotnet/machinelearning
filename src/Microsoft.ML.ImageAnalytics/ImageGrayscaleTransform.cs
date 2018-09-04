// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;

[assembly: LoadableClass(ImageGrayscaleTransform.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransform), typeof(ImageGrayscaleTransform.Arguments), typeof(SignatureDataTransform),
    ImageGrayscaleTransform.UserName, "ImageGrayscaleTransform", "ImageGrayscale")]

[assembly: LoadableClass(ImageGrayscaleTransform.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransform), null, typeof(SignatureLoadDataTransform),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageGrayscaleTransform), null, typeof(SignatureLoadModel),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageGrayscaleTransform), null, typeof(SignatureLoadRowMapper),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    // REVIEW: Should it be separate transform or part of ImageResizerTransform?
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> type in IDataView and
    /// convert them to greyscale representation of the same image.
    /// </summary>
    public sealed class ImageGrayscaleTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        internal const string Summary = "Convert image into grayscale.";

        internal const string UserName = "Image Greyscale Transform";
        public const string LoaderSignature = "ImageGrayscaleTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGGRAYT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImageGrayscale";

        public IReadOnlyCollection<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        public ImageGrayscaleTransform(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.Column, nameof(args.Column));

            return new ImageGrayscaleTransform(env, args.Column.Select(x => (x.Source ?? x.Name, x.Name)).ToArray())
                .MakeDataTransform(input);
        }

        public static ImageGrayscaleTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new ImageGrayscaleTransform(host, ctx);
        }

        private ImageGrayscaleTransform(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.SaveColumns(ctx);
        }

        private static readonly ColorMatrix _grayscaleColorMatrix = new ColorMatrix(
                new float[][]
                {
                    new float[] {.3f, .3f, .3f, 0, 0},
                    new float[] {.59f, .59f, .59f, 0, 0},
                    new float[] {.11f, .11f, .11f, 0, 0},
                    new float[] {0, 0, 0, 1, 0},
                    new float[] {0, 0, 0, 0, 1}
                });

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, schema);

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema.GetColumnType(srcCol) is ImageType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, "image", inputSchema.GetColumnType(srcCol).ToString());
        }

        private sealed class Mapper : MapperBase
        {
            private ImageGrayscaleTransform _parent;

            public Mapper(ImageGrayscaleTransform parent, ISchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            public override RowMapperColumnInfo[] GetOutputColumns()
                => _parent.ColumnPairs.Select((x, idx) => new RowMapperColumnInfo(x.output, InputSchema.GetColumnType(ColMapNewToOld[idx]), null)).ToArray();

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(Bitmap);
                var getSrc = input.GetGetter<Bitmap>(ColMapNewToOld[iinfo]);

                disposer =
                    () =>
                    {
                        if (src != null)
                        {
                            src.Dispose();
                            src = null;
                        }
                    };

                ValueGetter<Bitmap> del =
                    (ref Bitmap dst) =>
                    {
                        if (dst != null)
                            dst.Dispose();

                        getSrc(ref src);
                        if (src == null || src.Height <= 0 || src.Width <= 0)
                            return;

                        dst = new Bitmap(src.Width, src.Height);
                        ImageAttributes attributes = new ImageAttributes();
                        attributes.SetColorMatrix(_grayscaleColorMatrix);
                        var srcRectangle = new Rectangle(0, 0, src.Width, src.Height);
                        using (var g = Graphics.FromImage(dst))
                        {
                            g.DrawImage(src, srcRectangle, 0, 0, src.Width, src.Height, GraphicsUnit.Pixel, attributes);
                        }
                        Contracts.Assert(dst.Width == src.Width && dst.Height == src.Height);
                    };

                return del;
            }
        }
    }

    public sealed class ImageGrayscaleEstimator : TrivialEstimator<ImageGrayscaleTransform>
    {
        public ImageGrayscaleEstimator(IHostEnvironment env, params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageGrayscaleEstimator)), new ImageGrayscaleTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                var col = inputSchema.FindColumn(colInfo.input);

                if (col == null)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input, new ImageType().ToString(), col.GetTypeString());

                result[colInfo.output] = new SchemaShape.Column(colInfo.output, col.Kind, col.ItemType, col.IsKey, col.MetadataKinds);
            }

            return new SchemaShape(result.Values);
        }

        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        internal sealed class OutPipelineColumn<T> : Scalar<T>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutPipelineColumn(Scalar<T> input)
    : base(Reconciler.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(typeof(T) == typeof(Bitmap) || typeof(T) == typeof(UnknownSizeBitmap));
                Input = input;
            }
        }

        /// <summary>
        /// Reconciler to an <see cref="ImageGrayscaleEstimator"/> for the <see cref="PipelineColumn"/>.
        /// </summary>
        /// <remarks>Because we want to use the same reconciler for </remarks>
        /// <see cref="ImageStaticPipe.AsGrayscale(Scalar{Bitmap})"/>
        /// <see cref="ImageStaticPipe.AsGrayscale(Scalar{UnknownSizeBitmap})"/>
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
                var cols = new (string input, string output)[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (IColInput)toOutput[i];
                    cols[i] = (inputNames[outCol.Input], outputNames[toOutput[i]]);
                }
                return new ImageGrayscaleEstimator(env, cols);
            }
        }
    }
}
