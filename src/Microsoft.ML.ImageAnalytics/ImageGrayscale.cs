// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(ImageGrayscalingTransformer.Summary, typeof(IDataTransform), typeof(ImageGrayscalingTransformer), typeof(ImageGrayscalingTransformer.Options), typeof(SignatureDataTransform),
    ImageGrayscalingTransformer.UserName, "ImageGrayscaleTransform", "ImageGrayscale")]

[assembly: LoadableClass(ImageGrayscalingTransformer.Summary, typeof(IDataTransform), typeof(ImageGrayscalingTransformer), null, typeof(SignatureLoadDataTransform),
    ImageGrayscalingTransformer.UserName, ImageGrayscalingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageGrayscalingTransformer), null, typeof(SignatureLoadModel),
    ImageGrayscalingTransformer.UserName, ImageGrayscalingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageGrayscalingTransformer), null, typeof(SignatureLoadRowMapper),
    ImageGrayscalingTransformer.UserName, ImageGrayscalingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Image
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    // REVIEW: Should it be separate transform or part of ImageResizerTransform?
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="ImageGrayscalingEstimator" />.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="ITransformer.Transform(IDataView)"/> converts the image to grayscale.
    /// The images might be converted to grayscale to reduce the complexity of the model.
    /// The grayed out images contain less information to process than the colored images.
    /// Another use case for converting to grayscale is to generate new images out of the existing ones, so you can have a larger dataset,
    /// a technique known as <a href = "http://www.stat.harvard.edu/Faculty_Content/meng/JCGS01.pdf"> data augmentation</a>.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// <a href="https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started"> examples in the machinelearning-samples github repository.</a>
    /// <seealso cref = "ImageEstimatorsCatalog" />
    /// </remarks>
    public sealed class ImageGrayscalingTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal const string Summary = "Convert image into grayscale.";

        internal const string UserName = "Image Greyscale Transform";
        internal const string LoaderSignature = "ImageGrayscaleTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGGRAYT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageGrayscalingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "ImageGrayscale";

        /// <summary>
        /// The input and output column pairs passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>

        internal ImageGrayscalingTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.Columns, nameof(options.Columns));

            return new ImageGrayscalingTransformer(env, options.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ImageGrayscalingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new ImageGrayscalingTransformer(host, ctx);
        }

        private ImageGrayscalingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
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

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is ImageType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, "image", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private ImageGrayscalingTransformer _parent;

            public Mapper(ImageGrayscalingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select((x, idx) => new DataViewSchema.DetachedColumn(x.outputColumnName, InputSchema[ColMapNewToOld[idx]].Type, null)).ToArray();

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(Bitmap);
                var getSrc = input.GetGetter<Bitmap>(input.Schema[ColMapNewToOld[iinfo]]);

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

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> that converts the image to grayscale.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="ImageGrayscalingEstimator"/>.
    /// The images might be converted to grayscale to reduce the complexity of the model.
    /// The grayed out images contain less information to process than the colored images.
    /// Another use case for converting to grayscale is to generate new images out of the existing ones, so you can have a larger dataset,
    /// a technique known as <a href = "http://www.stat.harvard.edu/Faculty_Content/meng/JCGS01.pdf"> data augmentation</a>.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// <a href="https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started"> examples in the machinelearning-samples github repository.</a>
    /// <seealso cref = "ImageEstimatorsCatalog" />
    /// </remarks >
    public sealed class ImageGrayscalingEstimator : TrivialEstimator<ImageGrayscalingTransformer>
    {

        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        [BestFriend]
        internal ImageGrayscalingEstimator(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageGrayscalingEstimator)), new ImageGrayscalingTransformer(env, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, new ImageType().ToString(), col.GetTypeString());

                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, col.Kind, col.ItemType, col.IsKey, col.Annotations);
            }

            return new SchemaShape(result.Values);
        }
    }
}
