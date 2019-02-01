// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(ImageGrayscaleTransformer.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransformer), typeof(ImageGrayscaleTransformer.Arguments), typeof(SignatureDataTransform),
    ImageGrayscaleTransformer.UserName, "ImageGrayscaleTransform", "ImageGrayscale")]

[assembly: LoadableClass(ImageGrayscaleTransformer.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransformer), null, typeof(SignatureLoadDataTransform),
    ImageGrayscaleTransformer.UserName, ImageGrayscaleTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageGrayscaleTransformer), null, typeof(SignatureLoadModel),
    ImageGrayscaleTransformer.UserName, ImageGrayscaleTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageGrayscaleTransformer), null, typeof(SignatureLoadRowMapper),
    ImageGrayscaleTransformer.UserName, ImageGrayscaleTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    // REVIEW: Should it be separate transform or part of ImageResizerTransform?
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> type in IDataView and
    /// convert them to greyscale representation of the same image.
    /// </summary>
    public sealed class ImageGrayscaleTransformer : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
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

        public class Arguments : TransformInputBase
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
                loaderAssemblyName: typeof(ImageGrayscaleTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "ImageGrayscale";

        public IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>

        public ImageGrayscaleTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.Columns, nameof(args.Columns));

            return new ImageGrayscaleTransformer(env, args.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ImageGrayscaleTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new ImageGrayscaleTransformer(host, ctx);
        }

        private ImageGrayscaleTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
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

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is ImageType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, "image", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private ImageGrayscaleTransformer _parent;

            public Mapper(ImageGrayscaleTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select((x, idx) => new Schema.DetachedColumn(x.outputColumnName, InputSchema[ColMapNewToOld[idx]].Type, null)).ToArray();

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
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

    /// <summary>
    /// Converts the images to grayscale.
    /// </summary>
    public sealed class ImageGrayscalingEstimator : TrivialEstimator<ImageGrayscaleTransformer>
    {

        /// <summary>
        /// Converts the images to grayscale.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">The name of the columns containing the image paths(first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        public ImageGrayscalingEstimator(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageGrayscalingEstimator)), new ImageGrayscaleTransformer(env, columns))
        {
        }

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

                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, col.Kind, col.ItemType, col.IsKey, col.Metadata);
            }

            return new SchemaShape(result.Values);
        }
    }
}
