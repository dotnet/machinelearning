// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(ImageResizerTransformer.Summary, typeof(IDataTransform), typeof(ImageResizerTransformer), typeof(ImageResizerTransformer.Arguments),
    typeof(SignatureDataTransform), ImageResizerTransformer.UserName, "ImageResizerTransform", "ImageResizer")]

[assembly: LoadableClass(ImageResizerTransformer.Summary, typeof(IDataTransform), typeof(ImageResizerTransformer), null, typeof(SignatureLoadDataTransform),
    ImageResizerTransformer.UserName, ImageResizerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageResizerTransformer), null, typeof(SignatureLoadModel),
    ImageResizerTransformer.UserName, ImageResizerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageResizerTransformer), null, typeof(SignatureLoadRowMapper),
    ImageResizerTransformer.UserName, ImageResizerTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> and resize them to provided height and width.
    /// </summary>
    public sealed class ImageResizerTransformer : OneToOneTransformerBase
    {
        public enum ResizingKind : byte
        {
            [TGUI(Label = "Isotropic with Padding")]
            IsoPad = 0,

            [TGUI(Label = "Isotropic with Cropping")]
            IsoCrop = 1
        }

        public enum Anchor : byte
        {
            Right = 0,
            Left = 1,
            Top = 2,
            Bottom = 3,
            Center = 4
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the resized image", ShortName = "width")]
            public int? ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the resized image", ShortName = "height")]
            public int? ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Resizing method", ShortName = "scale")]
            public ResizingKind? Resizing;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Anchor for cropping", ShortName = "anchor")]
            public Anchor? CropAnchor;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ImageWidth != null || ImageHeight != null || Resizing != null || CropAnchor != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.Required, HelpText = "Resized width of the image", ShortName = "width")]
            public int ImageWidth;

            [Argument(ArgumentType.Required, HelpText = "Resized height of the image", ShortName = "height")]
            public int ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Resizing method", ShortName = "scale")]
            public ResizingKind Resizing = ResizingKind.IsoCrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Anchor for cropping", ShortName = "anchor")]
            public Anchor CropAnchor = Anchor.Center;
        }
        internal static class Defaults
        {
            public const ResizingKind Resizing = ResizingKind.IsoCrop;
            public const Anchor CropAnchor = Anchor.Center;
        }

        /// <summary>
        /// Describes how the transformer handles one image resize column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Name;
            public readonly string InputColumnName;

            public readonly int Width;
            public readonly int Height;
            public readonly ResizingKind Resizing;
            public readonly Anchor Anchor;
            public readonly ColumnType Type;

            /// <summary>
            /// Describes how the transformer handles one image resize column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="width">Width of resized image.</param>
            /// <param name="height">Height of resized image.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="resizing">What <see cref="ResizingKind"/> to use.</param>
            /// <param name="anchor">If <paramref name="resizing"/> set to <see cref="ResizingKind.IsoCrop"/> what anchor to use for cropping.</param>
            public ColumnInfo(string name, int width, int height, string inputColumnName = null, ResizingKind resizing = Defaults.Resizing, Anchor anchor = Defaults.CropAnchor)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckNonEmpty(inputColumnName, nameof(inputColumnName));
                Contracts.CheckUserArg(width > 0, nameof(Column.ImageWidth));
                Contracts.CheckUserArg(height > 0, nameof(Column.ImageHeight));
                Contracts.CheckUserArg(Enum.IsDefined(typeof(ResizingKind), resizing), nameof(Column.Resizing));
                Contracts.CheckUserArg(Enum.IsDefined(typeof(Anchor), anchor), nameof(Column.CropAnchor));

                Name = name;
                InputColumnName = inputColumnName;
                Width = width;
                Height = height;
                Resizing = resizing;
                Anchor = anchor;
                Type = new ImageType(Height, Width);
            }
        }

        internal const string Summary = "Scales an image to specified dimensions using one of the three scale types: isotropic with padding, "
            + "isotropic with cropping or anisotropic. In case of isotropic padding, transparent color is used to pad resulting image.";

        internal const string UserName = "Image Resizer Transform";
        internal const string LoaderSignature = "ImageScalerTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGSCALF",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verWrittenCur: 0x00010003, // No more sizeof(float)
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageResizerTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "ImageScaler";

        private readonly ColumnInfo[] _columns;

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        ///<summary>
        /// Resize image.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="imageWidth">Width of resized image.</param>
        /// <param name="imageHeight">Height of resized image.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="resizing">What <see cref="ResizingKind"/> to use.</param>
        /// <param name="cropAnchor">If <paramref name="resizing"/> set to <see cref="ResizingKind.IsoCrop"/> what anchor to use for cropping.</param>
        public ImageResizerTransformer(IHostEnvironment env, string outputColumnName,
            int imageWidth, int imageHeight, string inputColumnName = null, ResizingKind resizing = ResizingKind.IsoCrop, Anchor cropAnchor = Anchor.Center)
            : this(env, new ColumnInfo(outputColumnName, imageWidth, imageHeight, inputColumnName, resizing, cropAnchor))
        {
        }

        ///<summary>
        /// Resize image.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of image resizing for each column pair.</param>
        public ImageResizerTransformer(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Columns, nameof(args.Columns));

            var cols = new ColumnInfo[args.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Columns[i];
                cols[i] = new ColumnInfo(
                    item.Name,
                    item.ImageWidth ?? args.ImageWidth,
                    item.ImageHeight ?? args.ImageHeight,
                    item.Source ?? item.Name,
                    item.Resizing ?? args.Resizing,
                    item.CropAnchor ?? args.CropAnchor);
            }

            return new ImageResizerTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ImageResizerTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ImageResizerTransformer(host, ctx);
        }

        private ImageResizerTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>

            // for each added column
            //   int: width
            //   int: height
            //   byte: scaling kind
            //   byte: anchor

            _columns = new ColumnInfo[ColumnPairs.Length];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                int width = ctx.Reader.ReadInt32();
                Host.CheckDecode(width > 0);
                int height = ctx.Reader.ReadInt32();
                Host.CheckDecode(height > 0);
                var scale = (ResizingKind)ctx.Reader.ReadByte();
                Host.CheckDecode(Enum.IsDefined(typeof(ResizingKind), scale));
                var anchor = (Anchor)ctx.Reader.ReadByte();
                Host.CheckDecode(Enum.IsDefined(typeof(Anchor), anchor));
                _columns[i] = new ColumnInfo(ColumnPairs[i].outputColumnName, width, height, ColumnPairs[i].inputColumnName, scale, anchor);
            }
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

            // for each added column
            //   int: width
            //   int: height
            //   byte: scaling kind
            //   byte: anchor

            base.SaveColumns(ctx);

            foreach (var col in _columns)
            {
                ctx.Writer.Write(col.Width);
                ctx.Writer.Write(col.Height);
                Contracts.Assert((ResizingKind)(byte)col.Resizing == col.Resizing);
                ctx.Writer.Write((byte)col.Resizing);
                Contracts.Assert((Anchor)(byte)col.Anchor == col.Anchor);
                ctx.Writer.Write((byte)col.Anchor);
            }
        }

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is ImageType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _columns[col].InputColumnName, "image", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageResizerTransformer _parent;

            public Mapper(ImageResizerTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
                => _parent._columns.Select(x => new Schema.DetachedColumn(x.Name, x.Type, null)).ToArray();

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                var src = default(Bitmap);
                var getSrc = input.GetGetter<Bitmap>(ColMapNewToOld[iinfo]);
                var info = _parent._columns[iinfo];

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
                        if (src.Height == info.Height && src.Width == info.Width)
                        {
                            dst = src;
                            return;
                        }

                        int sourceWidth = src.Width;
                        int sourceHeight = src.Height;
                        int sourceX = 0;
                        int sourceY = 0;
                        int destX = 0;
                        int destY = 0;
                        int destWidth = 0;
                        int destHeight = 0;
                        float aspect = 0;
                        float widthAspect = 0;
                        float heightAspect = 0;

                        widthAspect = (float)info.Width / sourceWidth;
                        heightAspect = (float)info.Height / sourceHeight;

                        if (info.Resizing == ResizingKind.IsoPad)
                        {
                            widthAspect = (float)info.Width / sourceWidth;
                            heightAspect = (float)info.Height / sourceHeight;
                            if (heightAspect < widthAspect)
                            {
                                aspect = heightAspect;
                                destX = (int)((info.Width - (sourceWidth * aspect)) / 2);
                            }
                            else
                            {
                                aspect = widthAspect;
                                destY = (int)((info.Height - (sourceHeight * aspect)) / 2);
                            }

                            destWidth = (int)(sourceWidth * aspect);
                            destHeight = (int)(sourceHeight * aspect);
                        }
                        else
                        {
                            if (heightAspect < widthAspect)
                            {
                                aspect = widthAspect;
                                switch (info.Anchor)
                                {
                                    case Anchor.Top:
                                        destY = 0;
                                        break;
                                    case Anchor.Bottom:
                                        destY = (int)(info.Height - (sourceHeight * aspect));
                                        break;
                                    default:
                                        destY = (int)((info.Height - (sourceHeight * aspect)) / 2);
                                        break;
                                }
                            }
                            else
                            {
                                aspect = heightAspect;
                                switch (info.Anchor)
                                {
                                    case Anchor.Left:
                                        destX = 0;
                                        break;
                                    case Anchor.Right:
                                        destX = (int)(info.Width - (sourceWidth * aspect));
                                        break;
                                    default:
                                        destX = (int)((info.Width - (sourceWidth * aspect)) / 2);
                                        break;
                                }
                            }

                            destWidth = (int)(sourceWidth * aspect);
                            destHeight = (int)(sourceHeight * aspect);
                        }
                        dst = new Bitmap(info.Width, info.Height, src.PixelFormat);
                        var srcRectangle = new Rectangle(sourceX, sourceY, sourceWidth, sourceHeight);
                        var destRectangle = new Rectangle(destX, destY, destWidth, destHeight);
                        using (var g = Graphics.FromImage(dst))
                        {
                            g.DrawImage(src, destRectangle, srcRectangle, GraphicsUnit.Pixel);
                        }
                        Contracts.Assert(dst.Width == info.Width && dst.Height == info.Height);
                    };

                return del;
            }
        }
    }

    /// <summary>
    /// Estimator which resizes images.
    /// </summary>
    public sealed class ImageResizingEstimator : TrivialEstimator<ImageResizerTransformer>
    {
        /// <summary>
        /// Estimator which resizes images.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="imageWidth">Width of resized image.</param>
        /// <param name="imageHeight">Height of resized image.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="resizing">What <see cref="ImageResizerTransformer.ResizingKind"/> to use.</param>
        /// <param name="cropAnchor">If <paramref name="resizing"/> set to <see cref="ImageResizerTransformer.ResizingKind.IsoCrop"/> what anchor to use for cropping.</param>
        public ImageResizingEstimator(IHostEnvironment env,
            string outputColumnName,
            int imageWidth,
            int imageHeight,
            string inputColumnName = null,
            ImageResizerTransformer.ResizingKind resizing = ImageResizerTransformer.Defaults.Resizing,
            ImageResizerTransformer.Anchor cropAnchor = ImageResizerTransformer.Defaults.CropAnchor)
            : this(env, new ImageResizerTransformer(env, outputColumnName, imageWidth, imageHeight, inputColumnName, resizing, cropAnchor))
        {
        }

        /// <summary>
        /// Estimator which resizes images.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of image resizing for each column pair.</param>
        public ImageResizingEstimator(IHostEnvironment env, params ImageResizerTransformer.ColumnInfo[] columns)
            : this(env, new ImageResizerTransformer(env, columns))
        {
        }

        private ImageResizingEstimator(IHostEnvironment env, ImageResizerTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageResizingEstimator)), transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, new ImageType().ToString(), col.GetTypeString());

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Scalar, colInfo.Type, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
