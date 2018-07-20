// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(ImageResizerTransform.Summary, typeof(ImageResizerTransform), typeof(ImageResizerTransform.Arguments),
    typeof(SignatureDataTransform), ImageResizerTransform.UserName, "ImageResizerTransform", "ImageResizer")]

[assembly: LoadableClass(ImageResizerTransform.Summary, typeof(ImageResizerTransform), null, typeof(SignatureLoadDataTransform),
    ImageResizerTransform.UserName, ImageResizerTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> and resize them to provided height and width.
    /// </summary>
    public sealed class ImageResizerTransform : OneToOneTransformBase
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

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ImageWidth != null || ImageHeight != null || Resizing != null || CropAnchor != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Required, HelpText = "Resized width of the image", ShortName = "width")]
            public int ImageWidth;

            [Argument(ArgumentType.Required, HelpText = "Resized height of the image", ShortName = "height")]
            public int ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Resizing method", ShortName = "scale")]
            public ResizingKind Resizing = ResizingKind.IsoCrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Anchor for cropping", ShortName = "anchor")]
            public Anchor CropAnchor = Anchor.Center;
        }

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        private sealed class ColInfoEx
        {
            public readonly int Width;
            public readonly int Height;
            public readonly ResizingKind Scale;
            public readonly Anchor Anchor;
            public readonly ColumnType Type;

            public ColInfoEx(int width, int height, ResizingKind scale, Anchor anchor)
            {
                Contracts.CheckUserArg(width > 0, nameof(Column.ImageWidth));
                Contracts.CheckUserArg(height > 0, nameof(Column.ImageHeight));
                Contracts.CheckUserArg(Enum.IsDefined(typeof(ResizingKind), scale), nameof(Column.Resizing));
                Contracts.CheckUserArg(Enum.IsDefined(typeof(Anchor), anchor), nameof(Column.CropAnchor));

                Width = width;
                Height = height;
                Scale = scale;
                Anchor = anchor;
                Type = new ImageType(Height, Width);
            }
        }

        internal const string Summary = "Scales an image to specified dimensions using one of the three scale types: isotropic with padding, "
            + "isotropic with cropping or anisotropic. In case of isotropic padding, transparent color is used to pad resulting image.";

        internal const string UserName = "Image Resizer Transform";
        public const string LoaderSignature = "ImageScalerTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGSCALF",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImageScaler";

        // This is parallel to Infos.
        private readonly ColInfoEx[] _exes;

        /// Public constructor corresponding to SignatureDataTransform.
        public ImageResizerTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column, input, t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                var item = args.Column[i];
                _exes[i] = new ColInfoEx(
                    item.ImageWidth ?? args.ImageWidth,
                    item.ImageHeight ?? args.ImageHeight,
                    item.Resizing ?? args.Resizing,
                    item.CropAnchor ?? args.CropAnchor);
            }
            Metadata.Seal();
        }

        private ImageResizerTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // for each added column
            //   int: width
            //   int: height
            //   byte: scaling kind
            Host.AssertNonEmpty(Infos);

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                int width = ctx.Reader.ReadInt32();
                Host.CheckDecode(width > 0);
                int height = ctx.Reader.ReadInt32();
                Host.CheckDecode(height > 0);
                var scale = (ResizingKind)ctx.Reader.ReadByte();
                Host.CheckDecode(Enum.IsDefined(typeof(ResizingKind), scale));
                var anchor = (Anchor)ctx.Reader.ReadByte();
                Host.CheckDecode(Enum.IsDefined(typeof(Anchor), anchor));
                _exes[i] = new ColInfoEx(width, height, scale, anchor);
            }
            Metadata.Seal();
        }

        public static ImageResizerTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Single));
                    return new ImageResizerTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // for each added column
            //   int: width
            //   int: height
            //   byte: scaling kind
            ctx.Writer.Write(sizeof(Single));
            SaveBase(ctx);

            Host.Assert(_exes.Length == Infos.Length);
            for (int i = 0; i < _exes.Length; i++)
            {
                var ex = _exes[i];
                ctx.Writer.Write(ex.Width);
                ctx.Writer.Write(ex.Height);
                Host.Assert((ResizingKind)(byte)ex.Scale == ex.Scale);
                ctx.Writer.Write((byte)ex.Scale);
                Host.Assert((Anchor)(byte)ex.Anchor == ex.Anchor);
                ctx.Writer.Write((byte)ex.Anchor);
            }
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo && iinfo < Infos.Length);
            return _exes[iinfo].Type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            var src = default(Bitmap);
            var getSrc = GetSrcGetter<Bitmap>(input, iinfo);
            var ex = _exes[iinfo];

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
                    if (src.Height == ex.Height && src.Width == ex.Width)
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

                    widthAspect = (float)ex.Width / sourceWidth;
                    heightAspect = (float)ex.Height / sourceHeight;

                    if (ex.Scale == ResizingKind.IsoPad)
                    {
                        widthAspect = (float)ex.Width / sourceWidth;
                        heightAspect = (float)ex.Height / sourceHeight;
                        if (heightAspect < widthAspect)
                        {
                            aspect = heightAspect;
                            destX = (int)((ex.Width - (sourceWidth * aspect)) / 2);
                        }
                        else
                        {
                            aspect = widthAspect;
                            destY = (int)((ex.Height - (sourceHeight * aspect)) / 2);
                        }

                        destWidth = (int)(sourceWidth * aspect);
                        destHeight = (int)(sourceHeight * aspect);
                    }
                    else
                    {
                        if (heightAspect < widthAspect)
                        {
                            aspect = widthAspect;
                            switch (ex.Anchor)
                            {
                                case Anchor.Top:
                                    destY = 0;
                                    break;
                                case Anchor.Bottom:
                                    destY = (int)(ex.Height - (sourceHeight * aspect));
                                    break;
                                default:
                                    destY = (int)((ex.Height - (sourceHeight * aspect)) / 2);
                                    break;
                            }
                        }
                        else
                        {
                            aspect = heightAspect;
                            switch (ex.Anchor)
                            {
                                case Anchor.Left:
                                    destX = 0;
                                    break;
                                case Anchor.Right:
                                    destX = (int)(ex.Width - (sourceWidth * aspect));
                                    break;
                                default:
                                    destX = (int)((ex.Width - (sourceWidth * aspect)) / 2);
                                    break;
                            }
                        }

                        destWidth = (int)(sourceWidth * aspect);
                        destHeight = (int)(sourceHeight * aspect);
                    }
                    dst = new Bitmap(ex.Width, ex.Height);
                    var srcRectangle = new Rectangle(sourceX, sourceY, sourceWidth, sourceHeight);
                    var destRectangle = new Rectangle(destX, destY, destWidth, destHeight);
                    using (var g = Graphics.FromImage(dst))
                    {
                        g.DrawImage(src, destRectangle, srcRectangle, GraphicsUnit.Pixel);
                    }
                    Host.Assert(dst.Width == ex.Width && dst.Height == ex.Height);
                };

            return del;
        }
    }
}
