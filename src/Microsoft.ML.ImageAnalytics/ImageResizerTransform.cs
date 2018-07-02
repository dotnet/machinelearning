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
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.ImageAnalytics;


[assembly: LoadableClass(ImageResizerTransform.Summary, typeof(ImageResizerTransform), typeof(ImageResizerTransform.Arguments), typeof(SignatureDataTransform),
    ImageResizerTransform.UserName, "ImageResizerTransform", "ImageResizer")]

[assembly: LoadableClass(ImageResizerTransform.Summary, typeof(ImageResizerTransform), null, typeof(SignatureLoadDataTransform),
    ImageResizerTransform.UserName, ImageResizerTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW coeseanu: Rewrite as LambdaTransform to simplify.
    public sealed class ImageResizerTransform : OneToOneTransformBase
    {
        public enum ResizingKind : byte
        {
            [TGUI(Label = "Isotropic with Padding")]
            IsoPad = 0,

            [TGUI(Label = "Isotropic with Cropping")]
            IsoCrop = 1,

            [TGUI(Label = "Anisotropic")]
            Aniso = 2,
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the resized image", ShortName = "width")]
            public int? ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the resized image", ShortName = "height")]
            public int? ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Resizing method", ShortName = "scale")]
            public ResizingKind? Resizing;

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
                if (ImageWidth != null || ImageHeight != null || Resizing != null)
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
        }

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        private sealed class ColInfoEx
        {
            public readonly int Width;
            public readonly int Height;
            public readonly ResizingKind Scale;
            public readonly ColumnType Type;

            public ColInfoEx(int width, int height, ResizingKind scale)
            {
                Contracts.CheckUserArg(width > 0, nameof(Column.ImageWidth));
                Contracts.CheckUserArg(height > 0, nameof(Column.ImageHeight));
                Contracts.CheckUserArg(Enum.IsDefined(typeof(ResizingKind), scale), nameof(Column.Resizing));

                Width = width;
                Height = height;
                Scale = scale;
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
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImageScaler";

        // This is parallel to Infos.
        private readonly ColInfoEx[] _exes;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
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
                    item.Resizing ?? args.Resizing);
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
                _exes[i] = new ColInfoEx(width, height, scale);
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
            }
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo && iinfo < Infos.Length);
            return _exes[iinfo].Type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(ch, "ch");
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

            ValueGetter<Image> del =
                (ref Image dst) =>
                {
                    if (dst != null)
                        dst.Dispose();

                    getSrc(ref src);
                    if (src == null || src.Height <= 0 || src.Width <= 0)
                        return;

                    int x = 0;
                    int y = 0;
                    int w = ex.Width;
                    int h = ex.Height;
                    bool pad = ex.Scale == ResizingKind.IsoPad;
                    if (ex.Scale == ResizingKind.IsoPad || ex.Scale == ResizingKind.IsoCrop)
                    {
                        long wh = (long)src.Height * ex.Height;
                        long hw = (long)src.Width * ex.Width;

                        if (pad == (wh > hw))
                        {
                            h = checked((int)(hw / src.Height));
                            y = (ex.Height - h) / 2;
                        }
                        else
                        {
                            w = checked((int)(wh / src.Width));
                            x = (ex.Width - w) / 2;
                        }

                        // If we're not padding, the rectangle should fill everything.
                        Host.Assert(pad || x <= 0 && y <= 0 &&
                            h >= ex.Height && w >= ex.Width);
                    }
                    // Draw the image.
                    var srcRectangle = new Rectangle(0, 0, src.Width, src.Height);
                    if (pad)
                    {
                        using (var g = Graphics.FromImage(src))
                        {
                            g.DrawImage(dst, srcRectangle, new Rectangle(x, y, w, h), GraphicsUnit.Pixel);
                        }
                    }
                    else
                    {
                        using (var g = Graphics.FromImage(src))
                        {
                            g.DrawImage(dst, srcRectangle, new Rectangle(-x, -y, ex.Width, ex.Height), GraphicsUnit.Pixel);
                        }
                    }

                    Host.Assert(dst.Width == ex.Width && dst.Height == ex.Height);
                };

            return del;
        }
    }
}
