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
using System.Drawing.Imaging;

[assembly: LoadableClass(ImageGreyscaleTransform.Summary, typeof(ImageGreyscaleTransform), typeof(ImageGreyscaleTransform.Arguments), typeof(SignatureDataTransform),
    ImageGreyscaleTransform.UserName, "ImageResizerTransform", "ImageResizer")]

[assembly: LoadableClass(ImageGreyscaleTransform.Summary, typeof(ImageGreyscaleTransform), null, typeof(SignatureLoadDataTransform),
    ImageGreyscaleTransform.UserName, ImageGreyscaleTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    public sealed class ImageGreyscaleTransform : OneToOneTransformBase
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

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ImageGreyscaleTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column, input, t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));
            Metadata.Seal();
        }

        private ImageGreyscaleTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertValue(ctx);
            // *** Binary format ***
            // <base>
            Host.AssertNonEmpty(Infos);
            Metadata.Seal();
        }

        public static ImageGreyscaleTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ImageGreyscaleTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            SaveBase(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return Infos[iinfo].TypeSrc;
        }

        public ColorMatrix GreyscaleColorMatrix = new ColorMatrix(
                new float[][]
                {
                    new float[] {.3f, .3f, .3f, 0, 0},
                    new float[] {.59f, .59f, .59f, 0, 0},
                    new float[] {.11f, .11f, .11f, 0, 0},
                    new float[] {0, 0, 0, 1, 0},
                    new float[] {0, 0, 0, 0, 1}
                });

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(ch, "ch");
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            var src = default(Bitmap);
            var getSrc = GetSrcGetter<Bitmap>(input, iinfo);

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
                    dst.SetResolution(src.VerticalResolution, src.VerticalResolution);
                    ImageAttributes attributes = new ImageAttributes();
                    attributes.SetColorMatrix(GreyscaleColorMatrix);
                    var srcRectangle = new Rectangle(0, 0, src.Width, src.Height);
                    using (var g = Graphics.FromImage(dst))
                    {
                        g.DrawImage(src, srcRectangle, 0, 0, src.Width, src.Height, GraphicsUnit.Pixel, attributes);
                    }
                    Host.Assert(dst.Width == src.Width && dst.Height == src.Height);
                };

            return del;
        }
    }
}
