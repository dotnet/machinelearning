// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(ImageLoaderTransform.Summary, typeof(ImageLoaderTransform), typeof(ImageLoaderTransform.Arguments), typeof(SignatureDataTransform),
    ImageLoaderTransform.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoaderTransform.Summary, typeof(ImageLoaderTransform), null, typeof(SignatureLoadDataTransform),
   ImageLoaderTransform.UserName, ImageLoaderTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    /// <summary>
    /// Transform which takes one or many columns of type <see cref="DvText"/> and loads them as <see cref="ImageType"/>
    /// </summary>
    public sealed class ImageLoaderTransform : OneToOneTransformBase
    {
        public sealed class Column : OneToOneColumn
        {
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
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Folder where to search for images", ShortName = "folder")]
            public string ImageFolder;
        }

        internal const string Summary = "Load images from a files.";
        internal const string UserName = "Image Loader Transform";
        public const string LoaderSignature = "ImageLoaderTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGLOADT",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly ImageType _type;
        private readonly string _imageFolder;

        private const string RegistrationName = "ImageLoader";

        // Public constructor corresponding to SignatureDataTransform.
        public ImageLoaderTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column, input, TestIsText)
        {
            Host.AssertNonEmpty(Infos);
            _imageFolder = args.ImageFolder;
            Host.Assert(Infos.Length == Utils.Size(args.Column));
            _type = new ImageType();
            Metadata.Seal();
        }

        private ImageLoaderTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsText)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            _imageFolder = ctx.Reader.ReadString();
            _type = new ImageType();
            Metadata.Seal();
        }

        public static ImageLoaderTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ImageLoaderTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            ctx.Writer.Write(_imageFolder);
            SaveBase(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Check(0 <= iinfo && iinfo < Infos.Length);
            return _type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(ch, nameof(ch));
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var getSrc = GetSrcGetter<DvText>(input, iinfo);
            DvText src = default;
            ValueGetter<Bitmap> del =
                (ref Bitmap dst) =>
                {
                    if (dst != null)
                    {
                        dst.Dispose();
                        dst = null;
                    }

                    getSrc(ref src);

                    if (src.Length > 0)
                    {
                        // Catch exceptions and pass null through. Should also log failures...
                        try
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_imageFolder))
                                path = Path.Combine(_imageFolder, path);
                            dst = new Bitmap(path);
                        }
                        catch (Exception e)
                        {
                            // REVIEW: We catch everything since the documentation for new Bitmap(string)
                            // appears to be incorrect. When the file isn't found, it throws an ArgumentException,
                            // while the documentation says FileNotFoundException. Not sure what it will throw
                            // in other cases, like corrupted file, etc.

                            // REVIEW : Log failures.
                            ch.Info(e.Message);
                            ch.Info(e.StackTrace);
                            dst = null;
                        }
                    }
                };
            return del;
        }
    }
}
