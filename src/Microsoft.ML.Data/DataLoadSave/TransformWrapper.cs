﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.DataLoadSave;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(TransformWrapper), null, typeof(SignatureLoadModel),
    "Transform wrapper", TransformWrapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: this class is public, as long as the Wrappers.cs in tests still rely on it.
    // It needs to become internal.
    public sealed class TransformWrapper : ITransformer, ICanSaveModel
    {
        public const string LoaderSignature = "TransformWrapper";
        private const string TransformDirTemplate = "Step_{0:000}";

        private readonly IHost _host;
        private readonly IDataView _xf;

        public TransformWrapper(IHostEnvironment env, IDataView xf)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TransformWrapper));
            _host.CheckValue(xf, nameof(xf));
            _xf = xf;
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var dv = new EmptyDataView(_host, inputSchema);
            var output = ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, dv);
            return output.Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            var dataPipe = _xf;
            var transforms = new List<IDataTransform>();
            while (dataPipe is IDataTransform xf)
            {
                // REVIEW: a malicious user could construct a loop in the Source chain, that would
                // cause this method to iterate forever (and throw something when the list overflows). There's
                // no way to insulate from ALL malicious behavior.
                transforms.Add(xf);
                dataPipe = xf.Source;
                Contracts.AssertValue(dataPipe);
            }
            transforms.Reverse();

            ctx.SaveSubModel("Loader", c => BinaryLoader.SaveInstance(_host, c, dataPipe.Schema));

            ctx.Writer.Write(transforms.Count);
            for (int i = 0; i < transforms.Count; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(transforms[i], dirName);
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "XF  WRPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        // Factory for SignatureLoadModel.
        public TransformWrapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TransformWrapper));
            _host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());
            int n = ctx.Reader.ReadInt32();
            _host.CheckDecode(n >= 0);

            ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            IDataView data = loader;
            for (int i = 0; i < n; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                data = xf;
            }

            _xf = data;
        }

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, input);
    }

    /// <summary>
    /// Estimator for trained wrapped transformers.
    /// </summary>
    internal abstract class TrainedWrapperEstimatorBase : IEstimator<TransformWrapper>
    {
        private readonly IHost _host;

        protected TrainedWrapperEstimatorBase(IHost host)
        {
            Contracts.CheckValue(host, nameof(host));
            _host = host;
        }

        public abstract TransformWrapper Fit(IDataView input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var fakeSchema = new FakeSchema(_host, inputSchema);
            var transformer = Fit(new EmptyDataView(_host, fakeSchema));
            return SchemaShape.Create(transformer.GetOutputSchema(fakeSchema));
        }
    }

    /// <summary>
    /// Estimator for untrained wrapped transformers.
    /// </summary>
    public abstract class TrivialWrapperEstimator : TrivialEstimator<TransformWrapper>
    {
        protected TrivialWrapperEstimator(IHost host, TransformWrapper transformer)
            : base(host, transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var fakeSchema = new FakeSchema(Host, inputSchema);
            return SchemaShape.Create(Transformer.GetOutputSchema(fakeSchema));
        }
    }
}
