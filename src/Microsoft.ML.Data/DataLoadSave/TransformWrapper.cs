// Licensed to the .NET Foundation under one or more agreements.
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
            IsRowToRowMapper = IsChainRowToRowMapper(_xf);
        }

        public Schema GetOutputSchema(Schema inputSchema)
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TransformWrapper).Assembly.FullName);
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
            IsRowToRowMapper = IsChainRowToRowMapper(_xf);
        }

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, input);

        private static bool IsChainRowToRowMapper(IDataView view)
        {
            for (; view is IDataTransform xf; view = xf.Source)
            {
                if (!(xf is IRowToRowMapper))
                    return false;
            }
            return true;
        }

        public bool IsRowToRowMapper { get; }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var input = new EmptyDataView(_host, inputSchema);
            var revMaps = new List<IRowToRowMapper>();
            IDataView chain;
            for (chain = ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, input); chain is IDataTransform xf; chain = xf.Source)
            {
                // Everything in the chain ought to be a row mapper.
                _host.Assert(xf is IRowToRowMapper);
                revMaps.Add((IRowToRowMapper)xf);
            }
            // The walkback should have ended at the input.
            Contracts.Assert(chain == input);
            revMaps.Reverse();
            return new CompositeRowToRowMapper(inputSchema, revMaps.ToArray());
        }
    }

    /// <summary>
    /// Estimator for trained wrapped transformers.
    /// </summary>
    public abstract class TrainedWrapperEstimatorBase : IEstimator<TransformWrapper>
    {
        protected readonly IHost Host;

        protected TrainedWrapperEstimatorBase(IHost host)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
        }

        public abstract TransformWrapper Fit(IDataView input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var fakeSchema = Schema.Create(new FakeSchema(Host, inputSchema));
            var transformer = Fit(new EmptyDataView(Host, fakeSchema));
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
            var fakeSchema = Schema.Create(new FakeSchema(Host, inputSchema));
            return SchemaShape.Create(Transformer.GetOutputSchema(fakeSchema));
        }
    }
}
