// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataLoadSave;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(TransformWrapper), null, typeof(SignatureLoadModel),
    "Transform wrapper", TransformWrapper.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is a shim class to present the legacy <see cref="IDataTransform"/> interface as an <see cref="ITransformer"/>.
    /// Note that there are some important differences in usages that make this shimming somewhat non-seemless, so the goal
    /// would be gradual removal of this as we do away with <see cref="IDataTransform"/> based code.
    /// </summary>
    [BestFriend]
    internal sealed class TransformWrapper : ITransformer
    {
        internal const string LoaderSignature = "TransformWrapper";
        private const string TransformDirTemplate = "Step_{0:000}";

        private readonly IHost _host;
        private readonly IDataView _xf;
        private readonly bool _allowSave;
        private readonly bool _isRowToRowMapper;

        public TransformWrapper(IHostEnvironment env, IDataView xf, bool allowSave = false)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TransformWrapper));
            _host.CheckValue(xf, nameof(xf));
            _xf = xf;
            _allowSave = allowSave;
            _isRowToRowMapper = IsChainRowToRowMapper(_xf);
        }

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var dv = new EmptyDataView(_host, inputSchema);
            var output = ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, dv);
            return output.Schema;
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            if (!_allowSave)
                throw _host.Except("Saving is not permitted.");
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
        private TransformWrapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TransformWrapper));
            _host.CheckValue(ctx, nameof(ctx));
            _allowSave = true;
            ctx.CheckAtModel(GetVersionInfo());
            int n = ctx.Reader.ReadInt32();
            _host.CheckDecode(n >= 0);

            ctx.LoadModel<ILegacyDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            IDataView data = loader;
            for (int i = 0; i < n; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                data = xf;
            }

            _xf = data;
            _isRowToRowMapper = IsChainRowToRowMapper(_xf);
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

        bool ITransformer.IsRowToRowMapper => _isRowToRowMapper;

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
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
    internal abstract class TrainedWrapperEstimatorBase : IEstimator<TransformWrapper>
    {
        [BestFriend]
        private protected readonly IHost Host;

        [BestFriend]
        private protected TrainedWrapperEstimatorBase(IHost host)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
        }

        public abstract TransformWrapper Fit(IDataView input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var fakeSchema = FakeSchemaFactory.Create(inputSchema);
            var transformer = Fit(new EmptyDataView(Host, fakeSchema));
            return SchemaShape.Create(transformer.GetOutputSchema(fakeSchema));
        }
    }
}
