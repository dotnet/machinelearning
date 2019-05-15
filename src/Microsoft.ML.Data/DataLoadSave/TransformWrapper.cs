// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.DataLoadSave;
using Microsoft.ML.Runtime;

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
        private readonly bool _isRowToRowMapper;

        public TransformWrapper(IHostEnvironment env, IDataView xf)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TransformWrapper));
            _host.CheckValue(xf, nameof(xf));
            _xf = xf;
            _isRowToRowMapper = IsChainRowToRowMapper(_xf);
        }

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var dv = new EmptyDataView(_host, inputSchema);
            var output = ApplyTransformUtils.ApplyTransformToData(_host, (IDataTransform)_xf, dv);

            return output.Schema;
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => throw _host.Except("Saving is not permitted.");

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

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyTransformToData(_host, (IDataTransform)_xf, input);

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
            return new CompositeRowToRowMapper(inputSchema,
                new[] { (IRowToRowMapper)ApplyTransformUtils.ApplyTransformToData(_host, (IDataTransform)_xf, new EmptyDataView(_host, inputSchema)) });
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
