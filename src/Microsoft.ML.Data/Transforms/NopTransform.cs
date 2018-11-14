// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(NopTransformer.Summary, typeof(NopTransformer), null, typeof(SignatureLoadDataTransform),
    "", NopTransformer.LoaderSignature)]

[assembly: LoadableClass(NopTransformer.Summary, typeof(NopTransformer), null, typeof(SignatureLoadModel),
    NopTransformer.FriendlyName, NopTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NopTransformer), null, typeof(SignatureLoadRowMapper),
   NopTransformer.FriendlyName, NopTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(NopTransformer), null, typeof(SignatureEntryPointModule), "NopTransform")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A transform that does nothing.
    /// </summary>
    public sealed class NopTransformer : ITransformer, ICanSaveModel
    {
        private readonly IHost _host;

        internal const string Summary = "Does nothing.";
        internal const string LoaderSignature = "NopTransform";
        internal const string FriendlyName = "Nop Transform";
        internal static string RegistrationName = "NopTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NOOPNOOP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NopTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Creates a NopTransform if the input is not an IDataTransform.
        /// Otherwise it returns the input.
        /// </summary>
        public static IDataTransform CreateIfNeeded(IHostEnvironment env, IDataView input)
        {
            var dt = input as IDataTransform;
            if (dt != null)
                return dt;
            return new NopTransformer(env).MakeDataTransform(input);
        }

        public NopTransformer(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
        }

        internal NopTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : this(env)
        {
            // *** Binary format ***
            // Nothing :)
        }

        // Factory method for SignatureLoadDataTransform.
        internal static NopTransformer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NopTransformer(h, ctx));
        }

        // Factory method for SignatureLoadModel.
        internal static NopTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new NopTransformer(env, ctx);

        // Factory method for SignatureLoadRowMapper.
        internal static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Nothing :)
        }

        private RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(_host, input, MakeRowMapper(input.Schema), MakeRowMapper);
        }

        private IRowMapper MakeRowMapper(Schema schema)
            => new Mapper(this, schema);

        public bool IsRowToRowMapper => true;

        public class NopInput : TransformInputBase
        {
        }

        [TlcModule.EntryPoint(Name = "Transforms.NoOperation", Desc = Summary, UserName = "No Op", ShortName = "Nop")]
        public static CommonOutputs.TransformOutput Nop(IHostEnvironment env, NopInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Nop");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = CreateIfNeeded(host, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        public Schema GetOutputSchema(Schema inputSchema)
        {
            throw new NotImplementedException();
        }

        public IDataView Transform(IDataView input)
        {
            throw new NotImplementedException();
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            throw new NotImplementedException();
        }

        private sealed class Mapper : IRowMapper
        {
            private NopTransformer _parent;
            private Schema _inputSchema;

            public Mapper(NopTransformer parent, Schema inputSchema)
            {
                _parent = parent;
                _inputSchema = inputSchema;
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                return null;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
                => activeOutput;

            public Schema.Column[] GetOutputColumns()
                => null;

            public void Save(ModelSaveContext ctx)
                => _parent.Save(ctx);
        }
    }
}
