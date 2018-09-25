// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(NopTransform.Summary, typeof(NopTransform), null, typeof(SignatureLoadDataTransform),
    "", NopTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(NopTransform), null, typeof(SignatureEntryPointModule), "NopTransform")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A transform that does nothing.
    /// </summary>
    public sealed class NopTransform : IDataTransform, IRowToRowMapper
    {
        private readonly IHost _host;

        public IDataView Source { get; }

        ISchema IRowToRowMapper.InputSchema => Source.Schema;

        /// <summary>
        /// Creates a NopTransform if the input is not an IDataTransform.
        /// Otherwise it returns the input.
        /// </summary>
        public static IDataTransform CreateIfNeeded(IHostEnvironment env, IDataView input)
        {
            var dt = input as IDataTransform;
            if (dt != null)
                return dt;
            return new NopTransform(env, input);
        }

        private NopTransform(IHostEnvironment env, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            Source = input;
            _host = env.Register(RegistrationName);
        }

        internal const string Summary = "Does nothing.";

        public const string LoaderSignature = "NopTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NOOPNOOP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        internal static string RegistrationName = "NopTransform";

        public static NopTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NopTransform(h, ctx,  input));
        }

        private NopTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.AssertValue(host, "host");
            host.CheckValue(input, nameof(input));

            Source = input;
            _host = host;

            // *** Binary format ***
            // Nothing :)
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Nothing :)
        }

        public bool CanShuffle
        {
            get { return Source.CanShuffle; }
        }

        public ISchema Schema
        {
            get { return Source.Schema; }
        }

        public long? GetRowCount(bool lazy = true)
        {
            return Source.GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            return Source.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            return Source.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            return predicate;
        }

        public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(active, nameof(active));
            Contracts.CheckParam(input.Schema == Source.Schema, nameof(input), "Schema of input row must be the same as the schema the mapper is bound to");

            disposer = null;
            return input;
        }

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
    }
}
