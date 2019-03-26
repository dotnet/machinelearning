// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(NopTransform.Summary, typeof(NopTransform), null, typeof(SignatureLoadDataTransform),
    "", NopTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(NopTransform), null, typeof(SignatureEntryPointModule), "NopTransform")]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A transform that does nothing.
    /// </summary>
    [BestFriend]
    internal sealed class NopTransform : IDataTransform, IRowToRowMapper
    {
        private readonly IHost _host;

        public IDataView Source { get; }

        DataViewSchema IRowToRowMapper.InputSchema => Source.Schema;

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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NopTransform).Assembly.FullName);
        }

        internal static string RegistrationName = "NopTransform";

        public static NopTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NopTransform(h, ctx, input));
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

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Nothing :)
        }

        public bool CanShuffle => Source.CanShuffle;

        /// <summary>
        /// Explicit implementation prevents Schema from being accessed from derived classes.
        /// It's our first step to separate data produced by transform from transform.
        /// </summary>
        DataViewSchema IDataView.Schema => OutputSchema;

        /// <summary>
        /// Shape information of the produced output. Note that the input and the output of this transform (and their types) are identical.
        /// </summary>
        public DataViewSchema OutputSchema => Source.Schema;

        public long? GetRowCount()
        {
            return Source.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            => Source.GetRowCursor(columnsNeeded, rand);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            => Source.GetRowCursorSet(columnsNeeded, n, rand);

        /// <summary>
        /// Given a set of columns, return the input columns that are needed to generate those output columns.
        /// </summary>
        IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            => dependingColumns;

        DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(activeColumns, nameof(activeColumns));
            Contracts.CheckParam(input.Schema == Source.Schema, nameof(input), "Schema of input row must be the same as the schema the mapper is bound to");
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
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
    }
}