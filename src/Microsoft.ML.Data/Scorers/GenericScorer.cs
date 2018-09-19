// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;

[assembly: LoadableClass(typeof(GenericScorer), typeof(GenericScorer.Arguments), typeof(SignatureDataScorer),
    "Generic Scorer", GenericScorer.LoadName, "Generic")]

[assembly: LoadableClass(typeof(GenericScorer), null, typeof(SignatureLoadDataTransform),
    "Generic Scorer", GenericScorer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This class is a scorer that passes through all the ISchemaBound columns without adding any "derived columns".
    /// It also passes through all metadata (except for possibly changing the score column kind), and adds the
    /// score set id metadata.
    /// </summary>

    public sealed class GenericScorer : RowToRowScorerBase, ITransformCanSavePfa, ITransformCanSaveOnnx
    {
        public const string LoadName = "GenericScorer";

        public sealed class Arguments : ScorerArgumentsBase
        {
        }

        private sealed class Bindings : BindingsBase
        {
            /// <summary>
            /// The one and only constructor for Bindings.
            /// </summary>
            private Bindings(ISchema input, ISchemaBoundRowMapper mapper, string suffix, bool user)
                : base(input, mapper, suffix, user)
            {
                Contracts.Assert(DerivedColumnCount == 0);
            }

            /// <summary>
            /// Create the bindings given the input schema, bound mapper, and column name suffix.
            /// </summary>
            public static Bindings Create(ISchema input, ISchemaBoundRowMapper mapper, string suffix, bool user = true)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(mapper);
                Contracts.AssertValueOrNull(suffix);
                // We don't actually depend on this invariant, but if this assert fires it means the bindable
                // did the wrong thing.
                Contracts.Assert(mapper.InputSchema.Schema == input);

                return new Bindings(input, mapper, suffix, user);
            }

            /// <summary>
            /// Create the bindings given the env, bindable, input schema, column roles, and column name suffix.
            /// </summary>
            private static Bindings Create(IHostEnvironment env, ISchemaBindableMapper bindable, ISchema input,
                IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> roles, string suffix, bool user = true)
            {
                Contracts.AssertValue(env);
                Contracts.AssertValue(bindable);
                Contracts.AssertValue(input);
                Contracts.AssertValue(roles);
                Contracts.AssertValueOrNull(suffix);

                var mapper = bindable.Bind(env, new RoleMappedSchema(input, roles));
                // We don't actually depend on this invariant, but if this assert fires it means the bindable
                // did the wrong thing.
                Contracts.Assert(mapper.InputSchema.Schema == input);

                var rowMapper = mapper as ISchemaBoundRowMapper;
                Contracts.Check(rowMapper != null, "Predictor expected to be a RowMapper!");

                return Create(input, rowMapper, suffix, user);
            }

            /// <summary>
            /// Create a new Bindings from this one, but based on a potentially different schema.
            /// Used by the ITransformTemplate.ApplyToData implementation.
            /// </summary>
            public Bindings ApplyToSchema(IHostEnvironment env, ISchema input)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(env);

                var bindable = RowMapper.Bindable;
                var roles = RowMapper.GetInputColumnRoles();
                string suffix = Suffix;
                return Create(env, bindable, input, roles, suffix);
            }

            /// <summary>
            /// Deserialize the bindings, given the env, bindable and input schema.
            /// </summary>
            public static Bindings Create(ModelLoadContext ctx,
                IHostEnvironment env, ISchemaBindableMapper bindable, ISchema input)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // <base info>
                string suffix;
                var roles = LoadBaseInfo(ctx, out suffix);

                return Create(env, bindable, input, roles, suffix, user: false);
            }

            public override void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // <base info>
                SaveBase(ctx);
            }
        }

        public const string LoaderSignature = "GenericScoreTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GNRICSCR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "GenericScore";

        private readonly Bindings _bindings;
        protected override BindingsBase GetBindings() => _bindings;

        public bool CanSavePfa => (Bindable as ICanSavePfa)?.CanSavePfa == true;

        public bool CanSaveOnnx(OnnxContext ctx) => (Bindable as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        /// <summary>
        /// The <see cref="SignatureDataScorer"/> entry point for creating a <see cref="GenericScorer"/>.
        /// </summary>
        public GenericScorer(IHostEnvironment env, ScorerArgumentsBase args, IDataView data,
            ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            : base(env, data, RegistrationName, Contracts.CheckRef(mapper, nameof(mapper)).Bindable)
        {
            Host.CheckValue(args, nameof(args));
            Host.AssertValue(data, "data");
            Host.AssertValue(mapper, "mapper");

            var rowMapper = mapper as ISchemaBoundRowMapper;
            Host.CheckParam(rowMapper != null, nameof(mapper), "mapper should implement ISchemaBoundRowMapper");
            _bindings = Bindings.Create(data.Schema, rowMapper, args.Suffix);
        }

        /// <summary>
        /// Constructor for <see cref="ApplyToData"/> method.
        /// </summary>
        private GenericScorer(IHostEnvironment env, GenericScorer transform, IDataView data)
            : base(env, data, RegistrationName, transform.Bindable)
        {
            _bindings = transform._bindings.ApplyToSchema(env, data.Schema);
        }

        /// <summary>
        /// Constructor for deserialization.
        /// </summary>
        private GenericScorer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input)
        {
            Contracts.AssertValue(ctx);
            _bindings = Bindings.Create(ctx, host, Bindable, input.Schema);
        }

        /// <summary>
        /// <see cref="SignatureLoadDataTransform"/> entry point - for deserialization.
        /// </summary>
        public static GenericScorer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new GenericScorer(h, ctx, input));
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            _bindings.Save(ctx);
        }

        public void SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSavePfa);
            var pfaBindable = (IBindableCanSavePfa)Bindable;

            var schema = _bindings.RowMapper.InputSchema;
            Host.Assert(_bindings.DerivedColumnCount == 0);
            string[] outColNames = new string[_bindings.InfoCount];
            for (int iinfo = 0; iinfo < _bindings.InfoCount; ++iinfo)
                outColNames[iinfo] = _bindings.GetColumnName(_bindings.MapIinfoToCol(iinfo));

            pfaBindable.SaveAsPfa(ctx, schema, outColNames);
        }

        public void SaveAsOnnx(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSaveOnnx);
            var onnxBindable = (IBindableCanSaveOnnx)Bindable;

            var schema = _bindings.RowMapper.InputSchema;
            Host.Assert(_bindings.DerivedColumnCount == 0);
            string[] outVariableNames = new string[_bindings.InfoCount];
            for (int iinfo = 0; iinfo < _bindings.InfoCount; ++iinfo)
            {
                int colIndex = _bindings.MapIinfoToCol(iinfo);
                string colName = _bindings.GetColumnName(colIndex);
                colName = ctx.AddIntermediateVariable(_bindings.GetColumnType(colIndex), colName);
                outVariableNames[iinfo] = colName;
            }

            if (!onnxBindable.SaveAsOnnx(ctx, schema, outVariableNames))
            {
                foreach (var name in outVariableNames)
                    ctx.RemoveVariable(name, true);
            }
        }

        protected override bool WantParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            return _bindings.AnyNewColumnsActive(predicate);
        }

        public override IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Host.CheckValue(env, nameof(env));
            Host.CheckValue(newSource, nameof(newSource));

            return new GenericScorer(env, this, newSource);
        }

        protected override Delegate[] GetGetters(IRow output, Func<int, bool> predicate)
        {
            Host.Assert(_bindings.DerivedColumnCount == 0);
            Host.AssertValue(output);
            Host.AssertValue(predicate);
            Host.Assert(output.Schema == _bindings.RowMapper.OutputSchema);

            return GetGettersFromRow(output, predicate);
        }
    }
}