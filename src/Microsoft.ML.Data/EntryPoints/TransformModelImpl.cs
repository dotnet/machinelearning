// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// This encapsulates zero or more transform models. It does this by recording
    /// the initial schema, together with the sequence of transforms applied to that
    /// schema.
    /// </summary>
    [BestFriend]
    internal sealed class TransformModelImpl : TransformModel
    {
        // The cached schema of the root of the _chain.
        private readonly DataViewSchema _schemaRoot;

        /// <summary>
        /// This contains the transforms to save instantiated on an <see cref="IDataView"/> with
        /// appropriate initial schema. Note that the "root" of this is typically either
        /// an empty <see cref="IDataView"/> or a <see cref="BinaryLoader"/> with no rows. However, other root
        /// types are possible, since we don't insist on this when loading a model
        /// from a zip file. However, whenever we save, we force a <see cref="BinaryLoader"/> to
        /// be serialized for the root.
        /// </summary>
        private readonly IDataView _chain;

        /// <summary>
        /// The input schema that this transform model was originally instantiated on.
        /// Note that the schema may have columns that aren't needed by this transform model.
        /// If an <see cref="IDataView"/> exists with this schema, then applying this transform model to it
        /// shouldn't fail because of column type issues.
        /// REVIEW: Would be nice to be able to trim this to the minimum needed somehow. Note
        /// however that doing so may cause issues for composing transform models. For example,
        /// if transform model A needs column X and model B needs Y, that is NOT produced by A,
        /// then trimming A's input schema would cause composition to fail.
        /// </summary>
        internal override DataViewSchema InputSchema => _schemaRoot;

        /// <summary>
        /// The resulting schema once applied to this model. The <see cref="InputSchema"/> might have
        /// columns that are not needed by this transform and these columns will be seen in the
        /// <see cref="OutputSchema"/> produced by this transform.
        /// </summary>
        internal override DataViewSchema OutputSchema => _chain.Schema;

        /// <summary>
        /// Create a TransformModel containing the transforms from "result" back to "input".
        /// </summary>
        public TransformModelImpl(IHostEnvironment env, IDataView result, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(result, nameof(result));
            env.CheckValue(input, nameof(input));

            var root = new EmptyDataView(env, input.Schema);
            _schemaRoot = root.Schema;
            _chain = ApplyTransformUtils.ApplyAllTransformsToData(env, result, root, input);
        }

        private TransformModelImpl(IHostEnvironment env, DataViewSchema schemaRoot, IDataView chain)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schemaRoot);
            env.AssertValue(chain);
            _schemaRoot = schemaRoot;
            _chain = chain;
        }

        /// <summary>
        /// Create a TransformModel containing the given (optional) transforms applied to the
        /// given root schema.
        /// </summary>
        public TransformModelImpl(IHostEnvironment env, DataViewSchema schemaRoot, IDataTransform[] xfs)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(schemaRoot, nameof(schemaRoot));
            env.CheckValueOrNull(xfs);

            IDataView view = new EmptyDataView(env, schemaRoot);
            _schemaRoot = view.Schema;

            if (Utils.Size(xfs) > 0)
            {
                foreach (var xf in xfs)
                {
                    env.AssertValue(xf, "xfs", "Transforms should not be null");
                    view = ApplyTransformUtils.ApplyTransformToData(env, xf, view);
                }
            }

            _chain = view;
        }

        /// <summary>
        /// Load a transform model.
        /// </summary>
        public TransformModelImpl(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));

            // REVIEW: Should this preserve the "tags" for the transforms?
            using (var ch = env.Start("Loading transform model"))
            {
                _chain = ModelFileUtils.LoadPipeline(env, stream, new MultiFileSource(null), extractInnerPipe: true);
            }

            // Find the root schema.
            for (IDataView view = _chain; ;)
            {
                var xf = view as IDataTransform;
                if (xf == null)
                {
                    _schemaRoot = view.Schema;
                    break;
                }
                view = xf.Source;
                env.AssertValue(view);
            }
        }

        /// <summary>
        /// Apply this transform model to the given input data.
        /// </summary>
        internal override IDataView Apply(IHostEnvironment env, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            return ApplyTransformUtils.ApplyAllTransformsToData(env, _chain, input);
        }

        /// <summary>
        /// Apply this transform model to the given input transform model to produce a composite transform model.
        /// </summary>
        internal override TransformModel Apply(IHostEnvironment env, TransformModel input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            IDataView view;
            DataViewSchema schemaRoot = input.InputSchema;
            var mod = input as TransformModelImpl;
            if (mod != null)
                view = ApplyTransformUtils.ApplyAllTransformsToData(env, _chain, mod._chain);
            else
            {
                view = new EmptyDataView(env, schemaRoot);
                view = input.Apply(env, view);
                view = Apply(env, view);
            }

            return new TransformModelImpl(env, schemaRoot, view);
        }

        /// <summary>
        /// Save this transform model.
        /// </summary>
        internal override void Save(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));

            using (var ch = env.Start("Saving transform model"))
            {
                using (var rep = RepositoryWriter.CreateNew(stream, ch))
                {
                    ch.Trace("Saving root schema and transformations");
                    TrainUtils.SaveDataPipe(env, rep, _chain, blankLoader: true);
                    rep.Commit();
                }
            }
        }

        internal override IRowToRowMapper AsRowToRowMapper(IExceptionContext ectx)
        {
            return
                CompositeRowToRowMapper.IsCompositeRowToRowMapper(_chain)
                    ? new CompositeRowToRowMapper(ectx, _chain, _schemaRoot)
                    : null;
        }

        private sealed class CompositeRowToRowMapper : IRowToRowMapper
        {
            private readonly IDataView _chain;
            private readonly DataViewSchema _rootSchema;
            private readonly IExceptionContext _ectx;

            public DataViewSchema Schema => _chain.Schema;

            public DataViewSchema OutputSchema => Schema;

            public CompositeRowToRowMapper(IExceptionContext ectx, IDataView chain, DataViewSchema rootSchema)
            {
                Contracts.CheckValue(ectx, nameof(ectx));
                _ectx = ectx;
                _ectx.CheckValue(chain, nameof(chain));
                _ectx.CheckValue(rootSchema, nameof(rootSchema));

                _chain = chain;
                _rootSchema = rootSchema;
            }

            public static bool IsCompositeRowToRowMapper(IDataView chain)
            {
                var transform = chain as IDataTransform;
                while (transform != null)
                {
                    if (!(transform is IRowToRowMapper))
                        return false;
                    transform = transform.Source as IDataTransform;
                }
                return true;
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                _ectx.Assert(IsCompositeRowToRowMapper(_chain));

                var transform = _chain as IDataTransform;
                var cols = dependingColumns;
                while (transform != null)
                {
                    var mapper = transform as IRowToRowMapper;
                    _ectx.AssertValue(mapper);
                    cols = mapper.GetDependencies(cols);
                    transform = transform.Source as IDataTransform;
                }
                return cols;
            }

            public DataViewSchema InputSchema => _rootSchema;

            DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                _ectx.Assert(IsCompositeRowToRowMapper(_chain));
                _ectx.AssertValue(input);
                _ectx.AssertValue(activeColumns);

                _ectx.Check(input.Schema == InputSchema, "Schema of input row must be the same as the schema the mapper is bound to");

                var mappers = new List<IRowToRowMapper>();
                var actives = new List<IEnumerable<DataViewSchema.Column>>();
                var transform = _chain as IDataTransform;
                var activeCur = activeColumns;
                while (transform != null)
                {
                    var mapper = transform as IRowToRowMapper;
                    _ectx.AssertValue(mapper);
                    mappers.Add(mapper);
                    actives.Add(activeCur);
                    activeCur = mapper.GetDependencies(activeCur);
                    transform = transform.Source as IDataTransform;
                }

                mappers.Reverse();
                actives.Reverse();
                var row = input;
                for (int i = 0; i < mappers.Count; i++)
                    row = mappers[i].GetRow(row, actives[i]);

                return row;
            }
        }
    }
}
