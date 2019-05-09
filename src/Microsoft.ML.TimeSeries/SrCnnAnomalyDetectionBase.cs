// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    public class SrCnnAnomalyDetectionBaseWrapper : IStatefulTransformer, ICanSaveModel
    {
        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)InternalTransform).IsRowToRowMapper;

        /// <summary>
        /// Create a clone of the transformer. Used for taking the snapshot of the state.
        /// </summary>
        /// <returns></returns>
        IStatefulTransformer IStatefulTransformer.Clone() => InternalTransform.Clone();

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => InternalTransform.GetOutputSchema(inputSchema);

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="ITransformer.IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception should be thrown. If the input schema is in any way
        /// unsuitable for constructing the mapper, an exception should likewise be thrown.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
            => ((ITransformer)InternalTransform).GetRowToRowMapper(inputSchema);

        /// <summary>
        /// Same as <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> but also supports mechanism to save the state.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        public IRowToRowMapper GetStatefulRowToRowMapper(DataViewSchema inputSchema)
            => ((IStatefulTransformer)InternalTransform).GetStatefulRowToRowMapper(inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input) => InternalTransform.Transform(input);

        /// <summary>
        /// For saving a model into a repository.
        /// </summary>
        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            //TODO:
        }

        /// <summary>
        /// Creates a row mapper from Schema.
        /// </summary>
        internal IStatefulRowMapper MakeRowMapper(DataViewSchema schema) => InternalTransform.MakeRowMapper(schema);

        /// <summary>
        /// Creates an IDataTransform from an IDataView.
        /// </summary>
        internal IDataTransform MakeDataTransform(IDataView input) => InternalTransform.MakeDataTransform(input);

        internal SrCnnAnomalyDetectionBase InternalTransform;

        internal SrCnnAnomalyDetectionBaseWrapper(SrCnnArgumentBase args, string name, IHostEnvironment env)
        {
            InternalTransform = new SrCnnAnomalyDetectionBase(args, name, env, this);
        }

        internal SrCnnAnomalyDetectionBaseWrapper(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            InternalTransform = new SrCnnAnomalyDetectionBase(env, ctx, name, this);
        }

        internal sealed class SrCnnAnomalyDetectionBase : SrCnnTransformBase<Single, SrCnnAnomalyDetectionBase.State>
        {
            internal SrCnnAnomalyDetectionBaseWrapper Parent;

            public SrCnnAnomalyDetectionBase(SrCnnArgumentBase args, string name, IHostEnvironment env, SrCnnAnomalyDetectionBaseWrapper parent)
                : base(args, name, env)
            {
                //TODO:
            }

            public SrCnnAnomalyDetectionBase(IHostEnvironment env, ModelLoadContext ctx, string name, SrCnnAnomalyDetectionBaseWrapper parent)
                : base(env, ctx, name)
            {
                //TODO:
            }

            public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            {
                //TODO:
                throw new NotImplementedException();
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                //TODO:
            }

            internal sealed class State : SrCnnStateBase
            {
                public State()
                {
                }

                internal State(BinaryReader reader)
                {
                    //TODO:
                }

                internal override void Save(BinaryWriter writer)
                {
                    //TODO:
                }

                private protected override void CloneCore(State state)
                {
                    //TODO:
                }

                private protected override void LearnStateFromDataCore(FixedSizeQueue<float> data)
                {
                }
            }
        }
    }
}
