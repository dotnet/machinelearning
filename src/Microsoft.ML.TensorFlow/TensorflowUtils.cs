// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Dnn;
using Tensorflow;

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal static class TensorFlowUtils
    {
        /// <summary>
        /// Key to access operator's type (a string) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value describes the Tensorflow operator that produces this <see cref="DataViewSchema.Column"/>.
        /// </summary>
        internal const string TensorflowOperatorTypeKind = "TensorflowOperatorType";
        /// <summary>
        /// Key to access upstream operators' names (a string array) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value states operators that the associated <see cref="DataViewSchema.Column"/>'s generator depends on.
        /// </summary>
        internal const string TensorflowUpstreamOperatorsKind = "TensorflowUpstreamOperators";

        internal static DataViewSchema GetModelSchema(IExceptionContext ectx, Graph graph, string opType = null)
        {
            var schemaBuilder = new DataViewSchema.Builder();
            foreach (Operation op in graph)
            {
                if (opType != null && opType != op.OpType)
                    continue;

                var tfType = op.OutputType(0);
                // Determine element type in Tensorflow tensor. For example, a vector of floats may get NumberType.R4 here.
                var mlType = DnnUtils.Tf2MlNetTypeOrNull(tfType);

                // If the type is not supported in ML.NET then we cannot represent it as a column in an Schema.
                // We also cannot output it with a TensorFlowTransform, so we skip it.
                // Furthermore, operators which have NumOutputs <= 0 needs to be filtered.
                // The 'GetTensorShape' method crashes TensorFlow runtime
                // (https://github.com/dotnet/machinelearning/issues/2156) when the operator has no outputs.
                if (mlType == null || op.NumOutputs <= 0)
                    continue;

                // Construct the final ML.NET type of a Tensorflow variable.
                var tensorShape = op.output.TensorShape.dims;
                var columnType = new VectorDataViewType(mlType);
                if (!(Utils.Size(tensorShape) == 1 && tensorShape[0] <= 0) &&
                    (Utils.Size(tensorShape) > 0 && tensorShape.Skip(1).All(x => x > 0)))
                    columnType = new VectorDataViewType(mlType, tensorShape[0] > 0 ? tensorShape : tensorShape.Skip(1).ToArray());

                // There can be at most two metadata fields.
                //  1. The first field always presents. Its value is this operator's type. For example,
                //     if an output is produced by an "Softmax" operator, the value of this field should be "Softmax".
                //  2. The second field stores operators whose outputs are consumed by this operator. In other words,
                //     these values are names of some upstream operators which should be evaluated before executing
                //     the current operator. It's possible that one operator doesn't need any input, so this field
                //     can be missing.
                var metadataBuilder = new DataViewSchema.Annotations.Builder();
                // Create the first metadata field.
                metadataBuilder.Add(TensorflowOperatorTypeKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => value = op.OpType.AsMemory());
                if (op.NumInputs > 0)
                {
                    // Put upstream operators' names to an array (type: VBuffer) of string (type: ReadOnlyMemory<char>).
                    VBuffer<ReadOnlyMemory<char>> upstreamOperatorNames = default;
                    var bufferEditor = VBufferEditor.Create(ref upstreamOperatorNames, op.NumInputs);
                    for (int i = 0; i < op.NumInputs; ++i)
                        bufferEditor.Values[i] = op.inputs[i].op.name.AsMemory();
                    upstreamOperatorNames = bufferEditor.Commit(); // Used in metadata's getter.

                    // Create the second metadata field.
                    metadataBuilder.Add(TensorflowUpstreamOperatorsKind, new VectorDataViewType(TextDataViewType.Instance, op.NumInputs),
                        (ref VBuffer<ReadOnlyMemory<char>> value) => { upstreamOperatorNames.CopyTo(ref value); });
                }

                schemaBuilder.AddColumn(op.name, columnType, metadataBuilder.ToAnnotations());
            }
            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="DataViewSchema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TensorFlowTransformer"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="TensorflowOperatorTypeKind"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="TensorflowUpstreamOperatorsKind"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">Model to load.</param>
        internal static DataViewSchema GetModelSchema(IHostEnvironment env, string modelPath)
        {
            var model = LoadTensorFlowModel(env, modelPath);
            return GetModelSchema(env, model.Session.graph);
        }

        // A TensorFlow frozen model is a single file. An un-frozen (SavedModel) on the other hand has a well-defined folder structure.
        // Given a modelPath, this utility method determines if we should treat it as a SavedModel or not
        internal static bool IsSavedModel(IHostEnvironment env, string modelPath)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(modelPath, nameof(modelPath));
            FileAttributes attr = File.GetAttributes(modelPath);
            return attr.HasFlag(FileAttributes.Directory);
        }

        /// <summary>
        /// Load TensorFlow model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <returns></returns>
        internal static TensorFlowModel LoadTensorFlowModel(IHostEnvironment env, string modelPath)
        {
            var session = DnnUtils.GetSession(env, modelPath);
            return new TensorFlowModel(env, session, modelPath);
        }
    }
}
