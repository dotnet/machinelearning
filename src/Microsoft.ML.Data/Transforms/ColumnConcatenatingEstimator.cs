// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Concatenates columns in an <see cref="IDataView"/> into one single column. Estimator for the <see cref="ColumnConcatenatingTransformer"/>.
    /// </summary>
    public sealed class ColumnConcatenatingEstimator : IEstimator<ColumnConcatenatingTransformer>
    {
        private readonly IHost _host;
        private readonly string _name;
        private readonly string[] _source;

        /// <summary>
        /// Initializes a new instance of <see cref="ColumnConcatenatingEstimator"/>
        /// </summary>
        /// <param name="env">The local instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="outputColumnName">The name of the resulting column.</param>
        /// <param name="inputColumnNames">The columns to concatenate into one single column.</param>
        internal ColumnConcatenatingEstimator(IHostEnvironment env, string outputColumnName, params string[] inputColumnNames)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ColumnConcatenatingEstimator));

            _host.CheckNonEmpty(outputColumnName, nameof(outputColumnName));
            _host.CheckValue(inputColumnNames, nameof(inputColumnNames));
            _host.CheckParam(!inputColumnNames.Any(r => string.IsNullOrEmpty(r)), nameof(inputColumnNames),
                "Contained some null or empty items");

            _name = outputColumnName;
            _source = inputColumnNames;
        }

        /// <summary>
        /// Trains and returns a <see cref="ColumnConcatenatingTransformer"/>.
        /// </summary>
        public ColumnConcatenatingTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new ColumnConcatenatingTransformer(_host, _name, _source);
        }

        private bool HasCategoricals(SchemaShape.Column col)
        {
            _host.Assert(col.IsValid);
            if (!col.Annotations.TryFindColumn(AnnotationUtils.Kinds.CategoricalSlotRanges, out var mcol))
                return false;
            // The indices must be ints and of a definite size vector type. (Definite becuase
            // metadata has only one value anyway.)
            return mcol.Kind == SchemaShape.Column.VectorKind.Vector
                && mcol.ItemType == NumberDataViewType.Int32;
        }

        private SchemaShape.Column CheckInputsAndMakeColumn(
            SchemaShape inputSchema, string name, string[] sources)
        {
            _host.AssertNonEmpty(sources);

            var cols = new SchemaShape.Column[sources.Length];
            // If any input is a var vector, so is the output.
            bool varVector = false;
            // If any input is not normalized, the output is not normalized.
            bool isNormalized = true;
            // If any input has categorical indices, so will the output.
            bool hasCategoricals = false;
            // If any is scalar or had slot names, then the output will have slot names.
            bool hasSlotNames = false;

            // We will get the item type from the first column.
            DataViewType itemType = null;

            for (int i = 0; i < sources.Length; ++i)
            {
                if (!inputSchema.TryFindColumn(sources[i], out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", sources[i]);
                if (i == 0)
                    itemType = col.ItemType;
                // For the sake of an estimator I am going to have a hard policy of no keys.
                // Appending keys makes no real sense anyway.
                if (col.IsKey)
                {
                    throw _host.Except($"Column '{sources[i]}' is key." +
                        $"Concatenation of keys is unsupported.");
                }
                if (!col.ItemType.Equals(itemType))
                {
                    throw _host.Except($"Column '{sources[i]}' has values of {col.ItemType}" +
                        $"which is not the same as earlier observed type of {itemType}.");
                }
                varVector |= col.Kind == SchemaShape.Column.VectorKind.VariableVector;
                isNormalized &= col.IsNormalized();
                hasCategoricals |= HasCategoricals(col);
                hasSlotNames |= col.Kind == SchemaShape.Column.VectorKind.Scalar || col.HasSlotNames();
            }
            var vecKind = varVector ? SchemaShape.Column.VectorKind.VariableVector :
                    SchemaShape.Column.VectorKind.Vector;

            List<SchemaShape.Column> meta = new List<SchemaShape.Column>();
            if (isNormalized)
                meta.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
            if (hasCategoricals)
                meta.Add(new SchemaShape.Column(AnnotationUtils.Kinds.CategoricalSlotRanges, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Int32, false));
            if (hasSlotNames)
                meta.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));

            return new SchemaShape.Column(name, vecKind, itemType, false, new SchemaShape(meta));
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            result[_name] = CheckInputsAndMakeColumn(inputSchema, _name, _source);
            return new SchemaShape(result.Values);
        }
    }
}
