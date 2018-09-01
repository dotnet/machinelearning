// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe.Runtime
{
    /// <summary>
    /// A schema shape with names corresponding to a type parameter in one of the typed variants
    /// of the data pipeline structures. Used for validation.
    /// </summary>
    internal sealed class StaticSchemaShape
    {
        /// <summary>
        /// The enumeration of name/type pairs. Do not modify.
        /// </summary>
        public readonly KeyValuePair<string, Type>[] Pairs;

        private StaticSchemaShape(KeyValuePair<string, Type>[] pairs)
        {
            Contracts.AssertValue(pairs);
            Pairs = pairs;
        }

        /// <summary>
        /// Creates a new instance out of a parameter info, presumably fetched from a user specified delegate.
        /// </summary>
        /// <typeparam name="TTupleShape">The static tuple-shape type</typeparam>
        /// <param name="info">The parameter info on the method, whose type should be
        /// <typeparamref name="TTupleShape"/></param>
        /// <returns>A new instance with names and members types enumerated</returns>
        public static StaticSchemaShape Make<TTupleShape>(ParameterInfo info)
        {
            Contracts.AssertValue(info);
            var pairs = StaticPipeInternalUtils.GetNamesTypes<TTupleShape, PipelineColumn>(info);
            return new StaticSchemaShape(pairs);
        }

        /// <summary>
        /// Checks whether this object is consistent with an actual schema from a dynamic object,
        /// throwing exceptions if not.
        /// </summary>
        /// <param name="ectx">The context on which to throw exceptions</param>
        /// <param name="schema">The schema to check</param>
        public void Check(IExceptionContext ectx, ISchema schema)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(schema);

            foreach (var pair in Pairs)
            {
                if (!schema.TryGetColumnIndex(pair.Key, out int col))
                    throw ectx.ExceptParam(nameof(schema), $"Column named '{pair.Key}' was not found");
            }
            // REVIEW: Need more checking of types and whatnot.
        }

        /// <summary>
        /// Checks whether this object is consistent with an actual schema shape from a dynamic object,
        /// throwing exceptions if not.
        /// </summary>
        /// <param name="ectx">The context on which to throw exceptions</param>
        /// <param name="shape">The schema shape to check</param>
        public void Check(IExceptionContext ectx, SchemaShape shape)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(shape);

            foreach (var pair in Pairs)
            {
                var col = shape.FindColumn(pair.Key);
                if (col == null)
                    throw ectx.ExceptParam(nameof(shape), $"Column named '{pair.Key}' was not found");
            }
            // REVIEW: Need more checking of types and whatnot.
        }
    }
}
