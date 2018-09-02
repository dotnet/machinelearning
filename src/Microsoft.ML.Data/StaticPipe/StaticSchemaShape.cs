// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
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

        private static Type GetTypeOrNull(SchemaShape.Column col)
        {
            Contracts.AssertValue(col);

            Type vecType = null;
            switch (col.Kind)
            {
                case SchemaShape.Column.VectorKind.Scalar:
                    break; // Keep it null.
                case SchemaShape.Column.VectorKind.Vector:
                    // Assume that if the normalized metadata is indicated by the schema shape, it is bool and true.
                    vecType = col.MetadataKinds.Contains(MetadataUtils.Kinds.IsNormalized) ? typeof(NormVector<>) : typeof(Vector<>);
                    break;
                case SchemaShape.Column.VectorKind.VariableVector:
                    vecType = typeof(VarVector<>);
                    break;
                default:
                    // Not recognized. Not necessarily an error of the user, may just indicate this code ought to be updated.
                    Contracts.Assert(false);
                    return null;
            }

            if (col.IsKey)
            {
                Type physType = StaticKind(col.ItemType.RawKind);
                Contracts.Assert(physType == typeof(byte) || physType == typeof(ushort)
                    || physType == typeof(uint) || physType == typeof(ulong));
                // As of the time of this writing we cannot distinguish between multiple types of key value metadata,
                // so, we don't try. This is tracked in this issue: https://github.com/dotnet/machinelearning/issues/755.
                // Because Key<,> descends from Key<> the check will still work. Also the idiom here has no way of
                // representing variable size keys.
                var keyType = typeof(Key<>).MakeGenericType(physType);
                return vecType?.MakeGenericType(keyType) ?? keyType;
            }

            if (col.ItemType is PrimitiveType pt)
            {
                Type physType = StaticKind(pt.RawKind);
                // Though I am unaware of any existing instances, it is theoretically possible for a
                // primitive type to exist, have the same data kind as one of the existing types, and yet
                // not be one of the built in types. (E.g., an outside analogy to the key types.) For this
                // reason, we must be certain that when we return here we are covering one fo the builtin types.
                if (physType != null && (
                    pt == NumberType.I1 || pt == NumberType.I2 || pt == NumberType.I4 || pt == NumberType.I4 ||
                    pt == NumberType.U1 || pt == NumberType.U2 || pt == NumberType.U4 || pt == NumberType.U4 ||
                    pt == NumberType.R4 || pt == NumberType.R8 || pt == NumberType.UG || pt == BoolType.Instance ||
                    pt == DateTimeType.Instance || pt == DateTimeZoneType.Instance || pt == TimeSpanType.Instance ||
                    pt == TextType.Instance))
                {
                    return (vecType ?? typeof(Scalar<>)).MakeGenericType(physType);
                }
            }

            return null;
        }

        /// <summary>
        /// Returns a .NET type corresponding to the static pipelines that would tend to represent this column.
        /// Generally this will return <c>null</c> if it simply does not recognize the type but might throw if
        /// there is something seriously wrong with it.
        /// </summary>
        /// <param name="col">The column</param>
        /// <returns>The .NET type for the static pipelines that should be used to reflect this type, given
        /// both the characteristics of the <see cref="ColumnType"/> as well as one or two crucial pieces of metadata</returns>
        private static Type GetTypeOrNull(IColumn col)
        {
            Contracts.AssertValue(col);
            var t = col.Type;

            Type vecType = null;
            if (t is VectorType vt)
            {
                vecType = vt.VectorSize > 0 ? typeof(Vector<>) : typeof(VarVector<>);
                // Check normalized subtype of vectors.
                if (vt.VectorSize > 0)
                {
                    // Check to see if the column is normalized.
                    // Once we shift to metadata being a row globally we can also make this a bit more efficient:
                    var meta = col.Metadata;
                    if (meta.Schema.TryGetColumnIndex(MetadataUtils.Kinds.IsNormalized, out int normcol))
                    {
                        var normtype = meta.Schema.GetColumnType(normcol);
                        if (normtype == BoolType.Instance)
                        {
                            DvBool val = default;
                            meta.GetGetter<DvBool>(normcol)(ref val);
                            if (val.IsTrue)
                                vecType = typeof(NormVector<>);
                        }
                    }
                }
                t = t.ItemType;
                // Fall through to the non-vector case to handle subtypes.
            }
            Contracts.Assert(!t.IsVector);

            if (t is KeyType kt)
            {
                Type physType = StaticKind(kt.RawKind);
                Contracts.Assert(physType == typeof(byte) || physType == typeof(ushort)
                    || physType == typeof(uint) || physType == typeof(ulong));
                var keyType = kt.Count > 0 ? typeof(Key<>) : typeof(VarKey<>);
                keyType = keyType.MakeGenericType(physType);

                if (kt.Count > 0)
                {
                    // Check to see if we have key value metadata of the appropriate type, size, and whatnot.
                    var meta = col.Metadata;
                    if (meta.Schema.TryGetColumnIndex(MetadataUtils.Kinds.KeyValues, out int kvcol))
                    {
                        var kvType = meta.Schema.GetColumnType(kvcol);
                        if (kvType.VectorSize == kt.Count)
                        {
                            Contracts.Assert(kt.Count > 0);
                            var subtype = GetTypeOrNull(RowColumnUtils.GetColumn(meta, kvcol));
                            if (subtype != null && subtype.IsGenericType)
                            {
                                var sgtype = subtype.GetGenericTypeDefinition();
                                if (sgtype == typeof(NormVector<>) || sgtype == typeof(Vector<>))
                                {
                                    var args = subtype.GetGenericArguments();
                                    Contracts.Assert(args.Length == 1);
                                    keyType = typeof(Key<,>).MakeGenericType(physType, args[0]);
                                }
                            }
                        }
                    }
                }
                return vecType?.MakeGenericType(keyType) ?? keyType;
            }

            if (t is PrimitiveType pt)
            {
                Type physType = StaticKind(pt.RawKind);
                // Though I am unaware of any existing instances, it is theoretically possible for a
                // primitive type to exist, have the same data kind as one of the existing types, and yet
                // not be one of the built in types. (E.g., an outside analogy to the key types.) For this
                // reason, we must be certain that when we return here we are covering one fo the builtin types.
                if (physType != null && (
                    pt == NumberType.I1 || pt == NumberType.I2 || pt == NumberType.I4 || pt == NumberType.I4 ||
                    pt == NumberType.U1 || pt == NumberType.U2 || pt == NumberType.U4 || pt == NumberType.U4 ||
                    pt == NumberType.R4 || pt == NumberType.R8 || pt == NumberType.UG || pt == BoolType.Instance ||
                    pt == DateTimeType.Instance || pt == DateTimeZoneType.Instance || pt == TimeSpanType.Instance ||
                    pt == TextType.Instance))
                {
                    return (vecType ?? typeof(Scalar<>)).MakeGenericType(physType);
                }
            }

            return null;
        }

        /// <summary>
        /// Note that this can return a different type than the actual physical representation type, e.g., for
        /// <see cref="DataKind.Text"/> the return type is <see cref="string"/>, even though we do not use that
        /// type for communicating text.
        /// </summary>
        /// <returns>The basic type used to represent an item type</returns>
        private static Type StaticKind(DataKind kind)
        {
            switch (kind)
            {
                // The default kind is reserved for unknown types.
                case default(DataKind): return null;
                case DataKind.I1: return typeof(sbyte);
                case DataKind.I2: return typeof(short);
                case DataKind.I4: return typeof(int);
                case DataKind.I8: return typeof(long);

                case DataKind.U1: return typeof(byte);
                case DataKind.U2: return typeof(ushort);
                case DataKind.U4: return typeof(uint);
                case DataKind.U8: return typeof(ulong);
                case DataKind.U16: return typeof(UInt128);

                case DataKind.R4: return typeof(float);
                case DataKind.R8: return typeof(double);

                case DataKind.Text: return typeof(string);
                case DataKind.TimeSpan: return typeof(TimeSpan);
                case DataKind.DateTime: return typeof(DateTime);
                case DataKind.DateTimeZone: return typeof(DateTimeOffset);

                default:
                    throw Contracts.ExceptParam(nameof(kind), $"Unrecognized type '{kind}'");
            }
        }
    }
}
