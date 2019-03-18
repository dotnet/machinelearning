// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// A schema shape with names corresponding to a type parameter in one of the typed variants
    /// of the data pipeline structures. Instances of this class tend to be bundled with the statically
    /// typed variants of the dynamic structures (for example, <see cref="DataView{TShape}"/> and so forth),
    /// and their primary purpose is to ensure that the schemas of the dynamic structures and the
    /// statically declared structures are compatible.
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
        /// <typeparam name="TShape">The static shape type.</typeparam>
        /// <param name="info">The parameter info on the method, whose type should be
        /// <typeparamref name="TShape"/>.</param>
        /// <returns>A new instance with names and members types enumerated.</returns>
        public static StaticSchemaShape Make<TShape>(ParameterInfo info)
        {
            Contracts.AssertValue(info);
            var pairs = StaticPipeInternalUtils.GetNamesTypes<TShape, PipelineColumn>(info);
            return new StaticSchemaShape(pairs);
        }

        /// <summary>
        /// Checks whether this object is consistent with an actual schema from a dynamic object,
        /// throwing exceptions if not.
        /// </summary>
        /// <param name="ectx">The context on which to throw exceptions</param>
        /// <param name="schema">The schema to check</param>
        public void Check(IExceptionContext ectx, DataViewSchema schema)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(schema);

            foreach (var pair in Pairs)
            {
                if (!schema.TryGetColumnIndex(pair.Key, out int colIdx))
                    throw ectx.ExceptParam(nameof(schema), $"Column named '{pair.Key}' was not found");
                var col = schema[colIdx];
                var type = GetTypeOrNull(col);
                if ((type != null && !pair.Value.IsAssignableFromStaticPipeline(type)) || (type == null && IsStandard(ectx, pair.Value)))
                {
                    // When not null, we can use IsAssignableFrom to indicate we could assign to this, so as to allow
                    // for example Key<uint, string> to be considered to be compatible with Key<uint>.

                    // In the null case, while we cannot directly verify an unrecognized type, we can at least verify
                    // that the statically declared type should not have corresponded to a recognized type.
                    if (!pair.Value.IsAssignableFromStaticPipeline(type))
                    {
                        throw ectx.ExceptParam(nameof(schema),
                            $"Column '{pair.Key}' of type '{col.Type}' cannot be expressed statically as type '{pair.Value}'.");
                    }
                }
            }
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
                if (!shape.TryFindColumn(pair.Key, out var col))
                    throw ectx.ExceptParam(nameof(shape), $"Column named '{pair.Key}' was not found");
                var type = GetTypeOrNull(col);
                if ((type != null && !pair.Value.IsAssignableFromStaticPipeline(type)) || (type == null && IsStandard(ectx, pair.Value)))
                {
                    // When not null, we can use IsAssignableFrom to indicate we could assign to this, so as to allow
                    // for example Key<uint, string> to be considered to be compatible with Key<uint>.

                    // In the null case, while we cannot directly verify an unrecognized type, we can at least verify
                    // that the statically declared type should not have corresponded to a recognized type.
                    if (!pair.Value.IsAssignableFromStaticPipeline(type))
                    {
                        throw ectx.ExceptParam(nameof(shape),
                            $"Column '{pair.Key}' of type '{col.GetTypeString()}' cannot be expressed statically as type '{pair.Value}'.");
                    }
                }
            }
        }

        private static Type GetTypeOrNull(SchemaShape.Column col)
        {
            Contracts.Assert(col.IsValid);

            Type vecType = null;

            switch (col.Kind)
            {
                case SchemaShape.Column.VectorKind.Scalar:
                    break; // Keep it null.
                case SchemaShape.Column.VectorKind.Vector:
                    // Assume that if the normalized metadata is indicated by the schema shape, it is bool and true.
                    vecType = col.IsNormalized() ? typeof(NormVector<>) : typeof(Vector<>);
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
                Type physType = GetPhysicalType(col.ItemType);
                Contracts.Assert(physType == typeof(byte) || physType == typeof(ushort)
                    || physType == typeof(uint) || physType == typeof(ulong));
                var keyType = typeof(Key<>).MakeGenericType(physType);
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.KeyValues, out var kvMeta))
                {
                    var subtype = GetTypeOrNull(kvMeta);
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

                return vecType?.MakeGenericType(keyType) ?? keyType;
            }

            if (col.ItemType is PrimitiveDataViewType pt)
            {
                Type physType = GetPhysicalType(pt);
                // Though I am unaware of any existing instances, it is theoretically possible for a
                // primitive type to exist, have the same data kind as one of the existing types, and yet
                // not be one of the built in types. (For example, an outside analogy to the key types.) For this
                // reason, we must be certain that when we return here we are covering one fo the builtin types.
                if (physType != null && (
                    pt == NumberDataViewType.SByte || pt == NumberDataViewType.Int16 || pt == NumberDataViewType.Int32 || pt == NumberDataViewType.Int32 ||
                    pt == NumberDataViewType.Byte || pt == NumberDataViewType.UInt16 || pt == NumberDataViewType.UInt32 || pt == NumberDataViewType.UInt32 ||
                    pt == NumberDataViewType.Single || pt == NumberDataViewType.Double || pt == RowIdDataViewType.Instance || pt == BooleanDataViewType.Instance ||
                    pt == DateTimeDataViewType.Instance || pt == DateTimeOffsetDataViewType.Instance || pt == TimeSpanDataViewType.Instance ||
                    pt == TextDataViewType.Instance))
                {
                    return (vecType ?? typeof(Scalar<>)).MakeGenericType(physType);
                }
            }

            return null;
        }

        /// <summary>
        /// Returns true if the input type is something recognizable as being oen of the standard
        /// builtin types. This method will also throw if something is detected as being definitely
        /// wrong (for example, the input type does not descend from <see cref="PipelineColumn"/> at all,
        /// or a <see cref="Key{T}"/> is declared with a <see cref="string"/> type parameter or
        /// something.
        /// </summary>
        private static bool IsStandard(IExceptionContext ectx, Type t)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(t);
            if (!typeof(PipelineColumn).IsAssignableFrom(t))
            {
                throw ectx.ExceptParam(nameof(t), $"Type {t} was not even of {nameof(PipelineColumn)}");
            }
            var gt = t.IsGenericType ? t.GetGenericTypeDefinition() : t;
            if (gt != typeof(Scalar<>) && gt != typeof(Key<>) && gt != typeof(Key<,>) && gt != typeof(VarKey<>) &&
                gt != typeof(Vector<>) && gt != typeof(VarVector<>) && gt != typeof(NormVector<>) && gt != typeof(Custom<>))
            {
                throw ectx.ExceptParam(nameof(t),
                    $"Type {t} was not one of the standard subclasses of {nameof(PipelineColumn)}");
            }
            ectx.Assert(t.IsGenericType);
            var ga = t.GetGenericArguments();
            ectx.AssertNonEmpty(ga);

            if (gt == typeof(Key<>) || gt == typeof(Key<,>) || gt == typeof(VarKey<>))
            {
                ectx.Assert((gt == typeof(Key<,>) && ga.Length == 2) || ga.Length == 1);
                var kt = ga[0];
                if (kt != typeof(byte) && kt != typeof(ushort) && kt != typeof(uint) && kt != typeof(ulong))
                    throw ectx.ExceptParam(nameof(t), $"Type parameter {kt.Name} is not a valid type for key");
                return gt != typeof(Key<,>) || IsStandardCore(ga[1]);
            }

            ectx.Assert(ga.Length == 1);
            return IsStandardCore(ga[0]);
        }

        private static bool IsStandardCore(Type t)
        {
            Contracts.AssertValue(t);
            return t == typeof(float) || t == typeof(double) || t == typeof(string) || t == typeof(bool) ||
                t == typeof(sbyte) || t == typeof(short) || t == typeof(int) || t == typeof(long) ||
                t == typeof(byte) || t == typeof(ushort) || t == typeof(uint) || t == typeof(ulong) ||
                t == typeof(TimeSpan) || t == typeof(DateTime) || t == typeof(DateTimeOffset);
        }

        /// <summary>
        /// Returns a .NET type corresponding to the static pipelines that would tend to represent this column.
        /// Generally this will return <c>null</c> if it simply does not recognize the type but might throw if
        /// there is something seriously wrong with it.
        /// </summary>
        /// <param name="col">The column</param>
        /// <returns>The .NET type for the static pipelines that should be used to reflect this type, given
        /// both the characteristics of the <see cref="DataViewType"/> as well as one or two crucial pieces of metadata</returns>
        private static Type GetTypeOrNull(DataViewSchema.Column col)
        {
            var t = col.Type;

            Type vecType = null;
            if (t is VectorType vt)
            {
                vecType = vt.Size > 0 ? typeof(Vector<>) : typeof(VarVector<>);
                // Check normalized subtype of vectors.
                if (vt.Size > 0)
                {
                    // Check to see if the column is normalized.
                    // Once we shift to metadata being a row globally we can also make this a bit more efficient:
                    var meta = col.Annotations;
                    var normalizedColumn = meta.Schema.GetColumnOrNull(AnnotationUtils.Kinds.IsNormalized);
                    if (normalizedColumn.HasValue)
                    {
                        if (normalizedColumn.Value.Type == BooleanDataViewType.Instance)
                        {
                            bool val = default;
                            meta.GetGetter<bool>(normalizedColumn.Value)(ref val);
                            if (val)
                                vecType = typeof(NormVector<>);
                        }
                    }
                }
                t = vt.ItemType;
                // Fall through to the non-vector case to handle subtypes.
            }
            Contracts.Assert(!(t is VectorType));

            if (t is KeyType kt)
            {
                Type physType = GetPhysicalType(kt);
                Contracts.Assert(physType == typeof(byte) || physType == typeof(ushort)
                    || physType == typeof(uint) || physType == typeof(ulong));
                var keyType = kt.Count > 0 ? typeof(Key<>) : typeof(VarKey<>);
                keyType = keyType.MakeGenericType(physType);

                if (kt.Count > 0)
                {
                    // Check to see if we have key value metadata of the appropriate type, size, and whatnot.
                    var meta = col.Annotations;
                    if (meta.Schema.TryGetColumnIndex(AnnotationUtils.Kinds.KeyValues, out int kvcolIndex))
                    {
                        var kvcol = meta.Schema[kvcolIndex];
                        var kvType = kvcol.Type;
                        Contracts.Assert(kt.Count <= int.MaxValue);
                        if (kvType is VectorType kvVecType && kvVecType.Size == (int)kt.Count)
                        {
                            Contracts.Assert(kt.Count > 0);
                            var subtype = GetTypeOrNull(kvcol);
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

            if (t is PrimitiveDataViewType pt)
            {
                Type physType = GetPhysicalType(pt);
                // Though I am unaware of any existing instances, it is theoretically possible for a
                // primitive type to exist, have the same data kind as one of the existing types, and yet
                // not be one of the built in types. (For example, an outside analogy to the key types.) For this
                // reason, we must be certain that when we return here we are covering one fo the builtin types.
                if (physType != null && (
                    pt == NumberDataViewType.SByte || pt == NumberDataViewType.Int16 || pt == NumberDataViewType.Int32 || pt == NumberDataViewType.Int64 ||
                    pt == NumberDataViewType.Byte || pt == NumberDataViewType.UInt16 || pt == NumberDataViewType.UInt32 || pt == NumberDataViewType.UInt64 ||
                    pt == NumberDataViewType.Single || pt == NumberDataViewType.Double || pt == RowIdDataViewType.Instance || pt == BooleanDataViewType.Instance ||
                    pt == DateTimeDataViewType.Instance || pt == DateTimeOffsetDataViewType.Instance || pt == TimeSpanDataViewType.Instance ||
                    pt == TextDataViewType.Instance))
                {
                    return (vecType ?? typeof(Scalar<>)).MakeGenericType(physType);
                }
            }

            return null;
        }

        /// <summary>
        /// Note that this can return a different type than the actual physical representation type, for example, for
        /// <see cref="InternalDataKind.Text"/> the return type is <see cref="string"/>, even though we do not use that
        /// type for communicating text.
        /// </summary>
        /// <returns>The basic type used to represent an item type in the static pipeline</returns>
        private static Type GetPhysicalType(DataViewType columnType)
        {
            switch (columnType)
            {
                case NumberDataViewType numberType:
                case KeyType keyType:
                case TimeSpanDataViewType timeSpanType:
                case DateTimeDataViewType dateTimeType:
                case DateTimeOffsetDataViewType dateTimeOffsetType:
                case BooleanDataViewType boolType:
                    return columnType.RawType;
                case TextDataViewType textType:
                    return typeof(string);

                default:
                    return null;
            }
        }
    }
}
