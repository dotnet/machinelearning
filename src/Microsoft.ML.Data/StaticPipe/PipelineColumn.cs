﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe.Runtime;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// This class is used as a type marker for <see cref="IDataView"/> producing structures for use in the statically
    /// typed columnate pipeline building helper API. Users will not create these structures directly. Rather components
    /// will implement (hidden) subclasses of one of this classes subclasses (for example, <see cref="Scalar{T}"/>,
    /// <see cref="Vector{T}"/>), which will contain information that the builder API can use to construct an actual
    /// sequence of <see cref="IEstimator{TTransformer}"/> objects.
    /// </summary>
    public abstract class PipelineColumn
    {
        internal readonly Reconciler ReconcilerObj;
        internal readonly PipelineColumn[] Dependencies;

        private protected PipelineColumn(Reconciler reconciler, PipelineColumn[] dependencies)
        {
            Contracts.CheckValue(reconciler, nameof(reconciler));
            Contracts.CheckValueOrNull(dependencies);

            ReconcilerObj = reconciler;
            Dependencies = dependencies;
        }
    }

    /// <summary>
    /// For representing a non-key, non-vector <see cref="ColumnType"/>.
    /// </summary>
    /// <typeparam name="T">The scalar item type.</typeparam>
    public abstract class Scalar<T> : PipelineColumn
    {
        protected Scalar(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(Scalar<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a <see cref="VectorType"/> of known length.
    /// </summary>
    /// <typeparam name="T">The vector item type.</typeparam>
    public abstract class Vector<T> : PipelineColumn
    {
        protected Vector(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(Vector<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a <see cref="VectorType"/> that is normalized, that is, its <see cref="MetadataUtils.Kinds.IsNormalized"/>
    /// value is set with the value <c>true</c>.
    /// </summary>
    /// <typeparam name="T">The vector item type.</typeparam>
    public abstract class NormVector<T> : Vector<T>
    {
        protected NormVector(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(NormVector<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a <see cref="VectorType"/> of unknown length.
    /// </summary>
    /// <typeparam name="T">The vector item type.</typeparam>
    public abstract class VarVector<T> : PipelineColumn
    {
        protected VarVector(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(VarVector<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a <see cref="KeyType"/> of known cardinality, where the type of key is not specified.
    /// </summary>
    /// <typeparam name="T">The physical type representing the key, which should always be one of <see cref="byte"/>,
    /// <see cref="ushort"/>, <see cref="uint"/>, or <see cref="ulong"/></typeparam>
    /// <remarks>Note that a vector of keys type we would represent as <see cref="Vector{T}"/> with a
    /// <see cref="Key{T}"/> type parameter. Note also, if the type of the key is known then that should be represented
    /// by <see cref="Key{T, TVal}"/>.</remarks>
    public abstract class Key<T> : PipelineColumn
    {
        protected Key(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(Key<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a key-type of known cardinality that has key values over a particular type. This is used to
    /// represent a <see cref="KeyType"/> where it is known that it will have <see
    /// cref="MetadataUtils.Kinds.KeyValues"/> of a particular type <typeparamref name="TVal"/>.
    /// </summary>
    /// <typeparam name="T">The physical type representing the key, which should always be one of <see cref="byte"/>,
    /// <see cref="ushort"/>, <see cref="uint"/>, or <see cref="ulong"/></typeparam>
    /// <typeparam name="TVal">The type of values the key-type is enumerating. Commonly this is <see cref="string"/> but
    /// this is not necessary</typeparam>
    public abstract class Key<T, TVal> : Key<T>
    {
        protected Key(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(Key<T, TVal>)}<{typeof(T).Name}, {typeof(TVal).Name}>";
    }

    /// <summary>
    /// For representing a <see cref="KeyType"/> of unknown cardinality.
    /// </summary>
    /// <typeparam name="T">The physical type representing the key, which should always be one of <see cref="byte"/>,
    /// <see cref="ushort"/>, <see cref="uint"/>, or <see cref="ulong"/></typeparam>
    /// <remarks>Note that unlike the <see cref="Key{T}"/> and <see cref="Key{T, TVal}"/> duality, there is no
    /// type corresponding to this type but with key-values, since key-values are necessarily a vector of known
    /// size so any enumeration into that set would itself be a key-value of unknown cardinality.</remarks>
    public abstract class VarKey<T> : PipelineColumn
    {
        protected VarKey(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(VarKey<T>)}<{typeof(T).Name}>";
    }

    /// <summary>
    /// For representing a custom <see cref="ColumnType"/>.
    /// </summary>
    /// <typeparam name="T">The custom item type.</typeparam>
    public abstract class Custom<T>: PipelineColumn
    {
        protected Custom(Reconciler reconciler, params PipelineColumn[] dependencies)
            : base(reconciler, dependencies)
        {
        }

        public override string ToString() => $"{nameof(Custom<T>)}<{typeof(T).Name}>";
    }

}
