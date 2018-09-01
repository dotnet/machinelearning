// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe.Runtime
{
    /// <summary>
    /// An object for declaring a schema-shape. This is mostly commonly used in situations where a user is
    /// asserting that a dynamic object bears a certain specific static schema. For example: when phrasing
    /// the dynamically typed <see cref="IDataView"/> as being a specific <see cref="DataView{TTupleShape}"/>.
    /// It is never created by the user directly, but instead an instance is typically fed in as an argument
    /// to a delegate, and the user will call methods on this context to indicate a certain type is so.
    /// </summary>
    /// <remarks>
    /// All <see cref="PipelineColumn"/> objects are, deliberately, imperitavely useless as they are
    /// intended to be used only in a declarative fashion. The methods and properties of this class go one step
    /// further and return <c>null</c> for everything with a return type of <see cref="PipelineColumn"/>.
    ///
    /// Because <see cref="IDataView"/>'s type system is extensible, assemblies that declare their own types
    /// should allow users to assert typedness in their types by defining extension methods over this class.
    /// However, even failing the provision of such a helper, a user can still provide a workaround by just
    /// declaring the type as something like <c>default(Scalar&lt;TheCustomType&gt;</c>, without using the
    /// instance of this context.
    /// </remarks>
    public sealed class SchemaAssertionContext
    {
        /// <summary>Assertions over a column of <see cref="NumberType.I1"/>.</summary>
        public PrimitiveTypeAssertions<sbyte> I1 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.I2"/>.</summary>
        public PrimitiveTypeAssertions<short> I2 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.I4"/>.</summary>
        public PrimitiveTypeAssertions<int> I4 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.I8"/>.</summary>
        public PrimitiveTypeAssertions<long> I8 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.U1"/>.</summary>
        public PrimitiveTypeAssertions<byte> U1 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.U2"/>.</summary>
        public PrimitiveTypeAssertions<ushort> U2 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.U4"/>.</summary>
        public PrimitiveTypeAssertions<uint> U4 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.U8"/>.</summary>
        public PrimitiveTypeAssertions<ulong> U8 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.R4"/>.</summary>
        public NormalizableTypeAssertions<float> R4 { get; }

        /// <summary>Assertions over a column of <see cref="NumberType.R8"/>.</summary>
        public NormalizableTypeAssertions<double> R8 { get; }

        /// <summary>Assertions over a column of <see cref="TextType"/>.</summary>
        public PrimitiveTypeAssertions<string> Text { get; }

        /// <summary>Assertions over a column of <see cref="BoolType"/>.</summary>
        public PrimitiveTypeAssertions<bool> Bool { get; }

        /// <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U1"/> <see cref="ColumnType.RawKind"/>.</summary>
        public KeyTypeAssertions<byte> KeyU1 { get; }
        /// <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U2"/> <see cref="ColumnType.RawKind"/>.</summary>
        public KeyTypeAssertions<ushort> KeyU2 { get; }
        /// <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U4"/> <see cref="ColumnType.RawKind"/>.</summary>
        public KeyTypeAssertions<uint> KeyU4 { get; }
        /// <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U8"/> <see cref="ColumnType.RawKind"/>.</summary>
        public KeyTypeAssertions<ulong> KeyU8 { get; }

        internal SchemaAssertionContext()
        {
            I1 = new PrimitiveTypeAssertions<sbyte>();
            I2 = new PrimitiveTypeAssertions<short>();
            I4 = new PrimitiveTypeAssertions<int>();
            I8 = new PrimitiveTypeAssertions<long>();

            U1 = new PrimitiveTypeAssertions<byte>();
            U2 = new PrimitiveTypeAssertions<ushort>();
            U4 = new PrimitiveTypeAssertions<uint>();
            U8 = new PrimitiveTypeAssertions<ulong>();

            R4 = new NormalizableTypeAssertions<float>();
            R8 = new NormalizableTypeAssertions<double>();

            Text = new PrimitiveTypeAssertions<string>();
            Bool = new PrimitiveTypeAssertions<bool>();

            KeyU1 = new KeyTypeAssertions<byte>();
            KeyU2 = new KeyTypeAssertions<ushort>();
            KeyU4 = new KeyTypeAssertions<uint>();
            KeyU8 = new KeyTypeAssertions<ulong>();
        }

        // Until we have some transforms that use them, we might not expect to see too much interest in asserting
        // the time relevant datatypes.

        /// <summary>
        /// Holds assertions relating to a particular type.
        /// </summary>
        /// <typeparam name="T">The type corresponding to that type</typeparam>
        public abstract class TypeAssertionsBase<T>
        {
            protected internal TypeAssertionsBase() { }

            /// <summary>
            /// Asserts a type that is directly this <see cref="PrimitiveType"/>.
            /// </summary>
            public Scalar<T> Scalar => null;

            /// <summary>
            /// Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
            /// where <see cref="ColumnType.IsKnownSizeVector"/> is true.
            /// </summary>
            public Vector<T> Vector => null;

            /// <summary>
            /// Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
            /// where <see cref="ColumnType.IsKnownSizeVector"/> is true.
            /// </summary>
            public Scalar<T> VarVector => null;
        }

        public sealed class PrimitiveTypeAssertions<T> : TypeAssertionsBase<T> { internal PrimitiveTypeAssertions() { } }

        public sealed class NormalizableTypeAssertions<T> : TypeAssertionsBase<T>
        {
            internal NormalizableTypeAssertions() { }

            /// <summary>
            /// Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
            /// where <see cref="ColumnType.IsKnownSizeVector"/> is true, and the <see cref="MetadataUtils.Kinds.IsNormalized"/>
            /// metadata is defined with a Boolean <c>true</c> value.
            /// </summary>
            public NormVector<T> NormVector => null;
        }

        /// <summary>
        /// Assertions for key types of various forms.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public sealed class KeyTypeAssertions<T>
        {
            internal KeyTypeAssertions() { }

            /// <summary>
            /// Asserts a type corresponding to a <see cref="KeyType"/> where <see cref="KeyType.Count"/> is positive, that is, is of known cardinality,
            /// but that we are not asserting has any particular type of <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.
            /// </summary>
            public Key<T> NoValue => null;

            /// <summary>
            /// Asserts a type corresponding to a <see cref="KeyType"/> where <see cref="KeyType.Count"/> is zero, that is, is of unknown cardinality.
            /// </summary>
            public VarKey<T> UnknownCardinality => null;

            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I1"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, sbyte> I1Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I2"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, short> I2Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, int> I4Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, long> I8Values => null;

            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U1"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, byte> U1Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U2"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, ushort> U2Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, uint> U4Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, ulong> U8Values => null;

            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.R4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, float> R4Values => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.R8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, double> R8Values => null;

            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="TextType"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, string> TextValues => null;
            /// <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="BoolType"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
            public Key<T, bool> BoolValues => null;
        }
    }
}