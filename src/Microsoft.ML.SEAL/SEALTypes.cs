// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.Research.SEAL;

namespace Microsoft.ML.Transforms.SEAL
{
    public sealed class CiphertextTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create a ciphertext
        /// </summary>
        public CiphertextTypeAttribute()
        {
        }

        /// <summary>
        /// Ciphertext
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is CiphertextTypeAttribute otherCiphertext) return true;
            return false;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new CiphertextDataViewType(), typeof(Ciphertext[]), this);
        }
    }

    /// <summary>
    /// The standard Ciphertext type. The representation type of this is Ciphertext.
    /// </summary>
    public sealed class CiphertextDataViewType : StructuredDataViewType
    {
        public CiphertextDataViewType() : base(typeof(Ciphertext[]))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this) return true;
            return false;
        }

        public override string ToString() => "Ciphertext[]";
    }

    public sealed class GaloisKeysTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create Galois keys
        /// </summary>
        public GaloisKeysTypeAttribute()
        {
        }

        /// <summary>
        /// Galois keys
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is GaloisKeysTypeAttribute otherGaloisKeys) return true;
            return false;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new GaloisKeysDataViewType(), typeof(GaloisKeys), this);
        }
    }

    /// <summary>
    /// The standard GaloisKeys type. The representation type of this is GaloisKeys.
    /// </summary>
    public class GaloisKeysDataViewType : StructuredDataViewType
    {
        public GaloisKeysDataViewType() : base(typeof(GaloisKeys))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this) return true;
            return false;
        }

        public override string ToString() => "GaloisKeys";
    }
}
