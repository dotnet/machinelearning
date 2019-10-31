// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.Research.SEAL;

namespace Microsoft.ML.SEAL
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

    public sealed class CipherGaloisKeysTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create Galois keys
        /// </summary>
        public CipherGaloisKeysTypeAttribute()
        {
        }

        /// <summary>
        /// Galois keys
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is CipherGaloisKeysTypeAttribute otherCipherGaloisKeys) return true;
            return false;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new CipherGaloisKeysDataViewType(), typeof(Tuple<Ciphertext[], GaloisKeys>), this);
        }
    }

    /// <summary>
    /// The standard Ciphertext and Galois Keys type. The representation type of this is Tuple&lt;Ciphertext[], GaloisKeys&gt;.
    /// </summary>
    public class CipherGaloisKeysDataViewType : StructuredDataViewType
    {
        public CipherGaloisKeysDataViewType() : base(typeof(Tuple<Ciphertext[], GaloisKeys>))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this) return true;
            return false;
        }

        public override string ToString() => "Tuple<Ciphertext[], GaloisKeys>";
    }
}
