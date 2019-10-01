using System;
using System.Diagnostics;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.Research.SEAL;

namespace Microsoft.ML.DataView
{
    public class CiphertextDataViewType : StructuredDataViewType
    {
        private static volatile CiphertextDataViewType _instance;

        public static CiphertextDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new CiphertextDataViewType(), null) ??
                    _instance;
            }
        }

        private CiphertextDataViewType()
            : base(typeof(Ciphertext[]))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this) return true;
            Debug.Assert(!(other is CiphertextDataViewType));
            return false;
        }

        public override string ToString() => "Ciphertext[]";
    }

    public class CipherGaloisKeyDataViewType : StructuredDataViewType
    {
        private static volatile CipherGaloisKeyDataViewType _instance;

        public static CipherGaloisKeyDataViewType Instance
        {
            get
            {
                return _instance ??
                    Interlocked.CompareExchange(ref _instance, new CipherGaloisKeyDataViewType(), null) ??
                    _instance;
            }
        }

        private CipherGaloisKeyDataViewType()
            : base(typeof(Tuple<Ciphertext[], GaloisKeys>))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this) return true;
            Debug.Assert(!(other is CipherGaloisKeyDataViewType));
            return false;
        }

        public override string ToString() => "Tuple<Ciphertext[], GaloisKeys>";
    }
}
