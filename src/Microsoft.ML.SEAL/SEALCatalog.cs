using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.SEAL;

namespace Microsoft.ML
{
    public static class SealCatalog
    {
        public static SealEncryptionEstimator EncryptFeatures(this TransformsCatalog catalog,
            ulong polyModDegree,
            string sealPublicKeyFilePath,
            string outputColumnName,
            string inputColumnName = null)
            => new SealEncryptionEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                polyModDegree, sealPublicKeyFilePath, outputColumnName, inputColumnName);
    }
}
