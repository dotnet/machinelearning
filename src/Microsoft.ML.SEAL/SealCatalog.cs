using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.SEAL
{
    public static class SealCatalog
    {
        public static SealEstimator EncryptFeatures(this TransformsCatalog catalog,
            ulong polyModDegree,
            string sealPublicKeyFilePath,
            string outputColumnName,
            string inputColumnName = null)
            => new SealEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                polyModDegree, sealPublicKeyFilePath, outputColumnName, inputColumnName);
    }
}
