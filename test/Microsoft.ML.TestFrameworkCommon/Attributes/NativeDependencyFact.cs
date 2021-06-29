using Microsoft.ML.TestFrameworkCommon.Utility;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    public sealed class NativeDependencyFact : EnvironmentSpecificFactAttribute
    {
        private readonly string _library;

        public NativeDependencyFact(string library) : base($"This test requires a native library {library} that wasn't found.")
        {
            _library = library;
        }

        /// <inheritdoc />
        protected override bool IsEnvironmentSupported()
        {
            return NativeLibrary.NativeLibraryExists(_library);
        }
    }
}
