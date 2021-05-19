using Microsoft.ML.TestFrameworkCommon.Utility;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    public sealed class NativeDependencyFact : EnvironmentSpecificFactAttribute
    {
        private readonly string _library;

        public NativeDependencyFact(string library) : base($"This test requires a native library ${library} that wasn't found.")
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
