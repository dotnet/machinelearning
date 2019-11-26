using System.IO;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class ProjectFile : IProjectFile
    {
        public string Data { get; set; }

        public string Name { get; set; }

        public void WriteToDisk(string location)
        {
            var dir = Path.GetDirectoryName(location);
            var fileName = Path.GetFileName(location);
            Utilities.Utils.WriteOutputToFiles(Data, fileName, dir);
        }
    }
}