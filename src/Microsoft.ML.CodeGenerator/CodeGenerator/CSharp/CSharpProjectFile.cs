// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.CodeGenerator.Utilities;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    /// <summary>
    /// Class for CSharp Project file
    /// </summary>
    internal class CSharpProjectFile : ICSharpFile
    {
        private string _file;
        public string File
        {
            get => _file;
            set
            {
                _file = value;
            }
        }

        public string Name { get; set; }

        /// <summary>
        /// Write File To Disk
        /// </summary>
        /// <param name="location">full path of destinate directory</param>
        public void WriteToDisk(string location)
        {
            var extension = Path.GetExtension(Name);
            if (extension != ".csproj")
            {
                throw new Exception("CSharp project file extesion must be .csproj");
            }

            Utilities.Utils.WriteOutputToFiles(File, Name, location);
        }
    }
}
