// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.CodeGenerator.Utilities;

namespace Microsoft.ML.CodeGenerator.CodeGenerator
{
    internal class CSharpSolution : List<ICSharpProject>, ICSharpSolution
    {
        public string Name { get; set; }

        public void WriteToDisk(string folder)
        {
            foreach (var project in this)
            {
                project.WriteToDisk(Path.Combine(folder, project.Name));
            }

            // add project to solution
            Utils.CreateSolutionFile(Name, folder);
            var solutionPath = Path.Combine(folder, $"{Name}.sln");
            Utilities.Utils.AddProjectsToSolution(solutionPath, this.Select((project) => project.Name).ToArray());
        }
    }
}
