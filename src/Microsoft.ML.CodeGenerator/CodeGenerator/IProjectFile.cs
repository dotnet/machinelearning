// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CodeGenerator.CodeGenerator;

namespace Microsoft.ML.CodeGenerator
{
    public interface IProjectFile: IWritable
    {
        public string Name { get; set; }
    }
}