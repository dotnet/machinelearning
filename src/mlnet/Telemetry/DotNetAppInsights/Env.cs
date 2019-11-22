// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.DotNet.Cli.Utils
{
    public static class Env
    {
        private static IEnvironmentProvider _environment = new EnvironmentProvider();

        public static IEnumerable<string> ExecutableExtensions
        {
            get
            {
                return _environment.ExecutableExtensions;
            }
        }

        public static string GetCommandPath(string commandName, params string[] extensions)
        {
            return _environment.GetCommandPath(commandName, extensions);
        }

        public static string GetCommandPathFromRootPath(string rootPath, string commandName, params string[] extensions)
        {
            return _environment.GetCommandPathFromRootPath(rootPath, commandName, extensions);
        }

        public static string GetCommandPathFromRootPath(string rootPath, string commandName, IEnumerable<string> extensions)
        {
            return _environment.GetCommandPathFromRootPath(rootPath, commandName, extensions);
        }

        public static bool GetEnvironmentVariableAsBool(string name, bool defaultValue = false)
        {
            return _environment.GetEnvironmentVariableAsBool(name, defaultValue);
        }
    }
}
