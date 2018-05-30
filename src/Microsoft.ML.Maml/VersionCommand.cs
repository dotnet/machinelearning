// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.Tools;

[assembly: LoadableClass(VersionCommand.Summary, typeof(VersionCommand), null, typeof(SignatureCommand),
    "Version Command", "Version")]

namespace Microsoft.ML.Runtime.Tools
{
    public sealed class VersionCommand : ICommand
    {
        internal const string Summary = "Prints the TLC version.";

        private readonly IHost _host;

        public VersionCommand(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));

            _host = env.Register("Version");
        }

        public void Run()
        {
            using (var ch = _host.Start("Version"))
            {
                string version = typeof(VersionCommand).GetTypeInfo().Assembly.GetName().Version.ToString();
                ch.Info(version);
                ch.Done();
            }
        }
    }
}
