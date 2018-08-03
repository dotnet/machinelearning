// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.CodeAnalysis;
using System;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    /// <summary>
    /// Location where the diagnostic appears, as determined by path, line number, and column number.
    /// </summary>
    public struct DiagnosticResultLocation
    {
        public string Path { get; }
        public int Line { get; }
        public int Column { get; }

        public DiagnosticResultLocation(string path, int line, int column)
        {
            if (line < -1)
                throw new ArgumentOutOfRangeException(nameof(line), "Must be >= -1");

            if (column < -1)
                throw new ArgumentOutOfRangeException(nameof(column), "Must be >= -1");

            Path = path;
            Line = line;
            Column = column;
        }
    }

    /// <summary>
    /// Struct that stores information about a Diagnostic appearing in a source
    /// </summary>
    public struct DiagnosticResult
    {
        private DiagnosticResultLocation[] _locations;

        public DiagnosticResultLocation[] Locations {
            get => _locations ?? new DiagnosticResultLocation[0];
            set => _locations = value;
        }

        public DiagnosticSeverity Severity { get; set; }
        public string Id { get; set; }
        public string Message { get; set; }

        public DiagnosticResultLocation? Location {
            get => Locations.Length > 0 ? Locations[0] : (DiagnosticResultLocation?)null;
            set => _locations = value == null ? null : new DiagnosticResultLocation[] { value.Value };
        }

        public string Path =>
            Locations.Length > 0 ? Locations[0].Path : "";
        public int Line =>
            Locations.Length > 0 ? Locations[0].Line : -1;
        public int Column =>
            Locations.Length > 0 ? Locations[0].Column : -1;
    }
}