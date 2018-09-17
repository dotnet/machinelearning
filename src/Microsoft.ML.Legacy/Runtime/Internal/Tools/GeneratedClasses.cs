// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Internal.Tools
{
    internal sealed class GeneratedClasses
    {
        private sealed class ApiClass
        {
            public string OriginalName { get; set; }
            public string NewName { get; set; }
            public bool Generated { get; set; }
        }

        private readonly Dictionary<string, ApiClass> _typesSymbolTable;

        public GeneratedClasses()
        {
            _typesSymbolTable = new Dictionary<string, ApiClass>();
        }

        public string GetApiName(Type type, string rootNamespace)
        {
            string apiName = "";
            if (!_typesSymbolTable.TryGetValue(type.FullName, out ApiClass apiClass))
                apiName = GenerateIntenalName(type, rootNamespace);
            else
                apiName = apiClass.NewName;

            if (!string.IsNullOrEmpty(rootNamespace)&& apiName.StartsWith(rootNamespace))
                return apiName.Substring(rootNamespace.Length + 1);
            else return apiName;
        }

        private string GenerateIntenalName(Type type, string currentNamespace)
        {
            var fullTypeName = type.FullName;
            string name = currentNamespace != "" ? currentNamespace + '.' : "";

            int bracketIndex = fullTypeName.IndexOf('[');
            Type[] genericTypes = null;
            if (type.IsGenericType)
                genericTypes = type.GetGenericArguments();
            if (bracketIndex > 0)
            {
                Contracts.AssertValue(genericTypes);
                fullTypeName = fullTypeName.Substring(0, bracketIndex);
            }

            // When the type is nested, the names of the outer types are concatenated with a '+'.
            var nestedNames = fullTypeName.Split('+');
            var baseName = nestedNames[0];

            // We currently only handle generic types in the outer most class, support for generic inner classes
            // can be added if needed.
            int backTickIndex = baseName.LastIndexOf('`');
            int dotIndex = baseName.LastIndexOf('.');
            Contracts.Assert(dotIndex >= 0);
            if (backTickIndex < 0)
                name += baseName.Substring(dotIndex + 1);
            else
            {
                name += baseName.Substring(dotIndex + 1, backTickIndex - dotIndex - 1);
                Contracts.AssertValue(genericTypes);
                if (genericTypes != null)
                {
                    foreach (var genType in genericTypes)
                    {
                        var splitNames = genType.FullName.Split('+');
                        if (splitNames[0].LastIndexOf('.') >= 0)
                            splitNames[0] = splitNames[0].Substring(splitNames[0].LastIndexOf('.') + 1);
                        name += string.Join("", splitNames);
                    }
                }
            }

            for (int i = 1; i < nestedNames.Length; i++)
                name += nestedNames[i];

            Contracts.Assert(_typesSymbolTable.Values.All(apiclass => string.Compare(apiclass.NewName, name) != 0));
            _typesSymbolTable[type.FullName] = new ApiClass { OriginalName = type.FullName, Generated = false, NewName = name };
            return name;
        }

        internal bool IsGenerated(string fullName)
        {
            if (!_typesSymbolTable.ContainsKey(fullName))
                return false;
            return _typesSymbolTable[fullName].Generated;
        }

        internal void MarkAsGenerated(string fullName)
        {
            _typesSymbolTable[fullName].Generated = true;
        }
    }
}
