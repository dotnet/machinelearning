// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.CSharp;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Tools;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Tools;
using Newtonsoft.Json.Linq;
using static Microsoft.ML.Runtime.EntryPoints.CommonInputs;

[assembly: LoadableClass(typeof(CSharpApiGenerator), typeof(CSharpApiGenerator.Arguments), typeof(SignatureModuleGenerator),
    "CSharp API generator", "CSGenerator", "CS")]

namespace Microsoft.ML.Runtime.Internal.Tools
{
    public sealed class CSharpApiGenerator : IGenerator
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The path of the generated C# file")]
            public string CsFilename;

            [Argument(ArgumentType.Multiple, HelpText = "Entry points to exclude", ShortName = "excl")]
            public string[] Exclude;
        }

        private static class GeneratorUtils
        {
            public static string GetFullMethodName(ModuleCatalog.EntryPointInfo entryPointInfo)
            {
                return entryPointInfo.Name;
            }

            public static Tuple<string, string> GetClassAndMethodNames(ModuleCatalog.EntryPointInfo entryPointInfo)
            {
                var split = entryPointInfo.Name.Split('.');
                Contracts.Assert(split.Length == 2);
                return new Tuple<string, string>(split[0], split[1]);
            }

            public static string GetCSharpTypeName(Type type)
            {
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                    return GetCSharpTypeName(type.GetGenericArguments()[0]) + "?";

                string name;
                using (var p = new CSharpCodeProvider())
                    name = p.GetTypeOutput(new CodeTypeReference(type));
                return name;
            }

            public static string GetOutputType(Type outputType)
            {
                Contracts.Check(Var<int>.CheckType(outputType));

                if (outputType.IsArray)
                    return $"ArrayVar<{GetCSharpTypeName(outputType.GetElementType())}>";
                if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                    && outputType.GetGenericTypeArgumentsEx()[0] == typeof(string))
                {
                    return $"DictionaryVar<{GetCSharpTypeName(outputType.GetGenericTypeArgumentsEx()[1])}>";
                }

                return $"Var<{GetCSharpTypeName(outputType)}>";
            }

            public static string GetInputType(ModuleCatalog catalog, Type inputType,
                Dictionary<string, string> typesSymbolTable, string rootNameSpace = "")
            {
                if (inputType.IsGenericType && inputType.GetGenericTypeDefinition() == typeof(Var<>))
                    return $"Var<{GetCSharpTypeName(inputType.GetGenericTypeArgumentsEx()[0])}>";

                if (inputType.IsArray && Var<int>.CheckType(inputType.GetElementType()))
                    return $"ArrayVar<{GetCSharpTypeName(inputType.GetElementType())}>";

                if (inputType.IsGenericType && inputType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                    && inputType.GetGenericTypeArgumentsEx()[0] == typeof(string))
                {
                    return $"DictionaryVar<{GetCSharpTypeName(inputType.GetGenericTypeArgumentsEx()[1])}>";
                }

                if (Var<int>.CheckType(inputType))
                    return $"Var<{GetCSharpTypeName(inputType)}>";

                bool isNullable = false;
                bool isOptional = false;
                var type = inputType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                {
                    type = type.GetGenericArguments()[0];
                    isNullable = true;
                }
                else if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                {
                    type = type.GetGenericArguments()[0];
                    isOptional = true;
                }

                var typeEnum = TlcModule.GetDataType(type);
                switch (typeEnum)
                {
                    case TlcModule.DataKind.Float:
                    case TlcModule.DataKind.Int:
                    case TlcModule.DataKind.UInt:
                    case TlcModule.DataKind.Char:
                    case TlcModule.DataKind.String:
                    case TlcModule.DataKind.Bool:
                    case TlcModule.DataKind.DataView:
                    case TlcModule.DataKind.TransformModel:
                    case TlcModule.DataKind.PredictorModel:
                    case TlcModule.DataKind.FileHandle:
                        return GetCSharpTypeName(inputType);
                    case TlcModule.DataKind.Array:
                        return GetInputType(catalog, inputType.GetElementType(), typesSymbolTable) + "[]";
                    case TlcModule.DataKind.Component:
                        string kind;
                        bool success = catalog.TryGetComponentKind(type, out kind);
                        Contracts.Assert(success);
                        return $"{kind}";
                    case TlcModule.DataKind.Enum:
                        var enumName = GetEnumName(type, typesSymbolTable, rootNameSpace);
                        if (isNullable)
                            return $"{enumName}?";
                        if (isOptional)
                            return $"Optional<{enumName}>";
                        return $"{enumName}";
                    default:
                        if (isNullable)
                            return rootNameSpace + typesSymbolTable[type.FullName];
                        if (isOptional)
                            return $"Optional<{rootNameSpace + typesSymbolTable[type.FullName]}>";
                        if (typesSymbolTable.ContainsKey(type.FullName))
                            return rootNameSpace + typesSymbolTable[type.FullName];
                        else
                            return GetSymbolFromType(typesSymbolTable, type, rootNameSpace);
                }
            }

            public static bool IsComponent(Type inputType)
            {
                if (inputType.IsArray && Var<int>.CheckType(inputType.GetElementType()))
                    return false;

                if (inputType.IsGenericType && inputType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                    && inputType.GetGenericTypeArgumentsEx()[0] == typeof(string))
                {
                    return false;
                }

                if (Var<int>.CheckType(inputType))
                    return false;

                var type = inputType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                    type = type.GetGenericArguments()[0];
                else if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    type = type.GetGenericArguments()[0];

                var typeEnum = TlcModule.GetDataType(type);
                return typeEnum == TlcModule.DataKind.Component;
            }

            public static string Capitalize(string s)
            {
                if (string.IsNullOrEmpty(s))
                    return s;
                return char.ToUpperInvariant(s[0]) + s.Substring(1);
            }

            private static string GetCharValue(char value)
            {
                switch (value)
                {
                    case '\t':
                        return "\\t";
                    case '\n':
                        return "\\n";
                    case '\r':
                        return "\\r";
                    case '\\':
                        return "\\";
                    case '\"':
                        return "\"";
                    case '\'':
                        return "\\'";
                    case '\0':
                        return "\\0";
                    case '\a':
                        return "\\a";
                    case '\b':
                        return "\\b";
                    case '\f':
                        return "\\f";
                    case '\v':
                        return "\\v";
                    default:
                        return value.ToString();
                }
            }

            public static string GetValue(ModuleCatalog catalog, Type fieldType, object fieldValue,
                Dictionary<string, string> typesSymbolTable, string rootNameSpace = "")
            {
                if (fieldType.IsGenericType && fieldType.GetGenericTypeDefinition() == typeof(Var<>))
                    return $"new Var<{GetCSharpTypeName(fieldType.GetGenericTypeArgumentsEx()[0])}>()";

                if (fieldType.IsArray && Var<int>.CheckType(fieldType.GetElementType()))
                    return $"new ArrayVar<{GetCSharpTypeName(fieldType.GetElementType())}>()";

                if (fieldType.IsGenericType && fieldType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                    && fieldType.GetGenericTypeArgumentsEx()[0] == typeof(string))
                {
                    return $"new DictionaryVar<{GetCSharpTypeName(fieldType.GetGenericTypeArgumentsEx()[1])}>()";
                }

                if (Var<int>.CheckType(fieldType))
                    return $"new Var<{GetCSharpTypeName(fieldType)}>()";

                if (fieldValue == null)
                    return null;

                if (!fieldType.IsInterface)
                {
                    try
                    {
                        var defaultFieldValue = Activator.CreateInstance(fieldType);
                        if (defaultFieldValue == fieldValue)
                            return null;
                    }
                    catch (MissingMethodException)
                    {
                        // No parameterless constructor, ignore.
                    }
                }

                var typeEnum = TlcModule.GetDataType(fieldType);
                if (fieldType.IsGenericType && (fieldType.GetGenericTypeDefinition() == typeof(Optional<>) || fieldType.GetGenericTypeDefinition() == typeof(Nullable<>)))
                    fieldType = fieldType.GetGenericArguments()[0];
                switch (typeEnum)
                {
                    case TlcModule.DataKind.Array:
                        var arr = fieldValue as Array;
                        if (arr != null && arr.GetLength(0) > 0)
                            return $"{{ {string.Join(", ", arr.Cast<object>().Select(item => GetValue(catalog, fieldType.GetElementType(), item, typesSymbolTable)))} }}";
                        return null;
                    case TlcModule.DataKind.String:
                        var strval = fieldValue as string;
                        if (strval != null)
                            return Quote(strval);
                        return null;
                    case TlcModule.DataKind.Float:
                        if (fieldValue is double d)
                        {
                            if (double.IsPositiveInfinity(d))
                                return "double.PositiveInfinity";
                            if (double.IsNegativeInfinity(d))
                                return "double.NegativeInfinity";
                            if (d != 0)
                                return d.ToString("R") + "d";
                        }
                        else if (fieldValue is float f)
                        {
                            if (float.IsPositiveInfinity(f))
                                return "float.PositiveInfinity";
                            if (float.IsNegativeInfinity(f))
                                return "float.NegativeInfinity";
                            if (f != 0)
                                return f.ToString("R") + "f";
                        }
                        return null;
                    case TlcModule.DataKind.Int:
                        if (fieldValue is int i)
                        {
                            if (i != 0)
                                return i.ToString();
                        }
                        else if (fieldValue is long l)
                        {
                            if (l != 0)
                                return l.ToString();
                        }
                        return null;
                    case TlcModule.DataKind.Bool:
                        return (bool)fieldValue ? "true" : "false";
                    case TlcModule.DataKind.Enum:
                        return GetEnumName(fieldType, typesSymbolTable, rootNameSpace) + "." + fieldValue;
                    case TlcModule.DataKind.Char:
                        return $"'{GetCharValue((char)fieldValue)}'";
                    case TlcModule.DataKind.Component:
                        var type = fieldValue.GetType();
                        ModuleCatalog.ComponentInfo componentInfo;
                        if (!catalog.TryFindComponent(fieldType, type, out componentInfo))
                            return null;
                        object defaultComponent = null;
                        try
                        {
                            defaultComponent = Activator.CreateInstance(componentInfo.ArgumentType);
                        }
                        catch (MissingMethodException)
                        {
                            // No parameterless constructor, ignore.
                        }
                        var propertyBag = new List<string>();
                        if (defaultComponent != null)
                        {
                            foreach (var fieldInfo in componentInfo.ArgumentType.GetFields())
                            {
                                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                                    continue;
                                if (fieldInfo.FieldType == typeof(JArray) || fieldInfo.FieldType == typeof(JObject))
                                    continue;

                                var propertyValue = GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(fieldValue), typesSymbolTable);
                                var defaultPropertyValue = GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(defaultComponent), typesSymbolTable);
                                if (propertyValue != defaultPropertyValue)
                                    propertyBag.Add($"{GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} = {propertyValue}");
                            }
                        }
                        var properties = propertyBag.Count > 0 ? $" {{ {string.Join(", ", propertyBag)} }}" : "";
                        return $"new {GetComponentName(componentInfo)}(){properties}";
                    case TlcModule.DataKind.Unknown:
                        return $"new {rootNameSpace + typesSymbolTable[fieldType.FullName]}()";
                    default:
                        return fieldValue.ToString();
                }
            }

            private static string Quote(string src)
            {
                var dst = src.Replace("\\", @"\\").Replace("\"", "\\\"").Replace("\n", @"\n").Replace("\r", @"\r");
                return "\"" + dst + "\"";
            }

            public static string GetComponentName(ModuleCatalog.ComponentInfo component)
            {
                return $"{Capitalize(component.Name)}{component.Kind}";
            }

            public static string GetEnumName(Type type, Dictionary<string, string> typesSymbolTable, string rootNamespace = "")
            {
                if (typesSymbolTable.ContainsKey(type.FullName))
                    return rootNamespace + typesSymbolTable[type.FullName];
                else
                    return GetSymbolFromType(typesSymbolTable, type, rootNamespace);
            }

            public static string GetJsonFromField(string fieldName, Type fieldType)
            {
                if (fieldType.IsArray && Var<int>.CheckType(fieldType.GetElementType()))
                    return $"{{({fieldName}.IsValue ? {fieldName}.VarName : $\"'${{{fieldName}.VarName}}'\")}}";
                if (fieldType.IsGenericType && fieldType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                    && fieldType.GetGenericTypeArgumentsEx()[0] == typeof(string))
                {
                    return $"'${{{fieldName}.VarName}}'";
                }
                if (Var<int>.CheckType(fieldType))
                    return $"'${{{fieldName}.VarName}}'";

                var isNullable = false;
                var type = fieldType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                {
                    type = type.GetGenericArguments()[0];
                    isNullable = true;
                }
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    type = type.GetGenericArguments()[0];

                var typeEnum = TlcModule.GetDataType(type);
                switch (typeEnum)
                {
                    default:
                        if (isNullable)
                            return $"{{(!{fieldName}.HasValue ? \"null\" : $\"{{{fieldName}.Value}}\")}}";
                        return $"{{{fieldName}}}";
                    case TlcModule.DataKind.Enum:
                        if (isNullable)
                            return $"{{(!{fieldName}.HasValue ? \"null\" : $\"'{{{fieldName}.Value}}'\")}}";
                        return $"'{{{fieldName}}}'";
                    case TlcModule.DataKind.String:
                        return $"{{({fieldName} == null ? \"null\" : $\"'{{{fieldName}}}'\")}}";
                    case TlcModule.DataKind.Bool:
                        if (isNullable)
                            return $"{{(!{fieldName}.HasValue ? \"null\" : {fieldName}.Value ? \"true\" : \"false\")}}";
                        return $"'{{({fieldName} ? \"true\" : \"false\")}}'";
                    case TlcModule.DataKind.Component:
                    case TlcModule.DataKind.Unknown:
                        return $"{{({fieldName} == null ? \"null\" : {fieldName}.ToJson())}}";
                    case TlcModule.DataKind.Array:
                        return $"[{{({fieldName} == null ? \"\" : string.Join(\",\", {fieldName}.Select(f => $\"{GetJsonFromField("f", type.GetElementType())}\")))}}]";
                }
            }
        }

        private readonly IHost _host;
        private readonly string _csFilename;
        private readonly string _regenerate;
        private readonly HashSet<string> _excludedSet;
        private const string RegistrationName = "CSharpApiGenerator";
        public Dictionary<string, string> _typesSymbolTable = new Dictionary<string, string>();

        public CSharpApiGenerator(IHostEnvironment env, Arguments args, string regenerate)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.AssertValue(args, nameof(args));
            _host.AssertNonEmpty(regenerate, nameof(regenerate));
            Utils.CheckOptionalUserDirectory(args.CsFilename, nameof(args.CsFilename));

            _csFilename = args.CsFilename;
            if (string.IsNullOrWhiteSpace(_csFilename))
                _csFilename = "CSharpApi.cs";
            _regenerate = regenerate;
            _excludedSet = new HashSet<string>(args.Exclude);
        }

        public void Generate(IEnumerable<HelpCommand.Component> infos)
        {
            var catalog = ModuleCatalog.CreateInstance(_host);

            using (var sw = new StreamWriter(_csFilename))
            {
                var writer = IndentingTextWriter.Wrap(sw, "    ");

                // Generate header
                GenerateHeader(writer);

                foreach (var entryPointInfo in catalog.AllEntryPoints().Where(x => !_excludedSet.Contains(x.Name)).OrderBy(x => x.Name))
                {
                    // Generate method
                    GenerateMethod(writer, entryPointInfo, catalog);
                }

                // Generate footer
                GenerateFooter(writer);
                GenerateFooter(writer);

                foreach (var entryPointInfo in catalog.AllEntryPoints().Where(x => !_excludedSet.Contains(x.Name)).OrderBy(x => x.Name))
                {
                    // Generate input and output classes
                    GenerateInputOutput(writer, entryPointInfo, catalog);
                }

                writer.WriteLine("namespace Runtime");
                writer.WriteLine("{");
                writer.Indent();

                foreach (var kind in catalog.GetAllComponentKinds().OrderBy(x => x))
                {
                    // Generate kind base class
                    GenerateComponentKind(writer, kind);

                    foreach (var component in catalog.GetAllComponents(kind).OrderBy(x => x.Name))
                    {
                        // Generate component
                        GenerateComponent(writer, component, catalog);
                    }
                }

                GenerateFooter(writer);
                GenerateFooter(writer);
                writer.WriteLine("#pragma warning restore");
            }
        }

        private void GenerateHeader(IndentingTextWriter writer)
        {
            writer.WriteLine("//------------------------------------------------------------------------------");
            writer.WriteLine("// <auto-generated>");
            writer.WriteLine("//     This code was generated by a tool.");
            writer.WriteLine("//");
            writer.WriteLine("//     Changes to this file may cause incorrect behavior and will be lost if");
            writer.WriteLine("//     the code is regenerated.");
            writer.WriteLine("// </auto-generated>");
            writer.WriteLine("//------------------------------------------------------------------------------");
            //writer.WriteLine($"// This file is auto generated. To regenerate it, run: {_regenerate}");
            writer.WriteLine("#pragma warning disable");
            writer.WriteLine("using System.Collections.Generic;");
            writer.WriteLine("using Microsoft.ML.Runtime;");
            writer.WriteLine("using Microsoft.ML.Runtime.Data;");
            writer.WriteLine("using Microsoft.ML.Runtime.EntryPoints;");
            writer.WriteLine("using Newtonsoft.Json;");
            writer.WriteLine("using System;");
            writer.WriteLine("using System.Linq;");
            writer.WriteLine("using Microsoft.ML.Runtime.CommandLine;");
            writer.WriteLine();
            writer.WriteLine("namespace Microsoft.ML");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("namespace Runtime");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("public sealed partial class Experiment");
            writer.WriteLine("{");
            writer.Indent();
        }

        private void GenerateFooter(IndentingTextWriter writer)
        {
            writer.Outdent();
            writer.WriteLine("}");
        }

        private void GenerateInputOutput(IndentingTextWriter writer,
            ModuleCatalog.EntryPointInfo entryPointInfo,
            ModuleCatalog catalog)
        {
            var classAndMethod = GeneratorUtils.GetClassAndMethodNames(entryPointInfo);
            writer.WriteLine($"namespace {classAndMethod.Item1}");
            writer.WriteLine("{");
            writer.Indent();
            GenerateInput(writer, entryPointInfo, catalog);
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
        }

        /// <summary>
        /// This methods creates a unique name for a class/struct/enum, given a type and a namespace.
        /// It generates the name based on the <see cref="Type.FullName"/> property of the type
        /// (see description here https://msdn.microsoft.com/en-us/library/system.type.fullname(v=vs.110).aspx).
        /// Example: Assume we have the following structure in namespace X.Y:
        /// class A {
        ///   class B {
        ///     enum C {
        ///       Value1,
        ///       Value2
        ///     }
        ///   }
        /// }
        /// The full name of C would be X.Y.A+B+C. This method will generate the name "ABC" from it. In case
        /// A is generic with one generic type, then the full name of typeof(A&lt;float&gt;.B.C) would be X.Y.A`1+B+C[[System.Single]].
        /// In this case, this method will generate the name "ASingleBC".
        /// </summary>
        /// <param name="typesSymbolTable">A dictionary containing the names of the classes already generated.
        /// This parameter is only used to ensure that the newly generated name is unique.</param>
        /// <param name="type">The type for which to generate the new name.</param>
        /// <param name="currentNamespace">The namespace prefix to the new name.</param>
        /// <returns>A unique name derived from the given type and namespace.</returns>
        private static string GetSymbolFromType(Dictionary<string, string> typesSymbolTable, Type type, string currentNamespace)
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

            Contracts.Assert(typesSymbolTable.Select(kvp => kvp.Value).All(str => string.Compare(str, name) != 0));

            return name;
        }

        private void GenerateEnums(IndentingTextWriter writer, Type inputType, string currentNamespace)
        {
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;

                var type = fieldInfo.FieldType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                    type = type.GetGenericArguments()[0];
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    type = type.GetGenericArguments()[0];

                if (_typesSymbolTable.ContainsKey(type.FullName))
                    continue;

                if (!type.IsEnum)
                {
                    var typeEnum = TlcModule.GetDataType(type);
                    if (typeEnum == TlcModule.DataKind.Unknown)
                        GenerateEnums(writer, type, currentNamespace);
                    continue;
                }

                var enumType = Enum.GetUnderlyingType(type);

                _typesSymbolTable[type.FullName] = GetSymbolFromType(_typesSymbolTable, type, currentNamespace);
                if (enumType == typeof(int))
                    writer.WriteLine($"public enum {_typesSymbolTable[type.FullName].Substring(_typesSymbolTable[type.FullName].LastIndexOf('.') + 1)}");
                else
                {
                    Contracts.Assert(enumType == typeof(byte));
                    writer.WriteLine($"public enum {_typesSymbolTable[type.FullName].Substring(_typesSymbolTable[type.FullName].LastIndexOf('.') + 1)} : byte");
                }

                writer.Write("{");
                writer.Indent();
                var names = Enum.GetNames(type);
                var values = Enum.GetValues(type);
                string prefix = "";
                for (int i = 0; i < names.Length; i++)
                {
                    var name = names[i];
                    var value = values.GetValue(i);
                    writer.WriteLine(prefix);
                    if (enumType == typeof(int))
                        writer.Write($"{name} = {(int)value}");
                    else
                    {
                        Contracts.Assert(enumType == typeof(byte));
                        writer.Write($"{name} = {(byte)value}");
                    }
                    prefix = ",";
                }
                writer.WriteLine();
                writer.Outdent();
                writer.WriteLine("}");
                writer.WriteLine();
            }
        }

        string GetFriendlyTypeName(string currentNameSpace, string typeName)
        {
            Contracts.Assert(typeName.Length >= currentNameSpace.Length);

            int index = 0;
            for (index = 0; index < currentNameSpace.Length && currentNameSpace[index] == typeName[index]; index++) ;

            if (index == 0)
                return typeName;
            if (typeName[index - 1] == '.')
                return typeName.Substring(index);

            return typeName;
        }

        private void GenerateStructs(IndentingTextWriter writer,
            Type inputType,
            ModuleCatalog catalog,
            string currentNamespace)
        {
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;

                var type = fieldInfo.FieldType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                    type = type.GetGenericArguments()[0];
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    type = type.GetGenericArguments()[0];
                if (type.IsArray)
                    type = type.GetElementType();
                if (type == typeof(JArray) || type == typeof(JObject))
                    continue;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Var<>))
                    continue;
                if (type == typeof(CommonInputs.IEvaluatorInput))
                    continue;
                if (type == typeof(CommonOutputs.IEvaluatorOutput))
                    continue;
                var typeEnum = TlcModule.GetDataType(type);
                if (typeEnum == TlcModule.DataKind.State)
                    continue;
                if (typeEnum != TlcModule.DataKind.Unknown)
                    continue;

                if (_typesSymbolTable.ContainsKey(type.FullName))
                    continue;

                _typesSymbolTable[type.FullName] = GetSymbolFromType(_typesSymbolTable, type, currentNamespace);
                string classBase = "";
                if (type.IsSubclassOf(typeof(OneToOneColumn)))
                    classBase = $" : OneToOneColumn<{_typesSymbolTable[type.FullName].Substring(_typesSymbolTable[type.FullName].LastIndexOf('.') + 1)}>, IOneToOneColumn";
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                    classBase = $" : ManyToOneColumn<{_typesSymbolTable[type.FullName].Substring(_typesSymbolTable[type.FullName].LastIndexOf('.') + 1)}>, IManyToOneColumn";
                writer.WriteLine($"public sealed partial class {_typesSymbolTable[type.FullName].Substring(_typesSymbolTable[type.FullName].LastIndexOf('.') + 1)}{classBase}");
                writer.WriteLine("{");
                writer.Indent();
                GenerateInputFields(writer, type, catalog, _typesSymbolTable);
                writer.Outdent();
                writer.WriteLine("}");
                writer.WriteLine();
                GenerateStructs(writer, type, catalog, currentNamespace);
            }
        }

        private void GenerateLoaderAddInputMethod(IndentingTextWriter writer, string className)
        {
            //Constructor.
            writer.WriteLine("[JsonIgnore]");
            writer.WriteLine("private string _inputFilePath = null;");
            writer.WriteLine($"public {className}(string filePath)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("_inputFilePath = filePath;");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine("");

            //SetInput.
            writer.WriteLine($"public void SetInput(IHostEnvironment env, Experiment experiment)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("IFileHandle inputFile = new SimpleFileHandle(env, _inputFilePath, false, false);");
            writer.WriteLine("experiment.SetInput(InputFile, inputFile);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine("");

            //Apply.
            writer.WriteLine($"public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("Contracts.Assert(previousStep == null);");
            writer.WriteLine("");
            writer.WriteLine($"return new {className}PipelineStep(experiment.Add(this));");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine("");

            //Pipelinestep class.
            writer.WriteLine($"private class {className}PipelineStep : ILearningPipelineDataStep");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"public {className}PipelineStep (Output output)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("Data = output.Data;");
            writer.WriteLine("Model = null;");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
            writer.WriteLine("public Var<IDataView> Data { get; }");
            writer.WriteLine("public Var<ITransformModel> Model { get; }");
            writer.Outdent();
            writer.WriteLine("}");
        }

        private void GenerateColumnAddMethods(IndentingTextWriter writer,
            Type inputType,
            ModuleCatalog catalog,
            string className,
            out Type columnType)
        {
            columnType = null;
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;

                var type = fieldInfo.FieldType;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                    type = type.GetGenericArguments()[0];
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    type = type.GetGenericArguments()[0];
                var isArray = type.IsArray;
                if (isArray)
                    type = type.GetElementType();
                if (type == typeof(JArray) || type == typeof(JObject))
                    continue;
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Var<>))
                    continue;
                var typeEnum = TlcModule.GetDataType(type);
                if (typeEnum != TlcModule.DataKind.Unknown)
                    continue;

                if (type.IsSubclassOf(typeof(OneToOneColumn)))
                {
                    var fieldName = GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
                    writer.WriteLine($"public {className}()");
                    writer.WriteLine("{");
                    writer.WriteLine("}");
                    writer.WriteLine("");
                    writer.WriteLine($"public {className}(params string[] input{fieldName}s)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"if (input{fieldName}s != null)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"foreach (string input in input{fieldName}s)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"Add{fieldName}(input);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine("");
                    writer.WriteLine($"public {className}(params ValueTuple<string, string>[] inputOutput{fieldName}s)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"if (inputOutput{fieldName}s != null)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"foreach (ValueTuple<string, string> inputOutput in inputOutput{fieldName}s)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"Add{fieldName}(inputOutput.Item2, inputOutput.Item1);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine("");
                    writer.WriteLine($"public void Add{fieldName}(string source)");
                    writer.WriteLine("{");
                    writer.Indent();
                    if (isArray)
                    {
                        writer.WriteLine($"var list = {fieldName} == null ? new List<{_typesSymbolTable[type.FullName]}>() : new List<{_typesSymbolTable[type.FullName]}>({fieldName});");
                        writer.WriteLine($"list.Add(OneToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(source));");
                        writer.WriteLine($"{fieldName} = list.ToArray();");
                    }
                    else
                        writer.WriteLine($"{fieldName} = OneToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(source);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine();
                    writer.WriteLine($"public void Add{fieldName}(string name, string source)");
                    writer.WriteLine("{");
                    writer.Indent();
                    if (isArray)
                    {
                        writer.WriteLine($"var list = {fieldName} == null ? new List<{_typesSymbolTable[type.FullName]}>() : new List<{_typesSymbolTable[type.FullName]}>({fieldName});");
                        writer.WriteLine($"list.Add(OneToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(name, source));");
                        writer.WriteLine($"{fieldName} = list.ToArray();");
                    }
                    else
                        writer.WriteLine($"{fieldName} = OneToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(name, source);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine();

                    Contracts.Assert(columnType == null);

                    columnType = type;
                }
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                {
                    var fieldName = GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
                    writer.WriteLine($"public {className}()");
                    writer.WriteLine("{");
                    writer.WriteLine("}");
                    writer.WriteLine("");
                    writer.WriteLine($"public {className}(string output{fieldName}, params string[] input{fieldName}s)");
                    writer.WriteLine("{");
                    writer.Indent();
                    writer.WriteLine($"Add{fieldName}(output{fieldName}, input{fieldName}s);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine("");
                    writer.WriteLine($"public void Add{fieldName}(string name, params string[] source)");
                    writer.WriteLine("{");
                    writer.Indent();
                    if (isArray)
                    {
                        writer.WriteLine($"var list = {fieldName} == null ? new List<{_typesSymbolTable[type.FullName]}>() : new List<{_typesSymbolTable[type.FullName]}>({fieldName});");
                        writer.WriteLine($"list.Add(ManyToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(name, source));");
                        writer.WriteLine($"{fieldName} = list.ToArray();");
                    }
                    else
                        writer.WriteLine($"{fieldName} = ManyToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(name, source);");
                    writer.Outdent();
                    writer.WriteLine("}");
                    writer.WriteLine();

                    Contracts.Assert(columnType == null);

                    columnType = type;
                }
            }
        }

        private void GenerateInput(IndentingTextWriter writer,
            ModuleCatalog.EntryPointInfo entryPointInfo,
            ModuleCatalog catalog)
        {
            var classAndMethod = GeneratorUtils.GetClassAndMethodNames(entryPointInfo);
            string classBase = "";
            if (entryPointInfo.InputKinds != null)
            {
                classBase += $" : {string.Join(", ", entryPointInfo.InputKinds.Select(GeneratorUtils.GetCSharpTypeName))}";
                if (entryPointInfo.InputKinds.Any(t => typeof(ITrainerInput).IsAssignableFrom(t) || typeof(ITransformInput).IsAssignableFrom(t)))
                    classBase += ", Microsoft.ML.ILearningPipelineItem";
            }

            GenerateEnums(writer, entryPointInfo.InputType, classAndMethod.Item1);
            writer.WriteLine();
            GenerateStructs(writer, entryPointInfo.InputType, catalog, classAndMethod.Item1);
            writer.WriteLine("/// <summary>");
            foreach (var line in entryPointInfo.Description.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries))
                writer.WriteLine($"/// {line}");
            writer.WriteLine("/// </summary>");
            
            if(entryPointInfo.ObsoleteAttribute != null)
                writer.WriteLine($"[Obsolete(\"{entryPointInfo.ObsoleteAttribute.Message}\")]");
            
            writer.WriteLine($"public sealed partial class {classAndMethod.Item2}{classBase}");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine();
            if (entryPointInfo.InputKinds != null && entryPointInfo.InputKinds.Any(t => typeof(ILearningPipelineLoader).IsAssignableFrom(t)))
                GenerateLoaderAddInputMethod(writer, classAndMethod.Item2);

            GenerateColumnAddMethods(writer, entryPointInfo.InputType, catalog, classAndMethod.Item2, out Type transformType);
            writer.WriteLine();
            GenerateInputFields(writer, entryPointInfo.InputType, catalog, _typesSymbolTable);
            writer.WriteLine();

            GenerateOutput(writer, entryPointInfo, out HashSet<string> outputVariableNames);
            GenerateApplyFunction(writer, entryPointInfo, transformType, classBase, outputVariableNames);
            writer.Outdent();
            writer.WriteLine("}");
        }

        private static void GenerateApplyFunction(IndentingTextWriter writer, ModuleCatalog.EntryPointInfo entryPointInfo,
            Type type, string classBase, HashSet<string> outputVariableNames)
        {
            bool isTransform = false;
            bool isCalibrator = false;
            if (classBase.Contains("ITransformInput"))
                isTransform = true;
            else if (!classBase.Contains("ITrainerInput"))
                return;

            if (classBase.Contains("ICalibratorInput"))
                isCalibrator = true;

            string className = GeneratorUtils.GetClassAndMethodNames(entryPointInfo).Item2;
            writer.WriteLine("public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("if (!(previousStep is ILearningPipelineDataStep dataStep))");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine("throw new InvalidOperationException($\"{ nameof(" + className + ")} only supports an { nameof(ILearningPipelineDataStep)} as an input.\");");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();

            if (isTransform)
            {
                writer.WriteLine("Data = dataStep.Data;");
            }
            else
                writer.WriteLine("TrainingData = dataStep.Data;");

            string pipelineStep = $"{className}PipelineStep";
            writer.WriteLine($"Output output = experiment.Add(this);");
            writer.WriteLine($"return new {pipelineStep}(output);");
            writer.Outdent();
            writer.WriteLine("}");

            //Pipeline step.
            writer.WriteLine();
            if (isTransform && !isCalibrator)
                writer.WriteLine($"private class {pipelineStep} : ILearningPipelineDataStep");
            else
                writer.WriteLine($"private class {pipelineStep} : ILearningPipelinePredictorStep");

            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"public {pipelineStep}(Output output)");
            writer.WriteLine("{");
            writer.Indent();

            if (isTransform && !isCalibrator)
            {
                writer.WriteLine("Data = output.OutputData;");
                if (outputVariableNames.Contains("Model"))
                    writer.WriteLine("Model = output.Model;");
            }
            else
                writer.WriteLine("Model = output.PredictorModel;");

            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();

            if (isTransform && !isCalibrator)
            {
                writer.WriteLine("public Var<IDataView> Data { get; }");
                writer.WriteLine("public Var<ITransformModel> Model { get; }");
            }
            else
                writer.WriteLine("public Var<IPredictorModel> Model { get; }");

            writer.Outdent();
            writer.WriteLine("}");
        }

        private static void GenerateInputFields(IndentingTextWriter writer,
            Type inputType, ModuleCatalog catalog, Dictionary<string, string> typesSymbolTable, string rootNameSpace = "")
        {
            var defaults = Activator.CreateInstance(inputType);
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr =
                    fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;
                if (fieldInfo.FieldType == typeof(JObject))
                    continue;

                writer.WriteLine("/// <summary>");
                writer.WriteLine($"/// {inputAttr.HelpText}");
                writer.WriteLine("/// </summary>");
                if (fieldInfo.FieldType == typeof(JArray))
                {
                    writer.WriteLine($"public Experiment {GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} {{ get; set; }}");
                    writer.WriteLine();
                    continue;
                }

                var inputTypeString = GeneratorUtils.GetInputType(catalog, fieldInfo.FieldType, typesSymbolTable, rootNameSpace);
                if (GeneratorUtils.IsComponent(fieldInfo.FieldType))
                    writer.WriteLine("[JsonConverter(typeof(ComponentSerializer))]");
                if (GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name) != (inputAttr.Name ?? fieldInfo.Name))
                    writer.WriteLine($"[JsonProperty(\"{inputAttr.Name ?? fieldInfo.Name}\")]");

                // For range attributes on properties
                if (fieldInfo.GetCustomAttributes(typeof(TlcModule.RangeAttribute), false).FirstOrDefault()
                    is TlcModule.RangeAttribute ranAttr)
                    writer.WriteLine(ranAttr.ToString());

                // For obsolete/deprecated attributes
                if (fieldInfo.GetCustomAttributes(typeof(ObsoleteAttribute), false).FirstOrDefault()
                    is ObsoleteAttribute obsAttr)
                    writer.WriteLine($"[System.Obsolete(\"{obsAttr.Message}\")]");

                // For sweepable ranges on properties
                if (fieldInfo.GetCustomAttributes(typeof(TlcModule.SweepableParamAttribute), false).FirstOrDefault()
                    is TlcModule.SweepableParamAttribute sweepableParamAttr)
                {
                    if (string.IsNullOrEmpty(sweepableParamAttr.Name))
                        sweepableParamAttr.Name = fieldInfo.Name;
                    writer.WriteLine(sweepableParamAttr.ToString());
                }

                writer.Write($"public {inputTypeString} {GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} {{ get; set; }}");
                var defaultValue = GeneratorUtils.GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(defaults), typesSymbolTable, rootNameSpace);
                if (defaultValue != null)
                    writer.Write($" = {defaultValue};");
                writer.WriteLine();
                writer.WriteLine();
            }
        }

        private void GenerateOutput(IndentingTextWriter writer,
            ModuleCatalog.EntryPointInfo entryPointInfo,
            out HashSet<string> outputVariableNames)
        {
            outputVariableNames = new HashSet<string>();
            string classBase = "";
            if (entryPointInfo.OutputKinds != null)
                classBase = $" : {string.Join(", ", entryPointInfo.OutputKinds.Select(GeneratorUtils.GetCSharpTypeName))}";
            writer.WriteLine($"public sealed class Output{classBase}");
            writer.WriteLine("{");
            writer.Indent();

            var outputType = entryPointInfo.OutputType;
            if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(CommonOutputs.MacroOutput<>))
                outputType = outputType.GetGenericTypeArgumentsEx()[0];
            foreach (var fieldInfo in outputType.GetFields())
            {
                var outputAttr = fieldInfo.GetCustomAttributes(typeof(TlcModule.OutputAttribute), false)
                    .FirstOrDefault() as TlcModule.OutputAttribute;
                if (outputAttr == null)
                    continue;

                writer.WriteLine("/// <summary>");
                writer.WriteLine($"/// {outputAttr.Desc}");
                writer.WriteLine("/// </summary>");
                var outputTypeString = GeneratorUtils.GetOutputType(fieldInfo.FieldType);
                outputVariableNames.Add(GeneratorUtils.Capitalize(outputAttr.Name ?? fieldInfo.Name));
                writer.WriteLine($"public {outputTypeString} {GeneratorUtils.Capitalize(outputAttr.Name ?? fieldInfo.Name)} {{ get; set; }} = new {outputTypeString}();");
                writer.WriteLine();
            }

            writer.Outdent();
            writer.WriteLine("}");
        }

        private void GenerateMethod(IndentingTextWriter writer,
            ModuleCatalog.EntryPointInfo entryPointInfo,
            ModuleCatalog catalog)
        {
            var inputOuputClassName = GeneratorUtils.GetFullMethodName(entryPointInfo);
            inputOuputClassName = "Microsoft.ML." + inputOuputClassName;
            writer.WriteLine($"public {inputOuputClassName}.Output Add({inputOuputClassName} input)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"var output = new {inputOuputClassName}.Output();");
            writer.WriteLine("Add(input, output);");
            writer.WriteLine("return output;");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
            writer.WriteLine($"public void Add({inputOuputClassName} input, {inputOuputClassName}.Output output)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"_jsonNodes.Add(Serialize(\"{entryPointInfo.Name}\", input, output));");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
        }

        private void GenerateComponentKind(IndentingTextWriter writer, string kind)
        {
            writer.WriteLine($"public abstract class {kind} : ComponentKind {{}}");
            writer.WriteLine();
        }

        private void GenerateComponent(IndentingTextWriter writer, ModuleCatalog.ComponentInfo component, ModuleCatalog catalog)
        {
            GenerateEnums(writer, component.ArgumentType, "Runtime");
            writer.WriteLine();
            GenerateStructs(writer, component.ArgumentType, catalog, "Runtime");
            writer.WriteLine();
            writer.WriteLine("/// <summary>");
            writer.WriteLine($"/// {component.Description}");
            writer.WriteLine("/// </summary>");
            writer.WriteLine($"public sealed class {GeneratorUtils.GetComponentName(component)} : {component.Kind}");
            writer.WriteLine("{");
            writer.Indent();
            GenerateInputFields(writer, component.ArgumentType, catalog, _typesSymbolTable, "Microsoft.ML.");
            writer.WriteLine($"internal override string ComponentName => \"{component.Name}\";");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
        }
    }
}
