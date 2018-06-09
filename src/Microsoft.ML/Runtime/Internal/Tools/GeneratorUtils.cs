using Microsoft.CSharp;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json.Linq;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Tools
{
    internal static class GeneratorUtils
    {
        public static string GetFullMethodName(ModuleCatalog.EntryPointInfo entryPointInfo)
        {
            return entryPointInfo.Name;
        }

        public class EntryPointGenerationMetadata
        {
            public string Namespace { get; private set; }
            public string ClassName { get; private set; }
            public EntryPointGenerationMetadata(string @namespace, string className)
            {
                Namespace = @namespace;
                ClassName = className;
            }
        }

        public static EntryPointGenerationMetadata GetEntryPointMetadata(ModuleCatalog.EntryPointInfo entryPointInfo)
        {
            var split = entryPointInfo.Name.Split('.');
            Contracts.Assert(split.Length == 2);
            return new EntryPointGenerationMetadata(split[0], split[1]);
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

        public static string GetInputType(ModuleCatalog catalog, Type inputType, Dictionary<string, string> typesSymbolTable, string rootNameSpace)
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
                    return GetInputType(catalog, inputType.GetElementType(), typesSymbolTable, rootNameSpace) + "[]";
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
                        return GetEnumName(type, typesSymbolTable, rootNameSpace); ;
                    if (isOptional)
                        return $"Optional<{GetEnumName(type, typesSymbolTable, rootNameSpace)}>";
                    if (typesSymbolTable.ContainsKey(type.FullName))
                        return GetEnumName(type, typesSymbolTable, rootNameSpace);
                    else
                        return GetEnumName(type, typesSymbolTable, rootNameSpace); ;
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

        private static string GetCharAsString(char value)
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
                    return $"'{GetCharAsString((char)fieldValue)}'";
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
                    return $"new {GetEnumName(fieldType, typesSymbolTable, rootNameSpace)}()";
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

        public static string GetEnumName(Type type, Dictionary<string, string> typesSymbolTable, string rootNamespace)
        {
            if (!typesSymbolTable.TryGetValue(type.FullName, out string fullname))
                fullname = GetSymbolFromType(typesSymbolTable, type, rootNamespace);
            if (fullname.StartsWith(rootNamespace))
                return fullname.Substring(rootNamespace.Length + 1);
            else return fullname;
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
        public static string GetSymbolFromType(Dictionary<string, string> typesSymbolTable, Type type, string currentNamespace)
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
            typesSymbolTable[type.FullName] = name;
            return name;
        }

        public static void GenerateSummary(IndentingTextWriter writer, string summary)
        {
            if (string.IsNullOrEmpty(summary))
                return;
            writer.WriteLine("/// <summary>");
            foreach (var line in summary.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries))
                writer.WriteLine($"/// {line}");
            writer.WriteLine("/// </summary>");
        }

        public static void GenerateHeader(IndentingTextWriter writer)
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

        public static void GenerateFooter(IndentingTextWriter writer)
        {
            writer.Outdent();
            writer.WriteLine("}");
        }

    }
}
