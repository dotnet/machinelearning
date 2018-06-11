// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CSharp;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Internal.Tools
{
    internal static class CSharpGeneratorUtils
    {
        public sealed class EntryPointGenerationMetadata
        {
            public string Namespace { get; }
            public string ClassName { get; }
            public EntryPointGenerationMetadata(string classNamespace, string className)
            {
                Namespace = classNamespace;
                ClassName = className;
            }
        }

        public static EntryPointGenerationMetadata GetEntryPointMetadata(ModuleCatalog.EntryPointInfo entryPointInfo)
        {
            var split = entryPointInfo.Name.Split('.');
            Contracts.Assert(split.Length == 2);
            return new EntryPointGenerationMetadata(split[0], split[1]);
        }

        public static Type ExtractOptionalOrNullableType(Type type)
        {
            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(Optional<>) || type.GetGenericTypeDefinition() == typeof(Nullable<>)))
                type = type.GetGenericArguments()[0];

            return type;
        }

        public static Type ExtractOptionalOrNullableType(Type type, out bool isNullable, out bool isOptional)
        {
            isNullable = false;
            isOptional = false;
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
            return type;
        }

        public static string GetCSharpTypeName(Type type)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                return GetCSharpTypeName(type.GetGenericArguments()[0]) + "?";

            using (var p = new CSharpCodeProvider())
                return p.GetTypeOutput(new CodeTypeReference(type));
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

        public static string GetInputType(ModuleCatalog catalog, Type inputType, GeneratedClasses generatedClasses, string rootNameSpace)
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

            var type = ExtractOptionalOrNullableType(inputType, out bool isNullable, out bool isOptional);
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
                    return GetInputType(catalog, inputType.GetElementType(), generatedClasses, rootNameSpace) + "[]";
                case TlcModule.DataKind.Component:
                    string kind;
                    bool success = catalog.TryGetComponentKind(type, out kind);
                    Contracts.Assert(success);
                    return $"{kind}";
                case TlcModule.DataKind.Enum:
                    var enumName = generatedClasses.GetApiName(type, rootNameSpace);
                    if (isNullable)
                        return $"{enumName}?";
                    if (isOptional)
                        return $"Optional<{enumName}>";
                    return $"{enumName}";
                default:
                    if (isNullable)
                        return generatedClasses.GetApiName(type, rootNameSpace) + "?";
                    if (isOptional)
                        return $"Optional<{generatedClasses.GetApiName(type, rootNameSpace)}>";
                    return generatedClasses.GetApiName(type, rootNameSpace);
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

            var type = ExtractOptionalOrNullableType(inputType);
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
            GeneratedClasses generatedClasses, string rootNameSpace)
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
            fieldType = ExtractOptionalOrNullableType(fieldType, out bool isNullable, out bool isOptional);
            switch (typeEnum)
            {
                case TlcModule.DataKind.Array:
                    var arr = fieldValue as Array;
                    if (arr != null && arr.GetLength(0) > 0)
                        return $"{{ {string.Join(", ", arr.Cast<object>().Select(item => GetValue(catalog, fieldType.GetElementType(), item, generatedClasses, rootNameSpace)))} }}";
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
                    return generatedClasses.GetApiName(fieldType, rootNameSpace) + "." + fieldValue;
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

                            var propertyValue = GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(fieldValue), generatedClasses, rootNameSpace);
                            var defaultPropertyValue = GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(defaultComponent), generatedClasses, rootNameSpace);
                            if (propertyValue != defaultPropertyValue)
                                propertyBag.Add($"{Capitalize(inputAttr.Name ?? fieldInfo.Name)} = {propertyValue}");
                        }
                    }
                    var properties = propertyBag.Count > 0 ? $" {{ {string.Join(", ", propertyBag)} }}" : "";
                    return $"new {GetComponentName(componentInfo)}(){properties}";
                case TlcModule.DataKind.Unknown:
                    return $"new {generatedClasses.GetApiName(fieldType, rootNameSpace)}()";
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

        public static void GenerateMethod(IndentingTextWriter writer, string className, string defaultNamespace)
        {
            var inputOuputClassName = defaultNamespace + className;
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
            writer.WriteLine($"_jsonNodes.Add(Serialize(\"{className}\", input, output));");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
        }

        public static void GenerateLoaderAddInputMethod(IndentingTextWriter writer, string className)
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

            //GetInputData
            writer.WriteLine("public Var<IDataView> GetInputData() => null;");
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
    }
}
