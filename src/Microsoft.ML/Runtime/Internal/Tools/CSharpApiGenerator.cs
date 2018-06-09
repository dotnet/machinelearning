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

        private readonly IHost _host;
        private readonly string _csFilename;
        private readonly string _regenerate;
        private readonly HashSet<string> _excludedSet;
        private const string RegistrationName = "CSharpApiGenerator";
        private const string _defaultNamespace = "Microsoft.ML.";
        private Dictionary<string, string> _typesSymbolTable = new Dictionary<string, string>();

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
                GeneratorUtils.GenerateHeader(writer);

                foreach (var entryPointInfo in catalog.AllEntryPoints().Where(x => !_excludedSet.Contains(x.Name)).OrderBy(x => x.Name))
                {
                    // Generate method
                    GenerateMethod(writer, entryPointInfo, catalog);
                }

                // Generate footer
                GeneratorUtils.GenerateFooter(writer);
                GeneratorUtils.GenerateFooter(writer);

                foreach (var entryPointInfo in catalog.AllEntryPoints().Where(x => !_excludedSet.Contains(x.Name)).OrderBy(x => x.Name))
                {
                    // Generate input and output classes
                    GenerateInputOutput(writer, entryPointInfo, catalog);
                }

                writer.WriteLine("namespace Runtime");
                writer.WriteLine("{");
                writer.Indent();

                foreach (var kind in catalog.GetAllComponentKinds())
                {
                    // Generate kind base class
                    GenerateComponentKind(writer, kind);

                    foreach (var component in catalog.GetAllComponents(kind))
                    {
                        // Generate component
                        GenerateComponent(writer, component, catalog);
                    }
                }

                GeneratorUtils.GenerateFooter(writer);
                GeneratorUtils.GenerateFooter(writer);
                writer.WriteLine("#pragma warning restore");
            }
        }



        private void GenerateInputOutput(IndentingTextWriter writer,
            ModuleCatalog.EntryPointInfo entryPointInfo,
            ModuleCatalog catalog)
        {
            var classAndMethod = GeneratorUtils.GetEntryPointMetadata(entryPointInfo);
            writer.WriteLine($"namespace {classAndMethod.Namespace}");
            writer.WriteLine("{");
            writer.Indent();
            GenerateInput(writer, entryPointInfo, catalog);
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
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

                var symbolName = GeneratorUtils.GetSymbolFromType(_typesSymbolTable, type, currentNamespace);
                if (enumType == typeof(int))
                    writer.WriteLine($"public enum {symbolName.Substring(symbolName.LastIndexOf('.') + 1)}");
                else
                {
                    Contracts.Assert(enumType == typeof(byte));
                    writer.WriteLine($"public enum {symbolName.Substring(symbolName.LastIndexOf('.') + 1)} : byte");
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

        private void GenerateClasses(IndentingTextWriter writer,
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
                GenerateEnums(writer, type, currentNamespace);
                GenerateClasses(writer, type, catalog, currentNamespace);
                var symbolName = GeneratorUtils.GetSymbolFromType(_typesSymbolTable, type, currentNamespace);
                string classBase = "";
                if (type.IsSubclassOf(typeof(OneToOneColumn)))
                    classBase = $" : OneToOneColumn<{symbolName.Substring(symbolName.LastIndexOf('.') + 1)}>, IOneToOneColumn";
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                    classBase = $" : ManyToOneColumn<{symbolName.Substring(symbolName.LastIndexOf('.') + 1)}>, IManyToOneColumn";
                writer.WriteLine($"public sealed partial class {symbolName.Substring(symbolName.LastIndexOf('.') + 1)}{classBase}");
                writer.WriteLine("{");
                writer.Indent();
                GenerateInputFields(writer, type, catalog, currentNamespace);
                writer.Outdent();
                writer.WriteLine("}");
                writer.WriteLine();
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
                    columnType = GenerateOneToOneColumn(writer, className, columnType, fieldInfo, inputAttr, type, isArray);
                }
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                {
                    columnType = GenerateManyToOneColumn(writer, className, columnType, fieldInfo, inputAttr, type, isArray);
                }
            }
        }

        private Type GenerateManyToOneColumn(IndentingTextWriter writer, string className, Type columnType,
            System.Reflection.FieldInfo fieldInfo, ArgumentAttribute inputAttr, Type type, bool isArray)
        {
            var fieldName = GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
            var generatedType = _typesSymbolTable[type.FullName];
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
                writer.WriteLine($"var list = {fieldName} == null ? new List<{generatedType}>() : new List<{generatedType}>({fieldName});");
                writer.WriteLine($"list.Add(ManyToOneColumn<{generatedType}>.Create(name, source));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = ManyToOneColumn<{generatedType}>.Create(name, source);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();

            Contracts.Assert(columnType == null);

            columnType = type;
            return columnType;
        }

        private Type GenerateOneToOneColumn(IndentingTextWriter writer, string className, Type columnType,
            System.Reflection.FieldInfo fieldInfo, ArgumentAttribute inputAttr, Type type, bool isArray)
        {
            var fieldName = GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
            var generatedType = _typesSymbolTable[type.FullName];
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
                writer.WriteLine($"var list = {fieldName} == null ? new List<{generatedType}>() : new List<{generatedType}>({fieldName});");
                writer.WriteLine($"list.Add(OneToOneColumn<{generatedType}>.Create(source));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = OneToOneColumn<{generatedType}>.Create(source);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
            writer.WriteLine($"public void Add{fieldName}(string name, string source)");
            writer.WriteLine("{");
            writer.Indent();
            if (isArray)
            {
                writer.WriteLine($"var list = {fieldName} == null ? new List<{generatedType}>() : new List<{generatedType}>({fieldName});");
                writer.WriteLine($"list.Add(OneToOneColumn<{_typesSymbolTable[type.FullName]}>.Create(name, source));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = OneToOneColumn<{generatedType}>.Create(name, source);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();

            Contracts.Assert(columnType == null);

            columnType = type;
            return columnType;
        }

        private void GenerateInput(IndentingTextWriter writer, ModuleCatalog.EntryPointInfo entryPointInfo, ModuleCatalog catalog)
        {
            var entryPointMetadata = GeneratorUtils.GetEntryPointMetadata(entryPointInfo);
            string classBase = "";
            if (entryPointInfo.InputKinds != null)
            {
                classBase += $" : {string.Join(", ", entryPointInfo.InputKinds.Select(GeneratorUtils.GetCSharpTypeName))}";
                if (entryPointInfo.InputKinds.Any(t => typeof(ITrainerInput).IsAssignableFrom(t) || typeof(ITransformInput).IsAssignableFrom(t)))
                    classBase += ", Microsoft.ML.ILearningPipelineItem";
            }

            GenerateEnums(writer, entryPointInfo.InputType, _defaultNamespace + entryPointMetadata.Namespace);
            writer.WriteLine();
            GenerateClasses(writer, entryPointInfo.InputType, catalog, _defaultNamespace + entryPointMetadata.Namespace);
            GeneratorUtils.GenerateSummary(writer, entryPointInfo.Description);

            if (entryPointInfo.ObsoleteAttribute != null)
                writer.WriteLine($"[Obsolete(\"{entryPointInfo.ObsoleteAttribute.Message}\")]");

            writer.WriteLine($"public sealed partial class {entryPointMetadata.ClassName}{classBase}");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine();
            if (entryPointInfo.InputKinds != null && entryPointInfo.InputKinds.Any(t => typeof(ILearningPipelineLoader).IsAssignableFrom(t)))
                GenerateLoaderAddInputMethod(writer, entryPointMetadata.ClassName);

            GenerateColumnAddMethods(writer, entryPointInfo.InputType, catalog, entryPointMetadata.ClassName, out Type transformType);
            writer.WriteLine();
            GenerateInputFields(writer, entryPointInfo.InputType, catalog, _defaultNamespace + entryPointMetadata.Namespace);
            writer.WriteLine();

            GenerateOutput(writer, entryPointInfo, out HashSet<string> outputVariableNames);
            GenerateApplyFunction(writer, entryPointMetadata.ClassName, transformType, outputVariableNames, entryPointInfo.InputKinds);
            writer.Outdent();
            writer.WriteLine("}");
        }

        private static void GenerateApplyFunction(IndentingTextWriter writer, string className,
            Type type, HashSet<string> outputVariableNames, Type[] inputKinds)
        {
            if (inputKinds == null)
                return;

            bool isTransform = false;
            bool isCalibrator = false;

            if (inputKinds.Any(t => typeof(ITransformInput).IsAssignableFrom(t)))
                isTransform = true;
            else if (!inputKinds.Any(t => typeof(ITrainerInput).IsAssignableFrom(t)))
                return;

            if (inputKinds.Any(t => typeof(ICalibratorInput).IsAssignableFrom(t)))
                isCalibrator = true;

            if (isTransform)
                writer.WriteLine("public Var<IDataView> GetInputData() => Data;");
            else
                writer.WriteLine("public Var<IDataView> GetInputData() => TrainingData;");

            writer.WriteLine("");
            writer.WriteLine("public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)");
            writer.WriteLine("{");

            writer.Indent();
            writer.WriteLine("if (previousStep != null)");
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

            writer.Outdent();
            writer.WriteLine("}");

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

        private void GenerateInputFields(IndentingTextWriter writer, Type inputType, ModuleCatalog catalog, string rootNameSpace)
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

                GeneratorUtils.GenerateSummary(writer, inputAttr.HelpText);
                if (fieldInfo.FieldType == typeof(JArray))
                {
                    writer.WriteLine($"public Experiment {GeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} {{ get; set; }}");
                    writer.WriteLine();
                    continue;
                }

                var inputTypeString = GeneratorUtils.GetInputType(catalog, fieldInfo.FieldType, _typesSymbolTable, rootNameSpace);
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
                var defaultValue = GeneratorUtils.GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(defaults), _typesSymbolTable, rootNameSpace);
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

                GeneratorUtils.GenerateSummary(writer, outputAttr.Desc);
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
            var inputOuputClassName = _defaultNamespace + entryPointInfo.Name;
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
            GenerateClasses(writer, component.ArgumentType, catalog, "Runtime");
            writer.WriteLine();
            GeneratorUtils.GenerateSummary(writer, component.Description);
            writer.WriteLine($"public sealed class {GeneratorUtils.GetComponentName(component)} : {component.Kind}");
            writer.WriteLine("{");
            writer.Indent();
            GenerateInputFields(writer, component.ArgumentType, catalog, "Runtime");
            writer.WriteLine($"internal override string ComponentName => \"{component.Name}\";");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
        }
    }
}
