// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
        private const string _defaultNamespace = "Microsoft.ML.Legacy.";
        private readonly GeneratedClasses _generatedClasses;

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
            _generatedClasses = new GeneratedClasses();
        }

        public void Generate(IEnumerable<HelpCommand.Component> infos)
        {
            var catalog = _host.ComponentCatalog;

            using (var sw = new StreamWriter(_csFilename))
            {
                var writer = IndentingTextWriter.Wrap(sw, "    ");

                // Generate header
                CSharpGeneratorUtils.GenerateHeader(writer);

                foreach (var entryPointInfo in catalog.AllEntryPoints().Where(x => !_excludedSet.Contains(x.Name)).OrderBy(x => x.Name))
                {
                    // Generate method
                    CSharpGeneratorUtils.GenerateMethod(writer, entryPointInfo.Name, _defaultNamespace);
                }

                // Generate footer
                CSharpGeneratorUtils.GenerateFooter(writer);
                CSharpGeneratorUtils.GenerateFooter(writer);

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

                CSharpGeneratorUtils.GenerateFooter(writer);
                CSharpGeneratorUtils.GenerateFooter(writer);
                writer.WriteLine("#pragma warning restore");
            }
        }

        private void GenerateInputOutput(IndentingTextWriter writer, ComponentCatalog.EntryPointInfo entryPointInfo, ComponentCatalog catalog)
        {
            var classAndMethod = CSharpGeneratorUtils.GetEntryPointMetadata(entryPointInfo);
            writer.WriteLine($"namespace Legacy.{classAndMethod.Namespace}");
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
                var type = CSharpGeneratorUtils.ExtractOptionalOrNullableType(fieldInfo.FieldType);
                if (_generatedClasses.IsGenerated(type.FullName))
                    continue;

                if (!type.IsEnum)
                {
                    var typeEnum = TlcModule.GetDataType(type);
                    if (typeEnum == TlcModule.DataKind.Unknown)
                        GenerateEnums(writer, type, currentNamespace);
                    continue;
                }

                var enumType = Enum.GetUnderlyingType(type);

                var apiName = _generatedClasses.GetApiName(type, currentNamespace);
                if (enumType == typeof(int))
                    writer.WriteLine($"public enum {apiName}");
                else
                {
                    Contracts.Assert(enumType == typeof(byte));
                    writer.WriteLine($"public enum {apiName} : byte");
                }

                _generatedClasses.MarkAsGenerated(type.FullName);
                writer.Write("{");
                writer.Indent();
                var names = Enum.GetNames(type);
                var values = Enum.GetValues(type);
                string prefix = "";
                for (int i = 0; i < names.Length; i++)
                {
                    var name = names[i];
                    if (type.GetField(name).GetCustomAttribute<HideEnumValueAttribute>() != null)
                        continue;
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

        private void GenerateClasses(IndentingTextWriter writer, Type inputType, ComponentCatalog catalog, string currentNamespace)
        {
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;

                var type = fieldInfo.FieldType;
                type = CSharpGeneratorUtils.ExtractOptionalOrNullableType(type);
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

                if (_generatedClasses.IsGenerated(type.FullName))
                    continue;
                GenerateEnums(writer, type, currentNamespace);
                GenerateClasses(writer, type, catalog, currentNamespace);

                var apiName = _generatedClasses.GetApiName(type, currentNamespace);
                string classBase = "";
                if (type.IsSubclassOf(typeof(OneToOneColumn)))
                    classBase = $" : OneToOneColumn<{apiName}>, IOneToOneColumn";
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                    classBase = $" : ManyToOneColumn<{apiName}>, IManyToOneColumn";
                writer.WriteLine($"public sealed partial class {apiName}{classBase}");
                writer.WriteLine("{");
                writer.Indent();
                _generatedClasses.MarkAsGenerated(type.FullName);
                GenerateInputFields(writer, type, catalog, currentNamespace);
                writer.Outdent();
                writer.WriteLine("}");
                writer.WriteLine();
            }
        }

        private void GenerateColumnAddMethods(IndentingTextWriter writer, Type inputType, ComponentCatalog catalog,
            string className, out Type columnType)
        {
            columnType = null;
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;
                var type = CSharpGeneratorUtils.ExtractOptionalOrNullableType(fieldInfo.FieldType);
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
                    columnType = GenerateOneToOneColumn(writer, className, columnType, fieldInfo, inputAttr, type, isArray);
                else if (type.IsSubclassOf(typeof(ManyToOneColumn)))
                    columnType = GenerateManyToOneColumn(writer, className, columnType, fieldInfo, inputAttr, type, isArray);
            }
        }

        private Type GenerateManyToOneColumn(IndentingTextWriter writer, string className, Type columnType,
            System.Reflection.FieldInfo fieldInfo, ArgumentAttribute inputAttr, Type type, bool isArray)
        {
            var fieldName = CSharpGeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
            var apiName = _generatedClasses.GetApiName(type, "");
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
                writer.WriteLine($"var list = {fieldName} == null ? new List<{apiName}>() : new List<{apiName}>({fieldName});");
                writer.WriteLine($"list.Add(ManyToOneColumn<{apiName}>.Create(name, source));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = ManyToOneColumn<{apiName}>.Create(name, source);");
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
            var fieldName = CSharpGeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name);
            var generatedType = _generatedClasses.GetApiName(type, "");
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
            writer.WriteLine($"public {className}(params (string inputColumn, string outputColumn)[] inputOutput{fieldName}s)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"if (inputOutput{fieldName}s != null)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"foreach (var inputOutput in inputOutput{fieldName}s)");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine($"Add{fieldName}(inputOutput.outputColumn, inputOutput.inputColumn);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.Outdent();
            writer.WriteLine("}");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine("");
            writer.WriteLine($"public void Add{fieldName}(string inputColumn)");
            writer.WriteLine("{");
            writer.Indent();
            if (isArray)
            {
                writer.WriteLine($"var list = {fieldName} == null ? new List<{generatedType}>() : new List<{generatedType}>({fieldName});");
                writer.WriteLine($"list.Add(OneToOneColumn<{generatedType}>.Create(inputColumn));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = OneToOneColumn<{generatedType}>.Create(inputColumn);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();
            writer.WriteLine($"public void Add{fieldName}(string outputColumn, string inputColumn)");
            writer.WriteLine("{");
            writer.Indent();
            if (isArray)
            {
                writer.WriteLine($"var list = {fieldName} == null ? new List<{generatedType}>() : new List<{generatedType}>({fieldName});");
                writer.WriteLine($"list.Add(OneToOneColumn<{generatedType}>.Create(outputColumn, inputColumn));");
                writer.WriteLine($"{fieldName} = list.ToArray();");
            }
            else
                writer.WriteLine($"{fieldName} = OneToOneColumn<{generatedType}>.Create(outputColumn, inputColumn);");
            writer.Outdent();
            writer.WriteLine("}");
            writer.WriteLine();

            Contracts.Assert(columnType == null);

            columnType = type;
            return columnType;
        }

        private void GenerateInput(IndentingTextWriter writer, ComponentCatalog.EntryPointInfo entryPointInfo, ComponentCatalog catalog)
        {
            var entryPointMetadata = CSharpGeneratorUtils.GetEntryPointMetadata(entryPointInfo);
            string classBase = "";
            if (entryPointInfo.InputKinds != null)
            {
                classBase += $" : {string.Join(", ", entryPointInfo.InputKinds.Select(CSharpGeneratorUtils.GetCSharpTypeName))}";
                if (entryPointInfo.InputKinds.Any(t => typeof(ITrainerInput).IsAssignableFrom(t) || typeof(ITransformInput).IsAssignableFrom(t)))
                    classBase += ", Microsoft.ML.Legacy.ILearningPipelineItem";
            }

            GenerateEnums(writer, entryPointInfo.InputType, _defaultNamespace + entryPointMetadata.Namespace);
            writer.WriteLine();
            GenerateClasses(writer, entryPointInfo.InputType, catalog, _defaultNamespace + entryPointMetadata.Namespace);
            CSharpGeneratorUtils.GenerateSummary(writer, entryPointInfo.Description, entryPointInfo.XmlInclude);

            if (entryPointInfo.ObsoleteAttribute != null)
                writer.WriteLine($"[Obsolete(\"{entryPointInfo.ObsoleteAttribute.Message}\")]");

            writer.WriteLine($"public sealed partial class {entryPointMetadata.ClassName}{classBase}");
            writer.WriteLine("{");
            writer.Indent();
            writer.WriteLine();
            if (entryPointInfo.InputKinds != null && entryPointInfo.InputKinds.Any(t => typeof(Legacy.ILearningPipelineLoader).IsAssignableFrom(t)))
                CSharpGeneratorUtils.GenerateLoaderAddInputMethod(writer, entryPointMetadata.ClassName);

            GenerateColumnAddMethods(writer, entryPointInfo.InputType, catalog, entryPointMetadata.ClassName, out Type transformType);
            writer.WriteLine();
            GenerateInputFields(writer, entryPointInfo.InputType, catalog, _defaultNamespace + entryPointMetadata.Namespace);
            writer.WriteLine();

            GenerateOutput(writer, entryPointInfo, out HashSet<string> outputVariableNames);
            GenerateApplyFunction(writer, entryPointMetadata.ClassName, transformType, outputVariableNames, entryPointInfo.InputKinds);
            writer.Outdent();
            writer.WriteLine("}");
        }

        private static void GenerateApplyFunction(IndentingTextWriter writer, string className, Type type,
            HashSet<string> outputVariableNames, Type[] inputKinds)
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

        private void GenerateInputFields(IndentingTextWriter writer, Type inputType, ComponentCatalog catalog, string rootNameSpace)
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

                CSharpGeneratorUtils.GenerateSummary(writer, inputAttr.HelpText);
                if (fieldInfo.FieldType == typeof(JArray))
                {
                    writer.WriteLine($"public Experiment {CSharpGeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} {{ get; set; }}");
                    writer.WriteLine();
                    continue;
                }

                var inputTypeString = CSharpGeneratorUtils.GetInputType(catalog, fieldInfo.FieldType, _generatedClasses, rootNameSpace);
                if (CSharpGeneratorUtils.IsComponent(fieldInfo.FieldType))
                    writer.WriteLine("[JsonConverter(typeof(ComponentSerializer))]");
                if (CSharpGeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name) != (inputAttr.Name ?? fieldInfo.Name))
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

                writer.Write($"public {inputTypeString} {CSharpGeneratorUtils.Capitalize(inputAttr.Name ?? fieldInfo.Name)} {{ get; set; }}");
                var defaultValue = CSharpGeneratorUtils.GetValue(catalog, fieldInfo.FieldType, fieldInfo.GetValue(defaults), _generatedClasses, rootNameSpace);
                if (defaultValue != null)
                    writer.Write($" = {defaultValue};");
                writer.WriteLine();
                writer.WriteLine();
            }
        }

        private void GenerateOutput(IndentingTextWriter writer, ComponentCatalog.EntryPointInfo entryPointInfo, out HashSet<string> outputVariableNames)
        {
            outputVariableNames = new HashSet<string>();
            string classBase = "";
            if (entryPointInfo.OutputKinds != null)
                classBase = $" : {string.Join(", ", entryPointInfo.OutputKinds.Select(CSharpGeneratorUtils.GetCSharpTypeName))}";
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

                CSharpGeneratorUtils.GenerateSummary(writer, outputAttr.Desc);
                var outputTypeString = CSharpGeneratorUtils.GetOutputType(fieldInfo.FieldType);
                outputVariableNames.Add(CSharpGeneratorUtils.Capitalize(outputAttr.Name ?? fieldInfo.Name));
                writer.WriteLine($"public {outputTypeString} {CSharpGeneratorUtils.Capitalize(outputAttr.Name ?? fieldInfo.Name)} {{ get; set; }} = new {outputTypeString}();");
                writer.WriteLine();
            }

            writer.Outdent();
            writer.WriteLine("}");
        }

        private void GenerateComponentKind(IndentingTextWriter writer, string kind)
        {
            writer.WriteLine($"public abstract class {kind} : ComponentKind {{}}");
            writer.WriteLine();
        }

        private void GenerateComponent(IndentingTextWriter writer, ComponentCatalog.ComponentInfo component, ComponentCatalog catalog)
        {
            GenerateEnums(writer, component.ArgumentType, "Runtime");
            writer.WriteLine();
            GenerateClasses(writer, component.ArgumentType, catalog, "Runtime");
            writer.WriteLine();
            CSharpGeneratorUtils.GenerateSummary(writer, component.Description);
            writer.WriteLine($"public sealed class {CSharpGeneratorUtils.GetComponentName(component)} : {component.Kind}");
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
