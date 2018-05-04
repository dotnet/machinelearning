// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints.CodeGen
{
    internal abstract class EntryPointGeneratorBase : GeneratorBase
    {
        protected override void GenerateContent(IndentingTextWriter writer, string prefix, ComponentCatalog.LoadableClassInfo component, bool generateEnums, string moduleId)
        {
            if (generateEnums)
                GenerateEnums(writer, component);
            GenerateSummaryComment(writer, component);
            GenerateReturnComment(writer);
            GenerateModuleAttribute(writer, prefix, component, moduleId);
            GenerateOutputPort(writer);
            GenerateModuleType(writer, component);
            GenerateMethodSignature(writer, prefix, component);
            GenerateImplCall(writer, prefix, component);
        }

        protected abstract void GenerateSummaryComment(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component);

        protected void GenerateSummaryComment(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (arg.IsSubComponentItemType)
            {
                // We need to create a tree with all the subcomponents, unless the subcomponent is a Trainer.
                Contracts.Assert(arg.ItemType.GetGenericTypeDefinition() == typeof(SubComponent<,>));
                var types = arg.ItemType.GetGenericArguments();
                var baseType = types[0];
                var sigType = types[1];
                if (sigType == typeof(SignatureDataLoader))
                    return;
                GenerateParameterComment(w, arg.LongName + argSuffix, arg.HelpText);
                if (IsTrainer(sigType))
                    return;
                var infos = ComponentCatalog.GetAllDerivedClasses(baseType, sigType);
                foreach (var info in infos)
                {
                    var args = info.CreateArguments();
                    if (args == null)
                        continue;
                    var argInfo = CmdParser.GetArgInfo(args.GetType(), args);
                    foreach (var a in argInfo.Args)
                        GenerateSummaryComment(w, a, argSuffix + info.LoadNames[0]);
                }
            }
            else
                GenerateParameterComment(w, arg.LongName + argSuffix, arg.HelpText);
        }

        protected void GenerateParameterComment(IndentingTextWriter w, string name, string description)
        {
            w.WriteLine("/// <param name=\"{0}\">{1}</param>", name, description);
        }

        protected abstract void GenerateReturnComment(IndentingTextWriter w);

        protected abstract void GenerateModuleAttribute(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component, string moduleId);

        protected abstract void GenerateOutputPort(IndentingTextWriter w);

        protected void GenerateModuleType(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component)
        {
            string cat;
            if (component.IsOfType(typeof(SignatureBinaryClassifierTrainer)))
                cat = "BinaryClassifier";
            else if (component.IsOfType(typeof(SignatureMultiClassClassifierTrainer)))
                cat = "MultiClassClassifier";
            else if (component.IsOfType(typeof(SignatureRegressorTrainer)))
                cat = "Regression";
            else if (component.IsOfType(typeof(SignatureAnomalyDetectorTrainer)))
                cat = "AnomalyDetector";
            else
                cat = "None";
            w.WriteLine("[DataLabModuleType(Type = ModuleType.{0})]", cat);
        }

        protected abstract void GenerateMethodSignature(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component);

        protected void GenerateMethodSignature(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string parent, string parentType, string parentValue, ref string linePrefix, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (arg.IsSubComponentItemType)
            {
                // We need to create a tree with all the subcomponents, unless the subcomponent is a Trainer.
                Contracts.Assert(arg.ItemType.GetGenericTypeDefinition() == typeof(SubComponent<,>));
                var types = arg.ItemType.GetGenericArguments();
                var baseType = types[0];
                var sigType = types[1];
                if (sigType == typeof(SignatureDataLoader))
                    return;
                w.WriteLine(linePrefix);
                linePrefix = ",";
                if (IsTrainer(sigType))
                {
                    w.WriteLine("[DataLabInputPort(FriendlyName = \"Untrained model\", DisplayName = \"Untrained model\", IsOptional = false, DataTypes = WellKnownDataTypeIds.ILearnerDotNet, Description = \"An untrained model\")]");
                    GenerateParameter(w, "ILearner", arg.LongName + argSuffix);
                    return;
                }
                var typeName = EnumName(arg, sigType);
                GenerateDataLabParameterAttribute(w, arg.LongName, false, arg.LongName, arg.DefaultValue != null ? typeName + "." + arg.DefaultValue : null, arg.HelpText, parent, parentType, parentValue);
                GenerateParameter(w, typeName, arg.LongName + argSuffix);
                var infos = ComponentCatalog.GetAllDerivedClasses(baseType, sigType);
                foreach (var info in infos)
                {
                    var args = info.CreateArguments();
                    if (args == null)
                        continue;
                    var argInfo = CmdParser.GetArgInfo(args.GetType(), args);
                    foreach (var a in argInfo.Args)
                        GenerateMethodSignature(w, a, arg.LongName, typeName, info.LoadNames[0], ref linePrefix, argSuffix + info.LoadNames[0]);
                }
            }
            else
            {
                w.WriteLine(linePrefix);
                linePrefix = ",";
                if (IsColumnType(arg))
                {
                    GenerateDataLabParameterAttribute(w, arg.LongName, false, arg.LongName, null, arg.HelpText, parent, parentType, parentValue);
                    GenerateParameter(w, "string", arg.LongName + argSuffix);
                }
                else
                {
                    GenerateDataLabParameterAttribute(w, arg.LongName, IsOptional(arg), arg.LongName, Stringify(arg.DefaultValue), arg.HelpText, parent, parentType, parentValue);
                    GenerateParameter(w, GetCSharpTypeName(arg.ItemType), arg.LongName + argSuffix);
                }
            }
        }

        protected void GenerateDataLabParameterAttribute(IndentingTextWriter w, string friendlyName,
            bool isOptional, string displayName, object defaultValue, string description, string parent = null,
            string parentType = null, string parentValue = null)
        {
            string p = parent != null ? string.Format(" ParentParameter = \"{0}\",", parent) : "";
            string pv = parentValue != null
                ? string.Format(" ParentParameterValue = new object[] {{ {0}.{1} }},", parentType, parentValue)
                : "";
            string dv = defaultValue != null ? string.Format(" DefaultValue = {0},", defaultValue) : "";
            w.WriteLine(
                "[DataLabParameter(FriendlyName = \"{0}\", IsOptional = {1}, DisplayName = \"{2}\",{3}{4}{5} Description = \"{6}\")]",
                friendlyName, isOptional ? "true" : "false", displayName, p, pv, dv, description);
        }

        protected abstract void GenerateImplCall(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component);

        protected void GenerateImplCall(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (arg.IsSubComponentItemType)
            {
                Contracts.Assert(arg.ItemType.GetGenericTypeDefinition() == typeof(SubComponent<,>));
                var types = arg.ItemType.GetGenericArguments();
                var baseType = types[0];
                var sigType = types[1];
                if (sigType == typeof(SignatureDataLoader))
                    return;
                if (IsTrainer(sigType))
                {
                    w.WriteLine("builder.{0} = {1};", Capitalize(arg.LongName + argSuffix), arg.LongName + argSuffix);
                    return;
                }
                w.WriteLine("builder.{0} = {1};", Capitalize(arg.LongName + argSuffix), arg.LongName + argSuffix);
                var infos = ComponentCatalog.GetAllDerivedClasses(baseType, sigType);
                foreach (var info in infos)
                {
                    var args = info.CreateArguments();
                    if (args == null)
                        continue;
                    var argInfo = CmdParser.GetArgInfo(args.GetType(), args);
                    foreach (var a in argInfo.Args)
                        GenerateImplCall(w, a, argSuffix + info.LoadNames[0]);
                }
            }
            else
            {
                if (IsColumnType(arg))
                    w.WriteLine("builder.{0} = {1}.Split('|');", Capitalize(arg.LongName + argSuffix), arg.LongName + argSuffix);
                else
                    w.WriteLine("builder.{0} = {1}{2};", Capitalize(arg.LongName + argSuffix), CastIfNeeded(arg), arg.LongName + argSuffix);
            }
        }

        protected override void GenerateEnumValue(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo info)
        {
            var userName = info != null ? info.UserName : "None";
            var name = info != null ? info.LoadNames[0] : "None";
            w.WriteLine("[ItemInfo(FriendlyName = \"{0}\", DisplayValue = \"{1}\")]", userName,
                userName);
            w.Write("{0}", name);
        }

        protected override string GetCSharpTypeName(Type type)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
                return GetCSharpTypeName(type.GetGenericArguments()[0]) + "?";

            // REVIEW: How are long/uint params going to be surfaced in AML?
            // long and uint params are not supported, falling back to int
            if (type == typeof(long) || type == typeof(uint))
                return "int";

            return base.GetCSharpTypeName(type);
        }

        protected override void GenerateUsings(IndentingTextWriter w)
        {
            w.WriteLine("using System;");
            w.WriteLine("using System.Diagnostics.CodeAnalysis;");
            w.WriteLine("using System.Linq;");
            w.WriteLine("using Microsoft.Analytics.MachineLearning;");
            w.WriteLine("using Microsoft.Analytics.Modules;");
            w.WriteLine("using Microsoft.ML.Runtime;");
            w.WriteLine("using Microsoft.ML.Runtime.CommandLine;");
            w.WriteLine("using Microsoft.ML.Runtime.Data;");
            w.WriteLine("using Microsoft.ML.Runtime.Internal.Internallearn;");
        }
    }
}