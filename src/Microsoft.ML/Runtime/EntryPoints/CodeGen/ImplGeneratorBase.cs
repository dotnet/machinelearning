// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints.CodeGen
{
    internal abstract class ImplGeneratorBase : GeneratorBase
    {
        protected override void GenerateContent(IndentingTextWriter writer, string prefix, ComponentCatalog.LoadableClassInfo component, bool generateEnums, string moduleId)
        {
            GenerateImplFields(writer, component, (w, a) => GenerateFieldsOrProperties(w, a, "", GenerateField));
            if (generateEnums)
                GenerateEnums(writer, component);
            GenerateImplFields(writer, component, (w, a) => GenerateFieldsOrProperties(w, a, "", GenerateProperty));
            GenerateMethodSignature(writer, prefix, component);
            GenerateImplBody(writer, component);
        }

        protected void GenerateImplFields(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component,
            Action<IndentingTextWriter, CmdParser.ArgInfo.Arg> fieldGenerator)
        {
            var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
            var arguments = argumentInfo.Args.Where(a => !a.IsHidden).ToArray();
            foreach (var arg in arguments)
                fieldGenerator(w, arg);
        }

        /// <summary>
        /// Generates private fields and public properties for all the fields in the arguments.
        /// Recursively generate fields and properties for subcomponents.
        /// </summary>
        protected void GenerateFieldsOrProperties(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string argSuffix,
            Action<IndentingTextWriter, string, string, string, bool, string> oneFieldGenerator)
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
                    oneFieldGenerator(w, "Tuple<string, string>", arg.LongName + argSuffix, "", false, arg.HelpText);
                    return;
                }
                var typeName = EnumName(arg, sigType);
                string defVal = arg.DefaultValue != null ? string.Format(" = {0}.{1}", typeName, arg.DefaultValue) : "";
                oneFieldGenerator(w, typeName, arg.LongName + argSuffix, defVal, arg.ItemType == typeof(bool),
                    arg.HelpText);
                var infos = ComponentCatalog.GetAllDerivedClasses(baseType, sigType);
                foreach (var info in infos)
                {
                    var args = info.CreateArguments();
                    if (args == null)
                        continue;
                    var argInfo = CmdParser.GetArgInfo(args.GetType(), args);
                    foreach (var a in argInfo.Args)
                        GenerateFieldsOrProperties(w, a, argSuffix + info.LoadNames[0], oneFieldGenerator);
                }
            }
            else
            {
                object val = Stringify(arg.DefaultValue);
                string defVal = val == null ? "" : string.Format(" = {0}", val);
                var typeName = IsColumnType(arg) ? "string[]" : IsStringColumnType(arg) ? GetCSharpTypeName(arg.Field.FieldType) : GetCSharpTypeName(arg.ItemType);
                oneFieldGenerator(w, typeName, arg.LongName + argSuffix, defVal, arg.ItemType == typeof(bool), arg.HelpText);
            }
        }

        protected static void GenerateField(IndentingTextWriter w, string typeName, string argName, string defVal,
            bool isBool, string helpText)
        {
            w.WriteLine("private {0} {1}{2};", typeName, argName, defVal);
            w.WriteLine();
        }

        protected static void GenerateProperty(IndentingTextWriter w, string typeName, string argName, string defVal,
            bool isBool, string helpText)
        {
            var help = helpText ?? argName;
            help = help.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
            w.WriteLine("/// <summary> Gets or sets {0}{1} </summary>", isBool ? "a value indicating whether " : "", help);
            w.WriteLine("public {0} {1}", typeName, Capitalize(argName));
            w.WriteLine("{");
            using (w.Nest())
            {
                w.WriteLine("get {{ return {0}; }}", argName);
                w.WriteLine("set {{ {0} = value; }}", argName);
            }
            w.WriteLine("}");
            w.WriteLine();
        }

        protected abstract void GenerateMethodSignature(IndentingTextWriter w, string prefix,
            ComponentCatalog.LoadableClassInfo component);

        protected abstract void GenerateImplBody(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component);

        protected void GenerateImplBody(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string argSuffix)
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
                if (IsTrainer(sigType))
                {
                    if (arg.IsCollection)
                        w.WriteLine("args{0}.{1} = new[] {{ new SubComponent<{2}, {3}>({4}.Item1, {4}.Item2) }};",
                            argSuffix, arg.LongName, GetCSharpTypeName(baseType), GetCSharpTypeName(sigType),
                            arg.LongName + argSuffix);
                    else
                        w.WriteLine("args{0}.{1} = new SubComponent<{2}, {3}>({4}.Item1, {4}.Item2);", argSuffix,
                            arg.LongName, GetCSharpTypeName(baseType), GetCSharpTypeName(sigType),
                            arg.LongName + argSuffix);
                    return;
                }
                if (sigType == typeof(SignatureDataLoader))
                    return;
                var typeName = EnumName(arg, sigType);
                w.WriteLine("switch ({0})", arg.LongName + argSuffix);
                w.WriteLine("{");
                using (w.Nest())
                {
                    if (arg.NullName != null)
                    {
                        w.WriteLine("case {0}.None:", typeName);
                        using (w.Nest())
                        {
                            w.WriteLine("args{0}.{1} = null;", argSuffix, arg.LongName);
                            w.WriteLine("break;");
                        }
                    }
                    var infos = ComponentCatalog.GetAllDerivedClasses(baseType, sigType);
                    foreach (var info in infos)
                    {
                        w.WriteLine("case {0}.{1}:", typeName, info.LoadNames[0]);
                        using (w.Nest())
                        {
                            if (info.ArgType != null)
                            {
                                var newArgSuffix = argSuffix + info.LoadNames[0];
                                w.WriteLine("var args{0} = new {1}();", newArgSuffix, GetCSharpTypeName(info.ArgType));
                                w.WriteLine("var defs{0} = new {1}();", newArgSuffix, GetCSharpTypeName(info.ArgType));
                                var args = info.CreateArguments();
                                if (args != null)
                                {
                                    var argInfo = CmdParser.GetArgInfo(args.GetType(), args);
                                    foreach (var a in argInfo.Args)
                                        GenerateImplBody(w, a, newArgSuffix);
                                }
                                w.WriteLine(
                                    "args{0}.{1} = new {2}(\"{3}\", CmdParser.GetSettings(args{4}, defs{4}));",
                                    argSuffix, arg.LongName, GetCSharpTypeName(arg.ItemType), info.LoadNames[0],
                                    newArgSuffix);
                            }
                            else
                                w.WriteLine("args{0}.{1} = new {2}(\"{3}\");", argSuffix, arg.LongName, GetCSharpTypeName(arg.ItemType), info.LoadNames[0]);
                            w.WriteLine("break;");
                        }
                    }
                }
                w.WriteLine("}");
            }
            else if (arg.IsCollection)
            {
                if (IsColumnType(arg))
                    w.WriteLine("args{0}.{1} = {1}.Select({2}.Parse).ToArray();", argSuffix, arg.LongName, GetCSharpTypeName(arg.ItemType));
                else if (IsStringColumnType(arg))
                    w.WriteLine("args{0}.{1} = {2};", argSuffix, arg.LongName, arg.LongName + argSuffix);
                else
                    w.WriteLine("args{0}.{1} = new[] {{ {2} }};", argSuffix, arg.LongName, arg.LongName + argSuffix);
            }
            else
                w.WriteLine("args{0}.{1} = {2};", argSuffix, arg.LongName, arg.LongName + argSuffix);
        }

        protected override void GenerateEnumValue(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo info)
        {
            var name = info != null ? info.LoadNames[0] : "None";
            w.WriteLine("/// <summary> {0} option </summary>", name);
            w.Write("{0}", name);
        }

        protected override void GenerateUsings(IndentingTextWriter w)
        {
            w.WriteLine("using System;");
            w.WriteLine("using System.Linq;");
            w.WriteLine("using Microsoft.ML.Runtime;");
            w.WriteLine("using Microsoft.ML.Runtime.CommandLine;");
            w.WriteLine("using Microsoft.ML.Runtime.Data;");
            w.WriteLine("using Microsoft.ML.Runtime.Internal.Internallearn;");
        }
    }
}