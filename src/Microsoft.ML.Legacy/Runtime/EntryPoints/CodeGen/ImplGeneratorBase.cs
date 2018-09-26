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
        protected override void GenerateContent(IndentingTextWriter writer, string prefix, ComponentCatalog.LoadableClassInfo component, string moduleId)
        {
            GenerateImplFields(writer, component, (w, a) => GenerateFieldsOrProperties(w, a, "", GenerateField));
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

            object val = Stringify(arg.DefaultValue);
            string defVal = val == null ? "" : string.Format(" = {0}", val);
            var typeName = IsColumnType(arg) ? "string[]" : IsStringColumnType(arg) ? GetCSharpTypeName(arg.Field.FieldType) : GetCSharpTypeName(arg.ItemType);
            oneFieldGenerator(w, typeName, arg.LongName + argSuffix, defVal, arg.ItemType == typeof(bool), arg.HelpText);
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

            if (arg.IsCollection)
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