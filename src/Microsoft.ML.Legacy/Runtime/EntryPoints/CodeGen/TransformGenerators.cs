// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints.CodeGen
{
    internal sealed class TransformImplGenerator : ImplGeneratorBase
    {
        protected override void GenerateMethodSignature(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("/// <summary>");
            w.WriteLine("/// Creates a {0}", component.LoadNames[0]);
            w.WriteLine("/// </summary>");
            w.WriteLine("/// <param name=\"env\">The environment</param>");
            w.WriteLine("/// <param name=\"data\">The data set</param>");
            w.WriteLine("/// <returns>The transformed data.</returns>");
            w.WriteLine("public IDataView Create{0}{1}Impl(", prefix, component.LoadNames[0]);
            using (w.Nest())
            {
                w.WriteLine("IHostEnvironment env,");
                w.WriteLine("IDataView data)");
            }
        }

        protected override void GenerateImplBody(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("{");
            using (w.Nest())
            {
                if (component.ArgType == null)
                {
                    var call = GenerateCall(component);
                    w.WriteLine("return {0}(env, data);", call);
                }
                else
                {
                    w.WriteLine("var args = new {0}();", GetCSharpTypeName(component.ArgType));
                    var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                    foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                        GenerateImplBody(w, arg, "");
                    var call = GenerateCall(component);
                    w.WriteLine("return {0}(args, env, data);", call);
                }
            }
            w.WriteLine("}");
        }

        private string GenerateCall(ComponentCatalog.LoadableClassInfo component)
        {
            // The caller needs to ensure that the component has either a constructor or a create method.
            Contracts.Assert(component.Constructor != null || component.CreateMethod != null);
            string call;
            if (component.Constructor != null)
            {
                var type = GetCSharpTypeName(component.Constructor.DeclaringType);
                call = string.Format("new {0}", type);
            }
            else
            {
                var type = GetCSharpTypeName(component.CreateMethod.DeclaringType);
                var name = component.CreateMethod.Name;
                call = string.Format("{0}.{1}", type, name);
            }
            return call;
        }
    }

    internal sealed class TransformEntryPointGenerator : EntryPointGeneratorBase
    {
        protected override void GenerateSummaryComment(IndentingTextWriter w, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("/// <summary>");
            var desc = component.Summary ?? component.LoadNames[0];
            using (var sr = new StringReader(desc))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                    w.WriteLine("/// {0}", line);
            }
            w.WriteLine("/// </summary>");
            GenerateParameterComment(w, "data", "The data");
            var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
            foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                GenerateSummaryComment(w, arg, "");
        }

        protected override void GenerateReturnComment(IndentingTextWriter w)
        {
            w.WriteLine("/// <returns>A Tuple of transformed data and trained transform.</returns>");
        }

        protected override void GenerateModuleAttribute(IndentingTextWriter w, string prefix,
            ComponentCatalog.LoadableClassInfo component, string moduleId)
        {
            if (!string.IsNullOrEmpty(prefix))
                prefix += " ";
            w.WriteLine("[DataLabModule(FriendlyName = \"{0}{1}\",", prefix, component.UserName);
            using (w.Nest())
            {
                var desc = component.Summary ?? component.LoadNames[0];
                w.WriteLine("Description = \"{0}\",", desc.Replace("\n", "\\n").Replace("\r", "\\r"));
                w.WriteLine("IsBlocking = true,");
                w.WriteLine("IsDeterministic = true,");
                w.WriteLine("Version = \"2.0\",");
                w.WriteLine("Owner = \"Microsoft Corporation\",");
                w.WriteLine("FamilyId = \"{{{0}}}\",", moduleId.ToUpperInvariant());
                w.WriteLine("ReleaseState = States.Alpha)]");
            }
        }

        protected override void GenerateOutputPort(IndentingTextWriter w)
        {
            w.WriteLine(
                "[DataLabOutputPort(FriendlyName = \"Transformed IDataView\", DisplayName = \"Transformed IDataView\", Position = 0, DataType = WellKnownDataTypeIds.IDataViewDotNet, Description = \"Transformed data (IDataView)\")]");
            w.WriteLine(
                "[DataLabOutputPort(FriendlyName = \"Transformed data model\", DisplayName = \"Transformed data model\", Position = 1, DataType = WellKnownDataTypeIds.ITransformDotNet, Description = \"Transformed data model (ITransform)\")]");
        }

        protected override void GenerateMethodSignature(IndentingTextWriter w, string prefix,
            ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("public static Tuple<IDataView, DataTransform> Create{0}{1}(", prefix, component.LoadNames[0]);
            using (w.Nest())
            {
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                w.WriteLine("[DataLabInputPort(FriendlyName = \"IDataView\", DisplayName = \"IDataView\", IsOptional = false, DataTypes = WellKnownDataTypeIds.IDataViewDotNet, Description = \"Input data (IDataView)\")]");
                w.Write("IDataView data");
                var pre = ",";
                foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                    GenerateMethodSignature(w, arg, null, null, null, ref pre, "");
                w.WriteLine(")");
            }
        }

        protected override void GenerateImplCall(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("{");
            using (w.Nest())
            {
                var className = prefix + component.LoadNames[0];
                w.WriteLine("var builder = new {0}();", className);
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                    GenerateImplCall(w, arg, "");
                w.WriteLine("var env = new LocalEnvironment(1, verbose: true);");
                w.WriteLine("var view = builder.Create{0}{1}Impl(env, data);", prefix, component.LoadNames[0]);
                w.WriteLine("return new Tuple<IDataView, DataTransform>(view, new DataTransform(view));");
            }
            w.WriteLine("}");
        }
    }

    internal sealed class TransformModuleInstanceEntryPointGenerator : GeneratorBase
    {
        private string _compName;

        protected override void GenerateUsings(IndentingTextWriter w)
        {
            var allNamespaces = new HashSet<string>();
            foreach (var ns in Namespaces)
                allNamespaces.Add(ns);
            allNamespaces.Add("System.Collections.Generic");
            allNamespaces.Add("Microsoft.Analytics.Modules.Common");
            allNamespaces.Add("Microsoft.Analytics.Platform.ML.Models");
            allNamespaces.Add("Microsoft.ML.Runtime.Data");
            allNamespaces.Add("Microsoft.ML.Runtime.Modules.Contracts");
            allNamespaces.Add("Microsoft.ML.Runtime.Modules.Contracts.Attributes");
            allNamespaces.Add("Microsoft.ML.Runtime.Modules.Contracts.Types");
            var namespaces = allNamespaces.ToArray();
            Array.Sort(namespaces,
                (a, b) =>
                    a.StartsWith("System") && !b.StartsWith("System") ? -1
                    : !a.StartsWith("System") && b.StartsWith("System") ? 1
                    : string.CompareOrdinal(a, b));
            foreach (var ns in namespaces)
                w.WriteLine("using {0};", ns);
        }

        protected override void GenerateClassName(IndentingTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine();
            var className = prefix + component.LoadNames[0];
            w.WriteLine("/// <summary>Module: {0}</summary>", className);
            w.WriteLine("public static class {0}EntryPoint", className);
            w.WriteLine("{");
        }

        protected override void GenerateContent(IndentingTextWriter writer, string prefix,
            ComponentCatalog.LoadableClassInfo component, string moduleId)
        {
            writer.WriteLine("[Module(");
            _compName = prefix + component.LoadNames[0];
            var name = Name ?? PrettyPrintDisplayName(component.LoadNames[0]);
            using (writer.Nest())
            {
                writer.WriteLine("Name = \"{0}\",", name);
                writer.WriteLine("FamilyId = \"{0}\",", moduleId);
                writer.WriteLine("Owner = \"{0}\",", Owner);
                writer.WriteLine("ReleaseVersion = \"{0}\",", Version);
                writer.WriteLine("State = ModuleState.{0},", State);
                writer.WriteLine("Type = ModuleType.{0},", ModuleType);
                writer.WriteLine("Determinism = Determinism.{0},", Determinism);
                writer.WriteLine("Category = @\"{0}\")]", Category);
            }
            writer.WriteLine("public static IModule Create{0}(", _compName);
            using (writer.Nest())
            {
                writer.WriteLine("[Help(Display = @\"Dataset\", ToolTip = @\"Input dataset\")]");
                writer.WriteLine("[ModuleInputPort]");
                writer.WriteLine("IDataView idataset,");
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                    GenerateMethodSignature(writer, arg, null, null, null, "");
                writer.WriteLine("[Help(Display = @\"Results dataset\", ToolTip = @\"Transformed dataset\")]");
                writer.WriteLine("[ModuleOutputPort]");
                writer.WriteLine("IDataView odataset,");
                writer.WriteLine("[Help(Display = @\"{0}\", ToolTip = @\"{0}\")]", name);
                writer.WriteLine("[ModuleOutputPort]");
                writer.WriteLine("DataTransform otransform,");
                writer.WriteLine("[Context]");
                writer.WriteLine("IContext context)");
            }
            writer.WriteLine("{");
            using (writer.Nest())
            {
                writer.WriteLine("var instance = new {0}Module();", _compName);
                writer.WriteLine();
                writer.WriteLine("var ports = new Dictionary<string, object> { { \"idataset\", idataset } };");
                writer.WriteLine("var parameters = new Dictionary<string, object>");
                writer.WriteLine("{");
                using (writer.Nest())
                {
                    var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                    foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                        GenerateDictionaryEntry(writer, arg, "");
                }
                writer.WriteLine("};");
                writer.WriteLine();
                writer.WriteLine("instance.Context = context;");
                writer.WriteLine("instance.SetInputPorts(ports);");
                writer.WriteLine("instance.SetParameters(parameters);");
                writer.WriteLine();
                writer.WriteLine("return instance;");
            }
            writer.WriteLine("}");
            writer.WriteLine();
            writer.WriteLine("public class {0}Module : ModuleBase", _compName);
            writer.WriteLine("{");
            using (writer.Nest())
            {
                writer.WriteLine("private Dictionary<string, object> parameters;");
                writer.WriteLine("private Dictionary<string, object> ports;");
                writer.WriteLine();
                writer.WriteLine("public override Dictionary<string, object> Run()");
                writer.WriteLine("{");
                using (writer.Nest())
                {
                    writer.WriteLine("var view = ConstructTransform((IDataView)ports[\"idataset\"]);");
                    writer.WriteLine("return new Dictionary<string, object> { { \"odataset\", view }, { \"otransform\", new DataTransform(view) } };");
                }
                writer.WriteLine("}");
                writer.WriteLine();
                writer.WriteLine("public override void SetParameters(Dictionary<string, object> parameters)");
                writer.WriteLine("{");
                using (writer.Nest())
                    writer.WriteLine("this.parameters = parameters;");
                writer.WriteLine("}");
                writer.WriteLine();
                writer.WriteLine("public override void SetInputPorts(Dictionary<string, object> ports)");
                writer.WriteLine("{");
                using (writer.Nest())
                    writer.WriteLine("this.ports = ports;");
                writer.WriteLine("}");
                writer.WriteLine();
                writer.WriteLine("public override Dictionary<string, object> ComputeSchema(Dictionary<string, object> inputports)");
                writer.WriteLine("{");
                using (writer.Nest())
                {
                    writer.WriteLine("var view = ConstructTransform((IDataView)inputports[\"idataset\"]);");
                    writer.WriteLine("return new Dictionary<string, object> { { \"odataset\", view.Schema } };");
                }
                writer.WriteLine("}");
                writer.WriteLine();
                writer.WriteLine("private IDataView ConstructTransform(IDataView input)");
                writer.WriteLine("{");
                using (writer.Nest())
                {
                    writer.WriteLine("var builder = new {0}();", _compName);
                    var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                    foreach (var arg in argumentInfo.Args.Where(a => !a.IsHidden))
                        GenerateImplCall(writer, arg, null, null, null, "");
                    writer.WriteLine("return builder.Create{0}Impl(Host, input);", _compName);
                }
                writer.WriteLine("}");
            }
            writer.WriteLine("}");
        }

        protected override string EnumName(CmdParser.ArgInfo.Arg arg, Type sigType)
        {
            return _compName + "." + base.EnumName(arg, sigType);
        }

        private void GenerateMethodSignature(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string parent, string parentType, string parentValue, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (IsColumnType(arg))
            {
                GenerateParameterAttribute(w, arg.LongName, null, arg.HelpText, parent, parentType, parentValue);
                GenerateParameter(w, "string", arg.LongName + argSuffix);
            }
            else
            {
                GenerateParameterAttribute(w, arg.LongName, Stringify(arg.DefaultValue), arg.HelpText, parent, parentType, parentValue);
                GenerateParameter(w, GetCSharpTypeName(arg.ItemType), arg.LongName + argSuffix);
            }
        }

        private void GenerateDictionaryEntry(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (IsColumnType(arg))
                GenerateDictionaryEntry(w, "string", arg.LongName + argSuffix);
            else
                GenerateDictionaryEntry(w, GetCSharpTypeName(arg.ItemType), arg.LongName + argSuffix);
        }

        private void GenerateDictionaryEntry(IndentingTextWriter w, string type, string name)
        {
            w.WriteLine("{{ \"{0}\", {0} }},", name);
        }

        private void GenerateImplCall(IndentingTextWriter w, CmdParser.ArgInfo.Arg arg, string parent, string parentType, string parentValue, string argSuffix)
        {
            if (Exclude.Contains(arg.LongName))
                return;

            if (IsColumnType(arg) || IsStringColumnType(arg))
            {
                string name = arg.LongName + argSuffix;
                if (arg.IsCollection)
                    w.WriteLine("builder.{0} = ((string)parameters[\"{1}\"]).Split('|');", Capitalize(name), name);
                else
                    w.WriteLine("builder.{0} = (string)parameters[\"{1}\"];", Capitalize(name), name);
            }
            else
                GenerateImplCall(w, GetCSharpTypeName(arg.ItemType), arg.LongName + argSuffix);
        }

        private void GenerateImplCall(IndentingTextWriter w, string type, string name)
        {
            w.WriteLine("builder.{0} = ({1})parameters[\"{2}\"];", Capitalize(name), type, name);
        }

        protected override void GenerateParameter(IndentingTextWriter w, string type, string name)
        {
            w.WriteLine("{0} {1},", type, name);
        }

        private void GenerateParameterAttribute(IndentingTextWriter w, string displayName, object defaultValue, string description,
            string parent = null, string parentType = null, string parentValue = null)
        {
            w.WriteLine("[Help(Display = @\"{0}\", ToolTip = \"{1}\")]", PrettyPrintDisplayName(displayName), description);
            if (parent != null)
                w.WriteLine("[Relevancy(Key = \"{0}\", Values = new object[] {{ {1}.{2} }})]", parent, parentType, parentValue);
            if (defaultValue != null)
                w.WriteLine("[Domain(DefaultValue = {0})]", defaultValue);
            w.WriteLine("[ModuleParameter]");
        }

        private string PrettyPrintDisplayName(string displayName)
        {
            var sb = new StringBuilder();
            bool first = true;
            foreach (var c in Capitalize(displayName))
            {
                if (!first && c >= 'A' && c <= 'Z')
                    sb.Append(' ');
                first = false;
                sb.Append(c);
            }
            return sb.ToString();
        }
    }
}