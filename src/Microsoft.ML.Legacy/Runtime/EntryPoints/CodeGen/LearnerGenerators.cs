// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints.CodeGen
{
    internal class LearnerImplGenerator : ImplGeneratorBase
    {
        protected override void GenerateMethodSignature(IndentedTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("/// <summary>");
            w.WriteLine("/// Creates a {0}", component.LoadNames[0]);
            w.WriteLine("/// </summary>");
            w.WriteLine("/// <returns>A tuple containing learner name and settings.</returns>");
            w.WriteLine("public Tuple<string, string> GetTlcSettings()");
        }

        protected override void GenerateImplBody(IndentedTextWriter w, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("{");
            using (w.Nest())
            {
                w.WriteLine("var args = new {0}();", GetCSharpTypeName(component.ArgType));
                w.WriteLine("var defs = new {0}();", GetCSharpTypeName(component.ArgType));
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                var arguments = argumentInfo.Args.Where(a => !a.IsHidden).ToArray();
                foreach (var arg in arguments)
                    GenerateImplBody(w, arg, "");
                w.WriteLine("return new Tuple<string, string>(\"{0}\", CmdParser.GetSettings(args, defs));", component.LoadNames[0]);
            }
            w.WriteLine("}");
        }
    }

    internal sealed class LearnerEntryPointGenerator : EntryPointGeneratorBase
    {
        protected override void GenerateSummaryComment(IndentedTextWriter w, ComponentCatalog.LoadableClassInfo component)
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
            var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
            var arguments = argumentInfo.Args.Where(a => !a.IsHidden).ToArray();
            foreach (var arg in arguments)
                GenerateSummaryComment(w, arg, "");
        }

        protected override void GenerateReturnComment(IndentedTextWriter w)
        {
            w.WriteLine("/// <returns>An untrained model.</returns>");
        }

        protected override void GenerateModuleAttribute(IndentedTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component, string moduleId)
        {
            if (!string.IsNullOrEmpty(prefix))
                prefix += " ";
            w.WriteLine("[DataLabModule(FriendlyName = \"{0}{1}\",", prefix, component.UserName);
            using (w.Nest())
            {
                var desc = component.Summary ?? component.LoadNames[0];
                w.WriteLine("Description = \"{0}\",", desc.Replace("\n", "\\n").Replace("\r", "\\r"));
                string cat;
                if (component.IsOfType(typeof(SignatureBinaryClassifierTrainer)) ||
                    component.IsOfType(typeof(SignatureMultiClassClassifierTrainer)))
                {
                    cat = @"Machine Learning\Initialize Model\Classification";
                }
                else if (component.IsOfType(typeof(SignatureRegressorTrainer)))
                    cat = @"Machine Learning\Initialize Model\Regression";
                else if (component.IsOfType(typeof(SignatureAnomalyDetectorTrainer)))
                    cat = @"Machine Learning\Initialize Model\Anomaly Detection";
                else
                    cat = @"Machine Learning\Initialize Model";

                w.WriteLine("Category = @\"{0}\",", cat);
                w.WriteLine("IsBlocking = true,");
                w.WriteLine("IsDeterministic = true,");
                w.WriteLine("Version = \"2.0\",");
                w.WriteLine("Owner = \"Microsoft Corporation\",");
                w.WriteLine("FamilyId = \"{{{0}}}\",", Guid.NewGuid().ToString().ToUpperInvariant());
                w.WriteLine("ReleaseState = States.Alpha)]");
            }
        }

        protected override void GenerateOutputPort(IndentedTextWriter w)
        {
            w.WriteLine(
                "[DataLabOutputPort(FriendlyName = \"Untrained model\", DisplayName = \"Untrained model\", Position = 0, DataType = WellKnownDataTypeIds.ITrainerDotNet, Description = \"An untrained model.\")]");
        }

        protected override void GenerateMethodSignature(IndentedTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.Write("public static Tuple<ITrainer> Create{0}{1}(", prefix, component.LoadNames[0]);
            using (w.Nest())
            {
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                var arguments = argumentInfo.Args.Where(a => !a.IsHidden).ToArray();
                var pre = "";
                foreach (var arg in arguments)
                    GenerateMethodSignature(w, arg, null, null, null, ref pre, "");
                w.WriteLine(")");
            }
        }

        protected override void GenerateImplCall(IndentedTextWriter w, string prefix, ComponentCatalog.LoadableClassInfo component)
        {
            w.WriteLine("{");
            using (w.Nest())
            {
                var className = prefix + component.LoadNames[0];
                w.WriteLine("var builder = new {0}();", className);
                var argumentInfo = CmdParser.GetArgInfo(component.ArgType, component.CreateArguments());
                var arguments = argumentInfo.Args.Where(a => !a.IsHidden).ToArray();
                foreach (var arg in arguments)
                    GenerateImplCall(w, arg, "");
                w.WriteLine("var learner = builder.GetTlcSettings();");
                w.WriteLine("return new TlcTrainer(learner.Item1, learner.Item2);");
            }
            w.WriteLine("}");
        }
    }
}