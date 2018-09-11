// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Tools;

[assembly: LoadableClass(HelpCommand.Summary, typeof(HelpCommand), typeof(HelpCommand.Arguments), typeof(SignatureCommand),
    "MAML Help Command", "Help", "?")]

[assembly: LoadableClass(typeof(XmlGenerator), typeof(XmlGenerator.Arguments), typeof(SignatureModuleGenerator),
    "Xml generator", "XmlGenerator", "Xml")]

namespace Microsoft.ML.Runtime.Tools
{
    public interface IGenerator
    {
        void Generate(IEnumerable<HelpCommand.Component> infos);
    }

    public delegate void SignatureModuleGenerator(string regenerate);

    public sealed class HelpCommand : ICommand
    {
        public sealed class Arguments
        {
#pragma warning disable 649 // never assigned
            [DefaultArgument(ArgumentType.AtMostOnce, HelpText = "The component name to get help for")]
            public string Component;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The kind of component to look for", ShortName = "kind")]
            public string Kind;

            [Argument(ArgumentType.AtMostOnce, HelpText = "List the component kinds", ShortName = "list")]
            public bool ListKinds;

            [Argument(ArgumentType.AtMostOnce, ShortName = "all", Hide = true)]
            public bool AllComponents;

            // extra DLLs for dynamic loading
            [Argument(ArgumentType.Multiple, HelpText = "Extra DLLs", ShortName = "dll")]
            public string[] ExtraAssemblies;

            [Argument(ArgumentType.LastOccurenceWins, Hide = true, SignatureType = typeof(SignatureModuleGenerator))]
            public IComponentFactory<string, IGenerator> Generator;
#pragma warning restore 649 // never assigned
        }

        internal const string Summary = "Prints command line help.";

        private readonly IHostEnvironment _env;
        private readonly string _component;
        private readonly string _kind;
        private readonly bool _listKinds;
        private readonly bool _allComponents;
        private readonly string[] _extraAssemblies;
        private readonly IGenerator _generator;

        // REVIEW: Need to change the help command to use the provided host environment for output,
        // instead of assuming the console.
        public HelpCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            _env = env;
            _component = args.Component;
            if (string.IsNullOrWhiteSpace(_component))
                _component = null;

            _kind = args.Kind;
            if (string.IsNullOrWhiteSpace(_kind))
                _kind = null;

            _listKinds = args.ListKinds;
            _allComponents = args.AllComponents;

            _extraAssemblies = args.ExtraAssemblies;

            if (args.Generator != null)
            {
                _generator = args.Generator.CreateComponent(_env, "maml.exe ? " + CmdParser.GetSettings(env, args, new Arguments()));
            }
        }

        public void Run()
        {
            Run(null);
        }

        public void Run(int? columns)
        {
            ComponentCatalog.CacheClassesExtra(_extraAssemblies);

            using (var ch = _env.Start("Help"))
            using (var sw = new StringWriter(CultureInfo.InvariantCulture))
            using (var writer = IndentingTextWriter.Wrap(sw))
            {
                if (_listKinds)
                {
                    if (_component != null)
                        writer.WriteLine("Listing component kinds so ignoring specified component");
                    else if (_kind != null)
                        writer.WriteLine("Listing component kinds so ignoring specified kind");
                    ListKinds(writer);
                }
                else if (_component != null)
                    ShowHelp(writer, columns);
                else if (_allComponents)
                    ShowAllHelp(writer, columns);
                else
                    ShowComponents(writer);

                ch.Info(sw.ToString());
                ch.Done();
            }
        }

        private void ShowHelp(IndentingTextWriter writer, int? columns = null)
        {
            _env.AssertValue(_component);

            string name = _component.Trim();

            string sig = _kind?.ToLowerInvariant();

            // Note that we don't check IsHidden here. The current policy is when IsHidden is true, we don't
            // show the item in "list all" functionality, but will still show help when explicitly requested.

            var infos = ComponentCatalog.FindLoadableClasses(name)
                .OrderBy(x => ComponentCatalog.SignatureToString(x.SignatureTypes[0]).ToLowerInvariant());
            var kinds = new StringBuilder();
            var components = new List<Component>();
            foreach (var info in infos)
            {
                _env.AssertValue(info.SignatureTypes);
                kinds.Clear();
                bool foundSig = false;
                foreach (var signature in info.SignatureTypes)
                {
                    _env.Assert(signature.BaseType == typeof(MulticastDelegate));

                    string kind;
                    if (signature == typeof(SignatureDefault))
                    {
                        kind = "Component";
                        if (sig == null || "default".StartsWithInvariantCulture(sig))
                            foundSig = true;
                    }
                    else
                    {
                        kind = ComponentCatalog.SignatureToString(signature);
                        if (sig == null || kind.StartsWithInvariantCultureIgnoreCase(sig))
                            foundSig = true;
                    }

                    if (kinds.Length > 0)
                        kinds.Append(", ");
                    kinds.Append(kind);
                }
                if (foundSig)
                {
                    string kindsStr = kinds.ToString();
                    var args = info.CreateArguments();

                    ShowUsage(writer, kindsStr, info.Summary, info.LoadNames[0], info.LoadNames, args, columns);
                    components.Add(new Component(kindsStr, info, args));
                }
            }

            if (components.Count == 0)
                writer.WriteLine("Unknown component: '{0}'", name);
            else
                Serialize(components);
        }

        private void ShowAllHelp(IndentingTextWriter writer, int? columns = null)
        {
            string sig = _kind?.ToLowerInvariant();

            var infos = ComponentCatalog.GetAllClasses()
                .OrderBy(info => info.LoadNames[0].ToLowerInvariant())
                .ThenBy(info => ComponentCatalog.SignatureToString(info.SignatureTypes[0]).ToLowerInvariant());
            var components = new List<Component>();
            foreach (var info in infos)
            {
                // REVIEW: We should only be printing the usage once, not for every signature.
                _env.AssertValue(info.SignatureTypes);
                foreach (var signature in info.SignatureTypes)
                {
                    _env.Assert(signature.BaseType == typeof(MulticastDelegate));

                    string kind = ComponentCatalog.SignatureToString(signature);
                    if (sig != null && !kind.StartsWithInvariantCultureIgnoreCase(sig))
                        continue;

                    // Don't show classes that have no arguments.
                    var args = info.CreateArguments();
                    if (args == null)
                        continue;

                    ShowUsage(writer, kind, info.Summary, info.LoadNames[0], info.LoadNames, args, columns);
                    components.Add(new Component(kind, info, args));
                }
            }

            if (components.Count > 0)
                Serialize(components);
        }

        private void ShowUsage(IndentingTextWriter writer, string kind, string summary, string loadName,
            IReadOnlyList<string> loadNames, object args, int? columns)
        {
            _env.Assert(loadName == loadNames[0]);

            writer.WriteLine("Help for {0}: '{1}'", kind, loadName);
            using (writer.Nest())
                ShowAliases(writer, loadNames);

            writer.WriteLine();
            ShowFormattedSummary(writer, summary, columns);

            if (args == null)
            {
                writer.WriteLine("Component '{0}' is not configurable", loadName);
                writer.WriteLine();
            }
            else
                writer.WriteLine(CmdParser.ArgumentsUsage(_env, args.GetType(), args, false, columns));
        }

        private void ShowComponents(IndentingTextWriter writer)
        {
            Type typeSig;
            Type typeRes;
            string kind;

            if (_kind == null)
            {
                // Show commands.
                typeSig = typeof(SignatureCommand);
                typeRes = typeof(ICommand);
                kind = "Command";
                writer.WriteLine("Available commands:");
            }
            else
            {
                kind = _kind.ToLowerInvariant();
                var sigs = ComponentCatalog.GetAllSignatureTypes();
                typeSig = sigs.FirstOrDefault(t => ComponentCatalog.SignatureToString(t).ToLowerInvariant() == kind);
                if (typeSig == null)
                {
                    typeSig = sigs.FirstOrDefault(t => ComponentCatalog.SignatureToString(t).StartsWithInvariantCultureIgnoreCase(kind));
                    if (typeSig == null)
                    {
                        writer.WriteLine("Couldn't find kind '{0}'", kind);
                        ListKinds(writer);
                        return;
                    }
                }
                typeRes = typeof(object);
                writer.WriteLine("Available components for kind '{0}':", ComponentCatalog.SignatureToString(typeSig));
            }

            var infos = ComponentCatalog.GetAllDerivedClasses(typeRes, typeSig)
                .Where(x => !x.IsHidden)
                .OrderBy(x => x.LoadNames[0].ToLowerInvariant());
            using (writer.Nest())
            {
                var components = new List<Component>();
                foreach (var info in infos)
                {
                    _env.Assert(info.LoadNames.Count > 0);

                    writer.Write("{0}", info.LoadNames[0]);
                    if (!string.IsNullOrWhiteSpace(info.UserName))
                        writer.Write(": {0}", info.UserName);
                    writer.WriteLine();

                    using (writer.Nest())
                        ShowAliases(writer, info.LoadNames);
                    components.Add(new Component(kind, info, info.CreateArguments()));
                }

                if (components.Count > 0)
                    Serialize(components);
            }
        }

        private void Serialize(List<Component> components)
        {
            _env.AssertValue(components);

            if (_generator != null)
                GenerateModule(components);
        }

        private void ShowAliases(IndentingTextWriter writer, IReadOnlyList<string> names)
        {
            if (names.Count <= 1)
                return;

            string pre = "Aliases: ";
            for (int i = 1; i < names.Count; i++)
            {
                writer.Write(pre);
                pre = ", ";
                writer.Write(names[i]);
            }
            writer.WriteLine();
        }

        private void ListKinds(IndentingTextWriter writer)
        {
            var sigs = ComponentCatalog.GetAllSignatureTypes()
                .Select(ComponentCatalog.SignatureToString)
                .OrderBy(x => x);

            writer.WriteLine("Available component kinds:");
            using (writer.Nest())
            {
                foreach (var sig in sigs)
                    writer.WriteLine(sig);
            }
        }

        private void ShowFormattedSummary(IndentingTextWriter writer, string summary, int? columns)
        {
            _env.AssertValue(writer);

            if (string.IsNullOrWhiteSpace(summary))
                return;

            // REVIEW: should we replace consecutive spaces with a single space as a preprocessing step?
            int screenWidth = (columns ?? CmdParser.GetConsoleWindowWidth()) - 1;

            // GetConsoleWindowWidth returns 0 if command redirection operator is used
            if (screenWidth <= 0)
                screenWidth = 80;

            const int indentLen = 3;
            string indent = new string(' ', indentLen);
            var builder = new StringBuilder();

            // REVIEW: is using StringSplitOptions.RemoveEmptyEntries the right thing to do here?
            var blocks = summary.Split(new[] { "\n", "\r" }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < blocks.Length; i++)
                AppendFormattedText(builder, blocks[i], indent, screenWidth);

            writer.WriteLine("Summary:");
            writer.WriteLine(builder);
        }

        private void AppendFormattedText(StringBuilder builder, string text, string indent, int screenWidth)
        {
            _env.AssertValue(builder);
            _env.AssertNonEmpty(text);
            _env.AssertNonEmpty(indent);
            _env.Assert(screenWidth > 0);

            int textIdx = 0;
            while (textIdx < text.Length)
            {
                int screenLeft = screenWidth - indent.Length;
                int summaryLeft = text.Length - textIdx;
                if (summaryLeft <= screenLeft)
                {
                    builder.Append(indent).Append(text, textIdx, summaryLeft).AppendLine();
                    break;
                }

                int spaceIdx = text.LastIndexOf(' ', screenLeft + textIdx, screenLeft);
                if (spaceIdx < 0)
                {
                    // Print to the first space.
                    int startIdx = screenLeft + textIdx + 1;
                    spaceIdx = text.IndexOf(' ', startIdx, text.Length - startIdx);
                    if (spaceIdx < 0)
                    {
                        // Print to the end.
                        builder.Append(indent).Append(text, textIdx, summaryLeft).AppendLine();
                        break;
                    }
                }

                int appendCount = spaceIdx - textIdx;
                builder.Append(indent).Append(text, textIdx, appendCount).AppendLine();
                textIdx += appendCount + 1;
            }
        }

        public struct Component
        {
            public readonly string Kind;
            public readonly ComponentCatalog.LoadableClassInfo Info;
            public readonly object Args;

            public Component(string kind, ComponentCatalog.LoadableClassInfo info, object args)
            {
                Contracts.AssertNonEmpty(kind);
                Contracts.AssertValue(info);
                Contracts.AssertValueOrNull(args);

                Kind = kind;
                Info = info;
                Args = args;
            }
        }

        private void GenerateModule(List<Component> components)
        {
            Contracts.AssertValue(components);
            _generator.Generate(components);
        }
    }

    public sealed class XmlGenerator : IGenerator
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The path of the XML documentation file",
                ShortName = "xml", Hide = true)]
            public string XmlFilename;
        }

        private readonly string _xmlFilename;
        private readonly IHost _host;

        public XmlGenerator(IHostEnvironment env, Arguments args, string regenerate)
        {
            Contracts.CheckValue(env, nameof(env));
            env.AssertValue(args, nameof(args));
            env.AssertNonEmpty(regenerate, nameof(regenerate));

            _xmlFilename = args.XmlFilename;
            if (!string.IsNullOrWhiteSpace(_xmlFilename))
                Utils.CheckOptionalUserDirectory(_xmlFilename, nameof(args.XmlFilename));
            else
                _xmlFilename = null;
            _host = env.Register("XML Generator");
        }

        public void Generate(IEnumerable<HelpCommand.Component> infos)
        {
            if (_xmlFilename == null)
                return;
            using (var ch = _host.Start("Generating XML"))
            {
                var content = new XElement("Components",
                    from c in infos
                    where !string.IsNullOrWhiteSpace(c.Info.UserName)
                    select new XElement("Component",
                        new XAttribute("Kind", c.Kind),
                        new XElement("Name", c.Info.UserName),
                        string.IsNullOrWhiteSpace(c.Info.Summary) ? null : new XElement("Summary", c.Info.Summary),
                        new XElement("LoadNames",
                            from l in c.Info.LoadNames
                            select new XElement("LoadName", l)),
                        new XElement("Type", c.Info.Type.ToString()),
                        new XElement("SignatureTypes",
                            from s in c.Info.SignatureTypes
                            select new XElement("SignatureType", s.ToString())),
                        c.Args == null
                            ? null
                            : new XElement("Arguments",
                                from a in CmdParser.GetArgInfo(c.Args.GetType(), c.Args).Args
                                select new XElement("Argument",
                                    new XElement("LongName", a.LongName),
                                    a.ShortNames == null
                                        ? null
                                        : new XElement("ShortNames",
                                            from sn in a.ShortNames
                                            select new XElement("ShortName", sn)),
                                    new XElement("HelpText", a.HelpText),
                                    CreateDefaultValueElement(ch, c.Kind, a)))));
                File.WriteAllText(_xmlFilename, content.ToString());
                ch.Done();
            }
        }

        private XElement CreateDefaultValueElement(IChannel ch, string name, CmdParser.ArgInfo.Arg a)
        {
            if (a.DefaultValue == null)
                return null;
            if (a.DefaultValue is char)
            {
                char val = (char)a.DefaultValue;
                if (!char.IsLetterOrDigit(val) && !char.IsPunctuation(val) && !char.IsSymbol(val))
                {
                    ch.Warning("Unprintable default value for component {0}, character valued field {1}: {2}", name,
                        a.LongName, Convert.ToUInt16(val).ToString("X4", CultureInfo.InvariantCulture));

                    return null;
                }
            }
            return new XElement("DefaultValue", a.DefaultValue);
        }
    }
}
