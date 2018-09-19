// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints.CodeGen;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Tools;

[assembly: LoadableClass(typeof(ModuleGenerator), typeof(ModuleGenerator.Arguments), typeof(SignatureModuleGenerator),
    "Module generator", "ModuleGenerator", "Module")]

namespace Microsoft.ML.Runtime.EntryPoints.CodeGen
{
    public class ModuleGenerator : IGenerator
    {
        private readonly string _modulePrefix;
        private readonly bool _generateModule;
        private readonly bool _generateModuleInstance;
        private readonly string _regenerate;
        private readonly string _moduleId;
        private readonly string _moduleName;
        private readonly string _moduleOwner;
        private readonly string _moduleVersion;
        private readonly string _moduleState;
        private readonly string _moduleType;
        private readonly string _moduleDeterminism;
        private readonly string _moduleCategory;
        private readonly HashSet<string> _exclude;
        private readonly HashSet<string> _namespaces;
        private readonly IHost _host;

        public class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The prefix for the generated AML module", ShortName = "prefix")]
            public string ModulePrefix;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to generate the module entry point", ShortName = "module")]
            public bool GenerateModule = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to generate the module entry point as Module Instance", ShortName = "moduleinst")]
            public bool GenerateModuleInstance = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module id", ShortName = "id")]
            public string ModuleId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module name", ShortName = "name")]
            public string ModuleName;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module owner", ShortName = "owner")]
            public string ModuleOwner = "Microsoft";

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module version", ShortName = "version")]
            public string ModuleVersion = "0.0.0.0";

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module state", ShortName = "state")]
            public string ModuleState = "Alpha";

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module type", ShortName = "type")]
            public string ModuleType = "User";

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module determinism", ShortName = "determinism")]
            public string ModuleDeterminism = "Deterministic";

            [Argument(ArgumentType.AtMostOnce, HelpText = "The module category", ShortName = "category")]
            public string ModuleCategory = "V2\\Transforms";

            [Argument(ArgumentType.Multiple, HelpText = "Arguments to exclude", ShortName = "excl")]
            public string[] Exclude;

            [Argument(ArgumentType.Multiple, HelpText = "Extra namespaces", ShortName = "using")]
            public string[] Namespaces;
        }

        public ModuleGenerator(IHostEnvironment env, Arguments args, string regenerate)
        {
            Contracts.AssertValue(args, "args");
            Contracts.AssertNonEmpty(regenerate, "regenerate");

            _host = env.Register("ModuleGenerator");
            _modulePrefix = args.ModulePrefix;
            _regenerate = regenerate;
            _generateModule = args.GenerateModule;
            _generateModuleInstance = args.GenerateModuleInstance;
            _moduleId = string.IsNullOrEmpty(args.ModuleId) ? null : args.ModuleId;
            _moduleName = string.IsNullOrEmpty(args.ModuleName) ? null : args.ModuleName;
            _moduleOwner = args.ModuleOwner;
            _moduleVersion = args.ModuleVersion;
            _moduleState = args.ModuleState;
            _moduleType = args.ModuleType;
            _moduleDeterminism = args.ModuleDeterminism;
            _moduleCategory = args.ModuleCategory;
            _exclude = new HashSet<string>();
            foreach (var excl in args.Exclude)
            {
                if (!string.IsNullOrEmpty(excl))
                    _exclude.Add(excl);
            }
            _namespaces = new HashSet<string>();
            foreach (var ns in args.Namespaces)
            {
                if (!string.IsNullOrEmpty(ns))
                    _namespaces.Add(ns);
            }
        }

        private static volatile Dictionary<Type, GeneratorBase> _entryPointGeneratorMapping;
        private static volatile Dictionary<Type, GeneratorBase> _moduleInstanceEntryPointGeneratorMapping;
        private static volatile Dictionary<Type, GeneratorBase> _implGeneratorMapping;

        private static Dictionary<Type, GeneratorBase> EntryPointGeneratorMapping
        {
            get
            {
                if (_entryPointGeneratorMapping == null)
                {
                    var tmp = new Dictionary<Type, GeneratorBase>();
                    tmp.Add(typeof(SignatureTrainer), new LearnerEntryPointGenerator());
                    tmp.Add(typeof(SignatureDataTransform), new TransformEntryPointGenerator());
                    Interlocked.CompareExchange(ref _entryPointGeneratorMapping, tmp, null);
                }
                return _entryPointGeneratorMapping;
            }
        }

        private static Dictionary<Type, GeneratorBase> ModuleInstanceEntryPointGeneratorMapping
        {
            get
            {
                if (_moduleInstanceEntryPointGeneratorMapping == null)
                {
                    var tmp = new Dictionary<Type, GeneratorBase>();
                    tmp.Add(typeof(SignatureDataTransform), new TransformModuleInstanceEntryPointGenerator());
                    Interlocked.CompareExchange(ref _moduleInstanceEntryPointGeneratorMapping, tmp, null);
                }
                return _moduleInstanceEntryPointGeneratorMapping;
            }
        }

        private static Dictionary<Type, GeneratorBase> ImplGeneratorMapping
        {
            get
            {
                if (_implGeneratorMapping == null)
                {
                    var tmp = new Dictionary<Type, GeneratorBase>();
                    tmp.Add(typeof(SignatureTrainer), new LearnerImplGenerator());
                    tmp.Add(typeof(SignatureDataTransform), new TransformImplGenerator());
                    Interlocked.CompareExchange(ref _implGeneratorMapping, tmp, null);
                }
                return _implGeneratorMapping;
            }
        }

        public void Generate(IEnumerable<HelpCommand.Component> infos)
        {
            using (var ch = _host.Start("Generate"))
            {
                foreach (var info in infos.Select(c => c.Info))
                    GenerateFile(ch, info);
                ch.Done();
            }
        }

        private void GenerateFile(IChannel ch, ComponentCatalog.LoadableClassInfo info)
        {
            _host.AssertValue(ch);
            ch.AssertValue(info);

            string name = info.LoadNames[0];
            if (!info.IsOfType(typeof(SignatureTrainer)) && !info.IsOfType(typeof(SignatureDataTransform)))
            {
                ch.Warning("No generator available for {0}.", name);
                return;
            }

            if (info.Constructor == null && info.CreateMethod == null)
            {
                ch.Warning("No construction method available for {0}.", name);
                return;
            }

            if (_generateModule)
            {
                var entryPointFile = _modulePrefix + name + "EntryPoint.cs";
                if (_generateModuleInstance)
                    GenerateFile(info, entryPointFile, ModuleInstanceEntryPointGeneratorMapping);
                else
                    GenerateFile(info, entryPointFile, EntryPointGeneratorMapping);
            }

            var implFile = _modulePrefix + name + ".cs";
            GenerateFile(info, implFile, ImplGeneratorMapping);
        }

        private void GenerateFile(ComponentCatalog.LoadableClassInfo info, string filename, Dictionary<Type, GeneratorBase> mapping)
        {
            using (var sw = new StreamWriter(filename))
            {
                var writer = IndentingTextWriter.Wrap(sw, "    ");
                foreach (var kvp in mapping)
                {
                    if (info.IsOfType(kvp.Key))
                    {
                        kvp.Value.Generate(writer, _modulePrefix, _regenerate, info,
                            _moduleId ?? Guid.NewGuid().ToString(), _moduleName, _moduleOwner, _moduleVersion, _moduleState,
                            _moduleType, _moduleDeterminism, _moduleCategory, _exclude, _namespaces);
                        break;
                    }
                }
            }
        }
    }

    internal static class GeneratorUtils
    {
        public static bool IsOfType(this ComponentCatalog.LoadableClassInfo component, Type type)
        {
            return component.SignatureTypes != null && component.SignatureTypes.Contains(type);
        }
    }
}
