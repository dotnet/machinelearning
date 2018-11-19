// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(SavePfaCommand.Summary, typeof(SavePfaCommand), typeof(SavePfaCommand.Arguments), typeof(SignatureCommand),
    "Save PFA", "SavePfa", DocName = "command/SavePfa.md")]

namespace Microsoft.ML.Runtime.Model.Pfa
{
    internal sealed class SavePfaCommand : DataCommand.ImplBase<SavePfaCommand.Arguments>
    {
        public const string Summary = "Given a data model, write out the corresponding PFA.";
        public const string LoadName = "SavePfa";

        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output PFA too. Leave empty for stdout.", SortOrder = 1)]
            public string Pfa;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output PFA program. By default this will be the extension-less name ", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            // This option is a bit problematic, since you can only really set something if it has the same exact type.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should allow set operations.", ShortName = "set", SortOrder = 3, Hide = true)]
            public bool AllowSet;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma delimited list of input column names to drop", ShortName = "idrop", SortOrder = 4)]
            public string InputsToDrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma delimited list of output column names to drop", ShortName = "odrop", SortOrder = 5)]
            public string OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the inputs should also map to the outputs.", ShortName = "input", SortOrder = 6)]
            public bool KeepInput;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 7)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Format option for the JSON exporter.", ShortName = "format", SortOrder = 8)]
            public Formatting Formatting = Formatting.Indented;
        }

        private readonly string _outputModelPath;
        private readonly string _name;
        private readonly bool _allowSet;
        private readonly bool _keepInput;
        private readonly bool? _loadPredictor;
        private readonly HashSet<string> _inputsToDrop;
        private readonly HashSet<string> _outputsToDrop;
        private readonly Formatting _formatting;

        public SavePfaCommand(IHostEnvironment env, Arguments args)
                : base(env, args, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.Pfa, nameof(args.Pfa));
            _outputModelPath = string.IsNullOrWhiteSpace(args.Pfa) ? null : args.Pfa;
            if (args.Name == null && _outputModelPath != null)
                _name = Path.GetFileNameWithoutExtension(_outputModelPath);
            else if (!string.IsNullOrWhiteSpace(args.Name))
                _name = args.Name;
            _allowSet = args.AllowSet;
            _keepInput = args.KeepInput;
            _loadPredictor = args.LoadPredictor;

            _inputsToDrop = CreateDropMap(args.InputsToDrop);
            _outputsToDrop = CreateDropMap(args.OutputsToDrop);

            Host.CheckUserArg(Enum.IsDefined(typeof(Formatting), args.Formatting), nameof(args.Formatting), "Undefined value");
            _formatting = args.Formatting;
        }

        private static HashSet<string> CreateDropMap(string toDrop)
        {
            if (string.IsNullOrWhiteSpace(toDrop))
                return new HashSet<string>();
            return new HashSet<string>(toDrop.Split(','));
        }

        public override void Run()
        {
            using (var ch = Host.Start("Run"))
            {
                Run(ch);
            }
        }

        private void GetPipe(IChannel ch, IDataView end, out IDataView source, out IDataView trueEnd, out LinkedList<ITransformCanSavePfa> transforms)
        {
            Host.AssertValue(end);
            source = trueEnd = (end as CompositeDataLoader)?.View ?? end;
            IDataTransform transform = source as IDataTransform;
            transforms = new LinkedList<ITransformCanSavePfa>();
            while (transform != null)
            {
                ITransformCanSavePfa pfaTransform = transform as ITransformCanSavePfa;
                if (pfaTransform == null || !pfaTransform.CanSavePfa)
                {
                    ch.Warning("Had to stop walkback of pipeline at {0} since it cannot save itself as PFA", transform.GetType().Name);
                    return;
                }
                transforms.AddFirst(pfaTransform);
                transform = (source = transform.Source) as IDataTransform;
            }
            Host.AssertValue(source);
        }

        private void Run(IChannel ch)
        {
            IDataLoader loader;
            IPredictor rawPred;
            RoleMappedSchema trainSchema;

            if (string.IsNullOrEmpty(Args.InputModelFile))
            {
                loader = CreateLoader();
                rawPred = null;
                trainSchema = null;
                Host.CheckUserArg(Args.LoadPredictor != true, nameof(Args.LoadPredictor),
                    "Cannot be set to true unless " + nameof(Args.InputModelFile) + " is also specifified.");
            }
            else
                LoadModelObjects(ch, _loadPredictor, out rawPred, true, out trainSchema, out loader);

            // Get the transform chain.
            IDataView source;
            IDataView end;
            LinkedList<ITransformCanSavePfa> transforms;
            GetPipe(ch, loader, out source, out end, out transforms);
            Host.Assert(transforms.Count == 0 || transforms.Last.Value == end);

            // If we have a predictor, try to get the scorer for it.
            if (rawPred != null)
            {
                RoleMappedData data;
                if (trainSchema != null)
                    data = new RoleMappedData(end, trainSchema.GetColumnRoleNames());
                else
                {
                    // We had a predictor, but no roles stored in the model. Just suppose
                    // default column names are OK, if present.
                    data = new RoleMappedData(end, DefaultColumnNames.Label,
                        DefaultColumnNames.Features, DefaultColumnNames.GroupId, DefaultColumnNames.Weight, DefaultColumnNames.Name, opt: true);
                }

                var scorePipe = ScoreUtils.GetScorer(rawPred, data, Host, trainSchema);
                var scorePfa = scorePipe as ITransformCanSavePfa;
                if (scorePfa?.CanSavePfa == true)
                {
                    Host.Assert(scorePipe.Source == end);
                    end = scorePipe;
                    transforms.AddLast(scorePfa);
                }
                else
                {
                    Contracts.CheckUserArg(_loadPredictor != true,
                        nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but we do not know how to save it as PFA.");
                    ch.Warning("We do not know how to save the predictor as PFA. Ignoring.");
                }
            }
            else
            {
                Contracts.CheckUserArg(_loadPredictor != true,
                    nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but one was not present.");
            }

            var ctx = new BoundPfaContext(Host, source.Schema, _inputsToDrop, allowSet: _allowSet);
            foreach (var trans in transforms)
            {
                Host.Assert(trans.CanSavePfa);
                trans.SaveAsPfa(ctx);
            }

            var toExport = new List<string>();
            for (int i = 0; i < end.Schema.ColumnCount; ++i)
            {
                if (end.Schema.IsHidden(i))
                    continue;
                var name = end.Schema.GetColumnName(i);
                if (_outputsToDrop.Contains(name))
                    continue;
                if (!ctx.IsInput(name) || _keepInput)
                    toExport.Add(name);
            }
            JObject pfaDoc = ctx.Finalize(end.Schema, toExport.ToArray());
            if (_name != null)
                pfaDoc["name"] = _name;

            if (_outputModelPath == null)
                ch.Info(MessageSensitivity.Schema, pfaDoc.ToString(_formatting));
            else
            {
                using (var file = Host.CreateOutputFile(_outputModelPath))
                using (var stream = file.CreateWriteStream())
                using (var writer = new StreamWriter(stream))
                    writer.Write(pfaDoc.ToString(_formatting));
            }

            if (!string.IsNullOrWhiteSpace(Args.OutputModelFile))
            {
                ch.Trace("Saving the data pipe");
                // Should probably include "end"?
                SaveLoader(loader, Args.OutputModelFile);
            }
        }
    }
}
