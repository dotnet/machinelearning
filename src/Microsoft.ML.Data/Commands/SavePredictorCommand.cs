// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Tools;

// REVIEW: Fix these namespaces.
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(SavePredictorCommand.Summary, typeof(SavePredictorCommand), typeof(SavePredictorCommand.Arguments), typeof(SignatureCommand),
    "Save Predictor As", "SavePredictorAs", "SavePredictor", "SaveAs", "SaveModel")]

namespace Microsoft.ML.Runtime.Tools
{
    public sealed class SavePredictorCommand : ICommand
    {
        public sealed class Arguments
        {
#pragma warning disable 649 // never assigned
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model file containing the predictor", ShortName = "in")]
            public string InputModelFile;

            // output a textual summary of the model (may not be complete information to recreate the model)
            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save model summary", ShortName = "sum")]
            public string SummaryFile;

            // Output the model in human-readable text format
            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save in text format", ShortName = "text")]
            public string TextFile;

            // Output the model in Bing INI format
            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save in INI format", ShortName = "ini")]
            public string IniFile;

            // Output the model as C++/C# code
            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save in C++ code", ShortName = "code")]
            public string CodeFile;

            // Output the model in binary format (for fast loading)
            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save in binary format", ShortName = "bin")]
            public string BinaryFile;
#pragma warning restore 649 // never assigned
        }

        internal const string Summary = "Given a TLC model file with a predictor, we can output this same predictor in multiple export formats.";

        private readonly Arguments _args;
        private readonly IHost _host;

        public SavePredictorCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("SavePredictorCommand");
            _host.CheckValue(args, nameof(args));
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(args.InputModelFile), nameof(args.InputModelFile), "Must specify input model file");

            _args = args;
            CheckOutputDirectories();
        }

        private void CheckOutputDirectories()
        {
            Utils.CheckOptionalUserDirectory(_args.BinaryFile, nameof(_args.BinaryFile));
            Utils.CheckOptionalUserDirectory(_args.CodeFile, nameof(_args.CodeFile));
            Utils.CheckOptionalUserDirectory(_args.IniFile, nameof(_args.IniFile));
            Utils.CheckOptionalUserDirectory(_args.SummaryFile, nameof(_args.SummaryFile));
            Utils.CheckOptionalUserDirectory(_args.TextFile, nameof(_args.TextFile));
        }

        // REVIEW: Use the _env to emit messages instead of console.
        public void Run()
        {
            using (var file = _host.OpenInputFile(_args.InputModelFile))
            using (var strm = file.OpenReadStream())
            using (var binFile = CreateFile(_args.BinaryFile))
            using (var binStrm = CreateStrm(binFile))
            using (var sumFile = CreateFile(_args.SummaryFile))
            using (var sumStrm = CreateStrm(sumFile))
            using (var txtFile = CreateFile(_args.TextFile))
            using (var txtStrm = CreateStrm(txtFile))
            using (var iniFile = CreateFile(_args.IniFile))
            using (var iniStrm = CreateStrm(iniFile))
            using (var codFile = CreateFile(_args.CodeFile))
            using (var codStrm = CreateStrm(codFile))
                SavePredictorUtils.SavePredictor(_host, strm, binStrm, sumStrm, txtStrm, iniStrm, codStrm);
        }

        /// <summary>
        /// Create a file handle from path if it was not empty.
        /// </summary>
        private IFileHandle CreateFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return null;
            return _host.CreateOutputFile(path);
        }

        /// <summary>
        /// Create the write stream from the file, if not null.
        /// </summary>
        private Stream CreateStrm(IFileHandle file)
        {
            if (file == null)
                return null;
            return file.CreateWriteStream();
        }
    }

    public static class SavePredictorUtils
    {
        public static void SavePredictor(IHostEnvironment env, Stream modelStream, Stream binaryModelStream = null, Stream summaryModelStream = null,
            Stream textModelStream = null, Stream iniModelStream = null, Stream codeModelStream = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));

            bool needNames = codeModelStream != null
                || iniModelStream != null
                || summaryModelStream != null
                || textModelStream != null;

            IPredictor predictor;
            RoleMappedSchema schema;
            LoadModel(env, modelStream, needNames, out predictor, out schema);
            using (var ch = env.Start("Saving predictor"))
            {
                SavePredictor(ch, predictor, schema, binaryModelStream, summaryModelStream, textModelStream,
                    iniModelStream, codeModelStream);
            }
        }

        public static void SavePredictor(IChannel ch, IPredictor predictor, RoleMappedSchema schema,
            Stream binaryModelStream = null, Stream summaryModelStream = null, Stream textModelStream = null,
            Stream iniModelStream = null, Stream codeModelStream = null)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValue(schema, nameof(schema));

            int count = 0;
            if (binaryModelStream != null)
            {
                ch.Info("Saving predictor as binary");
                using (var writer = new BinaryWriter(binaryModelStream, Encoding.UTF8, true))
                    PredictorUtils.SaveBinary(ch, predictor, writer);
                count++;
            }

            ch.CheckValue(schema, nameof(schema));

            if (summaryModelStream != null)
            {
                ch.Info("Saving predictor summary");

                using (StreamWriter writer = Utils.OpenWriter(summaryModelStream))
                    PredictorUtils.SaveSummary(ch, predictor, schema, writer);
                count++;
            }

            if (textModelStream != null)
            {
                ch.Info("Saving predictor as text");
                using (StreamWriter writer = Utils.OpenWriter(textModelStream))
                    PredictorUtils.SaveText(ch, predictor, schema, writer);
                count++;
            }

            if (iniModelStream != null)
            {
                ch.Info("Saving predictor as ini");
                using (StreamWriter writer = Utils.OpenWriter(iniModelStream))
                {
                    // Test if our predictor implements the more modern INI export interface.
                    // If it does not, use the old utility method.
                    ICanSaveInIniFormat saver = predictor as ICanSaveInIniFormat;
                    if (saver == null)
                        PredictorUtils.SaveIni(ch, predictor, schema, writer);
                    else
                        saver.SaveAsIni(writer, schema);
                }
                count++;
            }

            if (codeModelStream != null)
            {
                ch.Info("Saving predictor as code");
                using (StreamWriter writer = Utils.OpenWriter(codeModelStream))
                    PredictorUtils.SaveCode(ch, predictor, schema, writer);
                count++;
            }

            // Note that we don't check for this case up front so this command can be used to simply
            // check that the predictor is loadable.
            if (count == 0)
                ch.Info("No files saved. Must specify at least one output file.");
        }

        public static void LoadModel(IHostEnvironment env, Stream modelStream, bool loadNames, out IPredictor predictor, out RoleMappedSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(modelStream, nameof(modelStream));

            schema = null;
            using (var rep = RepositoryReader.Open(modelStream, env))
            {
                ModelLoadContext.LoadModel<IPredictor, SignatureLoadModel>(env, out predictor, rep, ModelFileUtils.DirPredictor);

                if (loadNames)
                {
                    var roles = ModelFileUtils.LoadRoleMappingsOrNull(env, rep);
                    if (roles != null)
                    {
                        var emptyView = ModelFileUtils.LoadPipeline(env, rep, new MultiFileSource(null));
                        schema = new RoleMappedSchema(emptyView.Schema, roles, opt: true);
                    }
                    else
                    {
                        FeatureNameCollection names;
                        if (ModelFileUtils.TryLoadFeatureNames(out names, rep))
                            schema = names.Schema;
                    }
                }
            }
        }
    }
}