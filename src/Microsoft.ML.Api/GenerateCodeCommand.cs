// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using Microsoft.CSharp;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(GenerateCodeCommand), typeof(GenerateCodeCommand.Arguments), typeof(SignatureCommand),
    "Generate Sample Prediction Code", GenerateCodeCommand.LoadName, "codegen")]

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// Generates the sample prediction code for a given model file, with correct input and output classes.
    /// 
    /// REVIEW: Consider adding support for generating VBuffers instead of arrays, maybe for high dimensionality vectors.
    /// </summary>
    public sealed class GenerateCodeCommand : ICommand
    {
        public const string LoadName = "GenerateSamplePredictionCode";
        private const string CodeTemplatePath = "Microsoft.ML.Api.GeneratedCodeTemplate.csresource";

#pragma warning disable 0649 // The command is internal, suppress a warning about fields never assigned to.
        public sealed class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Input model file", ShortName = "in", IsInputFileName = true)]
            public string InputModelFile;

            [Argument(ArgumentType.Required, HelpText = "File to output generated C# code", ShortName = "cs")]
            public string CSharpOutput;

            /// <summary>
            /// Whether to use the <see cref="VBuffer{T}"/> to represent vector columns (supports sparse vectors).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use the VBuffer to represent vector columns (supports sparse vectors)",
                ShortName = "sparse", SortOrder = 102)]
            public bool SparseVectorDeclaration;

            // REVIEW: currently, it's only used in unit testing to not generate the paths into the test output folder. 
            // However, it might be handy for automation scenarios, so I've added this as a hidden option.
            [Argument(ArgumentType.AtMostOnce, HelpText = "A location of the model file to put into generated file", Hide = true)]
            public string ModelNameOverride;
        }
#pragma warning restore 0649

        private readonly IHost _host;
        private readonly Arguments _args;

        public GenerateCodeCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("GenerateCodeCommand");
            _host.CheckValue(args, nameof(args));
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(args.InputModelFile),
                   nameof(args.InputModelFile), "input model file is required");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(args.CSharpOutput),
                   nameof(args.CSharpOutput), "Output file is required");
            _args = args;
        }

        public void Run()
        {
            string template;
            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(CodeTemplatePath))
            using (var reader = new StreamReader(stream))
                template = reader.ReadToEnd();

            var codeProvider = new CSharpCodeProvider();
            using (var fs = File.OpenRead(_args.InputModelFile))
            {
                var transformPipe = ModelFileUtils.LoadPipeline(_host, fs, new MultiFileSource(null), true);
                var pred = _host.LoadPredictorOrNull(fs);

                IDataView root;
                for (root = transformPipe; root is IDataTransform; root = ((IDataTransform)root).Source)
                    ;

                // root is now the loader.
                _host.Assert(root is IDataLoader);

                // Loader columns.
                var loaderSb = new StringBuilder();
                for (int i = 0; i < root.Schema.ColumnCount; i++)
                {
                    if (root.Schema.IsHidden(i))
                        continue;
                    if (loaderSb.Length > 0)
                        loaderSb.AppendLine();

                    ColumnType colType = root.Schema.GetColumnType(i);
                    CodeGenerationUtils.AppendFieldDeclaration(codeProvider, loaderSb, i, root.Schema.GetColumnName(i), colType, true, _args.SparseVectorDeclaration);
                }

                // Scored example columns.
                IDataView scorer;
                if (pred == null)
                    scorer = transformPipe;
                else
                {
                    var roles = ModelFileUtils.LoadRoleMappingsOrNull(_host, fs);
                    scorer = roles != null
                        ? _host.CreateDefaultScorer(RoleMappedData.CreateOpt(transformPipe, roles), pred)
                        : _host.CreateDefaultScorer(_host.CreateExamples(transformPipe, "Features"), pred);
                }

                var nonScoreSb = new StringBuilder();
                var scoreSb = new StringBuilder();
                for (int i = 0; i < scorer.Schema.ColumnCount; i++)
                {
                    if (scorer.Schema.IsHidden(i))
                        continue;
                    bool isScoreColumn = scorer.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnSetId, i) != null;

                    var sb = isScoreColumn ? scoreSb : nonScoreSb;

                    if (sb.Length > 0)
                        sb.AppendLine();

                    ColumnType colType = scorer.Schema.GetColumnType(i);
                    CodeGenerationUtils.AppendFieldDeclaration(codeProvider, sb, i, scorer.Schema.GetColumnName(i), colType, false, _args.SparseVectorDeclaration);
                }

                // Turn model path into a C# identifier and insert it.
                var modelPath = !string.IsNullOrWhiteSpace(_args.ModelNameOverride) ? _args.ModelNameOverride : _args.InputModelFile;
                modelPath = CodeGenerationUtils.GetCSharpString(codeProvider, modelPath);
                modelPath = string.Format("modelPath = {0};", modelPath);

                // Replace values inside the template.
                var replacementMap =
                    new Dictionary<string, string>
                    {
                        { "EXAMPLE_CLASS_DECL", loaderSb.ToString() },
                        { "SCORED_EXAMPLE_CLASS_DECL", nonScoreSb.ToString() },
                        { "SCORE_CLASS_DECL", scoreSb.ToString() },
                        { "MODEL_PATH", modelPath }
                    };

                var classSource = CodeGenerationUtils.MultiReplace(template, replacementMap);
                File.WriteAllText(_args.CSharpOutput, classSource);
            }
        }
    }
}
