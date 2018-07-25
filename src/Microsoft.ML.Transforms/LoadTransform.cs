// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(LoadTransform.Summary, typeof(IDataTransform), typeof(LoadTransform), typeof(LoadTransform.Arguments), typeof(SignatureDataTransform),
    "Load Transform", "LoadTransform", "Load")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Load specific transforms from the specified model file. Allows one to 'cherry pick' transforms from
    /// a serialized chain, or to apply a pre-trained transform to a different (but still compatible) data view.
    /// </summary>
    public static class LoadTransform
    {
        public class Arguments
        {
            // REVIEW: make it not required, and make commands fill in the missing model file with the default
            // input model file. This requires some hacking in DataDiagnosticCommand.
            [Argument(ArgumentType.Required, HelpText = "Model file to load the transforms from", ShortName = "in",
                SortOrder = 1, IsInputFileName = true)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple, HelpText = "The tags (comma-separated) to be loaded (or omitted, if " + nameof(Complement) + "+)", SortOrder = 2)]
            public string[] Tag;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to load all transforms except those marked by tags", ShortName = "comp", SortOrder = 3)]
            public bool Complement = false;
        }

        internal const string Summary = "Loads specified transforms from the model file and applies them to current data.";

        /// <summary>
        /// A helper method to create <see cref="LoadTransform"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">Model file to load the transforms from.</param>
        /// <param name="tag">The tags (comma-separated) to be loaded (or omitted, if complement is true).</param>
        /// <param name="complement">Whether to load all transforms except those marked by tags.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string[] tag, bool complement = false)
        {
            var args = new Arguments()
            {
                ModelFile = modelFile,
                Tag = tag,
                Complement = complement
            };
            return Create(env, args, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("LoadTransform");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile), "File does not exist");

            IDataView currentView;

            // If there are no 'tag' parameters, we load everything, regardless of 'comp'.
            bool complement = args.Complement || Utils.Size(args.Tag) == 0;
            var allTags = new HashSet<string>();
            for (int i = 0; i < Utils.Size(args.Tag); i++)
            {
                var curList = args.Tag[i];
                if (string.IsNullOrWhiteSpace(curList))
                    continue;

                foreach (var tag in curList.Split(','))
                {
                    if (!string.IsNullOrWhiteSpace(tag))
                        allTags.Add(tag.ToLower());
                }
            }

            Func<string, bool> predicate =
                tag =>
                {
                    bool found = allTags.Contains(tag.ToLower());
                    return found == !complement;
                };

            using (var file = h.OpenInputFile(args.ModelFile))
            using (var strm = file.OpenReadStream())
            using (var rep = RepositoryReader.Open(strm, h))
            using (var pipeLoaderEntry = rep.OpenEntry(ModelFileUtils.DirDataLoaderModel, ModelLoadContext.ModelStreamName))
            using (var ctx = new ModelLoadContext(rep, pipeLoaderEntry, ModelFileUtils.DirDataLoaderModel))
            {
                currentView = CompositeDataLoader.LoadSelectedTransforms(ctx, input, h, predicate);

                if (currentView == input)
                {
                    // REVIEW: we are required to return an IDataTransform. Therefore, if we don't introduce a new transform
                    // on top of 'input', we must throw (since input may not be a data transform).
                    // We could of course introduce a 'no-op transform', or we could lift the requirement to always return an IDataTransform
                    // associated with SignatureDataTransform.

                    var criteria = string.Format(
                            complement
                                ? "transforms that don't have tags from the list: '{0}'"
                                : "transforms that have tags from the list: '{0}'",
                            string.Join(",", allTags));
                    throw h.ExceptUserArg(nameof(args.Tag), "No transforms were found that match the search criteria ({0})", criteria);
                }
            }

            h.Assert(currentView is IDataTransform);
            return (IDataTransform)currentView;
        }
    }
}