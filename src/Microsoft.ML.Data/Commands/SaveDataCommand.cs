// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(SaveDataCommand.Summary, typeof(SaveDataCommand), typeof(SaveDataCommand.Arguments), typeof(SignatureCommand),
    "Save Data", "SaveData", "save")]

[assembly: LoadableClass(ShowDataCommand.Summary, typeof(ShowDataCommand), typeof(ShowDataCommand.Arguments), typeof(SignatureCommand),
    "Show Data", "ShowData", "show")]

namespace Microsoft.ML.Data
{
    internal sealed class SaveDataCommand : DataCommand.ImplBase<SaveDataCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "The data saver to use", NullName = "<Auto>", SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> Saver;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File to save the data", ShortName = "dout")]
            public string OutputDataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to include hidden columns", ShortName = "keep")]
            public bool KeepHidden;
        }

        internal const string Summary = "Given input data, a loader, and possibly transforms, save the data to a new file as parameterized by a saver.";

        public SaveDataCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(SaveDataCommand))
        {
            Contracts.CheckNonEmpty(args.OutputDataFile, nameof(args.OutputDataFile));
            Utils.CheckOptionalUserDirectory(args.OutputDataFile, nameof(args.OutputDataFile));
        }

        public override void Run()
        {
            string command = "SaveData";
            using (var ch = Host.Start(command))
            {
                RunCore(ch);
            }
        }

        private void RunCore(IChannel ch)
        {
            Host.AssertValue(ch, "ch");
            IDataSaver saver;
            if (ImplOptions.Saver == null)
            {
                var ext = Path.GetExtension(ImplOptions.OutputDataFile);
                var isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                var isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);
                if (isBinary)
                {
                    saver = new BinarySaver(Host, new BinarySaver.Arguments());
                }
                else if (isTranspose)
                {
                    saver = new TransposeSaver(Host, new TransposeSaver.Arguments());
                }
                else
                {
                    saver = new TextSaver(Host, new TextSaver.Arguments());
                }
            }
            else
            {
                saver = ImplOptions.Saver.CreateComponent(Host);
            }

            ILegacyDataLoader loader = CreateAndSaveLoader();
            using (var file = Host.CreateOutputFile(ImplOptions.OutputDataFile))
                DataSaverUtils.SaveDataView(ch, saver, loader, file, ImplOptions.KeepHidden);
        }
    }

    internal sealed class ShowDataCommand : DataCommand.ImplBase<ShowDataCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Comma separated list of columns to display", ShortName = "cols")]
            public string Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of rows")]
            public int Rows = 20;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to include hidden columns", ShortName = "keep")]
            public bool KeepHidden;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force dense format")]
            public bool Dense;

            [Argument(ArgumentType.Multiple, HelpText = "The data saver to use", NullName = "<Auto>", SignatureType = typeof(SignatureDataSaver))]
            public IComponentFactory<IDataSaver> Saver;
        }

        internal const string Summary = "Given input data, a loader, and possibly transforms, display a sample of the data file.";

        public ShowDataCommand(IHostEnvironment env, Arguments args)
            : base(env, args, nameof(ShowDataCommand))
        {
        }

        public override void Run()
        {
            string command = "ShowData";
            using (var ch = Host.Start(command))
            {
                RunCore(ch);
            }
        }

        private void RunCore(IChannel ch)
        {
            Host.AssertValue(ch);
            IDataView data = CreateAndSaveLoader();

            if (!string.IsNullOrWhiteSpace(ImplOptions.Columns))
            {
                var keepColumns = ImplOptions.Columns
                    .Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
                if (Utils.Size(keepColumns) > 0)
                    data = ColumnSelectingTransformer.CreateKeep(Host, data, keepColumns);
            }

            IDataSaver saver;
            if (ImplOptions.Saver != null)
                saver = ImplOptions.Saver.CreateComponent(Host);
            else
                saver = new TextSaver(Host, new TextSaver.Arguments() { Dense = ImplOptions.Dense });
            var cols = new List<int>();
            for (int i = 0; i < data.Schema.Count; i++)
            {
                if (!ImplOptions.KeepHidden && data.Schema[i].IsHidden)
                    continue;
                var type = data.Schema[i].Type;
                if (saver.IsColumnSavable(type))
                    cols.Add(i);
                else
                    ch.Info(MessageSensitivity.Schema, "The column '{0}' will not be written as it has unsavable column type.", data.Schema[i].Name);
            }
            Host.NotSensitive().Check(cols.Count > 0, "No valid columns to save");

            // Send the first N lines to console.
            if (ImplOptions.Rows > 0)
            {
                var args = new SkipTakeFilter.TakeOptions() { Count = ImplOptions.Rows };
                data = SkipTakeFilter.Create(Host, args, data);
            }
            var textSaver = saver as TextSaver;
            // If it is a text saver, utilize a special utility for this purpose.
            if (textSaver != null)
                textSaver.WriteData(data, true, cols.ToArray());
            else
            {
                using (MemoryStream mem = new MemoryStream())
                {
                    using (Stream wrapStream = new SubsetStream(mem))
                        saver.SaveData(wrapStream, data, cols.ToArray());
                    mem.Seek(0, SeekOrigin.Begin);
                    using (StreamReader reader = new StreamReader(mem))
                    {
                        string result = reader.ReadToEnd();
                        ch.Info(MessageSensitivity.UserData | MessageSensitivity.Schema, result);
                    }
                }
            }
        }
    }

    [BestFriend]
    internal static class DataSaverUtils
    {
        public static void SaveDataView(IChannel ch, IDataSaver saver, IDataView view, IFileHandle file, bool keepHidden = false)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(saver, nameof(saver));
            ch.CheckValue(view, nameof(view));
            ch.CheckValue(file, nameof(file));
            ch.CheckParam(file.CanWrite, nameof(file), "Cannot write to file");

            using (var stream = file.CreateWriteStream())
                SaveDataView(ch, saver, view, stream, keepHidden);
        }

        public static void SaveDataView(IChannel ch, IDataSaver saver, IDataView view, Stream stream, bool keepHidden = false)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(saver, nameof(saver));
            ch.CheckValue(view, nameof(view));
            ch.CheckValue(stream, nameof(stream));

            var cols = new List<int>();
            for (int i = 0; i < view.Schema.Count; i++)
            {
                if (!keepHidden && view.Schema[i].IsHidden)
                    continue;
                var type = view.Schema[i].Type;
                if (saver.IsColumnSavable(type))
                    cols.Add(i);
                else
                    ch.Info(MessageSensitivity.Schema, "The column '{0}' will not be written as it has unsavable column type.", view.Schema[i].Name);
            }

            ch.Check(cols.Count > 0, "No valid columns to save");
            saver.SaveData(stream, view, cols.ToArray());
        }
    }
}