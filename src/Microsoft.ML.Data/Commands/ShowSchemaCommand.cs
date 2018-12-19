// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(ShowSchemaCommand.Summary, typeof(ShowSchemaCommand), typeof(ShowSchemaCommand.Arguments), typeof(SignatureCommand),
    "Show Schema", ShowSchemaCommand.LoadName, "schema")]

namespace Microsoft.ML.Runtime.Data
{
    internal sealed class ShowSchemaCommand : DataCommand.ImplBase<ShowSchemaCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show all steps in transform chain", ShortName = "steps")]
            public bool ShowSteps;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Show the metadata types", ShortName = "metaTypes")]
            public bool ShowMetadataTypes;

            // REVIEW: Support abbreviating long vector-valued metadata?
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show the metadata types and values", ShortName = "meta,metaVals,metaValues")]
            public bool ShowMetadataValues;

            // Note that showing metadata overrides this.
            // REVIEW: Should we just remove this or possibly support only showing metadata of specified kinds?
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show slot names", ShortName = "slots", Hide = true)]
            public bool ShowSlots;

#if !CORECLR
            // Note that this overrides other options.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Show JSON version of the schema", ShortName = "json", Hide = true)]
            public bool ShowJson;
#endif
        }

        internal const string LoadName = "ShowSchema";
        internal const string Summary = "Given input data, a loader, and possibly transforms, display the schema.";

        public ShowSchemaCommand(IHostEnvironment env, Arguments args)
                : base(env, args, nameof(ShowSchemaCommand))
        {
        }

        public override void Run()
        {
            using (var ch = Host.Start(LoadName))
            {
                RunCore(ch);
            }
        }

        private void RunCore(IChannel ch)
        {
            IDataLoader loader = CreateAndSaveLoader();
            using (var schemaWriter = new StringWriter())
            {
                RunOnData(schemaWriter, Args, loader);
                var str = schemaWriter.ToString();
                ch.AssertNonEmpty(str);
                ch.Info(str);
            }
        }

        /// <summary>
        /// This shows the schema of the given <paramref name="data"/>, ignoring the data specification
        /// in the <paramref name="args"/> parameter. Test code invokes this, hence it is internal.
        /// </summary>
        internal static void RunOnData(TextWriter writer, Arguments args, IDataView data)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValue(args);
            Contracts.AssertValue(data);

            if (args.ShowSteps)
            {
                IEnumerable<IDataView> viewChainReversed = GetViewChainReversed(data);
                foreach (var view in viewChainReversed.Reverse())
                {
                    writer.WriteLine("---- {0} ----", view.GetType().Name);
                    PrintSchema(writer, args, view.Schema, (view as ITransposeDataView)?.TransposeSchema);
                }
            }
            else
                PrintSchema(writer, args, data.Schema, (data as ITransposeDataView)?.TransposeSchema);
        }

        /// <summary>
        /// Returns the sequence of views passed through the transform chain, last to first.
        /// </summary>
        private static IEnumerable<IDataView> GetViewChainReversed(IDataView data)
        {
            Contracts.AssertValue(data);
            IDataView view = (data as CompositeDataLoader)?.View ?? data;
            while (view != null)
            {
                yield return view;
                var transform = view as IDataTransform;
                view = transform?.Source;
            }
        }

        private static void PrintSchema(TextWriter writer, Arguments args, Schema schema, ITransposeSchema tschema)
        {
            Contracts.AssertValue(writer);
            Contracts.AssertValue(args);
            Contracts.AssertValue(schema);
            Contracts.AssertValueOrNull(tschema);
#if !CORECLR
            if (args.ShowJson)
            {
                writer.WriteLine("Json Schema not supported.");
                return;
            }
#endif
            int colLim = schema.Count;

            var itw = new IndentedTextWriter(writer, "  ");
            itw.WriteLine("{0} columns:", colLim);
            using (itw.Nest())
            {
                var names = default(VBuffer<ReadOnlyMemory<char>>);
                for (int col = 0; col < colLim; col++)
                {
                    var name = schema[col].Name;
                    var type = schema[col].Type;
                    var slotType = tschema == null ? null : tschema.GetSlotType(col);
                    itw.WriteLine("{0}: {1}{2}", name, type, slotType == null ? "" : " (T)");

                    bool metaVals = args.ShowMetadataValues;
                    if (metaVals || args.ShowMetadataTypes)
                    {
                        ShowMetadata(itw, schema, col, metaVals);
                        continue;
                    }

                    if (!args.ShowSlots)
                        continue;
                    if (!type.IsKnownSizeVector)
                        continue;
                    ColumnType typeNames;
                    if ((typeNames = schema[col].Metadata.Schema.GetColumnOrNull(MetadataUtils.Kinds.SlotNames)?.Type) == null)
                        continue;
                    if (typeNames.VectorSize != type.VectorSize || !typeNames.ItemType.IsText)
                    {
                        Contracts.Assert(false, "Unexpected slot names type");
                        continue;
                    }
                    schema[col].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref names);
                    if (names.Length != type.VectorSize)
                    {
                        Contracts.Assert(false, "Unexpected length of slot names vector");
                        continue;
                    }

                    using (itw.Nest())
                    {
                        bool verbose = args.Verbose ?? false;
                        foreach (var kvp in names.Items(all: verbose))
                        {
                            if (verbose || !kvp.Value.IsEmpty)
                                itw.WriteLine("{0}:{1}", kvp.Key, kvp.Value);
                        }
                    }
                }
            }
        }

        private static void ShowMetadata(IndentedTextWriter itw, Schema schema, int col, bool showVals)
        {
            Contracts.AssertValue(itw);
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.Count);

            using (itw.Nest())
            {
                foreach (var metaColumn in schema[col].Metadata.Schema.OrderBy(mcol => mcol.Name))
                {
                    var type = metaColumn.Type;
                    itw.Write("Metadata '{0}': {1}", metaColumn.Name, type);
                    if (showVals)
                    {
                        if (!type.IsVector)
                            ShowMetadataValue(itw, schema, col, metaColumn.Name, type);
                        else
                            ShowMetadataValueVec(itw, schema, col, metaColumn.Name, type);
                    }
                    itw.WriteLine();
                }
            }
        }

        private static void ShowMetadataValue(IndentedTextWriter itw, Schema schema, int col, string kind, ColumnType type)
        {
            Contracts.AssertValue(itw);
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.Count);
            Contracts.AssertNonEmpty(kind);
            Contracts.AssertValue(type);
            Contracts.Assert(!type.IsVector);

            if (!type.IsStandardScalar && !type.IsKey)
            {
                itw.Write(": Can't display value of this type");
                return;
            }

            Action<IndentedTextWriter, Schema, int, string, ColumnType> del = ShowMetadataValue<int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.RawType);
            meth.Invoke(null, new object[] { itw, schema, col, kind, type });
        }

        private static void ShowMetadataValue<T>(IndentedTextWriter itw, Schema schema, int col, string kind, ColumnType type)
        {
            Contracts.AssertValue(itw);
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.Count);
            Contracts.AssertNonEmpty(kind);
            Contracts.AssertValue(type);
            Contracts.Assert(!type.IsVector);
            Contracts.Assert(type.RawType == typeof(T));

            var conv = Conversions.Instance.GetStringConversion<T>(type);

            var value = default(T);
            var sb = default(StringBuilder);
            schema[col].Metadata.GetValue(kind, ref value);
            conv(in value, ref sb);

            itw.Write(": '{0}'", sb);
        }

        private static void ShowMetadataValueVec(IndentedTextWriter itw, Schema schema, int col, string kind, ColumnType type)
        {
            Contracts.AssertValue(itw);
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.Count);
            Contracts.AssertNonEmpty(kind);
            Contracts.AssertValue(type);
            Contracts.Assert(type.IsVector);

            if (!type.ItemType.IsStandardScalar && !type.ItemType.IsKey)
            {
                itw.Write(": Can't display value of this type");
                return;
            }

            Action<IndentedTextWriter, Schema, int, string, ColumnType> del = ShowMetadataValueVec<int>;
            var meth = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
            meth.Invoke(null, new object[] { itw, schema, col, kind, type });
        }

        private static void ShowMetadataValueVec<T>(IndentedTextWriter itw, Schema schema, int col, string kind, ColumnType type)
        {
            Contracts.AssertValue(itw);
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.Count);
            Contracts.AssertNonEmpty(kind);
            Contracts.AssertValue(type);
            Contracts.Assert(type.IsVector);
            Contracts.Assert(type.ItemType.RawType == typeof(T));

            var conv = Conversions.Instance.GetStringConversion<T>(type.ItemType);

            var value = default(VBuffer<T>);
            schema[col].Metadata.GetValue(kind, ref value);

            itw.Write(": Length={0}, Count={0}", value.Length, value.GetValues().Length);

            using (itw.Nest())
            {
                var sb = default(StringBuilder);
                int count = 0;
                foreach (var item in value.Items())
                {
                    if ((count % 10) == 0)
                        itw.WriteLine();
                    else
                        itw.Write(", ");
                    var val = item.Value;
                    conv(in val, ref sb);
                    itw.Write("[{0}] '{1}'", item.Key, sb);
                    count++;
                }
            }
        }
    }
}
