// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    public static class PredictorUtils
    {
        /// <summary>
        /// Save the model summary
        /// </summary>
        public static void SaveSummary(IChannel ch, IPredictor predictor, RoleMappedSchema schema, TextWriter writer)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValueOrNull(schema);
            ch.CheckValue(writer, nameof(writer));

            var saver = predictor as ICanSaveSummary;
            if (saver != null)
                saver.SaveSummary(writer, schema);
            else
            {
                writer.WriteLine("'{0}' does not support saving summary", predictor.GetType().Name);
                ch.Error("'{0}' does not support saving summary", predictor.GetType().Name);
            }
        }

        /// <summary>
        /// Save the model in text format (if it can save itself)
        /// </summary>
        public static void SaveText(IChannel ch, IPredictor predictor, RoleMappedSchema schema, TextWriter writer)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValueOrNull(schema);
            ch.CheckValue(writer, nameof(writer));

            var textSaver = predictor as ICanSaveInTextFormat;
            if (textSaver != null)
            {
                textSaver.SaveAsText(writer, schema);
                return;
            }

            var summarySaver = predictor as ICanSaveSummary;
            if (summarySaver != null)
            {
                writer.WriteLine("'{0}' does not support saving in text format, writing out model summary instead", predictor.GetType().Name);
                ch.Error("'{0}' doesn't currently have standardized text format for /mt, will save model summary instead",
                    predictor.GetType().Name);
                summarySaver.SaveSummary(writer, schema);
            }
            else
            {
                writer.WriteLine("'{0}' does not support saving in text format", predictor.GetType().Name);
                ch.Error("'{0}' doesn't currently have standardized text format for /mt", predictor.GetType().Name);
            }
        }

        /// <summary>
        /// Save the model in binary format (if it can save itself)
        /// </summary>
        public static void SaveBinary(IChannel ch, IPredictor predictor, BinaryWriter writer)
        {
            Contracts.CheckValue(ch, nameof(ch));
            var saver = predictor as ICanSaveInBinaryFormat;
            if (saver == null)
            {
                ch.Error("'{0}' doesn't currently have standardized binary format for /mb", predictor.GetType().Name);
                return;
            }
            saver.SaveAsBinary(writer);
        }

        /// <summary>
        /// Save the model in text format (if it can save itself)
        /// </summary>
        public static void SaveIni(IChannel ch, IPredictor predictor, RoleMappedSchema schema, TextWriter writer)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValueOrNull(schema);
            ch.CheckValue(writer, nameof(writer));

            var iniSaver = predictor as ICanSaveInIniFormat;
            if (iniSaver != null)
            {
                iniSaver.SaveAsIni(writer, schema);
                return;
            }

            var summarySaver = predictor as ICanSaveSummary;
            if (summarySaver != null)
            {
                writer.WriteLine("'{0}' does not support saving in INI format, writing out model summary instead", predictor.GetType().Name);
                ch.Error("'{0}' doesn't currently have standardized INI format output, will save model summary instead",
                    predictor.GetType().Name);
                summarySaver.SaveSummary(writer, schema);
            }
            else
            {
                writer.WriteLine("'{0}' does not support saving in INI format", predictor.GetType().Name);
                ch.Error("'{0}' doesn't currently have standardized INI format output", predictor.GetType().Name);
            }
        }

        /// <summary>
        /// Save the model in text format (if it can save itself)
        /// </summary>
        public static void SaveCode(IChannel ch, IPredictor predictor, RoleMappedSchema schema, TextWriter writer)
        {
            Contracts.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValueOrNull(schema);
            ch.CheckValue(writer, nameof(writer));

            var saver = predictor as ICanSaveInSourceCode;
            if (saver != null)
                saver.SaveAsCode(writer, schema);
            else
            {
                writer.WriteLine("'{0}' does not support saving in code.", predictor.GetType().Name);
                ch.Error("'{0}' doesn't currently support saving the model as code", predictor.GetType().Name);
            }
        }
    }
}