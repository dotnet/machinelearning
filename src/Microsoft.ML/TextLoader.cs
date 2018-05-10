// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Microsoft.ML
{
    public class TextLoader<TInput> : ILearningPipelineLoader
    {
        private string _inputFilePath;
        private string CustomSchema;
        private Data.TextLoader ImportTextInput;

        /// <summary>
        /// Construct a TextLoader object
        /// </summary>
        /// <param name="inputFilePath">Data file path</param>
        /// <param name="useHeader">Does the file contains header?</param>
        /// <param name="separator">How the columns are seperated? 
        /// Options: separator="tab", separator="space", separator="comma" or separator=[single character]. 
        /// By default separator=null means "tab"</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values, 
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators 
        /// denote a missing value and an empty value is denoted by \"\". 
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations e.g. 
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero 
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        public TextLoader(string inputFilePath, bool useHeader = false, 
            string separator = null, bool allowQuotedStrings = true, 
            bool supportSparse = true, bool trimWhitespace = false)
        {
            _inputFilePath = inputFilePath;
            SetCustomStringFromType(useHeader, separator, allowQuotedStrings, supportSparse, trimWhitespace);
        }

        private IFileHandle GetTextLoaderFileHandle(IHostEnvironment env, string trainFilePath) =>
            new SimpleFileHandle(env, trainFilePath, false, false);

        private void SetCustomStringFromType(bool useHeader, string separator, 
            bool allowQuotedStrings, bool supportSparse, bool trimWhitespace)
        {
            StringBuilder schemaBuilder = new StringBuilder(CustomSchema);
            foreach (var field in typeof(TInput).GetFields())
            {
                var mappingAttr = field.GetCustomAttribute<ColumnAttribute>();
                if(mappingAttr == null)
                    throw Contracts.ExceptParam(field.Name, $"{field.Name} is missing ColumnAttribute");
                
                schemaBuilder.AppendFormat("col={0}:{1}:{2} ",
                    mappingAttr.Name ?? field.Name, 
                    TypeToName(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType), 
                    mappingAttr.Ordinal);
            }

            if (useHeader)
                schemaBuilder.Append(nameof(TextLoader.Arguments.HasHeader)).Append("+ ");

            if (separator != null)
                schemaBuilder.Append(nameof(TextLoader.Arguments.Separator)).Append("=").Append(separator).Append(" ");

            if (!allowQuotedStrings)
                schemaBuilder.Append(nameof(TextLoader.Arguments.AllowQuoting)).Append("- ");

            if (!supportSparse)
                schemaBuilder.Append(nameof(TextLoader.Arguments.AllowSparse)).Append("- ");

            if (trimWhitespace)
                schemaBuilder.Append(nameof(TextLoader.Arguments.TrimWhitespace)).Append("+ ");

            schemaBuilder.Length--;
            CustomSchema = schemaBuilder.ToString();
        }

        private string TypeToName(Type type)
        {
            if (type == typeof(string))
                return "TX";
            else if (type == typeof(float) || type == typeof(double))
                return "R4";
            else if (type == typeof(bool))
                return "BL";
            else
                throw new Exception("Type not implemented or supported."); //Add more types.
        }

        public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
        {
            Contracts.Assert(previousStep == null);

            ImportTextInput = new Data.TextLoader();
            ImportTextInput.CustomSchema = CustomSchema;
            var importOutput = experiment.Add(ImportTextInput);
            return new TextLoaderPipelineStep(importOutput.Data);
        }

        public void SetInput(IHostEnvironment env, Experiment experiment)
        {
            IFileHandle inputFile = GetTextLoaderFileHandle(env, _inputFilePath);
            experiment.SetInput(ImportTextInput.InputFile, inputFile);
        }

        private class TextLoaderPipelineStep : ILearningPipelineDataStep
        {
            public TextLoaderPipelineStep(Var<IDataView> data)
            {
                Data = data;
            }

            public Var<IDataView> Data { get; }
            public Var<ITransformModel> Model => null;
        }
    }
}
