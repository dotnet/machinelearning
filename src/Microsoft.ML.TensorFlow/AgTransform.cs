// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(AgTransform.Summary, typeof(IDataTransform), typeof(AgTransform),
    typeof(AgTransform.Arguments), typeof(SignatureDataTransform), AgTransform.UserName, AgTransform.ShortName)]

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(AgTransform.AgMapper), null, typeof(SignatureLoadRowMapper),
    "", AgTransform.AgMapper.LoaderSignature)]

[assembly: EntryPointModule(typeof(AgTransform))]

namespace Microsoft.ML.Transforms
{
    public static class AgTransform
    {
        internal sealed class AgMapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly string _outputColName;
            private readonly ColumnType _outputColType;
            private readonly int[] _inputColIndices;

            public const string LoaderSignature = "AgMapper";

            public AgMapper(IHostEnvironment env, ISchema inputSchema, string[] inputColNames, string outputColName)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckNonEmpty(inputColNames, nameof(inputColNames));
                _host.CheckNonEmpty(outputColName, nameof(outputColName));

                _outputColName = outputColName;
                _outputColType = new VectorType(NumberType.R4, 3);

                _inputColIndices = new int[inputColNames.Length];
                var colNames = new string[inputColNames.Length];
                for (int i = 0; i < inputColNames.Length; i++)
                {
                    colNames[i] = inputColNames[i];
                    if (!inputSchema.TryGetColumnIndex(colNames[i], out _inputColIndices[i]))
                        throw Contracts.Except($"Column '{colNames[i]}' does not exist");
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                var getters = new Delegate[1];
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = MakeGetter<float>(input);
                    ch.Done();
                    return getters;
                }
            }
            private Delegate MakeGetter<T>(IRow input)
            {
                _host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };
                return valuegetter;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                return new[] { new RowMapperColumnInfo(_outputColName, _outputColType, null) };
            }

            public void Save(ModelSaveContext ctx)
            {
                throw new NotImplementedException();
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 2)]
            public string OutputColumn;
        }

        public const string Summary = "Computes sum.";
        public const string UserName = "AgTransform";
        public const string ShortName = "AgTransform";
        private const string RegistrationName = "AgTransform";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column. Keep it same as in the TensorFlow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the TensorFlow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string name, params string[] source)
        {
            return Create(env, new Arguments() { InputColumns = source, OutputColumn = name }, input);
        }

        /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransform"]/*' />
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckUserArg(Utils.Size(args.InputColumns) > 0, nameof(args.InputColumns));

            for (int i = 0; i < args.InputColumns.Length; i++)
                host.CheckNonWhiteSpace(args.InputColumns[i], nameof(args.InputColumns));

            var mapper = new AgMapper(host, input.Schema, args.InputColumns, args.OutputColumn);
            return new RowToRowMapperTransform(host, input, mapper);
        }
    }
}