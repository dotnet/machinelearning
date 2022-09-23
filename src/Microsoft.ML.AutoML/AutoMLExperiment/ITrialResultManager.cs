// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal interface ITrialResultManager
    {
        IEnumerable<TrialResult> GetAllTrialResults();

        void AddOrUpdateTrialResult(TrialResult result);

        void Save();
    }

    /// <summary>
    /// trialResult Manager that saves and loads trial result as csv format.
    /// </summary>
    internal class CsvTrialResultManager : ITrialResultManager
    {
        private readonly string _filePath;
        private readonly IChannel _channel;
        private readonly HashSet<TrialResult> _trialResultsHistory;
        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly DataViewSchema _schema;
        private readonly HashSet<TrialResult> _newTrialResults;
        public CsvTrialResultManager(string filePath, SearchSpace.SearchSpace searchSpace, IChannel channel = null)
        {
            _filePath = filePath;
            _channel = channel;
            _searchSpace = searchSpace;
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("id", NumberDataViewType.Int32);
            schemaBuilder.AddColumn("loss", NumberDataViewType.Single);
            schemaBuilder.AddColumn("durationInMilliseconds", NumberDataViewType.Single);
            schemaBuilder.AddColumn("peakCpu", NumberDataViewType.Single);
            schemaBuilder.AddColumn("peakMemoryInMegaByte", NumberDataViewType.Single);
            schemaBuilder.AddColumn("parameter", new VectorDataViewType(NumberDataViewType.Double));
            _schema = schemaBuilder.ToSchema();

            // load from csv file.
            var trialResults = LoadFromCsvFile(filePath);
            _trialResultsHistory = new HashSet<TrialResult>(trialResults, new TrialResult());
            _newTrialResults = new HashSet<TrialResult>(new TrialResult());
        }

        public void AddOrUpdateTrialResult(TrialResult result)
        {
            if (_trialResultsHistory.Contains(result))
            {
                throw new ArgumentException("can't add or update result that already save to csv");
            }
            _newTrialResults.Remove(result);
            _newTrialResults.Add(result);
        }

        public IEnumerable<TrialResult> GetAllTrialResults()
        {
            return _trialResultsHistory.Concat(_newTrialResults);
        }

        /// <summary>
        /// save trial result to csv. This will not overwrite any existing records that already written in csv.
        /// </summary>
        public void Save()
        {
            // header (type)
            // | id (int) | loss (float) | durationInMilliseconds (float) | peakCpu (float) | peakMemoryInMegaByte (float) | parameter_i (float) |
            using (var fileStream = new FileStream(_filePath, FileMode.Append, FileAccess.Write))
            using (var writeStream = new StreamWriter(fileStream))
            {
                var sep = ",";

                if (_trialResultsHistory.Count == 0)
                {
                    // write header
                    var header = new string[]
                    {
                        "id",
                        "loss",
                        "durationInMilliseconds",
                        "peakCpu",
                        "peakMemoryInMegaByte"
                    }.Concat(Enumerable.Range(0, _searchSpace.FeatureSpaceDim).Select(i => $"parameter_{i}"));
                    writeStream.WriteLine(string.Join(sep, header));
                }

                foreach (var trialResult in _newTrialResults.OrderBy(res => res.TrialSettings.TrialId))
                {
                    var parameter = _searchSpace.MappingToFeatureSpace(trialResult.TrialSettings.Parameter);
                    var csvLine = string.Join(
                        sep,
                        new string[]
                        {
                            trialResult.TrialSettings.TrialId.ToString(CultureInfo.InvariantCulture),
                            trialResult.Loss.ToString("F3", CultureInfo.InvariantCulture),
                            trialResult.DurationInMilliseconds.ToString("F3", CultureInfo.InvariantCulture),
                            trialResult.PeakCpu?.ToString("F3", CultureInfo.InvariantCulture),
                            trialResult.PeakMemoryInMegaByte?.ToString("F3", CultureInfo.InvariantCulture),
                        }.Concat(parameter.Select(p => p.ToString("F3", CultureInfo.InvariantCulture))));
                    writeStream.WriteLine(csvLine);
                }

                writeStream.Flush();
                writeStream.Close();
            }

            foreach (var result in _newTrialResults)
            {
                _trialResultsHistory.Add(result);
            }

            _newTrialResults.Clear();
        }

        private IEnumerable<TrialResult> LoadFromCsvFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                return Array.Empty<TrialResult>();
            }

            // header (type)
            // | id (int) | loss (float) | durationInMilliseconds (float) | peakCpu (float) | peakMemoryInMegaByte (float) | parameter_i (float) |
            var context = new MLContext();
            var textLoaderColumns = new TextLoader.Column[]
            {
                new TextLoader.Column("id", DataKind.Int32, 0),
                new TextLoader.Column("loss", DataKind.Single, 1),
                new TextLoader.Column("durationInMilliseconds", DataKind.Single, 2),
                new TextLoader.Column("peakCpu", DataKind.Single, 3),
                new TextLoader.Column("peakMemoryInMegaByte", DataKind.Single, 4),
                new TextLoader.Column("parameter", DataKind.Double, 5, 5 + _searchSpace.FeatureSpaceDim - 1),
            };
            var res = new List<TrialResult>();
            var dataView = context.Data.LoadFromTextFile(filePath, textLoaderColumns, separatorChar: ',', hasHeader: true, allowQuoting: true);
            var rowCursor = dataView.GetRowCursor(_schema);

            var idGetter = rowCursor.GetGetter<int>(_schema["id"]);
            var lossGetter = rowCursor.GetGetter<float>(_schema["loss"]);
            var durationGetter = rowCursor.GetGetter<float>(_schema["durationInMilliseconds"]);
            var peakCpuGetter = rowCursor.GetGetter<float>(_schema["peakCpu"]);
            var peakMemoryGetter = rowCursor.GetGetter<float>(_schema["peakMemoryInMegaByte"]);
            var parameterGetter = rowCursor.GetGetter<VBuffer<double>>(_schema["parameter"]);

            while (rowCursor.MoveNext())
            {
                int id = 0;
                float loss = 0;
                float duration = 0;
                float peakCpu = 0;
                float peakMemory = 0;
                VBuffer<double> parameter = default;

                idGetter(ref id);
                lossGetter(ref loss);
                durationGetter(ref duration);
                peakCpuGetter(ref peakCpu);
                peakMemoryGetter(ref peakMemory);
                parameterGetter(ref parameter);
                var feature = parameter.DenseValues().ToArray();
                var trialResult = new TrialResult
                {
                    TrialSettings = new TrialSettings
                    {
                        TrialId = id,
                        Parameter = _searchSpace.SampleFromFeatureSpace(feature),
                    },
                    DurationInMilliseconds = duration,
                    Loss = loss,
                    PeakCpu = peakCpu,
                    PeakMemoryInMegaByte = peakMemory,
                };

                res.Add(trialResult);
            }

            _channel?.Trace($"load trial history from {filePath} successfully with {res.Count()} pieces of data");
            return res;
        }
    }
}
