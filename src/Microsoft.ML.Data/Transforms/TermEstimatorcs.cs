using Microsoft.ML.Core.Data;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.Runtime.Data.TermTransform;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class TermEstimator : IEstimator<TermTransform>
    {
        private readonly int _maxNumTerms;
        private readonly SortOrder _sort;
        private readonly Column[] _columns;
        private readonly IHost _host;

        public TermEstimator(IHostEnvironment env, string name, string source = null, int maxNumTerms = Defaults.MaxNumTerms, SortOrder sort = Defaults.Sort) :
           this(env, maxNumTerms, sort, new Column { Name = name, Source = source ?? name })
        {
        }

        public TermEstimator(IHostEnvironment env, int maxNumTerms = Defaults.MaxNumTerms, SortOrder sort = Defaults.Sort, params Column[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TermEstimator));
            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                if (newNames.Contains(column.Name))
                    throw Contracts.ExceptUserArg(nameof(columns), $"New column {column.Name} specified multiple times");
                newNames.Add(column.Name);
            }
            _columns = columns;
            _maxNumTerms = maxNumTerms;
            _sort = sort;
        }

        public TermTransform Fit(IDataView input)
        {
            // Invoke schema validation.
            GetOutputSchema(SchemaShape.Create(input.Schema));
            var args = new Arguments
            {
                Column = _columns,
                MaxNumTerms = _maxNumTerms,
                Sort = _sort
            };
            return new TermTransform(_host, args, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var column in _columns)
            {
                var originalColumn = inputSchema.FindColumn(column.Source);
                if (originalColumn != null)
                {
                    var col = new SchemaShape.Column(column.Name, originalColumn.Kind, DataKind.U4, true, originalColumn.MetadataKinds);
                    resultDic[column.Name] = col;
                }
                else
                {
                    throw _host.ExceptParam(nameof(inputSchema), $"{column.Source} not found in {nameof(inputSchema)}");
                }
            }
            return new SchemaShape(resultDic.Values.ToArray());
        }
    }
}
