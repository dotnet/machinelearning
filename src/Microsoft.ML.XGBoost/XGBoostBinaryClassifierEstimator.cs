using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Trainers.XGBoost
{

    public sealed class XGBoostBinaryClassificationTransformer : OneToOneTransformerBase
    {
        internal XGBoostBinaryClassificationTransformer(IHost host, params (string outputColumnName, string inputColumnName)[] columns) : base(host, columns)
        {
        }

        internal XGBoostBinaryClassificationTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            public Mapper(XGBoostBinaryClassificationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {

            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                throw new NotImplementedException();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                throw new NotImplementedException();
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                throw new NotImplementedException();
            }
        }
    }

    public sealed class XGBoostBinaryClassificationEstimator : TrivialEstimator<XGBoostBinaryClassificationTransformer>
    {
        private readonly IHost _host;
        public XGBoostBinaryClassificationEstimator(IHost host, XGBoostBinaryClassificationTransformer transformer) : base(host, transformer)
        {
            _host = Contracts.CheckRef(host, nameof(host)).Register(nameof(XGBoostBinaryClassificationEstimator));
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }
}
