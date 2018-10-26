using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Transforms
{
    public sealed class DnnImageFeaturizerTransform : OneToOneTransformerBase
    {
        public DnnImageFeaturizerTransform(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
        }

        public override void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
        {
            throw new NotImplementedException();
        }

        private sealed class Mapper : IRowMapper
        {
            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                throw new NotImplementedException();
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                throw new NotImplementedException();
            }

            public Schema.Column[] GetOutputColumns()
            {
                throw new NotImplementedException();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw new NotImplementedException();
            }
        }
    }

    public sealed class DnnImageFeaturizerEstimator : TrivialEstimator<DnnImageFeaturizerTransform>
    {
        public enum DnnModelType : byte
        {
            Resnet18 = 10,
            Resnet50 = 20,
            Resnet101 = 30,
            Alexnet = 100
        };

        public DnnImageFeaturizerEstimator(IHostEnvironment env, DnnModelType model, string input, string output)
        {
            EstimatorChain
        }

        /*public DnnImageFeaturizerEstimator(IHostEnvironment env, DnnImageFeaturizerTransform transformer) : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransform)), transformer)
        {
        }*/
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }
}
