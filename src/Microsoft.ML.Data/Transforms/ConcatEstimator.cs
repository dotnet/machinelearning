// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Data.StaticPipe.Runtime;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class ConcatEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly string _name;
        private readonly string[] _source;

        public ConcatEstimator(IHostEnvironment env, string name, params string[] source)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("ConcatEstimator");

            _host.CheckNonEmpty(name, nameof(name));
            _host.CheckNonEmpty(source, nameof(source));
            _host.CheckParam(!source.Any(r => string.IsNullOrEmpty(r)), nameof(source),
                "Contained some null or empty items");

            _name = name;
            _source = source;
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            var xf = new ConcatTransform(_host, input, _name, _source);
            var empty = new EmptyDataView(_host, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_host, xf, empty, input);
            return new ConcatTransformer(_host, chunk);
        }

        private bool HasCategoricals(SchemaShape.Column col)
        {
            _host.AssertValue(col);
            if (!col.Metadata.TryFindColumn(MetadataUtils.Kinds.CategoricalSlotRanges, out var mcol))
                return false;
            // The indices must be ints and of a definite size vector type. (Definite becuase
            // metadata has only one value anyway.)
            return mcol.Kind == SchemaShape.Column.VectorKind.Vector
                && mcol.ItemType == NumberType.I4;
        }

        private SchemaShape.Column CheckInputsAndMakeColumn(
            SchemaShape inputSchema, string name, string[] sources)
        {
            _host.AssertNonEmpty(sources);

            var cols = new SchemaShape.Column[sources.Length];
            // If any input is a var vector, so is the output.
            bool varVector = false;
            // If any input is not normalized, the output is not normalized.
            bool isNormalized = true;
            // If any input has categorical indices, so will the output.
            bool hasCategoricals = false;
            // If any is scalar or had slot names, then the output will have slot names.
            bool hasSlotNames = false;

            // We will get the item type from the first column.
            ColumnType itemType = null;

            for (int i = 0; i < sources.Length; ++i)
            {
                if (!inputSchema.TryFindColumn(sources[i], out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", sources[i]);
                if (i == 0)
                    itemType = col.ItemType;
                // For the sake of an estimator I am going to have a hard policy of no keys.
                // Appending keys makes no real sense anyway.
                if (col.IsKey)
                {
                    throw _host.Except($"Column '{sources[i]}' is key." +
                        $"Concatenation of keys is unsupported.");
                }
                if (!col.ItemType.Equals(itemType))
                {
                    throw _host.Except($"Column '{sources[i]}' has values of {col.ItemType}" +
                        $"which is not the same as earlier observed type of {itemType}.");
                }
                varVector |= col.Kind == SchemaShape.Column.VectorKind.VariableVector;
                isNormalized &= col.IsNormalized();
                hasCategoricals |= HasCategoricals(col);
                hasSlotNames |= col.Kind == SchemaShape.Column.VectorKind.Scalar || col.HasSlotNames();
            }
            var vecKind = varVector ? SchemaShape.Column.VectorKind.VariableVector :
                    SchemaShape.Column.VectorKind.Vector;

            List<SchemaShape.Column> meta = new List<SchemaShape.Column>();
            if (isNormalized)
                meta.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
            if (hasCategoricals)
                meta.Add(new SchemaShape.Column(MetadataUtils.Kinds.CategoricalSlotRanges, SchemaShape.Column.VectorKind.Vector, NumberType.I4, false));
            if (hasSlotNames)
                meta.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));

            return new SchemaShape.Column(name, vecKind, itemType, false, new SchemaShape(meta));
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            result[_name] = CheckInputsAndMakeColumn(inputSchema, _name, _source);
            return new SchemaShape(result.Values);
        }
    }

    // REVIEW: Note that the presence of this thing is a temporary measure only.
    // If it is cleaned up by code complete so much the better, but if not we will
    // have to wait a little bit.
    internal sealed class ConcatTransformer : ITransformer, ICanSaveModel
    {
        public const string LoaderSignature = "TransformWrapper";
        private const string TransformDirTemplate = "Step_{0:000}";

        private readonly IHostEnvironment _env;
        private readonly IDataView _xf;

        internal ConcatTransformer(IHostEnvironment env, IDataView xf)
        {
            _env = env;
            _xf = xf;
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            var dv = new EmptyDataView(_env, inputSchema);
            var output = ApplyTransformUtils.ApplyAllTransformsToData(_env, _xf, dv);
            return output.Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            var dataPipe = _xf;
            var transforms = new List<IDataTransform>();
            while (dataPipe is IDataTransform xf)
            {
                // REVIEW: a malicious user could construct a loop in the Source chain, that would
                // cause this method to iterate forever (and throw something when the list overflows). There's
                // no way to insulate from ALL malicious behavior.
                transforms.Add(xf);
                dataPipe = xf.Source;
                Contracts.AssertValue(dataPipe);
            }
            transforms.Reverse();

            ctx.SaveSubModel("Loader", c => BinaryLoader.SaveInstance(_env, c, dataPipe.Schema));

            ctx.Writer.Write(transforms.Count);
            for (int i = 0; i < transforms.Count; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(transforms[i], dirName);
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CCATWRPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public ConcatTransformer(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckAtModel(GetVersionInfo());
            int n = ctx.Reader.ReadInt32();

            ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            IDataView data = loader;
            for (int i = 0; i < n; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                data = xf;
            }

            _env = env;
            _xf = data;
        }

        public IDataView Transform(IDataView input) => ApplyTransformUtils.ApplyAllTransformsToData(_env, _xf, input);
    }

    /// <summary>
    /// The extension methods and implementation support for concatenating columns together.
    /// </summary>
    public static class ConcatStaticExtensions
    {
        /// <summary>
        /// Given a scalar vector, produce a vector of length one.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The scalar column.</param>
        /// <returns>The vector column, whose single item has the same value as the input.</returns>
        public static Vector<T> AsVector<T>(this Scalar<T> me)
            => new Impl<T>(Join(me, (PipelineColumn[])null));

        /// <summary>
        /// Given a bunch of normalized vectors, concatenate them together into a normalized vector.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static NormVector<T> ConcatWith<T>(this NormVector<T> me, params NormVector<T>[] others)
            => new ImplNorm<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns, concatenate them together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static Vector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVector<T>[] others)
            => new Impl<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns, concatenate them together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static Vector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVector<T>[] others)
            => new Impl<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        /// <summary>
        /// Given a set of columns including at least one variable sized vector column, concatenate them
        /// together into a vector valued column of the same type.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        /// <param name="me">The first input column.</param>
        /// <param name="others">Subsequent input columns.</param>
        /// <returns>The result of concatenating all input columns together.</returns>
        public static VarVector<T> ConcatWith<T>(this VarVector<T> me, params ScalarOrVectorOrVarVector<T>[] others)
            => new ImplVar<T>(Join(me, others));

        private interface IContainsColumn
        {
            PipelineColumn WrappedColumn { get; }
        }

        /// <summary>
        /// A wrapping object for the implicit conversions in <see cref="ConcatWith{T}(Scalar{T}, ScalarOrVector{T}[])"/>
        /// and other related methods.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        public sealed class ScalarOrVector<T> : ScalarOrVectorOrVarVector<T>
        {
            private ScalarOrVector(PipelineColumn col) : base(col) { }
            public static implicit operator ScalarOrVector<T>(Scalar<T> c) => new ScalarOrVector<T>(c);
            public static implicit operator ScalarOrVector<T>(Vector<T> c) => new ScalarOrVector<T>(c);
            public static implicit operator ScalarOrVector<T>(NormVector<T> c) => new ScalarOrVector<T>(c);
        }

        /// <summary>
        /// A wrapping object for the implicit conversions in <see cref="ConcatWith{T}(Scalar{T}, ScalarOrVectorOrVarVector{T}[])"/>
        /// and other related methods.
        /// </summary>
        /// <typeparam name="T">The value type.</typeparam>
        public class ScalarOrVectorOrVarVector<T> : IContainsColumn
        {
            public PipelineColumn WrappedColumn { get; }

            private protected ScalarOrVectorOrVarVector(PipelineColumn col)
            {
                Contracts.CheckValue(col, nameof(col));
                WrappedColumn = col;
            }

            public static implicit operator ScalarOrVectorOrVarVector<T>(VarVector<T> c)
               => new ScalarOrVectorOrVarVector<T>(c);
        }

        #region Implementation support
        private sealed class Rec : EstimatorReconciler
        {
            /// <summary>
            /// For the moment the concat estimator can only do one at a time, so I want to apply these operations
            /// one at a time, which means a separate reconciler. Otherwise there may be problems with name overwriting.
            /// If that is ever adjusted, then we can make a slightly more efficient reconciler, though this is probably
            /// not that important of a consideration from a runtime perspective.
            /// </summary>
            public static Rec Inst => new Rec();

            private Rec() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                // For the moment, the concat estimator can only do one concatenation at a time.
                // So we will chain the estimators.
                Contracts.AssertNonEmpty(toOutput);
                IEstimator<ITransformer> est = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var ccol = (IConcatCol)toOutput[i];
                    string[] inputs = ccol.Sources.Select(s => inputNames[s]).ToArray();
                    var localEst = new ConcatEstimator(env, outputNames[toOutput[i]], inputs);
                    if (i == 0)
                        est = localEst;
                    else
                        est = est.Append(localEst);
                }
                return est;
            }
        }

        private static PipelineColumn[] Join(PipelineColumn col, IContainsColumn[] cols)
        {
            if (Utils.Size(cols) == 0)
                return new[] { col };
            var retVal = new PipelineColumn[cols.Length + 1];
            retVal[0] = col;
            for (int i = 0; i < cols.Length; ++i)
                retVal[i + 1] = cols[i].WrappedColumn;
            return retVal;
        }

        private static PipelineColumn[] Join(PipelineColumn col, PipelineColumn[] cols)
        {
            if (Utils.Size(cols) == 0)
                return new[] { col };
            var retVal = new PipelineColumn[cols.Length + 1];
            retVal[0] = col;
            Array.Copy(cols, 0, retVal, 1, cols.Length);
            return retVal;
        }

        private interface IConcatCol
        {
            PipelineColumn[] Sources { get; }
        }

        private sealed class Impl<T> : Vector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public Impl(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }

        private sealed class ImplVar<T> : VarVector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public ImplVar(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }

        private sealed class ImplNorm<T> : NormVector<T>, IConcatCol
        {
            public PipelineColumn[] Sources { get; }
            public ImplNorm(PipelineColumn[] cols)
                : base(Rec.Inst, cols)
            {
                Sources = cols;
            }
        }
        #endregion
    }
}
