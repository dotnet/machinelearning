// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Model.Pfa
{
    public interface ICanSavePfa
    {
        /// <summary>
        /// Whether this object really is capable of saving itself as part of a PFA
        /// pipeline. An implementor of this object might implement this interface,
        /// but still return <c>false</c> if there is some characteristic of this object
        /// only detectable during runtime that would prevent its being savable. (E.g.,
        /// it may wrap some other object that may or may not be savable.)
        /// </summary>
        bool CanSavePfa { get; }
    }

    public interface ISaveAsPfa
    {
        /// <summary>
        /// Save as PFA. For any columns that are output, this interface should use
        /// <see cref="BoundPfaContext.DeclareVar(string, JToken)"/> to declare themselves,
        /// while any unwritable columns should be registered <see cref="BoundPfaContext.Hide"/>.
        /// If <see cref="ICanSavePfa.CanSavePfa"/> is <c>false</c> this should not be called.
        /// </summary>
        /// <param name="ctx">The PFA program being built</param>
        void SaveAsPfa(BoundPfaContext ctx);
    }

    /// <summary>
    /// This data model component is savable as PFA. See http://dmg.org/pfa/ .
    /// </summary>
    public interface ITransformCanSavePfa : ICanSavePfa, ISaveAsPfa, IDataTransform
    {

    }

    /// <summary>
    /// This <see cref="ISchemaBindableMapper"/> is savable as a PFA. Note that this is
    /// typically called within an <see cref="IDataScorerTransform"/> that is wrapping
    /// this mapper, and has already been bound to it.
    /// </summary>
    public interface IBindableCanSavePfa : ICanSavePfa, ISchemaBindableMapper
    {
        /// <summary>
        /// Save as PFA. If <see cref="ICanSavePfa.CanSavePfa"/> is
        /// <c>false</c> this should not be called. This method is intended to be called
        /// by the wrapping scorer transform, and is intended to produce enough information
        /// for that purpose.
        /// </summary>
        /// <param name="ctx">The PFA program being built</param>
        /// <param name="schema">The role mappings that was passed to this bindable
        /// object, when the <see cref="ISchemaBoundMapper"/> was created that this transform
        /// is wrapping</param>
        /// <param name="outputNames">Since this method is called from a scorer transform,
        /// it is that transform that controls what the output column names will be, of
        /// the outputs produced by this bindable mapper. This is the array that holds
        /// those names, so that implementors of this method know what to produce in
        /// <paramref name="ctx"/>.</param>
        void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames);
    }

    /// <summary>
    /// For simple mappers. Intended to be used for <see cref="IValueMapper"/> and
    /// <see cref="Microsoft.ML.Runtime.Internal.Calibration.ICalibrator"/> instances.
    /// </summary>
    public interface ISingleCanSavePfa : ICanSavePfa
    {
        /// <summary>
        /// Implementors of this method are responsible for providing the PFA expression that
        /// computes the output of this object. Note that this method does not control what name
        /// will be given to the output, and is not responsible for declaring the variable into
        /// which the output will be returned. (Though, the method may of course declare other
        /// variables, cells, or such to enable this computation.)
        /// </summary>
        /// <param name="ctx">The PFA context</param>
        /// <param name="input">The PFA token representing the input. In the case of
        /// a predictor, for example, this would be a reference to the variable holding
        /// the features array.</param>
        /// <returns>A PFA expression</returns>
        JToken SaveAsPfa(BoundPfaContext ctx, JToken input);
    }

    /// <summary>
    /// For simple mappers. Intended to be used for <see cref="IValueMapperDist"/>
    /// instances.
    /// </summary>
    public interface IDistCanSavePfa : ISingleCanSavePfa, IValueMapperDist
    {
        /// <summary>
        /// The call for distribution predictors. Unlike <see cref="ISingleCanSavePfa.SaveAsPfa"/>,
        /// this method requires this method to handle the declaration of the variables for their
        /// outputs, into the names <paramref name="score"/> and <paramref name="prob"/> provided.
        /// </summary>
        /// <param name="ctx">The PFA context</param>
        /// <param name="input">The PFA token representing the input. In nearly all cases this will
        /// be the name of the variable holding the features array.</param>
        /// <param name="score">The name of the column where the implementing method should
        /// save the expression, through <see cref="BoundPfaContext.DeclareVar(string, JToken)"/>,
        /// or if <c>null</c></param>
        /// <param name="scoreToken"></param>
        /// <param name="prob">Similar to <paramref name="score"/>, except the probability expression.</param>
        /// <param name="probToken"></param>
        void SaveAsPfa(BoundPfaContext ctx, JToken input,
            string score, out JToken scoreToken, string prob, out JToken probToken);
    }
}