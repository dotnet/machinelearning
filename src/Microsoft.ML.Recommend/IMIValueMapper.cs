//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Recommend
{
    /// <summary>
    /// Interface for mapping a two input value (of the two indicated ColumnTypes) to
    /// an output value (of an indicated ColumnType). This interface is commonly implemented
    /// by predictors. Note that the input and output ColumnTypes determine the proper
    /// type arguments for GetMapper, but typically contain additional information like
    /// vector lengths.
    /// </summary>
    public interface IMIValueMapper
    {
        /// <summary>
        /// The type of the input value for the matrix column index.
        /// </summary>
        ColumnType InputXType { get; }

        /// <summary>
        /// The type of the input value for the matrix row index.
        /// </summary>
        ColumnType InputYType { get; }

        /// <summary>
        /// The type of the output value, that is, the type of the matrix elements.
        /// </summary>
        ColumnType OutputType { get; }

        /// <summary>
        /// Get a delegate used for mapping from inputs to output values. Note that the delegate
        /// should only be used on a single thread - it should NOT be assumed to be safe for concurrency.
        /// </summary>
        ValueMapper<TXIn, TYIn, TOut> GetMapper<TXIn, TYIn, TOut>();
    }

    /// <summary>
    /// This is the interface for the 'items selection' mapper. It takes the user's ID, features and per-item affinities,
    /// and produces ranked list of candidates.
    ///
    /// The interface is extensive, but the actual models will ignore some (or all) of the inputs. For example:
    /// - the 'always suggest most popular items' model will ignore all inputs.
    /// - the matrix factorization will ignore all the inputs except the user ID.
    /// - SAR will ignore user ID and features and only read the item history.
    /// - Matchbox in 'user to item' mode will use user ID and features and ignore the item history.
    /// </summary>
    /// <param name="userId">The user ID</param>
    /// <param name="userFeatures">The user feature vector.</param>
    /// <param name="items">The vector of items purchased by the user.</param>
    /// <param name="weights">For each item in <paramref name="items"/>, the user's 'affinity' to the item.</param>
    /// <param name="recommendedItems">Output. The ranked list of candidates.</param>
    /// <param name="scores">Output, parallel to <paramref name="recommendedItems"/>. The scores of the ranked candidates.</param>
    public delegate void UserHistoryToItemsMapper(
        ref uint userId, ref VBuffer<Single> userFeatures, ref VBuffer<uint> items, ref VBuffer<Single> weights,
        ref VBuffer<uint> recommendedItems, ref VBuffer<Single> scores);

    /// <summary>
    /// The interface for the 'user history -> items' recommender.
    /// In the end-to-end recommendation pipeline, this predictor can be used both as a standalone recommender, and as a
    /// candidate selection model. In the second case, the recommended items will then be featurized and ranked by downstream
    /// component.
    /// </summary>
    public interface IUserHistoryToItemsRecommender: IPredictor
    {
        /// <summary>
        /// The number of user features. 0 means variable size, -1 means that the algorithm ignores the user features.
        /// </summary>
        int UserFeaturesSize { get; }

        /// <summary>
        /// The type of the user ID. This will be a U4 key or null, if the algorithm doesn't care about the user ID.
        /// </summary>
        KeyType UserIdType { get; }

        /// <summary>
        /// The type of the item ID, both inputs and outputs. This will be a U4 key.
        /// </summary>
        KeyType ItemIdType { get; }

        /// <summary>
        /// Returns a mapper to invoke for item recommendations.
        /// </summary>
        /// <param name="recommendationCount">Maximum number of recommendation to return.</param>
        /// <param name="includeHistory">Whether to recommend items in the user's history.</param>
        UserHistoryToItemsMapper GetRecommendMapper(int recommendationCount, bool includeHistory);
    }
}
