// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// An interface for exposing some number of items that can be opened for reading.
    /// </summary>
    /// REVIEW: Reconcile this with the functionality exposed by IHostEnvironment. For example,
    /// we could simply replace this with an array of IFileHandle.

    public interface IMultiStreamSource
    {
        /// <summary>
        /// Gets the number of items.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Return a string representing the "path" to the index'th stream. May return null.
        /// </summary>
        string GetPathOrNull(int index);

        /// <summary>
        /// Opens the indicated item and returns a readable stream on it.
        /// </summary>
        Stream Open(int index);

        /// <summary>
        /// Opens the indicated item and returns a text stream reader on it.
        /// </summary>
        /// REVIEW: Consider making this an extension method.
        TextReader OpenTextReader(int index);
    }

    /// <summary>
    /// Signature for creating an <see cref="ILegacyDataLoader"/>.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureDataLoader(IMultiStreamSource data);

    /// <summary>
    /// Signature for loading an <see cref="ILegacyDataLoader"/>.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureLoadDataLoader(ModelLoadContext ctx, IMultiStreamSource data);

    /// <summary>
    /// Interface for a data loader. An <see cref="ILegacyDataLoader"/> can save its model information
    /// and is instantiatable from arguments and an <see cref="IMultiStreamSource"/> .
    /// </summary>
    [BestFriend]
    internal interface ILegacyDataLoader : IDataView, ICanSaveModel
    {
    }

    [BestFriend]
    internal delegate void SignatureDataSaver();

    [BestFriend]
    internal interface IDataSaver
    {
        /// <summary>
        /// Check if the column can be saved.
        /// </summary>
        /// <returns>True if the column is savable.</returns>
        bool IsColumnSavable(DataViewType type);

        /// <summary>
        /// Save the data into the given stream. The stream should be kept open.
        /// </summary>
        /// <param name="stream">The stream that the data will be written.</param>
        /// <param name="data">The data to be saved.</param>
        /// <param name="cols">The list of column indices to be saved.</param>
        void SaveData(Stream stream, IDataView data, params int[] cols);
    }

    /// <summary>
    /// Signature for creating an <see cref="IDataTransform"/>.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureDataTransform(IDataView input);

    /// <summary>
    /// Signature for loading an <see cref="IDataTransform"/>.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureLoadDataTransform(ModelLoadContext ctx, IDataView input);

    /// <summary>
    /// Interface for a data transform. An <see cref="IDataTransform"/> can save its model information
    /// and is instantiatable from arguments and an input <see cref="IDataView"/>.
    /// </summary>
    [BestFriend]
    internal interface IDataTransform : IDataView, ICanSaveModel
    {
        IDataView Source { get; }
    }

    /// <summary>
    /// Data transforms need to be able to apply themselves to a different input IDataView.
    /// This interface allows them to implement custom rebinding logic.
    /// </summary>
    [BestFriend]
    internal interface ITransformTemplate : IDataTransform
    {
        // REVIEW: re-apply operation should support shallow schema modification,
        // like renaming source and destination columns.
        IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource);
    }
}
