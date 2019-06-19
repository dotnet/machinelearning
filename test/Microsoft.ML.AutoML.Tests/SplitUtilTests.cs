// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class SplitUtilTests
    {
        /// <summary>
        /// When there's only one row of data, assert that
        /// attempted cross validation throws (all splits should have empty
        /// train or test set).
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void CrossValSplitThrowsWhenNotEnoughData()
        {
            var mlContext = new MLContext();
            var dataViewBuilder = new ArrayDataViewBuilder(mlContext);
            dataViewBuilder.AddColumn("Number", NumberDataViewType.Single, 0f);
            dataViewBuilder.AddColumn("Label", NumberDataViewType.Single, 0f);
            var dataView = dataViewBuilder.GetDataView();
            SplitUtil.CrossValSplit(mlContext, dataView, 10, null);
        }

        /// <summary>
        /// When there are few rows of data, assert that
        /// cross validation succeeds, but # of splits is less than 10
        /// (splits with empty train or test sets should not be returned from this API).
        /// </summary>
        [TestMethod]
        public void CrossValSplitSmallDataView()
        {
            var mlContext = new MLContext(seed: 0);
            var dataViewBuilder = new ArrayDataViewBuilder(mlContext);
            dataViewBuilder.AddColumn("Number", NumberDataViewType.Single, new float[9]);
            dataViewBuilder.AddColumn("Label", NumberDataViewType.Single, new float[9]);
            var dataView = dataViewBuilder.GetDataView();
            const int requestedNumSplits = 10;
            var splits = SplitUtil.CrossValSplit(mlContext, dataView, requestedNumSplits, null);
            Assert.IsTrue(splits.trainDatasets.Any());
            Assert.IsTrue(splits.trainDatasets.Count() < requestedNumSplits);
            Assert.AreEqual(splits.trainDatasets.Count(), splits.validationDatasets.Count());
        }

        /// <summary>
        /// Assert that with many rows of data, cross validation produces the requested
        /// # of splits.
        /// </summary>
        [TestMethod]
        public void CrossValSplitLargeDataView()
        {
            var mlContext = new MLContext(seed: 0);
            var dataViewBuilder = new ArrayDataViewBuilder(mlContext);
            dataViewBuilder.AddColumn("Number", NumberDataViewType.Single, new float[10000]);
            dataViewBuilder.AddColumn("Label", NumberDataViewType.Single, new float[10000]);
            var dataView = dataViewBuilder.GetDataView();
            const int requestedNumSplits = 10;
            var splits = SplitUtil.CrossValSplit(mlContext, dataView, requestedNumSplits, null);
            Assert.IsTrue(splits.trainDatasets.Any());
            Assert.AreEqual(requestedNumSplits, splits.trainDatasets.Count());
            Assert.AreEqual(requestedNumSplits, splits.validationDatasets.Count());
        }
    }
}
