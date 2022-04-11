// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.ObjectModel;
using System.Linq;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public class DataFrameJoinTests
    {
        [Fact]
        public void DataFrameJoinTests_GetSortedListsIntersection_EmptyCollections_EmptyResult()
        {
            // Arrange

            var collection1 = new Collection<long>();
            var collection2 = new Collection<long>();

            // Act

            var intersection = DataFrameJoinExtensions.GetSortedListsIntersection(collection1, collection2);

            // Assert

            Assert.Equal(0, intersection.Count);
        }

        [Fact]
        public void DataFrameJoinTests_GetSortedListsIntersection_EmptyCollections_FirstIsNotEmpty_EmptyResult()
        {
            // Arrange

            var collection1 = new Collection<long>()
            {
                111,
                222,
                333
            };

            var collection2 = new Collection<long>();

            // Act

            var intersection = DataFrameJoinExtensions.GetSortedListsIntersection(collection1, collection2);

            // Assert

            Assert.Equal(0, intersection.Count);
        }

        [Fact]
        public void DataFrameJoinTests_GetSortedListsIntersection_EmptyCollections_SecondIsNotEmpty_EmptyResult()
        {
            // Arrange

            var collection1 = new Collection<long>();

            var collection2 = new Collection<long>()
            {
                111,
                222,
                333
            };

            // Act

            var intersection = DataFrameJoinExtensions.GetSortedListsIntersection(collection1, collection2);

            // Assert

            Assert.Equal(0, intersection.Count);
        }

        [Fact]
        public void DataFrameJoinTests_GetSortedListsIntersection_SortedCollections_WithoutIntersection_Success()
        {
            // Arrange

            var collection1 = new Collection<long>()
            {
                111,
                222,
                333,
                888,
                999
            };

            var collection2 = new Collection<long>()
            {
                444,
                555,
                666,
                777
            };

            // Act

            var intersection = DataFrameJoinExtensions.GetSortedListsIntersection(collection1, collection2);

            // Assert

            Assert.Equal(0, intersection.Count);
        }

        [Fact]
        public void DataFrameJoinTests_GetSortedListsIntersection_SortedCollections_WithIntersection_Success()
        {
            // Arrange

            var collection1 = new Collection<long>()
            {
                111,
                222,
                333,
                444,
                555,
                888
            };

            var collection2 = new Collection<long>()
            {
                444,
                555,
                666,
                777,
                888,
                999
            };

            var expected = new Collection<long>
            {
                444,
                555,
                888
            };

            // Act

            var intersection = DataFrameJoinExtensions.GetSortedListsIntersection(collection1, collection2);

            // Assert

            Assert.True(expected.SequenceEqual(intersection));
        }
    }
}
