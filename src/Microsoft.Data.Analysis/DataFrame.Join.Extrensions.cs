// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Microsoft.Data.Analysis
{
    public static class DataFrameJoinExtensions
    {
        public static IList<long> GetSortedListsIntersection(IReadOnlyList<long> sortedList1, IReadOnlyList<long> sortedList2)
        {
            var crossing = new Collection<long>();

            var i = 0;
            var j = 0;

            while (i < sortedList1.Count && j < sortedList2.Count)
            {
                var item1 = sortedList1[i];

                while (j < sortedList2.Count)
                {
                    var item2 = sortedList2[j];

                    if (item1 < item2)
                    {
                        i++;
                        break;
                    }
                    else if (item1 == item2)
                    {
                        crossing.Add(item1);
                        i++;
                        j++;
                        break;
                    }
                    else
                    {
                        j++;
                    }
                }
            }

            return crossing;
        }
    }
}
