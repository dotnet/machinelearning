// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.Analysis;


namespace Microsoft.ML.Fairlearn.reductions
{
    public class UtilityParity : ClassificationMoment
    {
        private const float _defaultDifferenceBound = 0.01F;

        private readonly float _epsilon;
        private readonly float _ratio;
        public UtilityParity(float differenceBound = Single.NaN, float ratioBond = Single.NaN, float ratioBoundSlack = 0.0f)
        {
            if (Single.NaN.Equals(differenceBound) && Single.NaN.Equals(ratioBond))
            {
                _epsilon = _defaultDifferenceBound;
                _ratio = 1.0F;
            }
            else if (!Single.NaN.Equals(differenceBound) && Single.NaN.Equals(ratioBond))
            {
                _epsilon = differenceBound;
                _ratio = 1.0F;
            }
            else if (Single.NaN.Equals(differenceBound) && !Single.NaN.Equals(ratioBond))
            {
                _epsilon = ratioBoundSlack;
                if (ratioBond <= 0.0f || ratioBond > 1.0f)
                {
                    throw new Exception("ratio must lie between (0.1]");
                }
                _ratio = ratioBond;
            }
            else
            {
                throw new Exception("Only one of difference_bound and ratio_bound can be used");
            }
        }
        //TODO: what should be the object type of X be? How can I make x capitilized to fit the whole data strcuture
        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="sensitiveFeature"></param>
        /// <param name="events"></param>
        /// <param name="utilities"></param>
        public void LoadData(IDataView x, DataFrameColumn y, StringDataFrameColumn sensitiveFeature, StringDataFrameColumn events, StringDataFrameColumn utilities = null)
        {
            base.LoadData(x, y, sensitiveFeature);
            Tags["event"] = events;
            Tags["utilities"] = utilities;

            if (utilities == null)
            {

            }

        }
        /// <summary>
        /// Calculate the degree to which constraints are currently violated by the predictor.
        /// </summary>
        /// <returns></returns>
        public new DataFrame Gamma(PrimitiveDataFrameColumn<float> yPred/*TODO: change to a predictor*/)
        {
            Tags["pred"] = yPred;
            //TODO: add the utility into the calculation of the violation, will be needed for other parity methods
            //TODO: also we need to add the events column to the returned gamma singed
            //calculate upper bound difference and lower bound difference
            var expectEvent = Tags["pred"].Mean();
            var expectGroupEvent = Tags.GroupBy("group_id").Mean()["pred"];
            var upperBoundDiff = _ratio * expectGroupEvent - expectEvent;
            var lowerBoundDiff = -1.0 /*to add a negative sign*/ * expectGroupEvent + _ratio * expectEvent;

            //the two diffs are going to be in the same column later on
            upperBoundDiff.SetName("value");
            lowerBoundDiff.SetName("value");

            //create the columns that hold the signs 
            StringDataFrameColumn posSign = new StringDataFrameColumn("sign", upperBoundDiff.Length);

            // a string column that has all the group names
            var groupID = Tags.GroupBy("group_id").Mean()["group_id"];

            // gSigned (gamma signed) is the dataframe that we return in the end that presents the uitility parity
            DataFrame gSigned = new DataFrame(posSign, groupID, upperBoundDiff);

            // plus sign for the upper bound
            gSigned["sign"].FillNulls("+", inPlace: true);

            // a temp dataframe that hold the utility rows for the lowerbound values
            StringDataFrameColumn negSign = new StringDataFrameColumn("sign", lowerBoundDiff.Length);
            DataFrame dfNeg = new DataFrame(negSign, groupID, lowerBoundDiff);
            dfNeg["sign"].FillNulls("-", inPlace: true);

            // stack the temp dataframe dfNeg to the bottom dataframe that we want to return
            dfNeg.Rows.ToList<DataFrameRow>().ForEach(row => { gSigned.Append(row, inPlace: true); });

            return gSigned;
        }

        //public float Signed
    }

    public class DemographicParity : UtilityParity
    {
    }
}
