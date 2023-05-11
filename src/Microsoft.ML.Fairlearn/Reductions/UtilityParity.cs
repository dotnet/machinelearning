// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.Analysis;

namespace Microsoft.ML.Fairlearn
{
    /// <summary>
    /// Modeled after the original <see href="https://github.com/fairlearn/fairlearn/blob/931963c40c0ba0cdd1a9e51c29adcc509da224a6/fairlearn/reductions/_moments/utility_parity.py#L45">repo</see>
    /// A generic moment for parity in utilities (or costs) under classification.
    /// This serves as the base class for <see cref="DemographicParity">Demographic Parity</see>
    /// can be used as difference-based constraints or ratio-based constraints.
    /// 
    /// Constraints compare the group-level mean utility for each group with the
    /// overall mean utility
    /// 
    /// </summary>
    public class UtilityParity : ClassificationMoment
    {
        private const float _defaultDifferenceBound = 0.01F;
        private readonly float _epsilon;
        private readonly float _ratio;

        public float ProbEvent { get; protected set; }

        public DataFrameColumn ProbGroupEvent { get; protected set; }

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
        /// <param name="x">The features</param>
        /// <param name="y">The label</param>
        /// <param name="sensitiveFeature">The sensitive groups</param>
        public override void LoadData(IDataView x, DataFrameColumn y, StringDataFrameColumn sensitiveFeature)//, StringDataFrameColumn events = null, StringDataFrameColumn utilities = null)
        {
            base.LoadData(x, y, sensitiveFeature);
            //Tags["event"] = events;
            //Tags["utilities"] = utilities;

            //if (utilities == null)
            //{
            //    // TODO: set up the default utitlity
            //}

            //probEvent will contain the probabilities for each of the event, since we are now focusing on
            //TODO: implementing the demography parity which has only one event, we will set it like this for now.
            ProbEvent = 1.0F;
            //ProbEvent = Tags.GroupBy("event").Count / TotalSamples; We should use this if we have an event

            //Here the "label" column is just a dummy column for the end goal of getting the number of data rows
            ProbGroupEvent = Tags.GroupBy("group_id").Count()["label"] / (TotalSamples * 1.0);
        }

        /// <summary>
        /// Calculate the degree to which constraints are currently violated by the predictor.
        /// </summary>
        /// <returns></returns>
        public override DataFrame Gamma(PrimitiveDataFrameColumn<float> yPred/* Maybe change this to a predictor (func)*/)
        {
            Tags["pred"] = yPred;
            //TODO: add the utility into the calculation of the violation, will be needed for other parity methods
            //TODO: also we need to add the events column to the returned gamma singed
            //calculate upper bound difference and lower bound difference
            var expectEvent = Tags["pred"].Mean();
            var expectGroupEvent = Tags.GroupBy("group_id").Mean("pred").OrderBy(("group_id"))["pred"];
            var upperBoundDiff = _ratio * expectGroupEvent - expectEvent;
            var lowerBoundDiff = -1.0 /*to add a negative sign*/ * expectGroupEvent + _ratio * expectEvent;

            //the two diffs are going to be in the same column later on
            upperBoundDiff.SetName("value");
            lowerBoundDiff.SetName("value");

            //create the columns that hold the signs 
            StringDataFrameColumn posSign = new StringDataFrameColumn("sign", upperBoundDiff.Length);

            // a string column that has all the group names

            // var groupID = DataFrameColumn.Create("group_id", Tags["group_id"].Cast<string>());
            var groupID = Tags.GroupBy("group_id").Mean("pred").OrderBy("group_id")["group_id"];
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

        public override float Bound()
        {
            return _epsilon;
        }

        public override DataFrameColumn SignedWeights(DataFrame lambdaVec)
        {
            //TODO: calculate the propper Lambda Event and ProbEvent.
            // In the case of Demographic Parity, LambdaEvent contains one value, and ProbEvent is just 1, so we will skip it for now
            // lambdaEvent = (lambdaVec["+"] - _ratio * lambdaVec["-"])

            var gPos = lambdaVec.Filter(lambdaVec["sign"].ElementwiseEquals("+")).OrderBy("group_id");
            var gNeg = lambdaVec.Filter(lambdaVec["sign"].ElementwiseEquals("-")).OrderBy("group_id");
            var lambdaEvent = (float)(gPos["value"] - _ratio * gNeg["value"]).Sum() / ProbEvent;
            var lambdaGroupEvent = (_ratio * gPos["value"] - gNeg["value"]) / ProbGroupEvent;

            DataFrameColumn adjust = lambdaEvent - lambdaGroupEvent;
            DataFrame lookUp = new DataFrame(gPos["group_id"], adjust);
            //TODO: chech for null values i.e., if any entry in adjust is 0, make the corrosponding of singed weight to 0
            //TODO: add utility calculation, for now it is just 1 for everything
            long dataSetLength = Tags.Rows.Count();
            float[] signedWeightsFloat = new float[dataSetLength];
            // iterate through the rows of the original dataset of features
            long i = 0;
            foreach (DataFrameRow row in Tags.Rows)
            {
                // we are creating a new array where it will store the weight according the the lookup table (adjust) we created
                // TODO: right now this only supports one event, we have to filter through the event column so that this supports multiple events
                signedWeightsFloat[i] = Convert.ToSingle(lookUp.Filter(lookUp["group_id"].ElementwiseEquals(row["group_id"]))["value"][0]);
                i++;
            }

            DataFrameColumn signedWeights = new PrimitiveDataFrameColumn<float>("signedWeight", signedWeightsFloat);

            return signedWeights;
        }
    }

    public class DemographicParity : UtilityParity
    {
    }
}
