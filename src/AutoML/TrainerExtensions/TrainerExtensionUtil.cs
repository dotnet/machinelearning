// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Microsoft.ML.Auto
{
    internal enum TrainerName
    {
        AveragedPerceptronBinary,
        AveragedPerceptronOva,
        FastForestBinary,
        FastForestOva,
        FastForestRegression,
        FastTreeBinary,
        FastTreeOva,
        FastTreeRegression,
        FastTreeTweedieRegression,
        LightGbmBinary,
        LightGbmMulti,
        LightGbmRegression,
        LinearSvmBinary,
        LinearSvmOva,
        LogisticRegressionBinary,
        LogisticRegressionOva,
        LogisticRegressionMulti,
        OnlineGradientDescentRegression,
        OrdinaryLeastSquaresRegression,
        PoissonRegression,
        SdcaBinary,
        SdcaMulti,
        SdcaRegression,
        StochasticGradientDescentBinary,
        StochasticGradientDescentOva,
        SymSgdBinary,
        SymSgdOva
    }

    internal static class TrainerExtensionUtil
    {
        public static T CreateOptions<T>(IEnumerable<SweepableParam> sweepParams)
        {
            var options = Activator.CreateInstance<T>();
            if(sweepParams != null)
            {
                UpdateFields(options, sweepParams);
            }
            return options;
        }

        private static string[] _lightGbmTreeBoosterParamNames = new[] { "RegLambda", "RegAlpha" };
        private const string LightGbmTreeBoosterPropName = "Booster";

        public static LightGBM.Options CreateLightGbmOptions(IEnumerable<SweepableParam> sweepParams)
        {
            var options = new LightGBM.Options();
            if(sweepParams != null)
            {
                var treeBoosterParams = sweepParams.Where(p => _lightGbmTreeBoosterParamNames.Contains(p.Name));
                var parentArgParams = sweepParams.Except(treeBoosterParams);
                UpdateFields(options, parentArgParams);
                UpdateFields(options.Booster, treeBoosterParams);
            }
            return options;
        }

        public static IDictionary<string, object> BuildPipelineNodeProps(TrainerName trainerName, IEnumerable<SweepableParam> sweepParams)
        {
            if (trainerName == TrainerName.LightGbmBinary || trainerName == TrainerName.LightGbmMulti ||
                trainerName == TrainerName.LightGbmRegression)
            {
                return BuildLightGbmPipelineNodeProps(sweepParams);
            }

            return sweepParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
        }

        private static IDictionary<string, object> BuildLightGbmPipelineNodeProps(IEnumerable<SweepableParam> sweepParams)
        {
            var treeBoosterParams = sweepParams.Where(p => _lightGbmTreeBoosterParamNames.Contains(p.Name));
            var parentArgParams = sweepParams.Except(treeBoosterParams);

            var treeBoosterProps = treeBoosterParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
            var treeBoosterCustomProp = new CustomProperty("Options.TreeBooster.Arguments", treeBoosterProps);

            var props = parentArgParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
            props[LightGbmTreeBoosterPropName] = treeBoosterCustomProp;

            return props;
        }

        public static ParameterSet BuildParameterSet(TrainerName trainerName, IDictionary<string, object> props)
        {
            if (trainerName == TrainerName.LightGbmBinary || trainerName == TrainerName.LightGbmMulti ||
                trainerName == TrainerName.LightGbmRegression)
            {
                return BuildLightGbmParameterSet(props);
            }

            var paramVals = props.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            return new ParameterSet(paramVals);
        }

        private static ParameterSet BuildLightGbmParameterSet(IDictionary<string, object> props)
        {
            var parentProps = props.Where(p => p.Key != LightGbmTreeBoosterPropName);
            var treeProps = ((CustomProperty)props[LightGbmTreeBoosterPropName]).Properties;
            var allProps = parentProps.Union(treeProps);
            var paramVals = allProps.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            return new ParameterSet(paramVals);
        }

        private static void SetValue(FieldInfo fi, IComparable value, object obj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                fi.SetValue(obj, value);
            else if (propertyType == typeof(double) && value is float)
                fi.SetValue(obj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                fi.SetValue(obj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                fi.SetValue(obj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of object instance based on the values in sweepParams
        /// </summary>
        public static void UpdateFields(object obj, IEnumerable<SweepableParam> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    if (param.RawValue == null)
                    {
                        continue;
                    }
                    var fi = obj.GetType().GetField(param.Name);
                    var propType = Nullable.GetUnderlyingType(fi.FieldType) ?? fi.FieldType;

                    if (param is SweepableDiscreteParam dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(fi.FieldType) != null)
                                fi.SetValue(obj, null);
                            else if (fi.FieldType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = fi.FieldType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(fi.FieldType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    fi.SetValue(obj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(fi, (IComparable)dp.Options[optIndex], obj, propType);
                    }
                    else
                        SetValue(fi, param.RawValue, obj, propType);
                }
                catch (Exception)
                {
                    throw new InvalidOperationException("cannot set learner parameter");
                }
            }
        }
    }
}
