// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public interface IPipelineNode<TModelType>
    {
        void SetInputData(Var<IDataView> data);

        void SetInputData(IDataView data, Experiment experiment);

        DataAndModel<TModelType> Add(Experiment experiment);

        T GetPropertyValueByName<T>(string name, T defaultValue);
    }

    public sealed class DataAndModel<TModel>
    {
        public Var<IDataView> OutData { get; }
        public Var<TModel> Model { get; }

        public DataAndModel(Var<IDataView> outData, Var<TModel> model)
        {
            OutData = outData;
            Model = model;
        }
    }

    public abstract class PipelineNodeBase
    {
        public virtual ParameterSet HyperSweeperParamSet { get; set; }

        protected virtual T1 CloneEntryPoint<T1>(T1 oldEp)
        {
            var newEp = oldEp.GetType().GetConstructor(new Type[] { })?.Invoke(new object[] { });
            if (newEp != null)
            {
                var propertyInfos = newEp.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static);
                foreach (var pi in propertyInfos)
                    pi.SetValue(newEp, pi.GetValue(oldEp));
            }
            return (T1)newEp;
        }

        protected string GetEpName(Type type)
        {
            string fullName = type.FullName;
            var epName = fullName?.Substring(fullName.Substring(0,
                                                 fullName.LastIndexOf(".", StringComparison.Ordinal)).LastIndexOf(".",
                                                 StringComparison.Ordinal) + 1) ?? type.Name;
            return epName;
        }

        protected void PropagateParamSetValues(ParameterSet hyperParams,
            TlcModule.SweepableParamAttribute[] sweepParams)
        {
            var spMap = sweepParams.ToDictionary(sp => sp.Name);

            foreach (var hp in hyperParams)
            {
                Contracts.Assert(spMap.ContainsKey(hp.Name));
                var sp = spMap[hp.Name];
                sp.SetUsingValueText(hp.ValueText);
            }
        }
    }

    public sealed class TransformPipelineNode : PipelineNodeBase, IPipelineNode<ITransformModel>
    {
        private readonly CommonInputs.ITransformInput _entryPointObj;
        private readonly CommonInputs.ITrainerInput _subTrainerObj;

        public TlcModule.SweepableParamAttribute[] SweepParams { get; }

        public TransformPipelineNode(CommonInputs.ITransformInput entryPointObj,
            IEnumerable<TlcModule.SweepableParamAttribute> sweepParams = null,
            CommonInputs.ITrainerInput subTrainerObj = null)
        {
            var newEp = CloneEntryPoint(entryPointObj);
            _entryPointObj = newEp ?? entryPointObj;
            if (subTrainerObj != null)
                _subTrainerObj = CloneEntryPoint(subTrainerObj);
            SweepParams = sweepParams?.Select(p => p.Clone()).ToArray() ??
                AutoMlUtils.GetSweepRanges(_entryPointObj.GetType());
        }

        public void SetInputData(Var<IDataView> data)
        {
            _entryPointObj.Data = data;
            if (_subTrainerObj != null)
                _subTrainerObj.TrainingData = data;
        }

        public void SetInputData(IDataView data, Experiment experiment)
        {
            experiment.SetInput(_entryPointObj.Data, data);
            if (_subTrainerObj != null)
                experiment.SetInput(_subTrainerObj.TrainingData, data);
        }

        public DataAndModel<ITransformModel> Add(Experiment experiment)
        {
            if (_subTrainerObj != null && _entryPointObj is CommonInputs.IFeaturizerInput epFeat)
                epFeat.PredictorModel = experiment.Add(_subTrainerObj).PredictorModel;
            var output = experiment.Add(_entryPointObj);
            return new DataAndModel<ITransformModel>(output.OutputData, output.Model);
        }

        public void UpdateProperties() => AutoMlUtils.UpdateProperties(_entryPointObj, SweepParams);

        public bool CheckEntryPointStateMatchesParamValues() => AutoMlUtils.CheckEntryPointStateMatchesParamValues(_entryPointObj, SweepParams);

        public string GetEpName()
        {
            return GetEpName(_entryPointObj.GetType());
        }

        public TransformPipelineNode Clone() => new TransformPipelineNode(_entryPointObj, SweepParams, _subTrainerObj);

        public override string ToString() => _entryPointObj.GetType().Name;

        public T GetPropertyValueByName<T>(string name, T defaultValue) =>
            (T)(_entryPointObj?.GetType().GetProperty(name)?.GetValue(_entryPointObj) ?? defaultValue);
    }

    public sealed class TrainerPipelineNode : PipelineNodeBase, IPipelineNode<IPredictorModel>
    {
        private readonly CommonInputs.ITrainerInput _entryPointObj;

        public TlcModule.SweepableParamAttribute[] SweepParams { get; }

        public TrainerPipelineNode(CommonInputs.ITrainerInput entryPointObj,
            IEnumerable<TlcModule.SweepableParamAttribute> sweepParams = null,
            ParameterSet hyperParameterSet = null)
        {
            var newEp = CloneEntryPoint(entryPointObj);
            _entryPointObj = newEp ?? entryPointObj;
            SweepParams = sweepParams?.Select(p => p.Clone()).ToArray() ??
                          AutoMlUtils.GetSweepRanges(_entryPointObj.GetType());
            HyperSweeperParamSet = hyperParameterSet?.Clone();

            // Make sure sweep params and param set are consistent.
            if (HyperSweeperParamSet != null)
            {
                PropagateParamSetValues(HyperSweeperParamSet, SweepParams);
                UpdateProperties();
            }
        }

        public void SetInputData(Var<IDataView> data)
        {
            _entryPointObj.TrainingData = data;
        }

        public void SetInputData(IDataView data, Experiment experiment)
        {
            experiment.SetInput(_entryPointObj.TrainingData, data);
        }

        public DataAndModel<IPredictorModel> Add(Experiment experiment)
        {
            var output = experiment.Add(_entryPointObj);
            return new DataAndModel<IPredictorModel>(_entryPointObj.TrainingData, output.PredictorModel);
        }

        public bool UpdateProperties() => AutoMlUtils.UpdateProperties(_entryPointObj, SweepParams);

        public bool CheckEntryPointStateMatchesParamValues() => AutoMlUtils.CheckEntryPointStateMatchesParamValues(_entryPointObj, SweepParams);

        public TrainerPipelineNode Clone() => new TrainerPipelineNode(_entryPointObj, SweepParams, HyperSweeperParamSet);

        public string GetEpName()
        {
            return GetEpName(_entryPointObj.GetType());
        }

        public override string ToString()
        {
            return $"{GetEpName()}{{{string.Join(", ", SweepParams.Select(p => $"{p.Name}:{p.ProcessedValue()}"))}}}";
        }

        public T GetPropertyValueByName<T>(string name, T defaultValue)
        {
            var value = _entryPointObj?.GetType().GetProperty(name)?.GetValue(_entryPointObj) ?? defaultValue;
            if (value.GetType() != typeof(T))
            {
                if (typeof(T) == typeof(double))
                    value = Convert.ToDouble(value);
                if (typeof(T) == typeof(long))
                    value = Convert.ToInt64(value);
                if (typeof(T) == typeof(int))
                    value = Convert.ToInt32(value);
                if (typeof(T) == typeof(float))
                    value = Convert.ToSingle(value);
            }
            return (T)value;
        }
    }
}
