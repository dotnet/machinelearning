// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security;
using System.Text;

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.OneDal
{


internal sealed class KnnDataView : IDataTransform
{
   
    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    private KnnAlgorithm _knn; // Expects the trained knn algorithm
    public KnnAlgorithm KNN { get { return _knn; }}

    public KnnDataView(IDataView source, KnnAlgorithm knn)
    {
        Source = source; // the "parent" dataview

	// FIXME -- copy from Source rather than construct from scratch
        var builder = new DataViewSchema.Builder();
        builder.AddColumn("Features", VectorDataViewType.Instance);
        builder.AddColumn("Label", NumberDataViewType.Instance);
        Schema = builder.ToSchema();
    }

    public long? GetRowCount() => null;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        => new Cursor(this, columnsNeeded.Any(c => c.Index == 0), columnsNeeded.Any(c => c.Index == 1));

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        => new[] { GetRowCursor(columnsNeeded, rand) };


     private sealed class Cursor : DataViewRowCursor
     {
         private KnnDataView _parent = Source;
         private bool _disposed;
         private long _position;
         private readonly Delegate[] _getters;

	 private int _featuresDimensionality;
	 private readonly IList< VBuffer<float> > _featureCache;
	 private int _itemsInCache = 0;
	 private readonly IList< int > _classCache;

	 private bool _parentIsEmpty = false;
	 private DataViewRowCursor _parentCursor;
	 private ValueGetter< VBuffer<float> > _featuresGetterFromCursor;
	 private int _currentIndex = -1;
	 private const int cacheCapacity = 1000;

         public override long Position => _position;
         public override long Batch => 0;
         public override DataViewSchema Schema { get; }

         public Cursor(KnnDataView parent, bool wantsFeatures, bool wantsLabels)
         {
	     _parent = parent;
             Schema = parent.Schema;
             _position = -1;
	     var featuresColumn = Schema["Features"];
     	     _featuresDimensionality = -1;
	     if (featuresColumn.Type is VectorDataViewType vt) {
	       _featuresDimensionality = vt.Size;
	     }
	     // FIXME -- report error when features column is not a vector (probably check from above)

	     _parentCursor = parent.Source.GetRowCursor(new [] { featuresColumn });
	     _featuresGetterFromCursor = _parentCursor.GetGetter< VBuffer<float> >(featuresColumn);
             _getters = new Delegate[]
             {
                 wantsFeatures ? (ValueGetter<VBuffer<float>>) FeaturesGetterImplementation : null,
                 wantsLabels ? (ValueGetter<int>) LabelsGetterImplementation : null

             };
         }

         protected override void Dispose(bool disposing)
         {
             if (_disposed)
                 return;
             if (disposing) {
                 // _enumerator.Dispose(); // FIXME -- dispose of parentcursor
                 _position = -1;
             }
             _disposed = true;
             base.Dispose(disposing);
         }

	 // FIXME
         private void FeaturesGetterImplementation(ref VBuffer<float> value) {
	   _featureCache[_currentIndex].CopyTo(value);
	 }
	 
         private void LabelsGetterImplementation(ref int value) => value = _classCache[_currentIndex];

         private void IdGetterImplementation(ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
         {
             if (!IsColumnActive(column))
                 throw new ArgumentOutOfRangeException(nameof(column));
             return (ValueGetter<TValue>)_getters[column.Index];
         }

         public override ValueGetter<DataViewRowId> GetIdGetter() => IdGetterImplementation;

         public override bool IsColumnActive(DataViewSchema.Column column)
             => _getters[column.Index] != null;

         public override bool MoveNext()
         {
             if (_disposed) return false;
	     if (_parentIsEmpty) {
	       Dispose();
	       return false;
	     }
	     
             if ((_itemsInCache == 0) || (_currentIndex >= _itemsInCache)) {
	     	_itemsInCache = FillCache();
		if (_itemsInCache <= 0) return false;
		_currentIndex = 0;
                 return true;
             }
	     _currentIndex = 0;
	     return true;
         }

	 private int FillCache()
	 {
	   if (_parentIsEmpty) return -1;
	   _featureCache.Clear();
	   int itemsInCache = 0;
	   VBuffer< float > featuresValue;
	   while (_parentCursor.MoveNext() && (itemsInCache < cacheCapacity)) {
	     _featuresGetterFromCursor( featuresValue );
	     _featureCache.Add(featuresValue);
	     itemsInCache++;
	   }

	   float [] featuresData = new float[itemsInCache * _featuresDimensionality];
	   Span<float> featuresSpan = new Span<float>(featuresData);
	   int index = 0;
	   foreach(var vb in _featureCache) {
	     int offset = index * _featuresDimensionality;
	     Span<float> target = featuresSpan.Slice(offset, _featuresDimensionality);
	     vb.GetValues().CopyTo(target);
	   }
	   float [] labelsTarget = new float[itemsInCache];
	   _parent.KNN.Predict(featuresData, labelsTarget); // FIXME -- should pass directly to Cursor
	   for (int i = 0; i < itemsInCache; i++) {
	     _classCache[i] = labelsTarget[i];              // FIXME -- settle on a name, labels or class
	   }

	   if (itemsInCache < cacheCapacity) _parentIsEmpty = true;
	   return itemsInCache;
	 }
    }
}
}
