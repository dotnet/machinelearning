// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    public class ObjectDetectionMetrics
    {
        /// <summary>
        /// Gets or sets mAP50 which means mean Average Precision(mAP) under IOU threshold 50%.
        /// </summary>
        public float MAP50 { get; set; }

        /// <summary>
        /// Gets or sets mAP , which is the average of mAP from IOU threshold 50% to 95% with step 5%, equaling to the
        /// average of mAP50, mAP55, mAP60... and mAP95.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
        public float MAP50_95 { get; set; }

        private class ObjectLabel
        {
            /// <summary>
            /// Gets or sets category for this sample.
            /// </summary>
            public string Category { get; set; }

            /// <summary>
            /// Gets or sets height of bounding box.
            /// </summary>
            public float Height { get; set; }

            /// <summary>
            /// Gets or sets width of bounding box.
            /// </summary>
            public float Width { get; set; }

            /// <summary>
            /// Gets or sets left border of bounding box.
            /// </summary>
            public float Left { get; set; }

            /// <summary>
            /// Gets or sets top border of bounding box.
            /// </summary>
            public float Top { get; set; }

            /// <summary>
            /// Gets or sets confidence score for model output.
            /// </summary>
            public float Confidence { get; set; }
        }

        private static List<List<ObjectLabel>> ThresholdFilter(List<List<ObjectLabel>> objectLabelListIn, float scoreThreshold)
        {
            int filterNum = 0;
            int total = 0;
            List<List<ObjectLabel>> objectLabelListOut = new();
            objectLabelListIn?.ForEach(objectLabelList =>
            {
                objectLabelListOut.Add(
                    objectLabelList.Where(objLabel => objLabel.Confidence >= scoreThreshold).ToList());
                filterNum += objectLabelList.Count - objectLabelListOut[^1].Count;
                total += objectLabelList.Count;
            });
            Console.WriteLine($"total : {total}, filtered: {filterNum}, filter ratio: {(total == 0 ? 0f : ((double)filterNum / (double)total)):P}");
            return objectLabelListOut;
        }

        internal static ObjectDetectionMetrics MeasureMetrics(IDataView dataView,
            DataViewSchema.Column labelCol,
            DataViewSchema.Column actualBoundingBoxColumn,
            DataViewSchema.Column predictedLabelCol,
            DataViewSchema.Column predictedBoundingBoxColumn,
            DataViewSchema.Column scoreCol
            )
        {
            var labelColEnumerable = dataView.GetColumn<VBuffer<ReadOnlyMemory<char>>>(labelCol);
            var labelSet = new HashSet<string>();

            foreach (var labelRow in labelColEnumerable)
            {
                foreach (var label in labelRow.DenseValues())
                    labelSet.Add(label.ToString());
            }
            var labels = labelSet.ToList();

            var expectedData = ActualIdvToObjLabl(dataView, labelCol, actualBoundingBoxColumn);
            var actualData = PredIdvToObjLabl(dataView, predictedLabelCol, predictedBoundingBoxColumn, scoreCol);

            actualData = ThresholdFilter(actualData, 0.05f);

            Dictionary<string, List<Tuple<float, float>>> iouDic = new();
            Dictionary<string, int> groundTruthBoxes = new();
            foreach (string label in labels)
            {
                iouDic.Add(label, new List<Tuple<float, float>>());
                groundTruthBoxes.Add(label, 0);
            }

            GenerateIOUDic(expectedData, actualData, iouDic, groundTruthBoxes);

            var evaluationMetrics = new ObjectDetectionMetrics
            {
                MAP50 = AP50(iouDic, groundTruthBoxes, 0.5f),
                MAP50_95 = AP50_95(iouDic, groundTruthBoxes)
            };

            return evaluationMetrics;
        }

        private static float AP50(Dictionary<string, List<Tuple<float, float>>> iouDic, Dictionary<string, int> groundTruthBoxes, float threshold = 0.5f)
        {
            int gt = groundTruthBoxes.Where(k => k.Value != 0).Count();
            if (gt == 0)
            {
                return 1.0f;   // no ground truth
            }

            float apSum = 0;
            foreach (string label in iouDic?.Keys)
            {
                apSum += LabelAP(iouDic[label], groundTruthBoxes[label], threshold);
            }

            return apSum / gt;
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
        private static float AP50_95(Dictionary<string, List<Tuple<float, float>>> iouDic, Dictionary<string, int> groundTruthBoxes)
        {
            int gt = groundTruthBoxes.Where(k => k.Value != 0).Count();
            if (gt == 0)
            {
                return 1.0f;   // no ground truth
            }

            float ap5095 = 0f;
            for (float thres = 0.5f; thres < 1f; thres += 0.05f)
            {
                float apSum = 0;
                foreach (string label in iouDic.Keys)
                {
                    apSum += LabelAP(iouDic[label], groundTruthBoxes[label], thres);
                }

                ap5095 += apSum / gt;
            }

            return ap5095 / ((1f - 0.5f) / 0.05f);
        }

        private static float CalculateIoU(ObjectLabel bbox, ObjectLabel gt)
        {
            // min overlap site
            float xleft = Math.Max(bbox.Left, gt.Left);
            float yleft = Math.Max(bbox.Top, gt.Top);
            float xright = Math.Min(bbox.Left + bbox.Width, gt.Left + gt.Width);
            float yright = Math.Min(bbox.Top + bbox.Height, gt.Top + gt.Height);

            if (xleft >= xright || yleft >= yright)
            {
                return 0f;
            }

            float overlap = (xright - xleft) * (yright - yleft);
            float sizePredict = bbox.Width * bbox.Height;
            float sizeGroundTrue = gt.Width * gt.Height;

            return overlap / (sizePredict + sizeGroundTrue - overlap);
        }

        private static void ImageIoU(List<ObjectLabel> objectLabel, List<ObjectLabel> annotation, Dictionary<string, List<Tuple<float, float>>> iouDic, Dictionary<string, int> groundTruthBoxes)
        {
            // calculations each two iou
            List<Tuple<int, int, float>> tupleList = new(); // annotaIndex, detectIndex, iouScore
            for (int annotaIndex = 0; annotaIndex < annotation.Count; annotaIndex++)
            {
                var gt = annotation[annotaIndex];
                groundTruthBoxes[gt.Category]++; // ground truth number
                for (int detectIndex = 0; detectIndex < objectLabel.Count; detectIndex++)
                {
                    var predBox = objectLabel[detectIndex];
                    if (predBox.Category != gt.Category)
                    {
                        continue;
                    }

                    float iou = CalculateIoU(predBox, gt);
                    if (iou != 0f)
                    {
                        tupleList.Add(Tuple.Create(annotaIndex, detectIndex, iou));
                    }
                }
            }

            tupleList.Sort((x, y) => y.Item3.CompareTo(x.Item3)); // descending sort

            bool[] annoSign = new bool[annotation.Count]; // whether a annotation bbox is used, default false
            bool[] predSign = new bool[objectLabel.Count]; // whether a predict bbox is used, default false
            foreach (var tuple in tupleList)
            {
                if (!annoSign[tuple.Item1] && !predSign[tuple.Item2])
                {
                    iouDic[annotation[tuple.Item1].Category].Add(
                        Tuple.Create(objectLabel[tuple.Item2].Confidence, tuple.Item3));
                    annoSign[tuple.Item1] = true;
                    predSign[tuple.Item2] = true;
                }
            }

            // add remain predict box as 0 iou
            for (int predSignIndex = 0; predSignIndex < predSign.Length; predSignIndex++)
            {
                if (!predSign[predSignIndex])
                {
                    iouDic[objectLabel[predSignIndex].Category].Add(
                        Tuple.Create(objectLabel[predSignIndex].Confidence, 0f));
                }
            }
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
        private static void GenerateIOUDic(List<List<ObjectLabel>> annotations, List<List<ObjectLabel>> objectLabelList, Dictionary<string, List<Tuple<float, float>>> iouDic, Dictionary<string, int> groundTruthBoxes)
        {
            // each image
            for (int annoIndex = 0; annoIndex < annotations?.Count; annoIndex++)
            {
                var objectLabels = objectLabelList?[annoIndex];
                var annotation = annotations[annoIndex];
                ImageIoU(objectLabels, annotation, iouDic, groundTruthBoxes);  // calculate iou of one kind
            }

            // sort list by label
            foreach (var detectionList in iouDic.Values)
            {
                detectionList.Sort((x, y) => y.Item1.CompareTo(x.Item1));   // compare score
            }
        }

        private static float LabelAP(List<Tuple<float, float>> detectionList, int groundTruthBoxesNum, float threshold = 0.5f)
        {
            if (groundTruthBoxesNum == 0)
            {
                return 0f; // should be 0 or 1 here?
            }

            if (detectionList?.Count == 0)
            {
                return 0f;
            }

            float tp = 0;   // true positive count
            Stack<Tuple<float, float>> prStack = new(); // precision and recall * groundTruthBoxesNum (= true positive)
            for (int index = 0; index < detectionList.Count; index++)
            {
                // compare iou
                if (detectionList[index].Item2 >= threshold)
                {
                    tp++;
                }

                prStack.Push(Tuple.Create(tp / (index + 1), tp));    // precision should be float
            }

            float precisionRecord = prStack.Peek().Item1;
            float recallRecord = prStack.Pop().Item2;
            float ap = 0f;  // ap value
            while (prStack.Count > 0)
            {
                Tuple<float, float> pr = prStack.Pop();
                float precision = pr.Item1;
                float recall = pr.Item2;
                if (precision > precisionRecord)
                {
                    ap += precisionRecord * (recallRecord - recall);
                    precisionRecord = precision;
                    recallRecord = recall;
                }
            }

            ap += precisionRecord * recallRecord;

            return ap / groundTruthBoxesNum;
        }

        private static List<List<ObjectLabel>> PredIdvToObjLabl(
            IDataView idv,
            DataViewSchema.Column predictedLabelCol,
            DataViewSchema.Column predictedBoundingBoxColumn,
            DataViewSchema.Column scoreCol)
        {
            var data = new List<List<ObjectLabel>>();
            var cursor = idv.GetRowCursor(predictedLabelCol, predictedBoundingBoxColumn, scoreCol);

            var predLabGet = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedLabelCol);
            var scoreGet = cursor.GetGetter<VBuffer<float>>(scoreCol);
            var boxGet = cursor.GetGetter<VBuffer<float>>(predictedBoundingBoxColumn);

            VBuffer<ReadOnlyMemory<char>> predLab = default;
            VBuffer<float> score = default;
            VBuffer<float> box = default;

            while (cursor.MoveNext())
            {
                predLabGet(ref predLab);
                scoreGet(ref score);
                boxGet(ref box);

                var l = new List<ObjectLabel>();
                var boxes = box.GetValues();
                var boxIdx = 0;

                for (int i = 0; i < score.Length; i++)
                {
                    var obj = new ObjectLabel();
                    obj.Confidence = score.GetValues()[i];
                    obj.Category = predLab.GetValues()[i].ToString();

                    obj.Left = boxes[boxIdx++];
                    obj.Top = boxes[boxIdx++];
                    obj.Width = (boxes[boxIdx++] - obj.Left + 1);
                    obj.Height = (boxes[boxIdx++] - obj.Top + 1);

                    l.Add(obj);
                }
                data.Add(l);
            }

            return data;
        }

        private static List<List<ObjectLabel>> ActualIdvToObjLabl(
            IDataView idv,
            DataViewSchema.Column labelCol,
            DataViewSchema.Column actualBoundingBoxColumn)
        {
            var data = new List<List<ObjectLabel>>();
            var cursor = idv.GetRowCursor(labelCol, actualBoundingBoxColumn);

            var predLabGet = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(labelCol);
            var boxGet = cursor.GetGetter<VBuffer<float>>(actualBoundingBoxColumn);

            VBuffer<ReadOnlyMemory<char>> predLab = default;
            VBuffer<float> box = default;

            while (cursor.MoveNext())
            {
                predLabGet(ref predLab);
                boxGet(ref box);

                var l = new List<ObjectLabel>();
                var boxes = box.GetValues();
                var boxIdx = 0;

                for (int i = 0; i < predLab.Length; i++)
                {
                    var obj = new ObjectLabel();
                    obj.Confidence = 1f;
                    obj.Category = predLab.GetValues()[i].ToString();

                    obj.Left = boxes[boxIdx++];
                    obj.Top = boxes[boxIdx++];
                    obj.Width = (boxes[boxIdx++] - obj.Left + 1);
                    obj.Height = (boxes[boxIdx++] - obj.Top + 1);

                    l.Add(obj);
                }
                data.Add(l);
            }

            return data;
        }
    }
}
