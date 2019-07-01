using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Contrib.Learn.Preprocessing
{
    public class VocabularyProcessor
    {
        private int _max_document_length;
        private int _min_frequency;

        public VocabularyProcessor(int max_document_length,
            int min_frequency)
        {
            _max_document_length = max_document_length;
            _min_frequency = min_frequency;
        }
    }
}
