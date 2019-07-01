using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.Estimator
{
    public class HyperParams
    {
        /// <summary>
        /// root dir
        /// </summary>
        public string data_root_dir { get; set; }

        /// <summary>
        /// results dir
        /// </summary>
        public string result_dir { get; set; } = "results";

        /// <summary>
        /// model dir
        /// </summary>
        public string model_dir { get; set; } = "model";

        public string eval_dir { get; set; } = "eval";

        public string test_dir { get; set; } = "test";

        public int dim { get; set; } = 300;
        public float dropout { get; set; } = 0.5f;
        public int num_oov_buckets { get; set; } = 1;
        public int epochs { get; set; } = 25;
        public int epoch_no_imprv { get; set; } = 3;
        public int batch_size { get; set; } = 20;
        public int buffer { get; set; } = 15000;
        public int lstm_size { get; set; } = 100;
        public string lr_method { get; set; } = "adam";
        public float lr { get; set; } = 0.001f;
        public float lr_decay { get; set; } = 0.9f;

        /// <summary>
        /// lstm on chars
        /// </summary>
        public int hidden_size_char { get; set; } = 100;

        /// <summary>
        /// lstm on word embeddings
        /// </summary>
        public int hidden_size_lstm { get; set; } = 300;

        /// <summary>
        /// is clipping
        /// </summary>
        public bool clip { get; set; } = false;

        public string filepath_dev { get; set; }
        public string filepath_test { get; set; }
        public string filepath_train { get; set; }

        public string filepath_words { get; set; }
        public string filepath_chars { get; set; }
        public string filepath_tags { get; set; }
        public string filepath_glove { get; set; }

        public HyperParams(string dataDir)
        {
            data_root_dir = dataDir;

            if (string.IsNullOrEmpty(data_root_dir))
                throw new ValueError("Please specifiy the root data directory");

            if (!Directory.Exists(data_root_dir))
                Directory.CreateDirectory(data_root_dir);

            result_dir = Path.Combine(data_root_dir, result_dir);
            if (!Directory.Exists(result_dir))
                Directory.CreateDirectory(result_dir);

            model_dir = Path.Combine(result_dir, model_dir);
            if (!Directory.Exists(model_dir))
                Directory.CreateDirectory(model_dir);

            test_dir = Path.Combine(result_dir, test_dir);
            if (!Directory.Exists(test_dir))
                Directory.CreateDirectory(test_dir);

            eval_dir = Path.Combine(result_dir, eval_dir);
            if (!Directory.Exists(eval_dir))
                Directory.CreateDirectory(eval_dir);
        }
    }
}
