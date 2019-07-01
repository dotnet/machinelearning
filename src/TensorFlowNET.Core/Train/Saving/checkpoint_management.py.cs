using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static Tensorflow.SaverDef.Types;

namespace Tensorflow
{
    public class checkpoint_management
    {
        /// <summary>
        /// Updates the content of the 'checkpoint' file.
        /// </summary>
        /// <param name="save_dir">Directory where the model was saved.</param>
        /// <param name="model_checkpoint_path">The checkpoint file.</param>
        /// <param name="all_model_checkpoint_paths">List of strings.</param>
        /// <param name="latest_filename"></param>
        /// <param name="save_relative_paths"></param>
        /// <param name="all_model_checkpoint_timestamps"></param>
        /// <param name="last_preserved_timestamp"></param>
        public static void update_checkpoint_state_internal(string save_dir,
            string model_checkpoint_path,
            List<string> all_model_checkpoint_paths = null,
            string latest_filename = "",
            bool save_relative_paths = false,
            List<float> all_model_checkpoint_timestamps = null,
            float? last_preserved_timestamp = null
            )
        {
            CheckpointState ckpt = null;

            // Writes the "checkpoint" file for the coordinator for later restoration.
            string coord_checkpoint_filename = _GetCheckpointFilename(save_dir, latest_filename);
            if (save_relative_paths)
            {
                throw new NotImplementedException("update_checkpoint_state_internal save_relative_paths");
            }
            else
            {
                ckpt = generate_checkpoint_state_proto(save_dir,
                    model_checkpoint_path,
                    all_model_checkpoint_paths,
                    all_model_checkpoint_timestamps,
                    last_preserved_timestamp);
            }

            if (coord_checkpoint_filename == ckpt.ModelCheckpointPath)
                throw new RuntimeError($"Save path '{model_checkpoint_path}' conflicts with path used for " +
                    "checkpoint state.  Please use a different save path.");

            File.WriteAllText(coord_checkpoint_filename, ckpt.ToString());
        }

        /// <summary>
        /// Returns a filename for storing the CheckpointState.
        /// </summary>
        /// <param name="save_dir">The directory for saving and restoring checkpoints.</param>
        /// <param name="latest_filename">
        /// Name of the file in 'save_dir' that is used
        /// to store the CheckpointState.
        /// </param>
        /// <returns>he path of the file that contains the CheckpointState proto.</returns>
        private static string _GetCheckpointFilename(string save_dir, string latest_filename)
        {
            if (string.IsNullOrEmpty(latest_filename))
                latest_filename = "checkpoint";

            return Path.Combine(save_dir, latest_filename);
        }

        private static CheckpointState generate_checkpoint_state_proto(string save_dir,
            string model_checkpoint_path,
            List<string> all_model_checkpoint_paths = null,
            List<float> all_model_checkpoint_timestamps = null,
            double? last_preserved_timestamp = null)
        {
            if (all_model_checkpoint_paths == null)
                all_model_checkpoint_paths = new List<string>();

            if (!all_model_checkpoint_paths.Contains(model_checkpoint_path))
                all_model_checkpoint_paths.Add(model_checkpoint_path);

            // Relative paths need to be rewritten to be relative to the "save_dir"
            // if model_checkpoint_path already contains "save_dir".

            var coord_checkpoint_proto = new CheckpointState()
            {
                ModelCheckpointPath = model_checkpoint_path
            };

            if (last_preserved_timestamp.HasValue)
                coord_checkpoint_proto.LastPreservedTimestamp = last_preserved_timestamp.Value;

            coord_checkpoint_proto.AllModelCheckpointPaths.AddRange(all_model_checkpoint_paths);
            if (all_model_checkpoint_timestamps != null)
                coord_checkpoint_proto.AllModelCheckpointTimestamps.AddRange(all_model_checkpoint_timestamps.Select(x => (double)x));

            return coord_checkpoint_proto;
        }

        /// <summary>
        /// Returns the meta graph filename.
        /// </summary>
        /// <param name="checkpoint_filename"></param>
        /// <param name="meta_graph_suffix"></param>
        /// <returns></returns>
        public static string meta_graph_filename(string checkpoint_filename, string meta_graph_suffix= "meta")
        {
            string basename = checkpoint_filename;
            string suffixed_filename = basename + "." + meta_graph_suffix;
            return suffixed_filename;
        }

        public static bool checkpoint_exists(string checkpoint_prefix)
        {
            string pathname = _prefix_to_checkpoint_path(checkpoint_prefix, CheckpointFormatVersion.V2);
            if (File.Exists(pathname))
                return true;
            else if (File.Exists(checkpoint_prefix))
                return true;
            else
                return false;
        }

        private static string _prefix_to_checkpoint_path(string prefix, CheckpointFormatVersion format_version)
        {
            if (format_version == CheckpointFormatVersion.V2)
                return prefix + ".index";
            return prefix;
        }
    }
}
