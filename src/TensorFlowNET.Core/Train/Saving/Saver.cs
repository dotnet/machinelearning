using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    /// <summary>
    /// Saves and restores variables.
    /// </summary>
    public class Saver
    {
        private VariableV1[] _var_list;
        private bool _reshape;
        private bool _sharded;
        private int _max_to_keep;
        private float _keep_checkpoint_every_n_hours;
        private string _name;
        private bool _restore_sequentially;
        private SaverDef _saver_def;
        private ISaverBuilder _builder;
        private bool _allow_empty;
        private bool _is_built;
        private SaverDef.Types.CheckpointFormatVersion _write_version;
        private bool _pad_step_number;
        private string _filename;
        private bool _is_empty;
        private float _next_checkpoint_time;
        private bool _save_relative_paths;
        private bool? _object_restore_saver;
        private Dictionary<string, float> _last_checkpoints;
        private Dictionary<string, float> _checkpoints_to_be_deleted;

        public Saver(VariableV1[] var_list = null,
            bool reshape = false,
            bool sharded = false,
            int max_to_keep = 5,
            float keep_checkpoint_every_n_hours = 10000,
            string name = null,
            bool restore_sequentially = false,
            SaverDef saver_def = null,
            ISaverBuilder builder = null,
            bool defer_build = false,
            bool allow_empty = false,
            SaverDef.Types.CheckpointFormatVersion write_version = SaverDef.Types.CheckpointFormatVersion.V2,
            bool pad_step_number = false,
            bool save_relative_paths = false,
            string filename = "")
        {
            _var_list = var_list;
            _reshape = reshape;
            _sharded = sharded;
            _max_to_keep = max_to_keep;
            _keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours;
            _name = name;
            _restore_sequentially = restore_sequentially;
            _saver_def = saver_def;
            _builder = builder;
            _is_built = false;
            _allow_empty = allow_empty;
            _write_version = write_version;
            _pad_step_number = pad_step_number;

            if (!defer_build)
                build();
            if(_saver_def != null)
            {
                _check_saver_def();
                _write_version = _saver_def.Version;
            }

            _save_relative_paths = save_relative_paths;
            _object_restore_saver = null;

            _last_checkpoints = new Dictionary<string, float>();
            _checkpoints_to_be_deleted = new Dictionary<string, float>();
        }

        public void build()
        {
            _build(_filename, build_save: true, build_restore: true);
        }

        private void _build(string checkpoint_path, bool build_save, bool build_restore)
        {
            if (_is_built) return;

            _is_built = true;

            if (_saver_def == null)
            {
                if (_builder == null)
                    _builder = new BulkSaverBuilder(_write_version);

                if (_var_list == null)
                    _var_list = variables._all_saveable_objects();

                if (_var_list == null || _var_list.Length == 0)
                {
                    if (_allow_empty)
                    {
                        _is_empty = true;
                        return;
                    }
                    else
                    {
                        throw new ValueError("No variables to save");
                    }
                }
                _is_empty = false;

                _saver_def = _builder._build_internal(_var_list,
                    reshape: _reshape,
                    sharded: _sharded,
                    max_to_keep: _max_to_keep,
                    keep_checkpoint_every_n_hours: _keep_checkpoint_every_n_hours,
                    name: _name,
                    restore_sequentially: _restore_sequentially,
                    filename: checkpoint_path,
                    build_save: build_save,
                    build_restore: build_restore);
            }
            else if (_saver_def != null && !string.IsNullOrEmpty(_name))
            {
                throw new NotImplementedException("Saver._build");
            }

            _check_saver_def();

            _next_checkpoint_time = Python.time() + _saver_def.KeepCheckpointEveryNHours * 3600;
        }

        private void _check_saver_def()
        {
            if (!tf.context.executing_eagerly())
            {
                if (string.IsNullOrEmpty(_saver_def.SaveTensorName))
                    throw new ValueError($"saver_def must specify the save_tensor_name: {_saver_def}");
                if (string.IsNullOrEmpty(_saver_def.RestoreOpName))
                    throw new ValueError($"saver_def must specify the restore_op_name: {_saver_def}");
            }
        }

        public string save(Session sess,
            string save_path,
            int global_step = -1,
            string latest_filename = "",
            string meta_graph_suffix = "meta",
            bool write_meta_graph = true,
            bool write_state = true,
            bool strip_default_attrs = false,
            bool save_debug_info = false)
        {
            if (string.IsNullOrEmpty(latest_filename))
                latest_filename = "checkpoint";
            string model_checkpoint_path = "";
            string checkpoint_file = "";

            if (global_step > 0)
                checkpoint_file = $"{save_path}-{global_step}";
            else
                checkpoint_file = save_path;

            var save_path_parent = Path.GetDirectoryName(save_path);

            if (!_is_empty)
            {
                model_checkpoint_path = sess.run(_saver_def.SaveTensorName,
                    new FeedItem(_saver_def.FilenameTensorName, checkpoint_file)
                );

                if (write_state)
                {
                    _RecordLastCheckpoint(model_checkpoint_path);
                    checkpoint_management.update_checkpoint_state_internal(
                        save_dir: save_path_parent,
                        model_checkpoint_path: model_checkpoint_path,
                        all_model_checkpoint_paths: _last_checkpoints.Keys.Select(x => x).ToList(),
                        latest_filename: latest_filename,
                        save_relative_paths: _save_relative_paths);
                    _MaybeDeleteOldCheckpoints(meta_graph_suffix: meta_graph_suffix);
                }
            }

            if (write_meta_graph)
            {
                string meta_graph_filename = checkpoint_management.meta_graph_filename(checkpoint_file, meta_graph_suffix: meta_graph_suffix);
                export_meta_graph(meta_graph_filename, strip_default_attrs: strip_default_attrs, save_debug_info: save_debug_info);
            }

            return _is_empty ? string.Empty : model_checkpoint_path;
        }

        public (Saver, object) import_meta_graph(string meta_graph_or_file, 
            bool clear_devices = false,
            string import_scope = "")
        {
            return saver._import_meta_graph_with_return_elements(meta_graph_or_file, clear_devices, import_scope);
        }

        /// <summary>
        /// Restores previously saved variables.
        /// 
        /// This method runs the ops added by the constructor for restoring variables.
        /// It requires a session in which the graph was launched.  The variables to
        /// restore do not have to have been initialized, as restoring is itself a way
        /// to initialize variables.
        /// </summary>
        /// <param name="sess">A `Session` to use to restore the parameters. None in eager mode.</param>
        /// <param name="save_path">Path where parameters were previously saved.</param>
        public void restore(Session sess, string save_path)
        {
            if (_is_empty)
                return;

            if (string.IsNullOrEmpty(save_path))
                throw new ValueError("Can't load save_path when it is None.");

            if (!checkpoint_management.checkpoint_exists(save_path))
                throw new ValueError($"The passed save_path is not a valid checkpoint: {save_path}");

            Console.WriteLine($"Restoring parameters from {save_path}");

            if (tf.context.executing_eagerly())
                ;
            else
                sess.run(_saver_def.RestoreOpName,
                    new FeedItem(_saver_def.FilenameTensorName, save_path));
        }

        /// <summary>
        /// Writes `MetaGraphDef` to save_path/filename.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="collection_list"></param>
        /// <param name="as_text"></param>
        /// <param name="export_scope"></param>
        /// <param name="clear_devices"></param>
        /// <param name="clear_extraneous_savers"></param>
        /// <param name="strip_default_attrs"></param>
        public MetaGraphDef export_meta_graph(string filename= "",
                        string[] collection_list = null,
                        string export_scope = "",
                        bool as_text = false,
                        bool clear_devices = false,
                        bool clear_extraneous_savers = false,
                        bool strip_default_attrs = false,
                        bool save_debug_info = false)
        {
            return export_meta_graph(
                filename: filename,
                graph_def: ops.get_default_graph().as_graph_def(add_shapes: true),
                saver_def: _saver_def,
                collection_list: collection_list,
                as_text: as_text,
                export_scope: export_scope,
                clear_devices: clear_devices,
                clear_extraneous_savers: clear_extraneous_savers,
                strip_default_attrs: strip_default_attrs);
        }

        public MetaGraphDef export_meta_graph(string filename = "",
            byte[] meta_info_def = null,
            GraphDef graph_def = null,
            SaverDef saver_def = null,
            string[] collection_list = null,
            bool as_text = false,
            bool clear_devices= false,
            bool clear_extraneous_savers= false,
            bool strip_default_attrs= false,
            string export_scope = "")
        {
            var meta_graph_def = meta_graph.export_scoped_meta_graph(
                filename: filename,
                meta_info_def: meta_info_def,
                graph_def: graph_def,
                saver_def: saver_def,
                // collection_list: collection_list,
                as_text: as_text,
                clear_devices: clear_devices,
                clear_extraneous_savers: clear_extraneous_savers,
                strip_default_attrs: strip_default_attrs);
            return meta_graph_def.Item1;
        }

        /// <summary>
        /// Manages the list of the latest checkpoints.
        /// </summary>
        /// <param name="latest_save_path"></param>
        private void _RecordLastCheckpoint(string latest_save_path)
        {
            if (_saver_def.MaxToKeep <= 0) return;

            // Remove first from list if the same name was used before.
            var _existed_checkpoints = _last_checkpoints.FirstOrDefault(p => latest_save_path == _CheckpointFilename((p.Key, p.Value)));
            if (_existed_checkpoints.Key != null)
                _last_checkpoints.Remove(_existed_checkpoints.Key);
            _last_checkpoints.Add(latest_save_path, time());

            // If more than max_to_keep, remove oldest.
            if (_last_checkpoints.Count > _saver_def.MaxToKeep)
            {
                var first = _last_checkpoints.First();
                _last_checkpoints.Remove(first.Key);
                _checkpoints_to_be_deleted[first.Key] = first.Value;
            }
        }

        private string _CheckpointFilename((string, float) p)
        {
            return p.Item1;
        }

        /// <summary>
        /// Deletes old checkpoints if necessary.
        /// </summary>
        /// <param name="meta_graph_suffix"></param>
        private void _MaybeDeleteOldCheckpoints(string meta_graph_suffix = "meta")
        {

        }
    }
}
