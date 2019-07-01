using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Train
{
    public abstract class Trackable
    {
        protected int _self_update_uid;

        /// <summary>
        /// Restore-on-create for a variable be saved with this `Checkpointable`.
        /// </summary>
        /// <returns></returns>
        protected virtual RefVariable _add_variable_with_custom_getter(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            Func<string, int[], TF_DataType, IInitializer, bool, RefVariable> getter = null,
            bool overwrite = false,
            bool trainable = false)
        {
            var checkpoint_initializer = true;
            var new_variable = getter(name, shape, dtype, initializer, trainable);

            // If we set an initializer and the variable processed it, tracking will not
            // assign again. It will add this variable to our dependencies, and if there
            // is a non-trivial restoration queued, it will handle that. This also
            // handles slot variables.
            if (!overwrite || new_variable is RefVariable)
                return _track_checkpointable(new_variable, name: name,
                                        overwrite: overwrite);
            else
                return new_variable;
        }

        /// <summary>
        /// Pop and load any deferred checkpoint restores into `trackable`.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="trackable"></param>
        protected void _handle_deferred_dependencies(string name, RefVariable trackable)
        {
            _maybe_initialize_trackable();
            // TODO
        }

        protected RefVariable _track_checkpointable(RefVariable checkpointable, string name, bool overwrite = false)
        {
            return checkpointable;
        }

        /// <summary>
        /// Initialize dependency management.
        /// </summary>
        protected void _maybe_initialize_trackable()
        {
            // _self_unconditional_checkpoint_dependencies = []
            _self_update_uid = -1;
        }
    }
}
