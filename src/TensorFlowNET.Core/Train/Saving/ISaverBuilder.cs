using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface ISaverBuilder
    {
        Operation save_op(Tensor filename_tensor, SaveableObject[] saveables);

        Tensor[] bulk_restore(Tensor filename_tensor, SaveableObject[] saveables, int preferred_shard, bool restore_sequentially);

        SaverDef _build_internal(VariableV1[] names_to_saveables, 
            bool reshape = false, 
            bool sharded = false, 
            int max_to_keep = 5,
            float keep_checkpoint_every_n_hours = 10000,
            string name = null,
            bool restore_sequentially = false,
            string filename = "model",
            bool build_save = true,
            bool build_restore = true);
    }
}
