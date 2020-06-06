#python train.py --gpu 0 --data_path ./data/completion_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 4

python train.py --gpu 0 --data_path ./data/completion_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 4


# usage: train.py [-h] [--gpu GPU] --data_path DATA_PATH --train_file_list
#                 TRAIN_FILE_LIST [--val_file_list VAL_FILE_LIST] [--save SAVE]
#                 [--retrain RETRAIN] [--input_dim INPUT_DIM]
#                 [--encoder_dim ENCODER_DIM]
#                 [--coarse_feat_dim COARSE_FEAT_DIM]
#                 [--refine_feat_dim REFINE_FEAT_DIM] [--no_pass_occ]
#                 [--no_pass_feats] [--use_skip_sparse USE_SKIP_SPARSE]
#                 [--use_skip_dense USE_SKIP_DENSE] [--no_logweight_target_sdf]
#                 [--num_hierarchy_levels NUM_HIERARCHY_LEVELS]
#                 [--num_iters_per_level NUM_ITERS_PER_LEVEL]
#                 [--truncation TRUNCATION] [--batch_size BATCH_SIZE]
#                 [--start_epoch START_EPOCH] [--max_epoch MAX_EPOCH]
#                 [--save_epoch SAVE_EPOCH] [--lr LR] [--decay_lr DECAY_LR]
#                 [--weight_decay WEIGHT_DECAY]
#                 [--weight_sdf_loss WEIGHT_SDF_LOSS]
#                 [--weight_missing_geo WEIGHT_MISSING_GEO] [--vis_dfs VIS_DFS]
#                 [--use_loss_masking] [--no_loss_masking]
#                 [--scheduler_step_size SCHEDULER_STEP_SIZE]
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --gpu GPU             which gpu to use
#   --data_path DATA_PATH
#                         path to data
#   --train_file_list TRAIN_FILE_LIST
#                         path to file list of train data
#   --val_file_list VAL_FILE_LIST
#                         path to file list of val data
#   --save SAVE           folder to output model checkpoints
#   --retrain RETRAIN     model to load from
#   --input_dim INPUT_DIM
#                         voxel dim.
#   --encoder_dim ENCODER_DIM
#                         pointnet feature dim
#   --coarse_feat_dim COARSE_FEAT_DIM
#                         feature dim
#   --refine_feat_dim REFINE_FEAT_DIM
#                         feature dim
#   --no_pass_occ
#   --no_pass_feats
#   --use_skip_sparse USE_SKIP_SPARSE
#                         use skip connections between sparse convs
#   --use_skip_dense USE_SKIP_DENSE
#                         use skip connections between dense convs
#   --no_logweight_target_sdf
#   --num_hierarchy_levels NUM_HIERARCHY_LEVELS
#                         #hierarchy levels (must be > 1).
#   --num_iters_per_level NUM_ITERS_PER_LEVEL
#                         #iters before fading in training for next level.
#   --truncation TRUNCATION
#                         truncation in voxels
#   --batch_size BATCH_SIZE
#                         input batch size
#   --start_epoch START_EPOCH
#                         start epoch
#   --max_epoch MAX_EPOCH
#                         number of epochs to train for
#   --save_epoch SAVE_EPOCH
#                         save every nth epoch
#   --lr LR               learning rate, default=0.001
#   --decay_lr DECAY_LR   decay learning rate by half every n epochs
#   --weight_decay WEIGHT_DECAY
#                         weight decay.
#   --weight_sdf_loss WEIGHT_SDF_LOSS
#                         weight sdf loss vs occ.
#   --weight_missing_geo WEIGHT_MISSING_GEO
#                         weight missing geometry vs rest of sdf.
#   --vis_dfs VIS_DFS     use df (iso 1) to visualize
#   --use_loss_masking
#   --no_loss_masking
#   --scheduler_step_size SCHEDULER_STEP_SIZE
#                         #iters before scheduler step (0 for each epoch)
