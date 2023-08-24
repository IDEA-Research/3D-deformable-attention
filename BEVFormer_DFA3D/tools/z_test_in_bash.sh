EPOCH="epoch_23"
DIR="bevformer_small_DFA3D_rerun3/"  # 
CONFIG="./projects/configs/bevformer/bevformer_small_DFA3D.py"  # bevformer_small_3DDef_MLVGDpt_6enc  bevformer_small_3DDef_MLVGDpt_6enc_1stage
CHECKPOINT="work_dirs/bevformer_small_DFA3D_rerun3/"$EPOCH".pth"
CUDA_VISIBLE_DEVICES=7 bash tools/dist_test.sh $CONFIG $CHECKPOINT 1
# >>"./work_dirs/"$DIR"/test/"$EPOCH"vis_dpt.log"
echo "config: "$CONFIG
echo "ckpt  : "$CHECKPOINT
