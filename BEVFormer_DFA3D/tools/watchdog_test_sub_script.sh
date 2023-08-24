#!/usr/bin/env bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time-min=5000
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lihongyang@idea.edu.cn
#SBATCH --qos=preemptive
#SBATCH --partition cvr
#SBATCH --gres=gpu:rtx:1  #* sync_gpu
#SBATCH --exclude cvr06,cvr07 
#SBATCH -N 1 # sync_node
#SBATCH --job-name=Test_py  #! sync_name
#SBATCH --output='./work_dirs/bevformer_small_DFA3D_rerun2/test/epoch_22.log'  #! sync_name #! sync_epoch
DIR='bevformer_small_DFA3D_rerun2'  #! sync_name
CONFIG='/comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/projects/configs/bevformer/bevformer_small_DFA3D.py'  #! sync_name
EPOCH='epoch_22'
WORK_DIR_ROOT='/comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/work_dirs/'
CHECKPOINT=$WORK_DIR_ROOT$DIR'/'$EPOCH'.pth'
JOB_NAME='bev_test'
WORK_DIR='./work_dirs/'$DIR'/test/'
OUTPUT_FILE=$WORK_DIR$EPOCH'.log'
echo ${CHECKPOINT}
echo ${CONFIG}
echo ${OUTPUT_FILE}
mkdir -p ${WORK_DIR}
function rand(){
        min=20000
        max=$(($60000-$min+1))
        num=$(($RANDOM+1000000000))
        echo $(($num%$max+$min))
}
p=$(rand)
PORT=${PORT:-$p}
PYTHONPATH='$(dirname $0)/..':$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT /comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox