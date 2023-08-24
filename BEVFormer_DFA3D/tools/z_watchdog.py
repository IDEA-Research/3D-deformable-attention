import time
import os
import random
import subprocess
from glob import glob
import bai
def test_epoch(name, config, epoch):
    GPUS="1"
    GPUS_PER_NODE="1"
    MEM_PER_NODE="20G"
    CPUS_PER_TASK="10"
    TIME_OUT_MIN="40"
    PORT = random.randint(20000, 39999)
    running_script = \
f"#!/usr/bin/env bash\n\
#SBATCH --cpus-per-task=10\n\
#SBATCH --mem=20G\n\
#SBATCH --time-min=5000\n\
#SBATCH --requeue\n\
#SBATCH --mail-type=ALL\n\
#SBATCH --mail-user=lihongyang@idea.edu.cn\n\
#SBATCH --qos=preemptive\n\
#SBATCH --partition cvr\n\
#SBATCH --gres=gpu:rtx:1  #* sync_gpu\n\
#SBATCH --exclude cvr06,cvr07 \n\
#SBATCH -N 1 # sync_node\n\
#SBATCH --job-name=Test_{config.split('.')[-1]}  #! sync_name\n\
#SBATCH --output='./work_dirs/{name}/test/{epoch}.log'  #! sync_name #! sync_epoch\n\
DIR='{name}'  #! sync_name\n\
CONFIG='/comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/projects/configs/bevformer/{config}'  #! sync_name\n\
EPOCH='{epoch}'\n\
WORK_DIR_ROOT='/comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/work_dirs/'\n\
CHECKPOINT=$WORK_DIR_ROOT$DIR'/'$EPOCH'.pth'\n\
JOB_NAME='bev_test'\n\
WORK_DIR='./work_dirs/'$DIR'/test/'\n\
OUTPUT_FILE=$WORK_DIR$EPOCH'.log'\n\
echo ${{CHECKPOINT}}\n\
echo ${{CONFIG}}\n\
echo ${{OUTPUT_FILE}}\n\
mkdir -p ${{WORK_DIR}}\n\
function rand(){{\n\
        min=20000\n\
        max=$(($60000-$min+1))\n\
        num=$(($RANDOM+1000000000))\n\
        echo $(($num%$max+$min))\n\
}}\n\
p=$(rand)\n\
PORT=${{PORT:-$p}}\n\
PYTHONPATH='$(dirname $0)/..':$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT /comp_robot/lihongyang/code/DFA3D_Opensource_Remote/BEVFormer_DFA3D/tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${{@:4}} --eval bbox\
"
    
    print(f'excuting: \n {running_script}')
    with open("./tools/watchdog_test_sub_script.sh", "w") as f:
        f.write(running_script)
        time.sleep(5)
    os.system("sbatch tools/watchdog_test_sub_script.sh")
    # subprocess.call(running_script, shell=True)
def check_iscomplete(log_file):
    complete = False
    error = False
    with open(log_file, "r") as f:
        results = f.readlines()
        for line in results[-100:]:
            if "Calculating metrics..." in line:
                complete = True
            if "RuntimeError" in line or "error" in line:
                error = True
    return complete, error
def send_log_results_to_host(log_file):
    with open(log_file, "r") as f:
        results = f.readlines()
        line_start = 100
        message = ""
        for l_id, line in enumerate(results[-100:]):
            if "mAP:" in line:
                line_start = l_id
            if (l_id >= line_start) and (l_id - line_start)<9:
                message += "\t"+line
    bai.text(f"Test complete:\nLogFile: {log_file}\nResults:\n{message}")
def check_storage(threshold_warning, threshold_force=None):
    results = os.popen("cd ~/code && du --max-depth 1 -h").read()
    storage_used = int(results.split("\n")[-2].split("G")[0])
    if storage_used > threshold_warning:
        bai.text("Disk quota is going to explode! \n"+results)

if __name__ == "__main__":
    config_file="./projects/configs/bevformer/bevformer_small_DFA3D.py"
    exp_name = "bevformer_small_DFA3D_rerun3"

    root_work_dir = "./work_dirs/"
    work_dir = root_work_dir + exp_name
    test_dir = work_dir + "/test/"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    count_no_new = 0
    print("listening to :", exp_name)
    while True:
        ckpts = glob(work_dir + "/*.pth")
        test_logs = glob(test_dir + "*.log")
        tested = [test_log.split("/")[-1].split(".")[0] for test_log in test_logs]
        for ckpt in ckpts:
            name_ckpt = ckpt.split("/")[-1].split(".")[0]
            if name_ckpt == "latest":
                continue
            if name_ckpt in tested:
                continue
            else:
                ckpt_file = os.path.join(work_dir, name_ckpt + ".pth")
                log_file = os.path.join(test_dir, name_ckpt + ".log")
                print(f"find new ckpt: {ckpt_file}, log the results at {log_file}")
                test_epoch(exp_name, config_file.split("/")[-1], name_ckpt)
                while True:
                    time.sleep(60)
                    if os.path.exists(log_file):
                        complete, error = check_iscomplete(log_file)
                        while error:
                            time.sleep(60)
                            test_epoch(exp_name, config_file.split("/")[-1], name_ckpt)
                            complete, error = check_iscomplete(log_file)
                        if complete:
                            send_log_results_to_host(log_file)
                            check_storage(140)
                            time.sleep(2*60)
                            break
                # else:
                #     while not check_iscomplete(log_file):
                #         print(f"Retesting ckpt: {ckpt_file}, log the results at {log_file}")
                #         os.remove(log_file)
                #         test_epoch(config_file, ckpt_file, log_file)
                count_no_new = 0
        count_no_new += 1
        if count_no_new == 120:
            print("sleeping warning exceed max_sleep")
        print("sleeping")
        time.sleep(5*60)
