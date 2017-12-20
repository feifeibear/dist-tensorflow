#!/bin/bash -l
#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=4
#SBATCH --constraint=gpu
#SBATCH --output=test-tf-%j.log
#SBATCH --error=test-tf-%j.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# env
module load daint-gpu
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3
export PYTHONPATH=/scratch/snx3000/youyang9/fjr/tf_workspace/distributed-compression-DNN/terngrad:$PYTHONPATH

set -x
set -e
PS=localhost
WORKER1=localhost
WORKER2=localhost

DATASET_NAME=cifar10 # imagenet or cifar10
DATA_DIR=${SCRATCH}/fjr/dataset/${DATASET_NAME}-data # dataset location
ROOT_WORKSPACE=${SCRATCH}/fjr/dataset/results/cifar10/ # the location to store summary and logs
if [ ! -d "$ROOT_WORKSPACE" ]; then
  echo "${ROOT_WORKSPACE} does not exsit!"
  exit
fi

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_training_data/
EVAL_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_eval_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
current_time=$(date)
current_time=`echo ${current_time} | sed 's/\ /-/g' | sed 's/://g'` #${current_time// /_}
#current_time=${current_time//:/-}
FOLDER_NAME=${DATASET_NAME}__${current_time}

export CUDA_VISIBLE_DEVICES=1
srun -n 4 python3 ./inception/cifar10_distributed_train.py \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 64 \
--num_epochs_per_decay 200 \
--max_steps 300000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--image_size 24 \
--data_dir=${DATA_DIR} \
--job_name='worker' \
--train_dir=/tmp/cifar10_distributed_train
#> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_w1_info.txt 2>&1 &

#export CUDA_VISIBLE_DEVICES=1
#python ./inception/cifar10_distributed_train.py \
#--optimizer adam \
#--initial_learning_rate 0.0002 \
#--batch_size 64 \
#--num_epochs_per_decay 200 \
#--max_steps 300000 \
#--seed 123 \
#--weight_decay 0.004 \
#--net cifar10_alexnet \
#--image_size 24 \
#--data_dir=${DATA_DIR} \
#--job_name='worker' \
#--task_id=0 \
#--ps_hosts="$PS:2222" \
#--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" \
#--train_dir=/tmp/cifar10_distributed_train #\
##> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_w2_info.txt 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=0
#python ./inception/cifar10_distributed_train.py \
#--job_name='ps' \
#--task_id=0 \
#--ps_hosts="$PS:2222" \
#--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" #\
##> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_ps_info.txt 2>&1 &
