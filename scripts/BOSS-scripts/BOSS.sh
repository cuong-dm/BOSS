NUM_CPUS=5
NUM_GPUS=1
for TASK_NAME in \
    ant \
    chembl \
    dkitty \
    superconductor \
    tf-bind-8 \
    tf-bind-10  ; do

  for ALGORITHM_NAME in \
      bo-qei-BOSS \
      cbas-BOSS \
      cma-es-BOSS \
      gradient-ascent-BOSS \
      gradient-ascent-mean-ensemble-BOSS \
      gradient-ascent-min-ensemble-BOSS \
      mins-BOSS \
      reinforce-BOSS ; do
  
    CUDA_VISIBLE_DEVICES=0 $ALGORITHM_NAME $TASK_NAME \
      --local-dir results/BOSS/$ALGORITHM_NAME-$TASK_NAME \
      --cpus $NUM_CPUS \
      --gpus $NUM_GPUS \
      --num-parallel 1 \
      --num-samples 8
    
  done
done