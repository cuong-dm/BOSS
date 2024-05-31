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
      bo-qei \
      cbas \
      cma-es \
      gradient-ascent \
      gradient-ascent-mean-ensemble \
      gradient-ascent-min-ensemble \
      mins \
      reinforce ; do
  
    CUDA_VISIBLE_DEVICES=0 $ALGORITHM_NAME $TASK_NAME \
      --local-dir results/baselines/$ALGORITHM_NAME-$TASK_NAME \
      --cpus $NUM_CPUS \
      --gpus $NUM_GPUS \
      --num-parallel 1 \
      --num-samples 8
    
  done
done