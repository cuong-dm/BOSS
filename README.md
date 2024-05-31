This repository is the official implementation of the paper **"Boosting Offline Optimizers with Surrogate Sensitivity"**. 

# Abstract 
Offline optimization is an important task in numerous material engineering domains where online experimentation to collect data is too expensive and needs to be replaced by an in silico maximization of a surrogate of the black-box function. Although such a surrogate can be learned from offline data, its prediction might not be reliable outside the offline data regime, which happens when the surrogate has narrow prediction margin and is (therefore) sensitive to small perturbations of its parameterization. This raises the following questions: (1) how to regulate the sensitivity of a surrogate model; and (2) whether conditioning an offline optimizer with such less sensitive surrogate will lead to better optimization performance. To address these questions, we develop an optimizable sensitivity measurement for the surrogate model, which then inspires a sensitivity-informed regularizer that is applicable to a wide range of offline optimizers. This development is both orthogonal and synergistic to prior research on offline optimization, which is demonstrated in our extensive experiment benchmark.

# Requirements

This repo is builts on original repository: https://github.com/brandontrabucco/design-baselines

To set up the environment, you may follow the instruction from  https://github.com/kaist-silab/design-baselines-fixes.

# Running experiments 
We provide the whole scripts to run baselines with and without BOSS on 6 tasks as below:
* ## Baselines
    Run files in the folder 'BOSS/scripts/baseline-scripts'

* ## Baselines + BOSS
    Run files in the folder'BOSS/scripts/BOSS-scripts'

# Extracting results
To extract results, run below commands:
* ## Extracting baseline results
    ```
    design-baselines make-table --dir results/baselines/$ALGORITHM_NAME-$TASK_NAME --percentile 100th
    ```

* ## Extracting BOSS results
    ```
    design-baselines make-table --dir results/BOSS/$ALGORITHM_NAME-$TASK_NAME --percentile 100th
    ```