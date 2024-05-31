from ray import tune
import click
import ray
import os
from random import randint


@click.group()
def cli():
    """A cleaned up implementation of Conservative Objective Models
    for reproducing some of our ICML 2021 rebuttal results.
    """


# CLEANED UP VERSION OF COMS - INTENDED FOR RELEASE POST-ICML 2021


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on DKittyMorphology-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "DKittyMorphology-v0",
        "task_kwargs": {"split_percentile": 40, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def ant(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on AntMorphology-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "AntMorphology-v0",
        "task_kwargs": {"split_percentile": 20, 'num_parallel': 2},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.4,
        "eval_beta": 0.4,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.4,
        "eval_beta": 0.4,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on Superconductor-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "Superconductor-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-molecule')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def molecule(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on MoleculeActivity-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "MoleculeActivity-v0",
        "task_kwargs": {'split_percentile': 80},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on GFP-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "GFP-v0",
        "task_kwargs": {'seed': tune.randint(1000)},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-tfbind8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def tfbind8(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on TfBind8-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "TfBind8-v0",
        "task_kwargs": {'split_percentile': 20},
        "is_discrete": True,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "discrete_clip": 0.6,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 50,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "train_beta": 0.4,
        "eval_beta": 0.4
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


# studying changes to COMs


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_tanh(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": 0.0,
        "final_tanh": True,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_tanh_no_cons(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 0.0,
        "alpha_lr": 0.0,
        "target_conservatism": 0.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": 0.0,
        "final_tanh": True,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_entropy(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": tune.grid_search([0.0, 0.01, 0.05, 0.1,
                                                 0.5, 1.0, 5.0, 10.0]),
        "final_tanh": False,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_entropy_no_cons(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": True,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 0.0,
        "alpha_lr": 0.0,
        "target_conservatism": 0.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": tune.grid_search([0.0, 0.01, 0.05, 0.1,
                                                 0.5, 1.0, 5.0, 10.0]),
        "final_tanh": False,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_tanh_denorm(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": 0.0,
        "final_tanh": True,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_tanh_no_cons_denorm(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 0.0,
        "alpha_lr": 0.0,
        "target_conservatism": 0.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": 0.0,
        "final_tanh": True,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_entropy_denorm(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 1.0,
        "alpha_lr": 0.01,
        "target_conservatism": -2.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": tune.grid_search([0.0, 0.01, 0.05, 0.1,
                                                 0.5, 1.0, 5.0, 10.0]),
        "final_tanh": False,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='coms-cleaned-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-7)
@click.option('--omega_mu_bound', type=float, default=1e-4)
@click.option('--omega_sigma_lower', type=float, default=1e-7)
@click.option('--omega_sigma_upper', type=float, default=1e-4)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha_rema', type=float, default=0.15)
@click.option('--path_hacked_task', type=str, default='')
def hopper_entropy_no_cons_denorm(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha_rema, path_hacked_task):
    """Evaluate Conservative Objective Models on HopperController-v0
    """

    from design_baselines.coms_cleaned_BOSS import coms_cleaned_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(coms_cleaned_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha_rema": alpha_rema,
        "logging_dir": "data",
        "task": "HopperController-v0",
        "task_kwargs": {},
        "is_discrete": False,
        "normalize_ys": True,
        "normalize_xs": False,
        "continuous_noise_std": 0.2,
        "val_size": 500,
        "batch_size": 128,
        "epochs": 20,
        "activations": ['leaky_relu', 'leaky_relu'],
        "hidden": 2048,
        "max_std": 0.2,
        "min_std": 0.1,
        "forward_model_lr": 0.0003,
        "train_beta": 0.0,
        "eval_beta": 0.0,
        "initial_alpha": 0.0,
        "alpha_lr": 0.0,
        "target_conservatism": 0.0,
        "inner_lr": 0.05,
        "outer_lr": 0.05,
        "inner_gradient_steps": 1,
        "outer_gradient_steps": 50,
        "entropy_coefficient": tune.grid_search([0.0, 0.01, 0.05, 0.1,
                                                 0.5, 1.0, 5.0, 10.0]),
        "final_tanh": False,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})

