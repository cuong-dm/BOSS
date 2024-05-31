from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-dkitty')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on DKittyMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "DKittyMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "exploration_std": 0.1,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 2048,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-ant')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on AntMorphology-Exact-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "AntMorphology-Exact-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "exploration_std": 0.1,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 2048,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-hopper')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on HopperController-Exact-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": "HopperController-Exact-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 0,
        "exploration_std": 0.1,
        "reinforce_lr": 0.1,
        "reinforce_batch_size": 2048,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-superconductor')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
@click.option('--oracle', type=str, default="RandomForest")
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, oracle):
    """Evaluate reinforce on Superconductor-RandomForest-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": True,
        "task": f"Superconductor-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "exploration_std": 0.1,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 2048,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-chembl')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
@click.option('--assay-chembl-id', type=str, default='CHEMBL3885882')
@click.option('--standard-type', type=str, default='MCHC')
def chembl(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha,
           assay_chembl_id, standard_type):
    """Evaluate reinforce on ChEMBL-ResNet-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": f"ChEMBL_{standard_type}_{assay_chembl_id}"
                f"_MorganFingerprint-RandomForest-v0",
        "task_kwargs": {"relabel": False,
                        "dataset_kwargs": dict(
                            assay_chembl_id=assay_chembl_id,
                            standard_type=standard_type)},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 100,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 2048,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.0003,
        "ensemble_epochs": 50,
        "reinforce_lr": 1.0,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-gfp')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
@click.option('--oracle', type=str, default="Transformer")
def gfp(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha, oracle):
    """Evaluate reinforce on GFP-Transformer-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": f"GFP-{oracle}-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-tf_bind_8')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def tf_bind_8(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on TFBind8-Transformer-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "TFBind8-Exact-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-utr')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def utr(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on UTR-ResNet-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "UTR-ResNet-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-tf_bind_10')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def tf_bind_10(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on TFBind10-Exact-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "TFBind10-Exact-v0",
        "task_kwargs": {"relabel": False, "dataset_kwargs": {"max_samples": 10000}},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": True},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})


@cli.command()
@click.option('--local-dir', type=str, default='reinforce-nas')
@click.option('--cpus', type=int, default=24)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
@click.option('--omega_mu_init', type=float, default=0.)
@click.option('--omega_sigma_init', type=float, default=1e-3)
@click.option('--omega_mu_bound', type=float, default=1e-3)
@click.option('--omega_sigma_lower', type=float, default=1e-5)
@click.option('--omega_sigma_upper', type=float, default=1e-2)
@click.option('--lambda_', type=float, default=1e-3)
@click.option('--n_gamma', type=int, default=100)
@click.option('--lr_omega', type=float, default=1e-2)
@click.option('--alpha', type=float, default=0.1)
def nas(local_dir, cpus, gpus, num_parallel, num_samples, omega_mu_init, omega_sigma_init, omega_mu_bound, omega_sigma_lower, omega_sigma_upper, lambda_, n_gamma, lr_omega, alpha):
    """Evaluate reinforce on CIFARNAS-Exact-v0
    """

    # Final Version

    from design_baselines.reinforce_BOSS import reinforce_BOSS
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))
    tune.run(reinforce_BOSS, config={
        "omega_mu_init": omega_mu_init, 
        "omega_sigma_init": omega_sigma_init, 
        "omega_mu_bound": omega_mu_bound, 
        "omega_sigma_lower": omega_sigma_lower, 
        "omega_sigma_upper": omega_sigma_upper, 
        "lambda_": lambda_, 
        "n_gamma": n_gamma, 
        "lr_omega": lr_omega, 
        "alpha": alpha,
        "logging_dir": "data",
        "normalize_ys": True,
        "normalize_xs": False,
        "task": "CIFARNAS-Exact-v0",
        "task_kwargs": {"relabel": False},
        "optimize_ground_truth": False,
        "bootstraps": 5,
        "val_size": 200,
        "ensemble_batch_size": 100,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "ensemble_lr": 0.001,
        "ensemble_epochs": 100,
        "reinforce_lr": 0.01,
        "reinforce_batch_size": 256,
        "iterations": 200,
        "solver_samples": 128, "do_evaluation": False},
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01})
