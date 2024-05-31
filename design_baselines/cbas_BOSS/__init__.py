from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.utils import save_object, load_object
from design_baselines.logger import Logger
from design_baselines.cbas_BOSS.trainers import Ensemble
from design_baselines.cbas_BOSS.trainers import WeightedVAE
from design_baselines.cbas_BOSS.trainers import CBAS
from design_baselines.cbas_BOSS.nets import ForwardModel
from design_baselines.cbas_BOSS.nets import Encoder
from design_baselines.cbas_BOSS.nets import DiscreteDecoder
from design_baselines.cbas_BOSS.nets import ContinuousDecoder
from design_baselines.cbas_BOSS.nets import PhiGammaModel

import tensorflow as tf
import numpy as np
import os
import random
import torch



def cbas_BOSS(config):
    """Optimize a design problem score using the algorithm CBAS
    otherwise known as Conditioning by Adaptive Sampling

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """
    logger = Logger(config['logging_dir'])
    task = StaticGraphTask(config['task'], **config['task_kwargs'])
    
    if task.is_discrete:
        task.map_to_integers()

    if config['normalize_ys']:
        task.map_normalize_y()
    if config['normalize_xs']:
        task.map_normalize_x()

    x = task.x
    y = task.y

    # create the training task and logger
    train_data, val_data = build_pipeline(
        x=x, y=y, w=np.ones_like(y),
        val_size=config['val_size'],
        batch_size=config['ensemble_batch_size'],
        bootstraps=config['bootstraps'])

    # make several keras neural networks with two hidden layers
    forward_models = [ForwardModel(
        task,
        embedding_size=config['embedding_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        initial_max_std=config['initial_max_std'],
        initial_min_std=config['initial_min_std'])
        for b in range(config['bootstraps'])]
    
    phi_gamma_model = PhiGammaModel()
    # create a trainer for a forward model with a conservative objective
    omega_0 = tf.constant([[config["omega_mu_init"], config["omega_sigma_init"]]])
    
    # create a trainer for a forward model with a conservative objective
    ensemble = Ensemble(
        phi_gamma_model,
        omega_0,
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config['ensemble_lr'])
    lambda_ = config["lambda_"]
    n_gamma = config["n_gamma"]
    lr_omega = config["lr_omega"]
    alpha = config["alpha"]
    omega_mu_bound = config["omega_mu_bound"]
    omega_sigma_lower = config["omega_sigma_lower"]
    omega_sigma_upper = config["omega_sigma_upper"]
    # train the model for an additional number of epochs
    print("="*20 + "     ensemble launch     " + "="*20)
    ensemble.launch(omega_mu_bound, 
                    omega_sigma_lower, 
                    omega_sigma_upper,
                    n_gamma,
                    lr_omega,
                    alpha,
                    lambda_,
                    train_data,
                    val_data,
                    logger,
                    config['ensemble_epochs'])

    # determine which arcitecture for the decoder to use
    decoder = DiscreteDecoder \
        if task.is_discrete else ContinuousDecoder

    # build the encoder and decoder distribution and the p model
    p_encoder = Encoder(task, config['latent_size'],
                        embedding_size=config['embedding_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        initial_max_std=config['initial_max_std'],
                        initial_min_std=config['initial_min_std'])
    p_decoder = decoder(task, config['latent_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        initial_max_std=config['initial_max_std'],
                        initial_min_std=config['initial_min_std'])
    p_vae = WeightedVAE(p_encoder, p_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=config['vae_lr'],
                        vae_beta=config['vae_beta'])

    # build a weighted data set
    train_data, val_data = build_pipeline(
        x=x, y=y, w=np.ones_like(task.y),
        batch_size=config['vae_batch_size'],
        val_size=config['val_size'])

    # train the initial vae fit to the original data distribution
    p_vae.launch(train_data,
                 val_data,
                 logger,
                 config['offline_epochs'])

    # build the encoder and decoder distribution and the p model
    q_encoder = Encoder(task, config['latent_size'],
                        embedding_size=config['embedding_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        initial_max_std=config['initial_max_std'],
                        initial_min_std=config['initial_min_std'])
    q_decoder = decoder(task, config['latent_size'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        initial_max_std=config['initial_max_std'],
                        initial_min_std=config['initial_min_std'])
    q_vae = WeightedVAE(q_encoder, q_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=config['vae_lr'],
                        vae_beta=config['vae_beta'])

    # create the cbas importance weight generator
    cbas = CBAS(ensemble,
                p_vae,
                q_vae,
                latent_size=config['latent_size'])

    # train and validate the q_vae using online samples
    q_encoder.set_weights(p_encoder.get_weights())
    q_decoder.set_weights(p_decoder.get_weights())
    for i in range(config['iterations']):

        # generate an importance weighted dataset
        x_t, y_t, w = cbas.generate_data(
            config['online_batches'],
            config['vae_batch_size'],
            config['percentile'])

        # build a weighted data set
        train_data, val_data = build_pipeline(
            x=x_t.numpy(),
            y=y_t.numpy(),
            w=w.numpy(),
            batch_size=config['vae_batch_size'],
            val_size=config['val_size'])

        # train a vae fit using weighted maximum likelihood
        start_epoch = config['online_epochs'] * i + \
                      config['offline_epochs']
        q_vae.launch(train_data,
                     val_data,
                     logger,
                     config['online_epochs'],
                     start_epoch=start_epoch)

    
    # sample designs from the prior
    z = tf.random.normal([config['solver_samples'], config['latent_size']])
    q_dx = q_decoder.get_distribution(z, training=False)
    x_t = q_dx.sample()
    np.save(os.path.join(config["logging_dir"],
                         f"solution.npy"), x_t.numpy())
    
    if config["do_evaluation"]:
        score = task.predict(x_t)
        if task.is_normalized_y:
            score = task.denormalize_y(score)
        logger.record("score",
                      score,
                      config['iterations'],
                      percentile=True)
