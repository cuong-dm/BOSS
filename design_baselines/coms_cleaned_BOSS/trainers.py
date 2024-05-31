from design_baselines.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class ConservativeObjectiveModel(tf.Module):

    def __init__(self, PhiGammaModel, omega_t,
                 forward_model,
                 forward_model_opt=tf.keras.optimizers.Adam,
                 PhiGammaModel_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001, alpha=1.0,
                 alpha_opt=tf.keras.optimizers.Adam,
                 alpha_lr=0.01, overestimation_limit=0.5,
                 particle_lr=0.05, particle_gradient_steps=50,
                 entropy_coefficient=0.9, noise_std=0.0):
        """A trainer class for building a conservative objective model
        by optimizing a model to make conservative predictions

        Arguments:

        forward_model: tf.keras.Model
            a tf.keras model that accepts designs from an MBO dataset
            as inputs and predicts their score
        forward_model_opt: tf.keras.Optimizer
            an optimizer such as the Adam optimizer that defines
            how to update weights using gradients
        forward_model_lr: float
            the learning rate for the optimizer used to update the
            weights of the forward model during training
        alpha: float
            the initial value of the lagrange multiplier in the
            conservatism objective of the forward model
        alpha_opt: tf.keras.Optimizer
            an optimizer such as the Adam optimizer that defines
            how to update the lagrange multiplier
        alpha_lr: float
            the learning rate for the optimizer used to update the
            lagrange multiplier during training
        overestimation_limit: float
            the degree to which the predictions of the model
            overestimate the true score function
        particle_lr: float
            the learning rate for the gradient ascent optimizer
            used to find adversarial solution particles
        particle_gradient_steps: int
            the number of gradient ascent steps used to find
            adversarial solution particles
        entropy_coefficient: float
            the entropy bonus added to the loss function when updating
            solution particles with gradient ascent
        noise_std: float
            the standard deviation of the gaussian noise added to
            designs when training the forward model
        """

        super().__init__()
        self.forward_model = forward_model
        self.PhiGammaModel = PhiGammaModel
        self.omega_t = omega_t
        self.forward_model_opt = \
            forward_model_opt(learning_rate=forward_model_lr)
        self.PhiGammaModel_optim = PhiGammaModel_optim(learning_rate=0.001)
        

        # lagrangian dual descent variables
        self.log_alpha = tf.Variable(np.log(alpha).astype(np.float32))
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.exp)
        self.alpha_opt = alpha_opt(learning_rate=alpha_lr)

        # algorithm hyper parameters
        self.overestimation_limit = overestimation_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

    @tf.function(experimental_relax_shapes=True)
    def optimize(self, x, steps, **kwargs):
        """Using gradient descent find adversarial versions of x
        that maximize the conservatism of the model

        Args:

        x: tf.Tensor
            the starting point for the optimizer that will be
            updated using gradient ascent
        steps: int
            the number of gradient ascent steps to take in order to
            find x that maximizes conservatism

        Returns:

        optimized_x: tf.Tensor
            a new design found by perform gradient ascent starting
            from the initial x provided as an argument
        """

        # gradient ascent on the conservatism
        def gradient_step(xt):
            with tf.GradientTape() as tape:
                tape.watch(xt)

                # shuffle the designs for calculating entropy
                shuffled_xt = tf.gather(
                    xt, tf.random.shuffle(tf.range(tf.shape(xt)[0])))

                # entropy using the gaussian kernel
                entropy = tf.reduce_mean((xt - shuffled_xt) ** 2)

                # the predicted score according to the forward model
                score = self.forward_model(xt, **kwargs)

                # the conservatism of the current set of particles
                loss = self.entropy_coefficient * entropy + score

            # update the particles to maximize the conservatism
            return tf.stop_gradient(
                xt + self.particle_lr * tape.gradient(loss, xt)),

        # use a while loop to perform gradient ascent on the score
        return tf.while_loop(
            lambda xt: True, gradient_step, (x,),
            maximum_iterations=steps)[0]

    @tf.function(experimental_relax_shapes=True)
    def update_omega(self, gamma_tensor, z_tensor, n_gamma):
        for gamma_epoch in range(5):
            with tf.GradientTape() as tape:
                pred = self.PhiGammaModel(gamma_tensor)
                loss = tf.keras.losses.BinaryCrossentropy()(pred, z_tensor)
        
            grads = tape.gradient(loss, self.PhiGammaModel.trainable_variables)
            self.PhiGammaModel_optim.apply_gradients(zip(grads, self.PhiGammaModel.trainable_variables))
            
        dl2_dmu = 0.
        dl2_dsigma = 0.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gamma_tensor)
            l2 = self.PhiGammaModel(gamma_tensor)
        dl2_dgammai = tape.gradient(l2, gamma_tensor)
        dl2_dgammai_sum = tf.math.reduce_sum(dl2_dgammai)
        dl2_dmu += dl2_dgammai_sum
        dl2_dsigma += tf.math.reduce_sum(dl2_dgammai_sum*(gamma_tensor - self.omega_t[0][0])/self.omega_t[0][1])
        dl2_dmu /= n_gamma
        dl2_dsigma /= n_gamma
        return tf.reshape(tf.stack([dl2_dmu, dl2_dsigma]), [1,2])
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y,
                   len_x_data,
                   n_gamma,
                   alpha_rema,
                   lambda_):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        # corrupt the inputs with noise
        x = x + self.noise_std * tf.random.normal(tf.shape(x))

        statistics = dict()
        gamma_list = []
        z_list = []
        with tf.GradientTape(persistent=True) as tape:
            sum_df_dbeta = 0.
            loss_2 = 0.
            with tf.GradientTape() as tape2:
                    tape2.watch(self.forward_model.trainable_variables)
                    forward_model_prediction = self.forward_model(x, training=True)
                    sum_forward_model_prediction = tf.reduce_sum(forward_model_prediction)/len_x_data
            df_dbeta = tape2.gradient(sum_forward_model_prediction, self.forward_model.trainable_variables)
            for df_dbeta_i in df_dbeta:
                if df_dbeta_i is not None:
                    sum_df_dbeta += tf.math.reduce_sum(df_dbeta_i)
        
            gamma = tf.random.normal([n_gamma], mean=self.omega_t[0][0], stddev=self.omega_t[0][1])
            loss_i = gamma * sum_df_dbeta
            loss_2 = tf.math.minimum(tf.ones_like(loss_i), tf.math.square((loss_i)/(alpha_rema*sum_forward_model_prediction)))
            loss_2 = tf.math.reduce_sum(loss_2)/n_gamma

            gamma_list.append(tf.reshape(gamma, [n_gamma, 1]))
            z_list.append(tf.reshape(tf.cast((tf.math.abs(loss_i) > tf.math.abs(alpha_rema*sum_forward_model_prediction)), tf.float32),  [n_gamma, 1]))

            # calculate the prediction error and accuracy of the model
            d_pos = self.forward_model(x, training=True)
            mse = tf.keras.losses.mean_squared_error(y, d_pos)
            statistics[f'train/mse'] = mse

            # evaluate how correct the rank fo the model predictions are
            rank_corr = spearman(y[:, 0], d_pos[:, 0])
            statistics[f'train/rank_corr'] = rank_corr

            # calculate negative samples starting from the dataset
            x_neg = self.optimize(
                x, self.particle_gradient_steps, training=False)

            # calculate the prediction error and accuracy of the model
            d_neg = self.forward_model(x_neg, training=False)
            overestimation = d_neg[:, 0] - d_pos[:, 0]
            statistics[f'train/overestimation'] = overestimation

            # build a lagrangian for dual descent
            alpha_loss = (self.alpha * self.overestimation_limit -
                          self.alpha * overestimation)
            # statistics[f'train/alpha'] = self.alpha

            # loss that combines maximum likelihood with a constraint
            model_loss = mse + self.alpha * overestimation
            total_loss = tf.reduce_mean(model_loss) + lambda_*loss_2
            alpha_loss = tf.reduce_mean(alpha_loss)

        # calculate gradients using the model
        alpha_grads = tape.gradient(alpha_loss, self.log_alpha)
        model_grads = tape.gradient(
            total_loss, self.forward_model.trainable_variables)

        # take gradient steps on the model
        self.alpha_opt.apply_gradients([[alpha_grads, self.log_alpha]])
        self.forward_model_opt.apply_gradients(zip(
            model_grads, self.forward_model.trainable_variables))

        return statistics, gamma_list, z_list

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(x, training=False)
        mse = tf.keras.losses.mean_squared_error(y, d_pos)
        statistics[f'validate/mse'] = mse

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d_pos[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr

        # calculate negative samples starting from the dataset
        x_neg = self.optimize(
            x, self.particle_gradient_steps, training=False)

        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg, training=False)
        overestimation = d_neg[:, 0] - d_pos[:, 0]
        statistics[f'validate/overestimation'] = overestimation
        return statistics

    def train(self, 
              omega_mu_bound, 
              omega_sigma_lower, 
              omega_sigma_upper,
              n_gamma,
              lr_omega,
              alpha_rema,
              lambda_, 
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """
        tf.config.run_functions_eagerly(True)

        statistics = defaultdict(list)
        for x, y in dataset:
            len_x_data = x.shape[0]
            stat, gamma_list, z_list = self.train_step(x, y,len_x_data, n_gamma, alpha_rema, lambda_)
            for name, tensor in stat.items():
                statistics[name].append(tensor)

            gamma_tensor = tf.concat(gamma_list, axis=0)
            z_tensor = tf.concat(z_list, axis=0)
            grad_omega = self.update_omega(gamma_tensor, z_tensor, n_gamma)
            self.omega_t = self.omega_t + lr_omega*grad_omega
            self.omega_t = tf.expand_dims(tf.stack([tf.clip_by_value(self.omega_t[0][0], clip_value_min=-omega_mu_bound, clip_value_max=omega_mu_bound), tf.clip_by_value(self.omega_t[0][1], clip_value_min=omega_sigma_lower, clip_value_max=omega_sigma_upper)], axis=0), axis=0)

        for name in statistics.keys():
            # if name == 'train/alpha':
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self, dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self, omega_mu_bound, 
               omega_sigma_lower, 
               omega_sigma_upper,
               n_gamma,
               lr_omega,
               alpha_rema,
               lambda_, train_data, validate_data, logger, epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            for name, loss in self.train(omega_mu_bound, omega_sigma_lower, omega_sigma_upper,n_gamma,lr_omega,alpha_rema,lambda_,train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)


class VAETrainer(tf.Module):

    def __init__(self,
                 vae,
                 optim=tf.keras.optimizers.Adam,
                 lr=0.001, beta=1.0):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        oracles: List[tf.keras.Model]
            a list of keras model that predict distributions over scores
        oracle_optim: __class__
            the optimizer class to use for optimizing the oracle model
        oracle__lr: float
            the learning rate for the oracle model optimizer
        """

        super().__init__()
        self.vae = vae
        self.beta = beta

        # create optimizers for each model in the ensemble
        self.vae_optim = optim(learning_rate=lr)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        with tf.GradientTape() as tape:

            latent = self.vae.encode(x, training=True)
            z = latent.mean()
            prediction = self.vae.decode(z)

            nll = -prediction.log_prob(x)

            kld = latent.kl_divergence(
                tfpd.MultivariateNormalDiag(
                    loc=tf.zeros_like(z), scale_diag=tf.ones_like(z)))

            total_loss = tf.reduce_mean(
                nll) + tf.reduce_mean(kld) * self.beta

        variables = self.vae.trainable_variables

        self.vae_optim.apply_gradients(zip(
            tape.gradient(total_loss, variables), variables))

        statistics[f'vae/train/nll'] = nll
        statistics[f'vae/train/kld'] = kld

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights

        Args:

        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        latent = self.vae.encode(x, training=True)
        z = latent.mean()
        prediction = self.vae.decode(z)

        nll = -prediction.log_prob(x)

        kld = latent.kl_divergence(
            tfpd.MultivariateNormalDiag(
                loc=tf.zeros_like(z), scale_diag=tf.ones_like(z)))

        statistics[f'vae/validate/nll'] = nll
        statistics[f'vae/validate/kld'] = kld

        return statistics

    def train(self,
              dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.train_step(x).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self,
                 dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights

        Args:

        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)
