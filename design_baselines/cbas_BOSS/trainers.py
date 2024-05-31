from design_baselines.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf


class Ensemble(tf.Module):

    def __init__(self,
                 PhiGammaModel,
                 omega_t,
                 forward_models,
                 PhiGammaModel_optim = tf.keras.optimizers.Adam,
                 forward_model_optim=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001):
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
        self.forward_models = forward_models
        self.PhiGammaModel = PhiGammaModel
        self.omega_t = omega_t 
        self.bootstraps = len(forward_models)

        # create optimizers for each model in the ensemble
        self.forward_model_optims = [
            forward_model_optim(learning_rate=forward_model_lr)
            for i in range(self.bootstraps)]
        self.PhiGammaModel_optim = PhiGammaModel_optim(learning_rate=0.001)
        

    def get_distribution(self,
                         x,
                         **kwargs):
        """Build the mixture distribution implied by the set of oracles
        that are trained in this module

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfpd.Distribution
            the mixture of gaussian distributions implied by the oracles
        """

        # get the distribution parameters for all models
        params = defaultdict(list)
        for fm in self.forward_models:
            for key, val in fm.get_params(x, **kwargs).items():
                params[key].append(val)

        # stack the parameters in a new component axis
        for key, val in params.items():
            params[key] = tf.stack(val, axis=-1)

        # build the mixture distribution using the family of component one
        weights = tf.fill([self.bootstraps], 1 / self.bootstraps)
        return tfpd.MixtureSameFamily(tfpd.Categorical(
            probs=weights), self.forward_models[0].distribution(**params))
    
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
    def train_step(self,
                   x,
                   y,
                   b,
                   w,
                   len_x_data,
                   n_gamma,
                   alpha,
                   lambda_):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        b: tf.Tensor
            bootstrap indicators shaped like [batch_size, num_oracles]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        gamma_list = []
        z_list = []
        for i in range(self.bootstraps):
            fm = self.forward_models[i]
            fm_optim = self.forward_model_optims[i]
            with tf.GradientTape(persistent=True) as tape:
                sum_df_dbeta = 0.
                loss_2 = 0.
                with tf.GradientTape() as tape2:
                        tape2.watch(fm.trainable_variables)
                        forward_model_prediction = fm.get_distribution(x, training=True).mean()
                        sum_forward_model_prediction = tf.reduce_sum(forward_model_prediction)/len_x_data
                df_dbeta = tape2.gradient(sum_forward_model_prediction, fm.trainable_variables)
                for df_dbeta_i in df_dbeta:
                    if df_dbeta_i is not None:
                        sum_df_dbeta += tf.math.reduce_sum(df_dbeta_i)
            
                gamma = tf.random.normal([n_gamma], mean=self.omega_t[0][0], stddev=self.omega_t[0][1])
                loss_i = gamma * sum_df_dbeta
                loss_2 = tf.math.minimum(tf.ones_like(loss_i), tf.math.square((loss_i)/(alpha*sum_forward_model_prediction)))
                loss_2 = tf.math.reduce_sum(loss_2)/n_gamma

                gamma_list.append(tf.reshape(gamma, [n_gamma, 1]))
                z_list.append(tf.reshape(tf.cast((tf.math.abs(loss_i) > tf.math.abs(alpha*sum_forward_model_prediction)), tf.float32),  [n_gamma, 1]))
                # calculate the prediction error and accuracy of the model
                d = fm.get_distribution(x, training=True)
                nll = -d.log_prob(y)[:, 0]

                # evaluate how correct the rank fo the model predictions are
                rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

                # build the total loss and weight by the bootstrap
                total_loss = tf.math.divide_no_nan(tf.reduce_sum(
                    w[:, 0] * b[:, i] * nll), tf.reduce_sum(b[:, i])) + lambda_*loss_2

            grads = tape.gradient(total_loss, fm.trainable_variables)
            fm_optim.apply_gradients(zip(grads, fm.trainable_variables)) 

            statistics[f'oracle_{i}/train/nll'] = nll
            statistics[f'oracle_{i}/train/rank_corr'] = rank_correlation

        return statistics, gamma_list, z_list

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x,
                      y):
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

        for i in range(self.bootstraps):
            fm = self.forward_models[i]

            # calculate the prediction error and accuracy of the model
            d = fm.get_distribution(x, training=False)
            nll = -d.log_prob(y)[:, 0]

            # evaluate how correct the rank fo the model predictions are
            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])

            statistics[f'oracle_{i}/validate/nll'] = nll
            statistics[f'oracle_{i}/validate/rank_corr'] = rank_correlation

        return statistics

    def train(self,
              omega_mu_bound, 
              omega_sigma_lower, 
              omega_sigma_upper,
              n_gamma,
              lr_omega,
              alpha,
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
        # tf.config.run_functions_eagerly(True)

        statistics = defaultdict(list)
        for x, y, b, w in dataset:
            len_x_data = x.shape[0]
            stat, gamma_list, z_list = self.train_step(x, y, b, w, len_x_data, n_gamma, alpha, lambda_)
            for name, tensor in stat.items():
                statistics[name].append(tensor)
            
            gamma_tensor = tf.concat(gamma_list, axis=0)
            z_tensor = tf.concat(z_list, axis=0)
            grad_omega = self.update_omega(gamma_tensor, z_tensor, n_gamma)
            self.omega_t = self.omega_t + lr_omega*grad_omega
            self.omega_t = tf.expand_dims(tf.stack([tf.clip_by_value(self.omega_t[0][0], clip_value_min=-omega_mu_bound, clip_value_max=omega_mu_bound), tf.clip_by_value(self.omega_t[0][1], clip_value_min=omega_sigma_lower, clip_value_max=omega_sigma_upper)], axis=0), axis=0)

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
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               omega_mu_bound, 
               omega_sigma_lower, 
               omega_sigma_upper,
               n_gamma,
               lr_omega,
               alpha,
               lambda_,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0):
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

        for e in range(start_epoch, start_epoch + epochs):
            print("="*20 + f"     epochs {e}     " + "="*20)
            for name, loss in self.train(omega_mu_bound, omega_sigma_lower, omega_sigma_upper,n_gamma,lr_omega,alpha,lambda_,train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = dict()
        for i in range(self.bootstraps):
            saveables[f'forward_model_{i}'] = self.forward_models[i]
            saveables[f'forward_model_optim_{i}'] = self.forward_model_optims[i]
        return saveables


class WeightedVAE(tf.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 vae_optim=tf.keras.optimizers.Adam,
                 vae_lr=0.001,
                 vae_beta=1.0):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        encoder: tf.keras.Model
            the encoder neural network that outputs parameters for a gaussian
        decoder: tf.keras.Model
            the decoder neural network that outputs parameters for a gaussian
        vae_optim: __class__
            the optimizer class to use for optimizing the oracle model
        vae_lr: float
            the learning rate for the oracle model optimizer
        vae_beta: float
            the variational beta for the oracle model optimizer
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optim = vae_optim(learning_rate=vae_lr)
        self.vae_beta = vae_beta

    @tf.function(experimental_relax_shapes=True)
    def train_step(self,
                   x,
                   y,
                   w):
        """Perform a training step of gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        x: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of training labels shaped like [batch_size, 1]
        w: tf.Tensor
            importance sampling weights shaped like [batch_size, 1]

        Returns:

        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()
        var_list = []
        var_list.extend(self.encoder.trainable_variables)
        var_list.extend(self.decoder.trainable_variables)

        with tf.GradientTape() as tape:

            # build distributions for the data x and latent variable z
            dz = self.encoder.get_distribution(x, training=True)
            z = dz.sample()
            dx = self.decoder.get_distribution(z, training=True)

            # build the reconstruction loss
            nll = -dx.log_prob(x)[..., tf.newaxis]
            while len(nll.shape) > 2:
                nll = tf.reduce_sum(nll, axis=1)
            prior = tfpd.MultivariateNormalDiag(
                loc=tf.zeros_like(z), scale_diag=tf.ones_like(z))

            # build the kl loss
            kl = dz.kl_divergence(prior)[:, tf.newaxis]
            total_loss = tf.reduce_mean(w * (nll + self.vae_beta * kl))

        grads = tape.gradient(total_loss, var_list)
        self.optim.apply_gradients(zip(grads, var_list))

        statistics[f'vae/train/nll'] = nll
        statistics[f'vae/train/kl'] = kl

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self,
                      x,
                      y):
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

        # build distributions for the data x and latent variable z
        dz = self.encoder.get_distribution(x, training=False)
        z = dz.sample()
        dx = self.decoder.get_distribution(z, training=False)

        # build the reconstruction loss
        nll = -dx.log_prob(x)[..., tf.newaxis]
        while len(nll.shape) > 2:
            nll = tf.reduce_sum(nll, axis=1)
        prior = tfpd.MultivariateNormalDiag(
            loc=tf.zeros_like(z), scale_diag=tf.ones_like(z))

        # build the kl loss
        kl = dz.kl_divergence(prior)[:, tf.newaxis]

        statistics[f'vae/validate/nll'] = nll
        statistics[f'vae/validate/kl'] = kl

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
        for x, y, w in dataset:
            for name, tensor in self.train_step(x, y, w).items():
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
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self,
               train_data,
               validate_data,
               logger,
               epochs,
               start_epoch=0):
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

        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.validate(validate_data).items():
                logger.record(name, loss, e)

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = dict()
        saveables['encoder'] = self.encoder
        saveables['decoder'] = self.decoder
        saveables['optim'] = self.optim
        return saveables


class CBAS(tf.Module):

    def __init__(self,
                 ensemble,
                 p_vae,
                 q_vae,
                 latent_size=20):
        """Build a trainer for an ensemble of probabilistic neural networks
        trained on bootstraps of a dataset

        Args:

        encoder: tf.keras.Model
            the encoder neural network that outputs parameters for a gaussian
        decoder: tf.keras.Model
            the decoder neural network that outputs parameters for a gaussian
        vae_optim: __class__
            the optimizer class to use for optimizing the oracle model
        vae_lr: float
            the learning rate for the oracle model optimizer
        vae_beta: float
            the variational beta for the oracle model optimizer
        """

        super().__init__()
        self.ensemble = ensemble
        self.p_vae = p_vae
        self.q_vae = q_vae
        self.latent_size = latent_size

    @tf.function(experimental_relax_shapes=True)
    def generate_data(self,
                      num_batches,
                      num_samples,
                      percentile):
        """A function for generating a data set of samples of a particular
        size using two adaptively sampled vaes

        Args:

        num_batches: int
            the number of batches of samples to generate
        num_samples: int
            the number of samples to generate all at once using the vae
        percentile: float
            the percentile in [0, 100] that determines importance weights

        Returns:

        xs: tf.Tensor
            the dataset x values sampled from the vaes
        ys: tf.Tensor
            the dataset y values predicted by the ensemble
        ws: tf.Tensor
            the dataset importance weights calculated using the vaes
        """

        xs = []
        ys = []
        ws = []

        for j in range(num_batches):

            # sample designs from the prior
            z = tf.random.normal([num_samples, self.latent_size])
            q_dx = self.q_vae.decoder.get_distribution(z, training=False)
            p_dx = self.p_vae.decoder.get_distribution(z, training=False)

            # evaluate the score and importance weights
            x = q_dx.sample()
            y = self.ensemble.get_distribution(x).mean()
            log_w = p_dx.log_prob(x)[..., tf.newaxis] - \
                    q_dx.log_prob(x)[..., tf.newaxis]
            while len(log_w.shape) > 2:
                log_w = tf.reduce_sum(log_w, axis=1)

            xs.append(x)
            ys.append(y)
            ws.append(tf.math.exp(log_w))

        # locate the cutoff for the scores below the percentile
        gamma = tfp.stats.percentile(ys, percentile)

        for j in range(num_batches):

            # re-weight by the cumulative probability of the score
            d = self.ensemble.get_distribution(xs[j])
            ws[j] *= 1.0 - d.cdf(tf.fill([num_samples, 1], gamma))

        return tf.concat(xs, axis=0), \
               tf.concat(ys, axis=0), \
               tf.concat(ws, axis=0)
