# Copyright 2025 the Aeneas Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aeneas: Contextualising ancient texts with generative neural networks."""

import copy
import functools
import glob
import os

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
import numpy as np
import optax
from predictingthepast.models.model import Model
from predictingthepast.train import dataloader
from predictingthepast.util import region_names
import predictingthepast.util.alphabet as alphabet_util
import predictingthepast.util.loss as loss_util
import predictingthepast.util.optim as optim_util
import tensorflow_datasets.public_api as tfds


FLAGS = flags.FLAGS


class Experiment(experiment.AbstractExperiment):
  """Aeneas experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Same random key on each device.
    self._rng_key = jl_utils.bcast_local_devices(self.init_rng)

    # Checkpointed experiment state.
    self._params = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    # Forward and update functions.
    self.forward = Model(**self.config.model)
    self._update_func = jax.pmap(self._update_func, axis_name='i')

    # Learning rate schedule
    self._learning_rate_fn = functools.partial(
        optim_util.linear_warmup_and_sqrt_decay,
        max_lr=self.config.optimizer.lr_schedule_kwargs.peak_value,
        warmup_steps=self.config.optimizer.lr_schedule_kwargs.warmup_steps,
    )

    self._opt_init, self._opt_update = self.optimizer()

    if 'use_jit' in self.config.evaluation and self.config.evaluation.use_jit:
      self._eval_batch = jax.jit(self._eval_batch)

    # Create alphabet
    alphabet_kwargs = dict(self.config.alphabet)
    if 'latin' in self.config.dataset.train_language:
      self._alphabet = alphabet_util.LatinAlphabet(**alphabet_kwargs)
    elif 'greek' in self.config.dataset.train_language:
      self._alphabet = alphabet_util.GreekAlphabet(**alphabet_kwargs)

    # Create region mapping
    self._region_map = {'names': [], 'names_inv': {}}
    if 'latin' in self.config.dataset.train_language:
      regions_latin = set()
      with open(self.config.dataset.latin_region_path, 'r') as fl:
        for region_name in fl.read().strip().split('\n'):
          regions_latin.add(region_names.region_name_filter(region_name))
      regions = regions_latin
      logging.info('Loaded %d Latin regions.', len(regions))

    elif 'greek' in self.config.dataset.train_language:
      regions_greek = set()
      with open(self.config.dataset.greek_region_path, 'r') as fg:
        for region_name in fg.read().strip().split('\n'):
          regions_greek.add(region_names.region_name_filter(region_name))
      regions = regions_greek
      logging.info('Loaded %d Greek regions.', len(regions))
    else:
      raise ValueError(
          f'Unsupported train language: {self.config.dataset.train_language}'
      )

    for r_i, r in enumerate(sorted(regions)):
      self._region_map['names_inv'][r] = r_i
      self._region_map['names'].append(r)

    # regions = regions_greek.union(regions_latin)
    logging.info('Union of %d regions in total.', len(regions))
    assert len(self._region_map['names']) == self.config.model.output_regions

  def optimizer(self):
    config_opt = self.config.optimizer

    kwargs = config_opt.kwargs.to_dict()
    kwargs['learning_rate'] = self._learning_rate_fn
    opt = getattr(optax, config_opt.name)(**kwargs)

    if getattr(self.config.optimizer, 'clip_adaptive', False):
      if config_opt.clip_level > 0.0:
        opt = optax.chain(optim_util.adaptive_grad_clip(config_opt.clip_level), opt)  # pytype: disable=wrong-arg-types
    elif config_opt.clip_level > 0.0:
      opt = optax.chain(optax.clip_by_global_norm(config_opt.clip_level), opt)
    return opt

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, **unused_args):
    """See base class."""

    if self._train_input is None:
      self._initialize_train(rng)

    batch = next(self._train_input)
    (self._params, self._opt_state, scalars) = self._update_func(
        self._params, self._opt_state, global_step, batch, rng
    )

    scalars = jl_utils.get_first(scalars)
    return scalars

  def _initialize_train(self, rng):
    # Check we haven't already restored params
    if self._params is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.'
      )
      batch = next(self._build_train_input())

      rng = jl_utils.get_first(rng)
      params_rng, dropout_rng = jax.random.split(rng)
      params_rng = jl_utils.bcast_local_devices(params_rng)
      dropout_rng = jl_utils.bcast_local_devices(dropout_rng)
      init_net = jax.pmap(
          functools.partial(self.forward.init, is_training=True)
      )
      self._params = init_net(
          {'params': params_rng, 'dropout': dropout_rng},
          text_char=batch['text_char'],
          vision_img=batch.get('vision_img'),
          vision_available=batch.get('vision_available'),
      )

      init_opt = jax.pmap(self._opt_init)
      self._opt_state = init_opt(self._params)

      self._train_input = jl_utils.py_prefetch(self._build_train_input)
      self._train_input = jl_utils.double_buffer_on_gpu(self._train_input)

  def _build_train_input(self):
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
    logging.info(
        'num_devices: %d, per_device_batch_size: %d, global_batch_size: %d',
        num_devices,
        per_device_batch_size,
        global_batch_size,
    )

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}'
      )

    config_dataset = self.config.dataset
    with (
        open(config_dataset.latin_dataset_path) as dl,
        open(config_dataset.greek_dataset_path) as dg,
    ):
      ds = dataloader.loader_tf(
          per_device_batch_size,
          config_dataset,
          self._region_map,
          alphabet=self._alphabet,
          latin_dataset_file=dl,
          greek_dataset_file=dg,
          mode='train',
          open_fn=open,
          glob_fn=glob.glob,
      )

    ds = ds.batch(jax.local_device_count())
    return iter(tfds.as_numpy(ds))

  def _loss_fn(
      self, params, batch, global_step, rng, norm_outputs=True, is_training=True
  ):
    del global_step
    text_len = batch['text_len']
    text_unmasked = batch['text_unmasked']
    text_mask = batch['text_mask']
    next_sentence_mask = batch['next_sentence_mask']
    next_sentence_label = batch['next_sentence_label']
    missing_unk_mask = batch['missing_unk_mask']
    missing_unk_label = batch['missing_unk_label']
    region_id = batch['region_id']
    region_available = batch['region_available']
    date_dist = batch['date_dist']
    date_available = batch['date_available']
    date_min = batch['date_min']
    date_max = batch['date_max']
    batch_size = batch['text_mask'].shape[0]
    eps = 1e-7

    (date_logits, region_logits, mask_logits, nsp_logits, unk_logits) = (
        self.forward.apply(
            params,
            text_char=batch['text_char'],
            text_char_onehot=None,
            vision_img=batch.get('vision_img'),
            vision_available=batch.get('vision_available'),
            is_training=is_training,
            rngs={'dropout': rng},
        )
    )

    date_loss = 0.0
    date_l1_loss = 0.0
    region_loss = 0.0
    region_accuracy = 0.0
    mask_loss = 0.0
    mask_accuracy = 0.0
    nsp_loss = 0.0
    nsp_accuracy = 0.0
    unk_loss = 0.0
    unk_accuracy = 0.0

    date_count = 0
    region_count = 0
    mask_count = 0
    nsp_count = 0
    unk_count = 0

    # Date loss
    if self.config.loss.date.enabled:
      date_count = jnp.sum(date_available)

      # L1
      date_pred_x = jnp.arange(
          self.config.dataset.date_min + self.config.dataset.date_interval / 2,
          self.config.dataset.date_max + self.config.dataset.date_interval / 2,
          self.config.dataset.date_interval,
      ).reshape(-1, 1)
      date_pred_val = jnp.dot(jax.nn.softmax(date_logits, axis=-1), date_pred_x)
      date_l1_loss = jnp.sum(
          jax.vmap(loss_util.date_loss_l1)(
              date_pred_val, date_min, date_max, date_available
          ),
          axis=0,
      )

      # KL
      date_loss_dist_ = loss_util.categorical_kl_divergence(
          date_dist, date_logits
      )
      jnp.nan_to_num(date_loss_dist_, copy=False)
      date_loss_dist_ *= date_available
      date_loss = jnp.sum(date_loss_dist_, axis=0)

      if norm_outputs:
        date_l1_loss /= date_count + eps
        date_loss /= batch_size

    # Region loss
    if self.config.loss.region.enabled:
      region_count = jnp.sum(region_available)
      region_loss = jnp.sum(
          loss_util.cross_entropy_label_smoothing_loss(
              region_logits,
              region_id,
              region_available,
              label_smoothing=self.config.loss.region.label_smoothing,
          ),
          0,
      )
      jnp.nan_to_num(region_loss, copy=False)
      region_all_accuracy = (jnp.argmax(region_logits, -1) == region_id).astype(
          region_logits.dtype
      ) * region_available.astype(region_logits.dtype)
      region_accuracy = jnp.sum(region_all_accuracy)
      if norm_outputs:
        region_loss /= batch_size
        region_accuracy /= region_count + eps

    # Mask loss
    if self.config.loss.mask.enabled:
      mask_count = jnp.sum(text_mask)
      mask_loss = jnp.sum(
          loss_util.cross_entropy_label_smoothing_loss(
              mask_logits,
              text_unmasked,
              text_mask,
              label_smoothing=self.config.loss.mask.label_smoothing,
          ),
          1,
      )  # [B]
      assert mask_loss.ndim == 1
      jnp.nan_to_num(mask_loss, copy=False)
      mask_loss = jnp.sum(mask_loss, 0)
      mask_all_accuracy = (jnp.argmax(mask_logits, -1) == text_unmasked).astype(
          mask_logits.dtype
      )
      assert mask_all_accuracy.ndim == 2
      mask_all_accuracy *= text_mask.astype(mask_logits.dtype)
      mask_accuracy = jnp.sum(mask_all_accuracy)
      if norm_outputs:
        mask_loss /= batch_size
        mask_accuracy /= mask_count + eps

    # NSP loss
    if self.config.loss.nsp.enabled:
      assert nsp_logits.ndim == 3 and nsp_logits.shape[-1] == 2
      nsp_count = jnp.sum(next_sentence_mask)
      nsp_loss = jnp.sum(
          loss_util.cross_entropy_label_smoothing_loss(
              nsp_logits,
              next_sentence_label,
              next_sentence_mask,
              label_smoothing=0,
          ),
          1,
      )  # [B]
      assert nsp_loss.ndim == 1
      jnp.nan_to_num(nsp_loss, copy=False)
      nsp_loss = jnp.sum(nsp_loss, 0)
      nsp_all_accuracy = (
          jnp.argmax(nsp_logits, -1) == next_sentence_label
      ).astype(nsp_logits.dtype)
      assert nsp_all_accuracy.ndim == 2
      nsp_all_accuracy *= next_sentence_mask.astype(nsp_logits.dtype)
      nsp_accuracy = jnp.sum(nsp_all_accuracy)
      if norm_outputs:
        nsp_loss /= batch_size
        nsp_accuracy /= nsp_count + eps

    # UNK loss
    if self.config.loss.unk.enabled:
      assert unk_logits.ndim == 3 and unk_logits.shape[-1] == 2
      unk_count = jnp.sum(missing_unk_mask)
      unk_loss = jnp.sum(
          loss_util.cross_entropy_label_smoothing_loss(
              unk_logits,
              missing_unk_label,
              missing_unk_mask,
              label_smoothing=0,
          ),
          1,
      )  # [B]
      assert unk_loss.ndim == 1
      jnp.nan_to_num(unk_loss, copy=False)
      unk_loss = jnp.sum(unk_loss, 0)
      unk_all_accuracy = (
          jnp.argmax(unk_logits, -1) == missing_unk_label
      ).astype(unk_logits.dtype)
      assert unk_all_accuracy.ndim == 2
      unk_all_accuracy *= missing_unk_mask.astype(unk_logits.dtype)
      unk_accuracy = jnp.sum(unk_all_accuracy)
      if norm_outputs:
        unk_loss /= batch_size
        unk_accuracy /= unk_count + eps

    # Text len
    out_len = jnp.mean(text_len) if norm_outputs else jnp.sum(text_len)
    loss_weights = [
        self.config.loss.date.weight,
        self.config.loss.region.weight,
        self.config.loss.mask.weight,
        self.config.loss.nsp.weight,
        self.config.loss.unk.weight,
    ]
    loss = (
        date_loss * self.config.loss.date.weight
        + region_loss * self.config.loss.region.weight
        + mask_loss * self.config.loss.mask.weight
        + nsp_loss * self.config.loss.nsp.weight
        + unk_loss * self.config.loss.unk.weight
    )
    scaled_loss = loss / jax.device_count()
    # NOTE: We use scaled_loss for grads and unscaled for logging.
    return scaled_loss, (
        loss,
        loss_weights,
        date_loss,
        date_l1_loss,
        date_count,
        region_loss,
        region_accuracy,
        region_count,
        mask_loss,
        mask_accuracy,
        mask_count,
        nsp_loss,
        nsp_accuracy,
        nsp_count,
        unk_loss,
        unk_accuracy,
        unk_count,
        out_len,
    )

  def _update_func(self, params, opt_state, global_step, batch, rng):
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (
        loss,
        loss_weights,
        date_loss,
        date_l1_loss,
        _,
        region_loss,
        region_accuracy,
        _,
        mask_loss,
        mask_accuracy,
        _,
        nsp_loss,
        nsp_accuracy,
        _,
        unk_loss,
        unk_accuracy,
        _,
        text_len,
    ) = grad_loss_fn(params, batch, global_step, rng)

    scaled_grads = jax.tree.map(jnp.nan_to_num, scaled_grads)
    grads = jax.lax.psum(scaled_grads, axis_name='i')

    # Compute and apply updates via our optimizer.
    learning_rate = self._learning_rate_fn(global_step)
    updates, opt_state = self._opt_update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {
        'loss/train': loss,
        'loss/date': date_loss,
        'loss/date_l1': date_l1_loss,
        'loss/region': region_loss,
        'loss/mask': mask_loss,
        'loss/nsp': nsp_loss,
        'loss/unk': unk_loss,
        'loss_weights/date': loss_weights[0],
        'loss_weights/region': loss_weights[1],
        'loss_weights/mask': loss_weights[2],
        'loss_weights/nsp': loss_weights[3],
        'loss_weights/unk': loss_weights[4],
        'accuracy/region': region_accuracy,
        'accuracy/mask': mask_accuracy,
        'accuracy/nsp': nsp_accuracy,
        'accuracy/unk': unk_accuracy,
        'opt/learning_rate': learning_rate,
        'opt/grad_norm': optax.global_norm(grads),
        'opt/param_norm': optax.global_norm(params),
        'stats/text_len': text_len,
    }
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_kwargs):
    """See base class."""

    if self._eval_input is None:
      self._initialize_eval()

    global_step = np.array(jl_utils.get_first(global_step))
    summary = self._eval_epoch(jl_utils.get_first(rng))

    for k, v in summary.items():
      summary[k] = np.array(v)

    score = summary['score/eval']
    logging.info('[Step %d] eval_score=%.2f', global_step, score)

    # Log outputs
    checkpoint_dir = jl_utils.get_checkpoint_dir(
        FLAGS.config, jax.process_index()
    )
    score_path = os.path.join(checkpoint_dir, 'best_score.txt')

    # Check for preexisting outputs
    best_score = None
    best_step = None
    if os.path.exists(score_path):
      with open(score_path, 'r') as f:
        tok = f.read().strip().split(' ')
        best_step = int(tok[0])
        best_score = float(tok[1])

    # Store outputs if score is better
    if best_score is None or (score > best_score and global_step > best_step):
      best_score = score

      with open(score_path, 'w') as f:
        f.write(f'{global_step} {best_score}')

    # Log best score
    summary['score/eval_best'] = best_score

    return summary

  def _initialize_eval(self):
    self._eval_input = self._build_eval_input()

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""
    datasets = []
    for lang in self.config.dataset.eval_language:
      # Create a copy of the config
      config_dataset = copy.deepcopy(self.config.dataset)
      config_dataset.eval_language = [lang]
      with (
          open(config_dataset.latin_dataset_path) as dl,
          open(config_dataset.greek_dataset_path) as dg,
      ):
        ds = dataloader.loader_tf(
            self.config.evaluation.batch_size,
            config_dataset,
            self._region_map,
            alphabet=self._alphabet,
            latin_dataset_file=dl,
            greek_dataset_file=dg,
            mode=self.config.evaluation.mode,
            open_fn=open,
            glob_fn=glob.glob,
        )
        datasets.append(tfds.as_numpy(ds))

    return datasets

  def _eval_batch(self, params, batch, rng):
    """Evaluates a batch."""
    (
        _,
        loss_weights,
        date_loss,
        date_l1_loss,
        date_count,
        region_loss,
        region_accuracy,
        region_count,
        mask_loss,
        mask_accuracy,
        mask_count,
        nsp_loss,
        nsp_accuracy,
        nsp_count,
        unk_loss,
        unk_accuracy,
        unk_count,
        text_len,
    ) = self._loss_fn(
        params, batch, None, rng, norm_outputs=False, is_training=False
    )[
        1
    ]

    # Outputs
    return {
        'loss/date': date_loss,
        'loss/date_l1': date_l1_loss,
        'loss/region': region_loss,
        'loss/mask': mask_loss,
        'loss/nsp': nsp_loss,
        'loss/unk': unk_loss,
        'loss_weights/date': loss_weights[0],
        'loss_weights/region': loss_weights[1],
        'loss_weights/mask': loss_weights[2],
        'loss_weights/nsp': loss_weights[3],
        'loss_weights/unk': loss_weights[4],
        'count/date': date_count,
        'count/region': region_count,
        'count/mask': mask_count,
        'count/nsp': nsp_count,
        'count/unk': unk_count,
        'stats/text_len': text_len,
        'accuracy/region': region_accuracy,
        'accuracy/mask': mask_accuracy,
        'accuracy/nsp': nsp_accuracy,
        'accuracy/unk': unk_accuracy,
    }

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    summary = {}
    total_num_sequences = 0

    # Prepare directories for storing model log
    checkpoint_dir = jl_utils.get_checkpoint_dir(
        FLAGS.config, jax.process_index()
    )
    model_log_path = os.path.join(checkpoint_dir, 'model_log')
    if self.config.evaluation.store_model_log:
      if os.path.isdir(model_log_path):
        map(os.remove, glob.glob(model_log_path + '/*'))
      else:
        os.makedirs(model_log_path)

    # Checkpoints broadcast for each local device
    params = jl_utils.get_first(self._params)

    # Iterate over datasets
    for lang, ds in zip(self.config.dataset.eval_language, self._eval_input):  # pytype: disable=wrong-arg-types
      # Iterate over batches
      for batch in ds:
        # Make sure that the input has batch_dim=1
        assert batch['text_char'].shape[0] == 1

        summary_batch = self._eval_batch(params, batch, rng)

        # Append batch values to dictionary
        for k, v in summary_batch.items():
          summary[f'{lang}/{k}'] = summary.get(f'{lang}/{k}', 0) + v

        total_num_sequences += self.config.evaluation.batch_size

      # Normalise and concatenate
      summary[f'{lang}/stats/text_len'] /= total_num_sequences

      summary[f'{lang}/loss/date'] /= summary[f'{lang}/count/date']
      summary[f'{lang}/loss/date_l1'] /= summary[f'{lang}/count/date']

      summary[f'{lang}/loss/region'] /= summary[f'{lang}/count/region']
      summary[f'{lang}/accuracy/region'] /= summary[f'{lang}/count/region']

      summary[f'{lang}/loss/mask'] /= summary[f'{lang}/count/mask']
      summary[f'{lang}/accuracy/mask'] /= summary[f'{lang}/count/mask']

      summary[f'{lang}/loss/nsp'] /= summary[f'{lang}/count/nsp']
      summary[f'{lang}/accuracy/nsp'] /= summary[f'{lang}/count/nsp']

      summary[f'{lang}/loss/unk'] /= summary[f'{lang}/count/unk']
      summary[f'{lang}/accuracy/unk'] /= summary[f'{lang}/count/unk']

      summary[f'{lang}/score/eval'] = (
          summary[f'{lang}/accuracy/mask']
          + summary[f'{lang}/accuracy/region']
          - summary[f'{lang}/loss/date_l1'] * 0.01
      )
      summary[f'{lang}/loss/eval'] = (
          summary[f'{lang}/loss/mask']
          + summary[f'{lang}/loss/date']
          + summary[f'{lang}/loss/region']
      )

    return summary


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
