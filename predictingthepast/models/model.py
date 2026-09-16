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
"""Aeneas model."""

import functools
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import bigbird
from . import common_layers
from . import resnet
from . import t5_layers


class T5DecoderLayer(nn.Module):
  """Transformer decoder layer."""

  emb_dim: int
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32  # pyrefly: ignore[bad-assignment]
  causal_mask: bool = False  # Unused
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      deterministic: bool,
      padding_mask: Optional[jnp.ndarray] = None,
  ):
    """Applies decoder block module."""
    assert inputs.shape[-1] == self.emb_dim

    # `inputs` is layer input with a shape [batch, length, emb_dim].
    x = t5_layers.LayerNorm(
        dtype=self.dtype, name='pre_self_attention_layer_norm'
    )(inputs)

    # Decoder Mask
    decoder_mask = t5_layers.make_attention_mask(
        padding_mask, padding_mask, pairwise_fn=jnp.logical_and, dtype=jnp.bool  # pyrefly: ignore[bad-argument-type]
    )

    # Self-attention block
    x = t5_layers.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        head_dim=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        float32_logits=True,
        name='self_attention',
    )(x, x, mask=decoder_mask, deterministic=deterministic)
    x = nn.Dropout(
        rate=self.dropout_rate,
        broadcast_dims=(-2,),
        name='post_self_attention_dropout',
    )(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5_layers.LayerNorm(dtype=self.dtype, name='pre_mlp_layer_norm')(x)
    y = t5_layers.MlpBlock(
        intermediate_dim=self.mlp_dim,
        intermediate_dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=self.dropout_rate, broadcast_dims=(-2,), name='post_mlp_dropout'
    )(y, deterministic=deterministic)
    y = y + x

    return y


class Model(nn.Module):
  """Transformer Model for sequence tagging."""

  vocab_char_size: int = 164
  output_regions: int = 85
  output_date: int = 160
  output_date_dist: bool = True
  use_output_mlp: bool = True
  num_heads: int = 8
  num_layers: int = 6
  word_char_emb_dim: int = 192
  emb_word_disable: bool = False
  emb_init: str = 'variance_scaling'
  emb_norm: bool = True
  emb_decoder_type: str = 'tied'  # 'tied', 'no'
  emb_dim: int = 512
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 1024
  causal_mask: bool = False
  feature_combine_type: str = 'concat'
  posemb_combine_type: str = 'add'
  region_date_pooling: str = 'first'
  learn_pos_emb: bool = True
  use_bfloat16: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  activation_fn: str = 'gelu'
  model_type: str = 't5'
  retrieval: bool = False
  vision: bool = False
  prepend_sos: int = 1
  modality_drop: float = 0.1

  def setup(self):
    # Embedding initialization
    if self.emb_init == 'variance_scaling':
      emb_init = nn.initializers.variance_scaling(
          1.0, 'fan_in', 'normal', in_axis=1, out_axis=0
      )
    elif self.emb_init == 'normal':
      if 'bigbird' in self.model_type:
        emb_init = nn.initializers.normal(stddev=1.0)
      else:
        emb_init = nn.initializers.normal(stddev=1e-3)
    else:
      raise ValueError('Wrong emb_init value.')

    self.text_char_emb = nn.Embed(
        num_embeddings=self.vocab_char_size,
        features=self.emb_dim,
        embedding_init=emb_init,
        name='char_embeddings',
    )

  @nn.compact
  def __call__(
      self,
      text_char=None,
      text_char_onehot=None,
      text_char_emb=None,
      vision_img=None,
      vision_available=None,
      padding=None,
      output_return_emb=False,
      is_training=False,
  ):
    """Applies Aeneas model on the inputs."""

    if text_char is not None and padding is None:
      padding = jnp.where(text_char > 0, 1, 0)
    elif text_char_onehot is not None and padding is None:
      padding = jnp.where(text_char_onehot.argmax(-1) > 0, 1, 0)
    padding_mask = padding  # [..., jnp.newaxis]
    text_len = jnp.sum(padding, 1)  # pyrefly: ignore[bad-argument-type]

    # Character embeddings
    if text_char is not None:
      text_char_emb_x = self.text_char_emb(text_char)
    elif text_char_onehot is not None:
      text_char_emb_x = self.text_char_emb.attend(text_char_onehot)
    elif text_char_emb is not None:
      text_char_emb_x = text_char_emb
    else:
      raise ValueError('Wrong text_char value.')

    if self.emb_norm:
      text_char_emb_x = common_layers.LayerNorm(name='text_char_emb_norm')(
          text_char_emb_x
      )
    x = text_char_emb_x

    # Set floating point
    if self.use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Vision
    if self.vision:
      # Generate embedding
      if vision_img is not None:
        x_img = resnet.ResNet8(num_classes=self.emb_dim, name='resnet')(
            vision_img,
            train=is_training,
            stop_gradient=False,
        ).astype(dtype)
        x_img = common_layers.LayerNorm(name='x_img_norm')(x_img)

        # Zero out embeddings for non-available images
        x_img = x_img * jnp.expand_dims(vision_available, -1)  # pyrefly: ignore[bad-argument-type]
      else:
        x_img = jnp.zeros((x.shape[0], self.emb_dim))

    if self.model_type == 'bigbird':
      # Add an extra dimension to the padding mask
      padding_mask = padding_mask[..., jnp.newaxis]  # pyrefly: ignore[unsupported-operation]

      # Positional embeddings
      if self.posemb_combine_type == 'add':
        posemb_dim = None
      elif self.posemb_combine_type == 'concat':
        posemb_dim = self.emb_dim // 2
      else:
        raise ValueError('Wrong feature_combine_type value.')

      pe_init = (
          common_layers.sinusoidal_init(max_len=self.max_len)
          if self.learn_pos_emb
          else None
      )
      x = common_layers.AddPositionEmbs(
          posemb_dim=posemb_dim,
          posemb_init=pe_init,
          max_len=self.max_len,
          combine_type=self.posemb_combine_type,
          name='posembed_input',
      )(x)
      x = nn.Dense(self.emb_dim, name='input_emb_dim_dense')(x)
      model_block = functools.partial(
          bigbird.BigBirdBlock,
          rope_pos_emb=False,
      )
    elif self.model_type == 'bigbird_rope':
      # Add an extra dimension to the padding mask
      padding_mask = padding_mask[..., jnp.newaxis]  # pyrefly: ignore[unsupported-operation]

      model_block = functools.partial(
          bigbird.BigBirdBlock,
          rope_pos_emb=True,
      )
    elif self.model_type == 't5':
      model_block = T5DecoderLayer
    else:
      raise ValueError('Wrong model type specified.')

    # Pre transformer dropout
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

    # Transformer
    for lyr in range(self.num_layers):
      x = model_block(
          emb_dim=self.emb_dim,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,  # pyrefly: ignore[bad-argument-type]
          causal_mask=self.causal_mask,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
      )(x, deterministic=not is_training, padding_mask=padding_mask)
    x = common_layers.LayerNorm(
        dtype=dtype, name=f'encoder_norm_{self.num_layers - 1}'
    )(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
    torso_output = x

    # Mask logits
    if self.emb_decoder_type == 'tied':
      x_mask_out_dim = self.text_char_emb.features
    elif self.emb_decoder_type == 'no':
      x_mask_out_dim = self.vocab_char_size
    else:
      raise ValueError('Wrong emb_decoder_type value.')

    if self.use_output_mlp:
      x_mask = common_layers.MlpBlock(
          out_dim=x_mask_out_dim,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          use_norm=True,
          out_dropout=True,
          dropout_rate=self.dropout_rate,
          activation_fn=self.activation_fn,
          deterministic=not is_training,
          name='x_mask',
      )(x)
    else:
      x_mask = nn.Dense(x_mask_out_dim, name='x_mask')(x)

    if self.emb_decoder_type == 'no':
      logits_mask = x_mask
    else:
      char_embeddings = self.text_char_emb.embedding

      char_embeddings = nn.Dropout(
          rate=self.dropout_rate,  # broadcast_dims=(-2,)
      )(char_embeddings, deterministic=not is_training)

      # Use the transpose of embedding matrix for the logit transform.
      logits_mask = jnp.matmul(x_mask, jnp.transpose(char_embeddings))
      # Correctly normalize pre-softmax logits for this shared case.
      logits_mask = logits_mask / jnp.sqrt(x_mask.shape[-1])

    # Missing unk count prediction
    if self.use_output_mlp:
      logits_unk = common_layers.MlpBlock(
          out_dim=2,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          use_norm=True,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          activation_fn=self.activation_fn,
          deterministic=not is_training,
          name='logits_unk',
      )(x)
    else:
      logits_unk = nn.Dense(2, name='logits_unk')(x)

    # Next sentence prediction
    if self.use_output_mlp:
      logits_nsp = common_layers.MlpBlock(
          out_dim=2,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          use_norm=True,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          activation_fn=self.activation_fn,
          deterministic=not is_training,
          name='logits_nsp',
      )(x)
    else:
      logits_nsp = nn.Dense(2, name='logits_nsp')(x)

    # Average over temporal dimension
    if self.region_date_pooling == 'average':
      x = jnp.multiply(padding_mask.astype(jnp.float32), x)  # pyrefly: ignore[missing-attribute]
      x = jnp.sum(x, 1) / text_len.astype(jnp.float32)[..., None]
      x_date = x
      x_region = x
    elif self.region_date_pooling == 'sum':
      x = jnp.multiply(padding_mask.astype(jnp.float32), x)  # pyrefly: ignore[missing-attribute]
      x = jnp.sum(x, 1)
      x_date = x
      x_region = x
    elif self.region_date_pooling == 'first':
      if self.prepend_sos == 1:
        x_date = x[:, 0, :]
        x_region = x[:, 0, :]
      elif self.prepend_sos == 2:
        x_date = x[:, 0, :]
        x_region = x[:, 1, :]
      else:
        raise ValueError('Wrong prepend_sos value.')
    else:
      raise ValueError('Wrong pooling type specified.')

    # Date pred
    if self.output_date_dist:
      output_date_dim = self.output_date
    else:
      output_date_dim = 1

    # Concatenate vision and text
    if self.vision:
      x_region = jax.lax.concatenate([x_region, x_img], 1) / jnp.sqrt(2)  # pyrefly: ignore[unbound-name]

    if self.use_output_mlp:
      logits_date = common_layers.MlpBlock(
          out_dim=output_date_dim,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          use_norm=True,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          activation_fn=self.activation_fn,
          deterministic=not is_training,
          name='logits_date',
      )(x_date)
    else:
      logits_date = nn.Dense(output_date_dim, name='logits_date')(x_date)

    # Region logits
    if self.use_output_mlp:
      logits_subregion = common_layers.MlpBlock(
          out_dim=self.output_regions,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          use_norm=True,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          activation_fn=self.activation_fn,
          deterministic=not is_training,
          name='logits_region',
      )(x_region)
    else:
      logits_subregion = nn.Dense(self.output_regions, name='logits_region')(
          x_region
      )

    outputs = (
        logits_date,
        logits_subregion,
        logits_mask,
        logits_nsp,
        logits_unk,
    )
    if output_return_emb:
      return outputs, torso_output
    return outputs
