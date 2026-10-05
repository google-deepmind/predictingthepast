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
"""Eval utils."""

from typing import List, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import predictingthepast.util.text as util_text
import tqdm


def date_loss_l1(pred, target_min, target_max):
  """L1 loss function for dates."""
  loss = 0.0
  loss += np.abs(pred - target_min) * np.less(pred, target_min).astype(
      pred.dtype
  )
  loss += np.abs(pred - target_max) * np.greater(pred, target_max).astype(
      pred.dtype
  )
  return loss


def date_loss_l1_twoside(pred_date_min, pred_date_max, date_min, date_max):
  """Calculates the two-sided L1 loss between two date ranges."""
  assert pred_date_min <= pred_date_max and date_min <= date_max

  loss = 0.0
  if pred_date_min > date_max:
    loss += abs(date_max - pred_date_max)
  elif pred_date_max < date_min:
    loss += abs(date_min - pred_date_min)
  else:
    if pred_date_min < date_min:
      loss += abs(date_min - pred_date_min)
    if pred_date_max > date_max:
      loss += abs(date_max - pred_date_max)
  return loss


def grad_to_saliency_char(gradient_char, text_char_onehot, text_len, alphabet):
  """Generates saliency map."""
  saliency_char = np.linalg.norm(gradient_char, axis=2)[0, : text_len[0]]

  text_char = np.array(text_char_onehot).argmax(axis=-1)
  idx_mask = np.logical_or(
      text_char[0, : text_len[0]] > alphabet.alphabet_end_idx,
      text_char[0, : text_len[0]] < alphabet.alphabet_start_idx,
  )
  idx_unmask = np.logical_not(idx_mask)

  saliency_char_tmp = saliency_char.copy()
  saliency_char_tmp[idx_mask] = 0.0
  if idx_unmask.any():
    saliency_char_tmp[idx_unmask] = (
        saliency_char[idx_unmask] - saliency_char[idx_unmask].min()
    ) / (
        saliency_char[idx_unmask].max() - saliency_char[idx_unmask].min() + 1e-8
    )
  return saliency_char_tmp


def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def log_softmax(x, axis=-1):
  """Log-Softmax function."""
  shifted = x - x.max(axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis, keepdims=True))


def to_nucleus_logits(
    logits: np.ndarray,
    top_p: float,
) -> np.ndarray:
  """Remaps logits for nucleus sampling."""
  clipped_top_p = max(top_p, 5e-6)
  sorted_logits = np.sort(logits)  # , is_stable=False)
  sorted_probs = softmax(sorted_logits)
  threshold_idx = np.argmax(
      np.cumsum(sorted_probs, -1) >= 1 - clipped_top_p, axis=-1
  )
  threshold_largest_logits = np.take_along_axis(
      sorted_logits, threshold_idx[..., None], axis=-1
  )
  assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
  logits = np.where(logits >= threshold_largest_logits, logits, -1e12)
  return logits


class BeamEntry(NamedTuple):
  text_pred: str
  mask_idx: Union[set[int], list[int]]
  pred_len: int
  unk_len: int
  pred_logprob: float
  text_history: List[str]


def beam_search_batch(
    forward,
    params,
    alphabet,
    text_pred,
    mask_idx,
    vision_img=None,
    vision_available=False,
    beam_width=20,
    temperature=1.0,
    nucleus=False,
    nucleus_top_p=0.95,
    a_penalty=0.6,
    max_len=768,
    max_iterations=None,
    skip_double_space=True,
    display_progress=False,
    sequential_decoding=True,
) -> List[BeamEntry]:
  """Non-sequential beam search."""

  mask_idx = set(mask_idx)
  initial_text_pred = text_pred.rstrip(alphabet.pad)
  beam = [
      BeamEntry(initial_text_pred, mask_idx, 0, 0, 0.0, [initial_text_pred])
  ]
  beam_top = {}

  space_idx = alphabet.char2idx[alphabet.space]

  # Initialise tqdm bar
  if display_progress:
    pbar = tqdm.tqdm(total=len(mask_idx))

  iteration = 0
  while beam and (max_iterations is None or iteration < max_iterations):
    iteration += 1
    beam_tmp = []
    beam_batch = []

    text_chars = []

    # Determine the maximum length among all texts
    text_pred_len_max = max(len(entry.text_pred) for entry in beam)

    for entry in beam:
      # Append padding (if needed)
      pad_len = text_pred_len_max - len(entry.text_pred)
      text_pred_pad = entry.text_pred + alphabet.pad * pad_len

      current_mask_idx = entry.mask_idx.copy()
      text_char = util_text.text_to_idx(text_pred_pad, alphabet).reshape(1, -1)
      text_chars.append(text_char)
      beam_batch.append(entry._replace(mask_idx=current_mask_idx))
    text_chars = np.vstack(text_chars)

    batch_size = text_chars.shape[0]
    _, _, mask_logits, _, unk_logits = forward(
        params,
        text_char=text_chars,
        text_char_onehot=None,
        vision_img=None
        if vision_img is None
        else jnp.repeat(vision_img, batch_size, axis=0),
        vision_available=None
        if vision_available is None
        else jnp.repeat(vision_available, batch_size, axis=0),
    )

    # Compute log probabilities
    mask_logits = np.array(mask_logits / temperature)
    unk_logits = np.array(unk_logits)

    # Iterate over batch elements
    for batch_i, current_beam_entry in enumerate(beam_batch):
      text_len = len(current_beam_entry.text_pred)

      mask_logits_i = mask_logits[batch_i, :text_len]
      mask_logprob_i = log_softmax(mask_logits_i)
      unk_logprob_i = log_softmax(unk_logits[batch_i, :text_len])
      if nucleus:
        mask_logits_nucleus_i = to_nucleus_logits(mask_logprob_i, nucleus_top_p)

      # Keep only predictions for mask
      mask_idx_loop = []
      for text_char_pos in current_beam_entry.mask_idx:
        # Skip if already restored
        if current_beam_entry.text_pred[text_char_pos] not in [
            alphabet.missing,
            alphabet.missing_unk,
        ]:
          continue
        mask_idx_loop.append(text_char_pos)
      if sequential_decoding:
        mask_idx_loop = sorted(mask_idx_loop)[:1]

      for text_char_pos in mask_idx_loop:

        # Process unknown length missing characters
        if current_beam_entry.text_pred[text_char_pos] == alphabet.missing_unk:

          # Case: add a missing character and shift all indexes by 1
          text_pred_i = (
              current_beam_entry.text_pred[: text_char_pos + 1]
              + alphabet.missing
              + current_beam_entry.text_pred[text_char_pos + 1 :]
          )
          mask_idx_i = set(
              m_i + 1 if m_i > text_char_pos else m_i
              for m_i in current_beam_entry.mask_idx
          )
          mask_idx_i.add(text_char_pos + 1)
          pred_logprob_i = (
              current_beam_entry.pred_logprob + unk_logprob_i[text_char_pos, 1]
          )

          # Skip if it exceeds the maximum length
          if len(text_pred_i) <= max_len:
            beam_tmp.append(
                BeamEntry(
                    text_pred_i,
                    mask_idx_i,
                    current_beam_entry.pred_len,
                    current_beam_entry.unk_len + 1,
                    pred_logprob_i,
                    current_beam_entry.text_history + [text_pred_i],
                )
            )

          # Case: add a replace missing unk with single missing character
          mask_idx_i = current_beam_entry.mask_idx.copy()
          text_pred_i = (
              current_beam_entry.text_pred[:text_char_pos]
              + alphabet.missing
              + current_beam_entry.text_pred[text_char_pos + 1 :]
          )
          pred_logprob_i = (
              current_beam_entry.pred_logprob + unk_logprob_i[text_char_pos, 0]
          )
          beam_tmp.append(
              BeamEntry(
                  text_pred_i,
                  mask_idx_i,
                  current_beam_entry.pred_len,
                  current_beam_entry.unk_len + 1,
                  pred_logprob_i,
                  current_beam_entry.text_history + [text_pred_i],
              )
          )
        elif current_beam_entry.text_pred[text_char_pos] == alphabet.missing:

          # Iterate over possible characters
          for text_char_id in [space_idx] + list(
              range(
                  alphabet.alphabet_start_idx,
                  alphabet.alphabet_end_idx + 1,
                  # alphabet.char2idx[alphabet.punctuation[-1]] + 1,
              )
          ):

            # Skip expanding the beam if logprob too small
            if nucleus and np.isclose(
                mask_logits_nucleus_i[text_char_pos, text_char_id], -1e12  # pyrefly: ignore[unbound-name]
            ):
              continue

            # Create a copy of the text
            text_pred_i_list = list(current_beam_entry.text_pred)
            text_pred_i_list[text_char_pos] = alphabet.idx2char[text_char_id]
            text_pred_i = ''.join(text_pred_i_list)

            # Ignore possible cases where two spaces are next to each other
            if text_pred_i_list[text_char_pos] == ' ' and skip_double_space:
              if text_char_pos == 0:
                if text_pred_i_list[text_char_pos + 1] == ' ':
                  continue
              elif text_char_pos == text_len - 1:
                if text_pred_i_list[text_char_pos - 1] == ' ':
                  continue
              elif (
                  text_pred_i_list[text_char_pos + 1] == ' '
                  or text_pred_i_list[text_char_pos - 1] == ' '
              ):
                continue

            pred_logprob_i = (
                current_beam_entry.pred_logprob
                + mask_logprob_i[text_char_pos, text_char_id]
            )

            # Check completed condition
            mask_idx_i = current_beam_entry.mask_idx.copy()
            finished = all(
                text_pred_i[m_idx]
                not in {alphabet.missing, alphabet.missing_unk}
                for m_idx in current_beam_entry.mask_idx
            )

            if finished:
              if (text_pred_i not in beam_top) or (
                  text_pred_i in beam_top
                  and beam_top[text_pred_i].pred_logprob < pred_logprob_i
              ):
                beam_top[text_pred_i] = BeamEntry(
                    text_pred_i,
                    sorted(mask_idx_i),
                    current_beam_entry.pred_len + 1,
                    current_beam_entry.unk_len,
                    pred_logprob_i,
                    current_beam_entry.text_history + [text_pred_i],
                )
            else:
              beam_tmp.append(
                  BeamEntry(
                      text_pred_i,
                      mask_idx_i,
                      current_beam_entry.pred_len + 1,
                      current_beam_entry.unk_len,
                      pred_logprob_i,
                      current_beam_entry.text_history + [text_pred_i],
                  )
              )

    # Order all candidates by score
    beam_tmp_kv = {}
    for entry in beam_tmp:
      if (entry.text_pred not in beam_tmp_kv) or (
          entry.text_pred in beam_tmp_kv
          and beam_tmp_kv[entry.text_pred].pred_logprob > entry.pred_logprob
      ):
        beam_tmp_kv[entry.text_pred] = entry
    beam_tmp = sorted(
        list(beam_tmp_kv.values()),
        key=lambda e: e.pred_logprob
        / (1 + e.pred_len + e.unk_len) ** a_penalty,
        reverse=True,
    )

    # Select top k beams
    beam = beam_tmp[:beam_width]

    # Update progress bar
    if display_progress:
      pbar.update(1)  # pyrefly: ignore[unbound-name]

  # Order final candidates by score
  return sorted(
      list(beam_top.values()),
      key=lambda e: e.pred_logprob / (1 + e.pred_len + e.unk_len) ** a_penalty,
      reverse=True,
  )[:beam_width]


def saliency_loss_subregion(
    forward, params, text_char_emb, padding, subregion=None
):
  """Saliency map for subregion."""

  _, subregion_logits, _, _, _ = forward(
      params,
      text_char_emb=text_char_emb,
      padding=padding,
  )
  if subregion is None:
    subregion = subregion_logits.argmax(axis=-1)[0]
  return subregion_logits[0, subregion]


def saliency_loss_date(forward, params, text_char_emb, padding):
  """saliency_loss_date."""

  date_pred, _, _, _, _ = forward(
      params,
      text_char_emb=text_char_emb,
      padding=padding,
  )

  date_pred_argmax = date_pred.argmax(axis=-1)
  return date_pred[0, date_pred_argmax[0]]


def predicted_dates(date_pred_probs, date_min, date_max, date_interval):
  """Returns mode and mean prediction."""
  date_years = np.arange(
      date_min + date_interval / 2, date_max + date_interval / 2, date_interval
  )

  # Compute mode:
  date_pred_argmax = (
      date_pred_probs.argmax() * date_interval + date_min + date_interval // 2
  )

  # Compute mean:
  date_pred_avg = np.dot(date_pred_probs, date_years)

  return date_pred_argmax, date_pred_avg


def compute_attribution_saliency_maps(
    text_char,
    text_len,
    padding,
    forward,
    params,
    alphabet,
    vocab_char_size,
    subregion_loss_kwargs=None,
):
  """Compute saliency maps for subregions and dates."""

  if subregion_loss_kwargs is None:
    subregion_loss_kwargs = {}

  # Get saliency gradients
  dtype = params['params']['char_embeddings']['embedding'].dtype
  text_char_onehot = jax.nn.one_hot(text_char, vocab_char_size).astype(dtype)
  text_char_emb = jnp.matmul(
      text_char_onehot, params['params']['char_embeddings']['embedding']
  )
  gradient_subregion_char = jax.grad(saliency_loss_subregion, 2)(
      forward, params, text_char_emb, padding, **subregion_loss_kwargs
  )
  gradient_date_char = jax.grad(saliency_loss_date, 2)(
      forward, params, text_char_emb, padding=padding
  )

  # Generate saliency maps for subregions
  input_grad_subregion_char = np.multiply(
      gradient_subregion_char, text_char_emb
  )  # grad x input
  subregion_saliency = grad_to_saliency_char(
      input_grad_subregion_char,
      text_char_onehot,
      text_len=text_len,
      alphabet=alphabet,
  )
  subregion_saliency = subregion_saliency / np.max(subregion_saliency[1:])

  # Generate saliency maps for dates
  input_grad_date_char = np.multiply(
      gradient_date_char, text_char_emb
  )  # grad x input
  date_saliency = grad_to_saliency_char(
      input_grad_date_char,
      text_char_onehot,
      text_len=text_len,
      alphabet=alphabet,
  )
  date_saliency = date_saliency / np.max(date_saliency[1:])

  return date_saliency, subregion_saliency


def saliency_loss_mask(
    forward, params, text_char_emb, padding, char_pos, char_idx
):
  """Saliency map for mask."""

  _, _, mask_logits, _, _ = forward(
      params,
      text_char_emb=text_char_emb,
      text_char_onehot=None,
      padding=padding,
  )
  return mask_logits[0, char_pos, char_idx]


class SequentialRestorationSaliencyResult(NamedTuple):
  text: str  # predicted text string so far
  pred_char_pos: int  # newly restored character's position
  saliency_map: np.ndarray  # saliency map for the newly added character


def sequential_restoration_saliency(
    text_history: List[str],
    forward,
    params,
    alphabet,
    vocab_char_size,
):
  """Generates per-step saliency maps based on a given text_history.

  This function iterates through a provided sequence of text states (trajectory)
  and calculates the saliency map for the single character change/insertion
  that transformed the text from state i to state i+1.

  Args:
    text_history: A list of strings representing the sequence of predicted
      texts. text_history[0] is the initial state, text_history[-1] is the
      final. These strings should be user-visible, without SOS/EOS or padding.
    forward: The model's forward pass function.
    params: The model parameters.
    alphabet: The Alphabet object.
    vocab_char_size: The size of the character vocabulary.

  Yields:
    SequentialRestorationSaliencyResult for each step in the history.
  """
  if not text_history or len(text_history) < 2:
    return

  for i in range(len(text_history) - 1):
    text_str_prev_user = text_history[i]  # User-visible string, no SOS
    text_str_curr_user = text_history[i + 1]  # User-visible string, no SOS

    # Determine the changed character and its position in text_str_curr_user
    pred_char_pos_in_curr_user = -1

    min_len_texts = min(len(text_str_prev_user), len(text_str_curr_user))
    found_diff = False
    for k in range(min_len_texts):
      if text_str_prev_user[k] != text_str_curr_user[k]:
        pred_char_pos_in_curr_user = k
        found_diff = True
        break

    if not found_diff:
      if len(text_str_curr_user) > len(text_str_prev_user):  # Insertion
        pred_char_pos_in_curr_user = len(text_str_prev_user)  # New char at end
      elif len(text_str_prev_user) > len(
          text_str_curr_user
      ):  # Deletion (unexpected)
        continue
      else:  # Identical texts
        continue

    if pred_char_pos_in_curr_user == -1:  # Should be caught by identical check
      continue

    char_at_pred_pos = text_str_curr_user[pred_char_pos_in_curr_user]
    pred_char_idx_for_loss = alphabet.char2idx[char_at_pred_pos]

    # Prepare model inputs based on text_str_curr_user.
    # util_text.text_to_idx adds SOS.
    # text_char_model_input is text_str_curr_user converted to indices, WITH SOS
    text_char_model_input = util_text.text_to_idx(
        text_str_curr_user, alphabet
    ).reshape(1, -1)
    # pred_char_pos_for_loss is index in text_char_model_input
    # (accounts for SOS)
    pred_char_pos_for_loss = pred_char_pos_in_curr_user + 1

    current_padding_model_input = jnp.where(text_char_model_input > 0, 1, 0)
    text_len_model_input = text_char_model_input.shape[1]

    # Gradients for saliency map
    text_char_onehot_model_input = jax.nn.one_hot(
        text_char_model_input, vocab_char_size
    ).astype(jnp.float32)
    text_char_emb_model_input = jnp.matmul(
        text_char_onehot_model_input,
        params['params']['char_embeddings']['embedding'],
    )
    gradient_mask_char = jax.grad(saliency_loss_mask, 2)(
        forward,
        params,
        text_char_emb_model_input,
        current_padding_model_input,
        char_pos=pred_char_pos_for_loss,
        char_idx=pred_char_idx_for_loss,
    )

    # Use gradient x input for visuaslizing saliency
    input_grad_mask_char = np.multiply(
        gradient_mask_char, text_char_emb_model_input
    )

    # Return visualization-ready saliency maps
    saliency_map = grad_to_saliency_char(
        input_grad_mask_char,  # From text_str_curr_user
        text_char_onehot_model_input,  # From text_str_curr_user
        [text_len_model_input],  # Length of text_str_curr_user with SOS
        alphabet,
    )  # normalize, etc.

    # Normalize saliency map (excluding SOS part if present)
    # This matches the normalization style in the original greedy version.
    if saliency_map.shape[0] > 1 and np.max(saliency_map[1:]) > 1e-9:
      saliency_map_normalized = saliency_map.copy()
      saliency_map_normalized[1:] = saliency_map_normalized[1:] / np.max(
          saliency_map_normalized[1:]
      )
    else:
      saliency_map_normalized = saliency_map

    # Yield result: text is user-visible, pred_char_pos is user-visible index.
    # Saliency map corresponds to the user-visible part of the text.
    saliency_map_for_user_text = saliency_map_normalized[1:text_len_model_input]

    yield SequentialRestorationSaliencyResult(
        text=text_str_curr_user,
        pred_char_pos=pred_char_pos_in_curr_user,
        saliency_map=saliency_map_for_user_text,
    )


def replace_mask_idx_with_unk(text, mask_idx, missing_unk='_'):
  """Replaces the missing characters with unk missing characters."""
  if not mask_idx:
    return text
  mask_unk = []

  # Sort the mask index
  mask_idx = mask_idx.copy()
  mask_idx.sort()

  # Convert text to a list for easier manipulation
  text_list = list(text)

  # Start with the first group
  text_list[mask_idx[0]] = missing_unk
  mask_unk.append(mask_idx[0])

  # Expand to the rest of the list
  count_empty = 0
  for i in range(1, len(mask_idx)):
    if mask_idx[i] == mask_idx[i - 1] + 1:
      text_list[mask_idx[i]] = ''
      count_empty += 1
    else:
      text_list[mask_idx[i]] = missing_unk
      mask_unk.append(mask_idx[i] - count_empty)

  return ''.join(text_list), mask_unk
