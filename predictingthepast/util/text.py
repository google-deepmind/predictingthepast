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
"""Text processing functions."""

import random
import re
import unicodedata

import numpy as np


def idx_to_text(idxs, alphabet, strip_sos=True, strip_pad=True):
  """Converts a list of indices to a string."""
  idxs = np.array(idxs)
  out = ''
  for i in range(idxs.size):
    idx = idxs[i]
    if strip_pad and idx == alphabet.pad_idx:
      break
    elif strip_sos and idx == alphabet.sos_idx:
      pass
    else:
      out += alphabet.idx2char[idx]
  return out


def idx_to_text_batch(idxs, alphabet, lengths=None):
  """Converts batched lists of indices to strings."""
  b = []
  for i in range(idxs.shape[0]):
    idxs_i = idxs[i]
    if lengths:
      idxs_i = idxs_i[: lengths[i]]
    b.append(idx_to_text(idxs_i, alphabet))
  return b


def random_mask_span(t, geometric_p=0.2, limit_chars=None):
  """Masks a span of sequential words."""

  # Sample a random span length using a geomteric distribution
  if geometric_p and limit_chars:
    span_len = min(np.random.geometric(geometric_p), limit_chars)
  elif geometric_p:
    span_len = np.random.geometric(geometric_p)
  elif limit_chars:
    span_len = limit_chars
  else:
    raise ValueError('geometric_p or limit_chars should be set.')

  # Obtain span indexes (indlusive)
  span_idx = [
      (ele.start(), ele.end())
      for ele in re.finditer(r'[a-zA-Z0α-ωΑ-Ω\s]+', t)
      if ele.end() - ele.start() >= span_len
  ]
  if not span_idx:
    return []

  # Select a span to mask
  span_start, span_end = random.choice(span_idx)

  # Pick a random start index
  span_start = np.random.randint(span_start, span_end - span_len + 1)
  assert span_start + span_len <= span_end

  # Create mask indices
  mask_idx = np.arange(span_start, span_start + span_len, dtype=np.int32)

  return mask_idx


def random_sentence_swap(sentences, p):
  """Swaps sentences with probability p."""

  if len(sentences) < 2:
    return sentences  # Can't swap if there's less than 2 sentences

  sentences = sentences.copy()
  for i in range(len(sentences) - 1):
    if np.random.uniform() < p:
      sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]  # Swap

  return sentences


def random_char_delete(sentence, p):
  """Deletes a char from a sentence with probability p."""

  chars = list(sentence)

  # Randomly drop a char
  new_chars = [c for c in chars if np.random.uniform() > p]

  if not new_chars:
    return sentence

  return ''.join(new_chars)


def random_word_delete(sentence, p):
  """Deletes a word from a sentence with probability p."""

  words = sentence.strip().split(' ')

  if len(words) == 1:
    return words[0]

  # Randomly drop a word
  new_words = [word for word in words if np.random.uniform() > p]

  # Cover the case where all words are deleted
  return ' '.join(new_words) or random.choice(words)


def random_word_abbr(sentence, p):
  """Abbreviates from a sentence with probability p."""

  words = sentence.strip().split(' ')
  for i in range(len(words)):
    if np.random.uniform() < p and len(words[i]) > 1:
      words[i] = words[i][0]

  return ' '.join(words)


def random_word_swap(sentence, p):
  """Swaps words from a sentence with probability p."""

  words = sentence.strip().split(' ')

  if len(words) < 2:
    return sentence  # Can't swap if there's less than 2 words

  new_words = words.copy()
  for i in range(len(words) - 1):
    if np.random.uniform() < p:
      new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]  # Swap

  return ' '.join(new_words)


def strip_accents(s):
  return ''.join(
      c
      for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )


def text_to_idx(t, alphabet):
  """Converts a string to character indices."""
  return np.array([alphabet.char2idx[c] for c in t], dtype=np.int32)


def hasalnum(s):
  for c in s:
    if c.isalnum():
      return True
  return False


def inject_missing_unk(t, geometric_p=0.2, missing_unk='#', min_len=None):
  """Indejects missing unk in string."""

  # Pick a span length to mask
  span_len = np.random.geometric(geometric_p) - 1

  # If resulting text too short skip
  if (min_len and len(t) - span_len < min_len) or span_len == 0:
    return t, 0

  # Pick start and end indices
  start_idx = np.random.randint(0, len(t) - span_len + 1)
  end_idx = start_idx + span_len

  # Replace the text
  t_out = t[:start_idx] + missing_unk + t[end_idx:]

  return t_out, span_len
