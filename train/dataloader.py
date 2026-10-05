# Copyright 2025 the Aeneas Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataloader functions."""

import concurrent.futures
import glob
import io
import itertools
import json
import random
import re

from absl import logging
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from predictingthepast.util import dates as util_dates
from predictingthepast.util import region_names
from predictingthepast.util import text as util_text
import tensorflow.compat.v1 as tf


DATASET_GREEK = 0
DATASET_LATIN = 1


def append_id(dataset, dataset_id):
  for i in range(len(dataset)):
    dataset[i]['dataset_id'] = dataset_id
  return dataset


def img_skew(img, max_skew=0.2):
  width, height = img.size
  x_shift = np.random.uniform(-max_skew, max_skew) * width
  y_shift = np.random.uniform(-max_skew, max_skew) * height
  skew_matrix = (1, x_shift / height, 0, y_shift / width, 1, 0)
  return img.transform(img.size, Image.Transform.AFFINE, skew_matrix)


def img_add_random_noise(img, noise_level=0.05):
  noise_level_rand = np.random.uniform(0.0, noise_level)
  np_img = np.array(img).astype('float32')  # Convert to numpy array
  np_img += np.random.normal(scale=noise_level_rand * 255, size=np_img.shape)
  np_img = np.clip(np_img, 0, 255).astype('uint8')  # Clip values to valid range
  return Image.fromarray(np_img)


def random_img_aug(
    img,
    output_size=(128, 128),
    mode='train',
    zoom_factor=4,
    zoom_sampling_log=True,
):
  """Randomly augment an image."""
  # Convert to grayscale
  img = img.convert('L')

  if mode == 'train':

    # Random skew
    img = img_skew(img, max_skew=0.1)

    # Random rotation
    rotation_degree = np.random.randint(-30, 30 + 1)
    img = img.rotate(rotation_degree)

    # Random brightness
    brightness_factor = np.random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # Random contrast
    contrast_factor = np.random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Random slight blur
    blur_radius = np.random.uniform(0, 2)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Random noise
    img = img_add_random_noise(img, noise_level=0.05)

    # Random sharpen
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(np.random.uniform(0.0, 2.0))

  # Get image size
  width, height = img.size

  # Determine the longer side of the image
  max_side = max(img.size)
  min_side = min(img.size)

  # Create a square canvas with side length max_side
  padded_img = Image.new('L', (max_side, max_side), (0,))

  # Compute padding offsets
  offset_x = (max_side - width) // 2
  offset_y = (max_side - height) // 2

  # Paste image onto canvas
  padded_img.paste(img, (offset_x, offset_y))

  if mode == 'train':

    # Determine the size of the crop
    min_crop_size = int(min_side / zoom_factor)
    max_crop_size = max_side

    if zoom_sampling_log:
      crop_size = int(
          np.exp(
              np.random.uniform(np.log(min_crop_size), np.log(max_crop_size))
          )
      )
    else:
      crop_size = np.random.randint(min_crop_size, max_crop_size + 1)

    # Choose a random starting point for the crop
    left_start = max(offset_x - crop_size // 2, 0)
    upper_start = max(offset_y - crop_size // 2, 0)
    left_end = min(left_start + width, padded_img.width - crop_size)
    upper_end = min(upper_start + height, padded_img.height - crop_size)

    left = np.random.randint(left_start, left_end + 1)
    upper = np.random.randint(upper_start, upper_end + 1)
    right = left + crop_size
    lower = upper + crop_size
  else:
    crop_size = max_side

    left = 0
    upper = 0
    right = left + crop_size
    lower = upper + crop_size

  # Crop the image
  cropped_img = padded_img.crop((left, upper, right, lower))

  # Resize
  cropped_img = cropped_img.resize(output_size)

  return cropped_img


def should_process_sample(sample_id, mode, allow_list=None):
  """Determines if a sample should be processed based on its ID, mode, and allow list."""
  last_digit = sample_id % 10
  if allow_list and sample_id in allow_list:
    return True
  elif not allow_list:
    if mode == 'test' and last_digit == 3:
      return True
    elif mode == 'valid' and last_digit == 4:
      return True
    elif mode == 'train' and last_digit not in (3, 4):
      return True
  return False


def generate_and_yield(*args, **kwargs):
  # with generate_semaphore:  # Acquire semaphore within the worker function
  s = generate_sample(*args, **kwargs)
  if s:
    yield s


def get_vision_img(config, record_number, vision_img_cache, mode):
  """Returns vision image and embedding."""
  # Load image
  path = f'{config.vision.path}{record_number}.jpg'

  # Use image cache if available.
  if vision_img_cache.get(path) is not None:

    cv2_img = vision_img_cache[path]

    # Convert OpenCV image (numpy array) back to PIL
    img = Image.fromarray(cv2_img)

    # Generate random augmented image
    vision_img = random_img_aug(
        img,
        mode=mode,
        output_size=config.vision.output_size,
        zoom_factor=config.vision.zoom_factor,
        zoom_sampling_log=config.vision.zoom_sampling_log,
    )

    # Standardize image values
    vision_img = np.array(vision_img)[..., None]
    vision_img = vision_img / 127.5 - 1.0

    # Mark as available
    return vision_img, True
  else:
    return None, False


def generate_sample(
    config,
    alphabet,
    region_map,
    sample,
    vision_img_cache=None,
    mode='train',
):
  """Generates a new TF dataset sample."""

  # Sample id
  inscription_id = int(sample['id'])
  if 'record_number' in sample:
    record_number = sample['record_number']
  else:
    record_number = None

  # Vision
  if 'vision' in config and config.vision.enabled:
    assert vision_img_cache is not None
    future_vision = None

    # If training load the image with 50% chance
    if (mode == 'train' and np.random.choice([True, False])) or mode != 'train':

      vision_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      future_vision = vision_executor.submit(
          get_vision_img,
          config,
          record_number,
          vision_img_cache,
          mode,
      )

  # Get text
  text = sample['text']
  text = alphabet.filter(text)
  text = text.lower().strip()
  sentence_delimiters = ''.join(set(alphabet.punctuation))

  # Random word swap
  if mode == 'train' and config.random_word_swap > 0:
    text = util_text.random_word_swap(text, config.random_word_swap)

  # Random word swap
  if mode == 'train' and config.random_word_abbr > 0:
    text = util_text.random_word_abbr(text, config.random_word_abbr)

  # Random word delete
  if mode == 'train' and config.random_word_delete > 0:
    text = util_text.random_word_delete(text, config.random_word_delete)

  # Random char delete
  if mode == 'train' and config.random_char_delete > 0:
    text = util_text.random_char_delete(text, config.random_char_delete)

  # Inject missing unk and mask text
  unk_len = 0
  if mode in ['train', 'valid'] and config.inject_missing_unk_p > 0:
    text, unk_len = util_text.inject_missing_unk(
        text,
        geometric_p=config.inject_missing_unk_p,
        missing_unk='*',
        min_len=config.context_char_min,
    )

  # Add punctuation to the end
  if text[-1] not in sentence_delimiters:
    text = text + '.'

  # Split sentences and append punctuation
  tok = re.split(rf'([{sentence_delimiters}]+)', text)
  sentences = [
      tok[i] + tok[i + 1] if i + 1 < len(tok) else tok[i]
      for i in range(0, len(tok), 2)
  ]

  # Strip spaces
  sentences = list(map(str.strip, sentences))
  # Filter blank sentences
  sentences = list(filter(None, sentences))
  # Remove sentences without alphanumerics
  sentences = list(filter(util_text.hasalnum, sentences))
  # Generate indexes
  sentence_idx = np.arange(len(sentences), dtype=np.int32)

  # Random sentence shuffling
  if mode == 'train' and config.random_sentence_swap > 0:
    # Shuffle indexes
    sentence_idx = util_text.random_sentence_swap(
        sentence_idx, config.random_sentence_swap
    )
    # Reshuffle sentences
    sentences = np.array(sentences)[sentence_idx].tolist()

  # Join text
  text = ' '.join(sentences)

  # Randomly delete all punctuation
  if config.punctuation_delete:
    text = re.sub(rf'[{sentence_delimiters}]+', '', text)

  # Computer start for prepending start of sentence character
  start_sample_idx = int(config.prepend_sos)

  if (
      mode in ['train']  # , 'valid'
      and config.context_char_random
      and len(text) >= config.context_char_min
  ):
    # During training pick random context length
    context_char_len = np.random.randint(
        config.context_char_min,
        min(len(text), config.context_char_max - start_sample_idx) + 1,
    )

    start_idx = 0
    if context_char_len < len(text):
      start_idx = np.random.randint(0, len(text) - context_char_len + 1)
    text = text[start_idx : start_idx + context_char_len - start_sample_idx]

  elif config.context_char_max and len(text) > (
      config.context_char_max - start_sample_idx
  ):
    # Clip text by maximum length
    start_idx = np.random.randint(
        0, len(text) - (config.context_char_max - start_sample_idx) + 1
    )
    text = text[
        start_idx : start_idx + config.context_char_max - start_sample_idx
    ]

  # Prepend start of sentence character
  text = alphabet.sos * config.prepend_sos + text

  # Get missing unk len
  missing_unk_label = np.zeros(len(text), dtype=int)
  missing_unk_mask = np.zeros(len(text), dtype=bool)
  if unk_len > 0:
    missing_unk_idx = text.find('*')
    if missing_unk_idx > 0:
      missing_unk_mask[missing_unk_idx] = True
      missing_unk_label[missing_unk_idx] = int(unk_len > 1)
      text = text.replace('*', alphabet.missing_unk)

  # Unmasked text
  text_unmasked_idx = util_text.text_to_idx(text, alphabet)

  # Mask text
  text_mask = np.zeros(len(text), dtype=bool)
  char_mask_idx = []
  text_list = list(text)

  if mode == 'train':
    # Non missing idx (avoid removing start of sentence character)
    non_missing_idx = []
    for i in range(start_sample_idx, len(text_list)):
      if (
          text_list[i]
          not in [alphabet.missing, alphabet.missing_unk] + alphabet.punctuation
      ):
        non_missing_idx.append(i)

    # Skip sample if there are no usable characters
    if not non_missing_idx:
      return

    if config.char_mask_rate_max > 0.0:
      # Compute rate
      char_mask_rate = np.random.uniform(
          config.char_mask_rate_min, config.char_mask_rate_max
      )

      # Fix masking in valid mode for comparing experiments
      span_mask_geometric_p = config.span_mask_geometric_p
      mask_num_total = int(char_mask_rate * len(non_missing_idx))
      mask_num_span = int(mask_num_total * config.span_mask_ratio)
      mask_num_char = mask_num_total - mask_num_span

      # Mask random indices
      if mask_num_char > 0:
        char_mask_idx = np.random.choice(
            non_missing_idx, mask_num_char, replace=False
        ).tolist()

      # Mask random spans
      if mask_num_span > 0:
        span_mask_idx = []
        for _ in range(1000):
          span_mask_idx_ = util_text.random_mask_span(
              text,
              geometric_p=span_mask_geometric_p,
              limit_chars=mask_num_span - len(span_mask_idx),
          )
          if len(span_mask_idx_) + len(span_mask_idx) <= mask_num_span:
            span_mask_idx.extend(span_mask_idx_)
          elif len(span_mask_idx_) + len(span_mask_idx) > mask_num_span:
            break
        char_mask_idx.extend(span_mask_idx)
  elif mode == 'valid':
    for _ in range(1000):
      mask_num_span = np.random.randint(1, config.span_mask_eval_len + 1)
      char_mask_idx = util_text.random_mask_span(
          text,
          geometric_p=None,
          limit_chars=mask_num_span,
      )
      if len(char_mask_idx) == mask_num_span:
        break
      elif char_mask_idx and len(char_mask_idx) != mask_num_span:
        raise ValueError(
            f'Error in mask length generation. Text: {text}, char mask index:'
            f' {char_mask_idx}, mask num span: {mask_num_span}'
        )

  # Mask text
  for idx in set(char_mask_idx):
    text_mask[idx] = True
    text_list[idx] = alphabet.missing
  text = ''.join(text_list)

  # Text missing mask
  text_np = np.array(list(text))
  text_missing_mask = np.logical_or(
      text_np == alphabet.missing, text_np == alphabet.missing_unk
  )

  # Convert to indices
  text_idx = util_text.text_to_idx(text, alphabet)
  text_idx_len = len(text_idx)

  if text_idx_len < config.context_char_min:
    logging.info('Short context: %d, %d', inscription_id, text_idx_len)
    return None

  # Skip if region does not exist in map
  region_id = 0
  region_available = False
  region_name = None
  if 'region_sub' in sample:
    region_name = region_names.region_name_filter(sample['region_sub'])

  if region_name in region_map['names']:
    region_id = region_map['names_inv'][region_name]
    region_available = True

  # Dates
  if (
      sample['date_min']
      and sample['date_max']
      and int(sample['date_min']) <= int(sample['date_max'])
      and int(sample['date_min']) >= config.date_min
      and int(sample['date_max']) < config.date_max
  ):
    date_available = True
    date_min = float(sample['date_min'])
    date_max = float(sample['date_max'])
    date_dist = util_dates.date_range_to_dist(
        date_min,
        date_max,
        config.date_min,
        config.date_max,
        config.date_interval,
        config.date_bins,
    )
  else:
    date_available = False
    date_min = 0.0
    date_max = 0.0
    date_dist = util_dates.date_range_to_dist(
        None,
        None,
        config.date_min,
        config.date_max,
        config.date_interval,
        config.date_bins,
    )

  # Vision
  if 'vision' in config and config.vision.enabled:
    vision_available = False
    vision_img = np.zeros((
        config.vision.output_size[0],
        config.vision.output_size[1],
        1,
    ))
    if future_vision is not None:  # pyrefly: ignore[unbound-name]
      out_vision = future_vision.result()
      if out_vision[0] is not None:
        vision_img = out_vision[0]
        vision_available = out_vision[1]
      vision_executor.shutdown()  # pyrefly: ignore[unbound-name]

  out = {
      'id': inscription_id,  # 'text_str': text,
      'text_char': text_idx,
      'text_mask': text_mask,
      'text_missing_mask': text_missing_mask,
      'text_len': text_idx_len,
      'text_unmasked': text_unmasked_idx,
      'next_sentence_mask': np.zeros(len(text_idx), dtype=bool),
      'next_sentence_label': np.zeros(len(text_idx), dtype=np.int32),
      'missing_unk_mask': missing_unk_mask,
      'missing_unk_label': missing_unk_label,
      'region_available': region_available,
      'region_id': region_id,
      'date_available': date_available,
      'date_min': date_min,
      'date_max': date_max,
      'date_dist': date_dist,
      'dataset_id': sample['dataset_id'],
  }
  if 'vision' in config and config.vision.enabled:
    out.update({
        'vision_img': vision_img.astype(np.float16),  # pyrefly: ignore[unbound-name]
        'vision_available': vision_available,  # pyrefly: ignore[unbound-name]
    })
  return out


def loader_tf(
    batch_size,
    config,
    region_map,
    alphabet=None,
    latin_dataset_file=None,
    greek_dataset_file=None,
    mode='train',
    open_fn=open,
    glob_fn=glob.glob,
):
  """TF dataloader."""

  dataset = {}
  if mode == 'train':
    for l in config.train_language:
      dataset[l] = []
  else:
    for l in config.eval_language:
      dataset[l] = []

  # Allowlist
  allow_list = set()
  if getattr(config, 'allow_list', []):
    allow_list = set(map(int, config.allow_list))

  # Blocklist
  block_list = set()
  if getattr(config, 'block_list', []):
    logging.info('Ignore list texts: %d.', len(config.block_list))
    block_list.update(config.block_list)

  if (mode == 'train' and 'greek' in config.train_language) or (
      mode != 'train' and 'greek' in config.eval_language
  ):
    # Load Greek dataset
    logging.info('Loading Greek dataset')
    dataset_tmp = {d['id']: d for d in json.load(greek_dataset_file)}  # pyrefly: ignore[bad-argument-type]

    # Find duplicate inscriptions
    rev_dataset = {}

    for key in sorted(dataset_tmp.keys()):
      value = dataset_tmp[key]
      rev_dataset.setdefault(value['text'], set()).add(key)
      if len(rev_dataset[value['text']]) > 1:
        block_list.add(int(value['id']))
    del rev_dataset
    logging.info('Texts filtered: %d.', len(block_list))

    # Create deduplicated dataset
    for d in dataset_tmp.values():
      if int(d['id']) not in block_list:
        dataset['greek'].append(d)
    del dataset_tmp

    dataset['greek'] = append_id(dataset['greek'], DATASET_GREEK)

    logging.info('Greek dataset texts: %d.', len(dataset['greek']))

  # Load Latin dataset
  if (mode == 'train' and 'latin' in config.train_language) or (
      mode != 'train' and 'latin' in config.eval_language
  ):
    dataset['latin'] = []
    dataset_tmp = json.load(latin_dataset_file)  # pyrefly: ignore[bad-argument-type]
    for d in dataset_tmp:
      if d['id'] not in block_list:
        dataset['latin'].append(d)
    dataset['latin'] = append_id(dataset['latin'], DATASET_LATIN)
    logging.info('Latin dataset texts: %d.', len(dataset['latin']))

  dataset_len = sum([len(v) for v in dataset.values()])
  logging.info('Total dataset texts: %d.', dataset_len)

  # Setup vision embedder.
  vision_img_cache = {}
  if 'vision' in config and config.vision.enabled:

    # Load image bytes
    def load_image(path):
      """Loads an image from the given path and returns its bytes."""
      with open_fn(path, 'rb') as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).copy()  # pyrefly: ignore[bad-argument-type]
        img = img.convert('L')

        # Convert PIL image to OpenCV (numpy array)
        cv2_img = np.array(img).astype(np.uint8)
        cv2_img = cv2.bilateralFilter(
            cv2_img, d=9, sigmaColor=40, sigmaSpace=40
        )
        cv2_img = cv2.fastNlMeansDenoising(
            cv2_img, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        return path, cv2_img

    # Preload images in parallel
    available_paths = set(glob_fn(config.vision.path + '/*.jpg'))
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for d in dataset.values():
        futures = []
        for sample in d:
          if 'record_number' in sample:

            # Check if image exists
            record_number = sample['record_number']
            path = f'{config.vision.path}{record_number}.jpg'

            if path in available_paths:
              futures.append(executor.submit(load_image, path))
            else:
              vision_img_cache[path] = None

        # Fetch results
        logging.info('Preloading images')
        for future in concurrent.futures.as_completed(futures):
          path, image_bytes = future.result()
          vision_img_cache[path] = image_bytes
        logging.info('Preloading images finished')

  # Sample generator function
  def generate_samples():

    dataset_ = list(itertools.chain.from_iterable(dataset.values()))

    # Generate indices and sample equal number of samples from both datasets.
    if mode == 'train':
      dataset_max_len = max([len(dataset[l]) for l in dataset])
      for l, v in dataset.items():
        # Add more samples to balance datasets
        if len(v) < dataset_max_len:
          v_idx = np.random.randint(0, len(v), dataset_max_len - len(v))
          for idx in v_idx:
            dataset_.append(v[idx])

    # Breaks dataset correlated order
    logging.info('Shuffling dataset')
    random.shuffle(dataset_)

    for sample in dataset_:
      # Replace guess signs with missing chars
      if 'region_sub' in sample:
        sample['text'] = re.sub(
            r'\[(.*?)\]', lambda m: '-' * len(m.group(1)), sample['text']
        )
        sample['text'] = (
            sample['text'].replace(alphabet.sog, '').replace(alphabet.eog, '')  # pyrefly: ignore[missing-attribute]
        )

      # Filter by text length
      if (
          len(
              sample['text']
              .replace(alphabet.missing, '')  # pyrefly: ignore[missing-attribute]
              .replace(alphabet.missing_unk, '')  # pyrefly: ignore[missing-attribute]
              .strip()
          )
          < config.context_char_min
      ):
        continue

      # Last digit 3 -> test, 4 -> valid, the rest are the training set
      sample_id = int(sample['id'])
      if should_process_sample(sample_id, mode, allow_list):

        yield from generate_and_yield(
            config,
            alphabet,
            region_map,
            sample,
            vision_img_cache,
            mode,
        )

  # Create dataset from generator.
  with tf.device('/cpu:0'):
    output_signature = {
        'id': tf.TensorSpec(shape=(), dtype=tf.int32),
        'text_char': tf.TensorSpec(shape=(None), dtype=tf.int32),
        'text_mask': tf.TensorSpec(shape=(None), dtype=tf.bool),
        'text_missing_mask': tf.TensorSpec(shape=(None), dtype=tf.bool),
        'text_unmasked': tf.TensorSpec(shape=(None), dtype=tf.int32),
        'next_sentence_mask': tf.TensorSpec(shape=(None), dtype=tf.bool),
        'next_sentence_label': tf.TensorSpec(shape=(None), dtype=tf.int32),
        'missing_unk_mask': tf.TensorSpec(shape=(None), dtype=tf.bool),
        'missing_unk_label': tf.TensorSpec(shape=(None), dtype=tf.int32),
        'text_len': tf.TensorSpec(shape=(), dtype=tf.int32),
        'region_available': tf.TensorSpec(shape=(), dtype=tf.bool),
        'region_id': tf.TensorSpec(shape=(), dtype=tf.int32),
        'date_available': tf.TensorSpec(shape=(), dtype=tf.bool),
        'date_min': tf.TensorSpec(shape=(), dtype=tf.float32),
        'date_max': tf.TensorSpec(shape=(), dtype=tf.float32),
        'date_dist': tf.TensorSpec(shape=(config.date_bins), dtype=tf.float32),
        'dataset_id': tf.TensorSpec(shape=(), dtype=tf.int32),
    }

    if 'vision' in config and config.vision.enabled:
      output_signature.update({
          'vision_img': tf.TensorSpec(
              (config.vision.output_size[0], config.vision.output_size[1], 1),
              dtype=tf.float16,
          ),
          'vision_available': tf.TensorSpec(shape=(), dtype=tf.bool),
      })

    ds = tf.data.Dataset.from_generator(
        generate_samples,
        output_signature=output_signature,
    )

  # Shuffle and repeat.
  if mode == 'train':
    if config.repeat_train == -1:
      ds = ds.repeat()
    elif config.repeat_train >= 1:
      ds = ds.repeat(config.repeat_train)
  else:
    if config.repeat_eval == -1:
      ds = ds.repeat()
    elif config.repeat_eval >= 1:
      ds = ds.repeat(config.repeat_eval)

  # Batch and pad.
  max_len = config.context_char_max
  padded_shapes = {
      'id': [],
      'text_char': [max_len],
      'text_mask': [max_len],
      'text_missing_mask': [max_len],
      'text_unmasked': [max_len],
      'next_sentence_mask': [max_len],
      'next_sentence_label': [max_len],
      'missing_unk_mask': [max_len],
      'missing_unk_label': [max_len],
      'text_len': [],
      'region_available': [],
      'region_id': [],
      'date_available': [],
      'date_min': [],
      'date_max': [],
      'date_dist': [config.date_bins],
      'dataset_id': [],
  }
  padding_values = {
      'id': 0,
      'text_char': alphabet.pad_idx,  # pyrefly: ignore[missing-attribute]
      'text_mask': False,
      'text_missing_mask': True,
      'text_unmasked': alphabet.pad_idx,  # pyrefly: ignore[missing-attribute]
      'next_sentence_mask': False,
      'next_sentence_label': 0,
      'missing_unk_mask': False,
      'missing_unk_label': 0,
      'text_len': 0,
      'region_available': False,
      'region_id': 0,
      'date_available': False,
      'date_min': 0.0,
      'date_max': 0.0,
      'date_dist': 0.0,
      'dataset_id': 0,
  }

  if 'vision' in config and config.vision.enabled:
    padded_shapes.update({
        'vision_img': [
            config.vision.output_size[0],
            config.vision.output_size[1],
            1,
        ],
        'vision_available': [],
    })
    padding_values.update({
        'vision_img': np.float16(0.0),
        'vision_available': False,
    })

  ds = ds.padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      padding_values=padding_values,
  )

  ds = ds.prefetch(tf.data.AUTOTUNE)

  return ds
