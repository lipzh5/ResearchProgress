# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import clip
import torch
import numpy as np
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp

import optax
from CliportVariant import TransporterNets, n_params
from CollectDemonst import collect_demonst
import Utils
import os
import flax


# Coordinate map (i.e. position encoding).
coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)

#@markdown Define Transporter Nets train and eval functions

# Train with InfoNCE loss over pick and place positions.
@jax.jit
def train_step(optimizer, batch):
  def loss_fn(params):
    batch_size = batch['img'].shape[0]
    pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'], batch['pick_yx'])

    # InfoNCE pick loss.
    pick_logits = pick_logits.reshape(batch_size, -1)
    pick_onehot = batch['pick_onehot'].reshape(batch_size, -1)
    pick_loss = jnp.mean(optax.softmax_cross_entropy(logits=pick_logits, labels=pick_onehot), axis=0)

    # InfoNCE place loss.
    place_logits = place_logits.reshape(batch_size, -1)
    place_onehot = batch['place_onehot'].reshape(batch_size, -1)
    place_loss = jnp.mean(optax.softmax_cross_entropy(logits=place_logits, labels=place_onehot), axis=0)

    loss = pick_loss + place_loss
    return loss, (pick_logits, place_logits)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, grad, logits

@jax.jit
def eval_step(params, batch):
  pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'])
  return pick_logits, place_logits


#@markdown Compute CLIP features for text in the dataset.

def precompute_CLIP_features():
  # Precompute CLIP features for all text in training dataset.
  dataset, dataset_size = collect_demonst()
  text_tokens = clip.tokenize(dataset['text']).cuda()
  text_i = 0
  data_text_feats = np.zeros((0, 512), dtype=np.float32)
  from ClipModel import clip_model_loader
  clip_model = clip_model_loader.load_instance()

  while text_i < len(text_tokens):
    batch_size = min(len(text_tokens) - text_i, 512)
    text_batch = text_tokens[text_i:text_i+batch_size]
    with torch.no_grad():
      batch_feats = clip_model.encode_text(text_batch).float()
    batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
    batch_feats = np.float32(batch_feats.cpu())
    data_text_feats = np.concatenate((data_text_feats, batch_feats), axis=0)
    text_i += batch_size
  return dataset, dataset_size, data_text_feats


def train_or_load_model(load_pretrained=True):
  #@markdown Train your own model, or load a pretrained one.
  # load_pretrained = True  #@param {type:"boolean"}

  # Initialize model weights using dummy tensors.
  rng = jax.random.PRNGKey(0)
  rng, key = jax.random.split(rng)
  init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
  init_text = jnp.ones((4, 512), jnp.float32)
  init_pix = jnp.zeros((4, 2), np.int32)
  init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
  print(f'Model parameters: {n_params(init_params):,}')
  optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)

  if load_pretrained:
    ckpt_path = f'ckpt_{40000}'
    if not os.path.exists(ckpt_path):
      Utils.download_with_id('1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP')
      # !gdown --id 1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP
    optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    print('Loaded:', ckpt_path)
  else:
    # Training loop.
    dataset, dataset_size, data_text_feats = precompute_CLIP_features()
    batch_size = 8
    for train_iter in range(1, 40001):
      batch_i = np.random.randint(dataset_size, size=batch_size)
      text_feat = data_text_feats[batch_i, ...]
      img = dataset['image'][batch_i, ...] / 255
      img = np.concatenate((img, np.broadcast_to(coords[None, ...], (batch_size,) + coords.shape)), axis=3)

      # Get onehot label maps.
      pick_yx = np.zeros((batch_size, 2), dtype=np.int32)
      pick_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
      place_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
      for i in range(len(batch_i)):
        pick_y, pick_x  = dataset['pick_yx'][batch_i[i], :]
        place_y, place_x = dataset['place_yx'][batch_i[i], :]
        pick_onehot[i, pick_y, pick_x] = 1
        place_onehot[i, place_y, place_x] = 1
        # pick_onehot[i, ...] = scipy.ndimage.gaussian_filter(pick_onehot[i, ...], sigma=3)
        # Data augmentation (random translation).
        roll_y, roll_x = np.random.randint(-112, 112, size=2)
        img[i, ...] = np.roll(img[i, ...], roll_y, axis=0)
        img[i, ...] = np.roll(img[i, ...], roll_x, axis=1)
        pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_y, axis=0)
        pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_x, axis=1)
        place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_y, axis=0)
        place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_x, axis=1)
        pick_yx[i, 0] = pick_y + roll_y
        pick_yx[i, 1] = pick_x + roll_x

      import datetime
      from flax.metrics import tensorboard
      #@markdown **TensorBoard:** Displays an interactive TensorBoard.
      name = datetime.datetime.now().strftime(f'%Y-%m-%d-%H:%M:%S-cliport')
      logdir = os.path.join("logs", name)
      writer = tensorboard.SummaryWriter(logdir)
      # %tensorboard --logdir logs
      # Backpropagate.
      batch = {}
      batch['img'] = jnp.float32(img)
      batch['text'] = jnp.float32(text_feat)
      batch['pick_yx'] = jnp.int32(pick_yx)
      batch['pick_onehot'] = jnp.float32(pick_onehot)
      batch['place_onehot'] = jnp.float32(place_onehot)
      rng, batch['rng'] = jax.random.split(rng)
      optim, loss, _, _ = train_step(optim, batch)
      writer.scalar('train/loss', loss, train_iter)

      if train_iter % np.power(10, min(4, np.floor(np.log10(train_iter)))) == 0:
        print(f'Train Step: {train_iter} Loss: {loss}')

      if train_iter % 1000 == 0:
        checkpoints.save_checkpoint('.', optim, train_iter, prefix='ckpt_', keep=100000, overwrite=True)
  return optim


# if __name__ == "__main__":
#   testx = np.linspace(-1, 1, 10)
#   testxx = testx[..., None]
#   print(testx, type(testx), testx.shape)
#   print('===\n')
#   print(testxx, testxx.shape)
#   y = np.expand_dims(testx, -1)
#   print(y, y.shape)