import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from scipy.spatial.transform import Rotation as R


class ImplicitSO3(nn.Module):
     def __init__(self, len_visual_description, number_fourier_components, mlp_layer_sizes,
                 so3_sampling_mode, number_train_queries, number_eval_queries):
          super(ImplicitSO3, self).__init__()
          self.len_rotation = 9
          self.number_fourier_components = number_fourier_components
          self.frequencies = 2 ** torch.arange(number_fourier_components, dtype=torch.float32)
          self.so3_sampling_mode = so3_sampling_mode
          self.number_train_queries = number_train_queries
          self.number_eval_queries = number_eval_queries

          if number_fourier_components == 0:
               self.len_query = self.len_rotation
          else:
               self.len_query = self.len_rotation * number_fourier_components * 2

          self.grids = {}

          # Initialize the MLP model
          self.query_embedding_layer = nn.Linear(self.len_query, mlp_layer_sizes[0])
          self.hidden_layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(mlp_layer_sizes[:-1], mlp_layer_sizes[1:])])
          self.output_layer = nn.Linear(mlp_layer_sizes[-1], 1)

          if self.so3_sampling_mode == 'grid':
               self.get_closest_available_grid(self.number_train_queries)
          self.get_closest_available_grid(self.number_eval_queries)

     def forward(self, vision_description, rotation_matrix, training=False):
          if training:
               query_rotations = self.generate_queries(self.number_train_queries, mode=self.so3_sampling_mode)
          else:
               query_rotations = self.generate_queries(self.number_eval_queries, mode='grid')
          query_rotations = query_rotations.to(vision_description.device) # move to cuda

          delta_rot = query_rotations[-1].T @ rotation_matrix  # [B, 3, 3]
          query_rotations = torch.einsum('aij,bjk->baik', query_rotations, delta_rot)  # [B, num, 3, 3]
          query_rotations = query_rotations.reshape(-1, self.number_train_queries, self.len_rotation)  # [B, num, 9]
          query_rotations = self.positional_encoding(query_rotations)  # [B, num, 72]

          vision_embedding = vision_description
          query_embedding = self.query_embedding_layer(query_rotations)
          output = vision_embedding + query_embedding   # element-wise addition
          output = F.relu(output)

          for layer in self.hidden_layers:
               output = F.relu(layer(output))
          prob = self.output_layer(output)   # [B, num_query_rotation, 1]

          logits = prob[..., 0]   # [B, num_query_rotation]
          probabilities = torch.softmax(logits, dim=-1) # [B, num_query_rotation]
          scaling_factor = 1
          scaled_probabilities = probabilities * scaling_factor  # Use a new tensor
          loss = -torch.mean(torch.log(scaled_probabilities[:, -1]))
          return loss

     def predict_probability(self, vision_description, number_train_queries, take_softmax=True):
          batch_size = vision_description.shape[0]
          query_rotations_org = self.generate_queries(number_train_queries, mode=self.so3_sampling_mode)          
          query_rotations = query_rotations_org.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, num, 3, 3]
          query_rotations_embedded = query_rotations.reshape(-1, query_rotations.shape[1], self.len_rotation) 
          query_rotations_embedded = query_rotations_embedded.to(vision_description.device) # move to cuda
          query_rotations = self.positional_encoding(query_rotations_embedded)

          vision_embedding = vision_description
          query_embedding = self.query_embedding_layer(query_rotations)
          output = vision_embedding + query_embedding
          output = F.relu(output)
          for layer in self.hidden_layers:
               output = F.relu(layer(output))
          prob = self.output_layer(output)
          out = prob[..., 0]
          if take_softmax:
               probabilities = torch.softmax(out, dim=-1) # [B, num_query_rotation]
               best_prob = probabilities.argmax(dim=1).detach().cpu().numpy() # [B]
               best_rotation = query_rotations_org[best_prob].to(vision_description.device)   # [B, 3, 3]
               max_prob = probabilities.max().item()
               return query_rotations_org, out, best_rotation, max_prob
          else:
               return query_rotations_org, out, None, None
          
     def positional_encoding(self, query_rotations):
          """
          Handles the positional encoding.

          Args:
          query_rotations: tensor of shape [N, len_rotation] or
               [bs, N, len_rotation].

          Returns:
          Tensor of shape [N, len_query] or [bs, N, len_query].
          """
          if self.frequencies.shape[0] == 0:
               return query_rotations

          # Define encoding function that applies sine and cosine transformations
          def _enc(freq):
               return torch.cat([
                    torch.sin(query_rotations * freq),
                    torch.cos(query_rotations * freq)
               ], dim=-1)
          
          # Apply encoding across frequencies
          query_rotations_encoded = torch.stack([_enc(freq) for freq in self.frequencies], dim=0)
          
          # Adjust shape based on input dimensions
          if query_rotations.dim() == 3:
               # Input shape [bs, N, len_rotation]
               query_rotations_encoded = query_rotations_encoded.permute(1, 2, 0, 3)  # Shape [bs, N, num_freq, len_rotation]
               query_rotations_encoded = query_rotations_encoded.reshape(query_rotations.shape[0], query_rotations.shape[1], -1)
          else:
               # Input shape [N, len_rotation]
               query_rotations_encoded = query_rotations_encoded.permute(1, 0, 2)  # Shape [N, num_freq, len_rotation]
               query_rotations_encoded = query_rotations_encoded.reshape(query_rotations.shape[0], -1)
          
          return query_rotations_encoded

     def get_closest_available_grid(self, number_queries=None):
          if not number_queries:
               number_queries = self.number_eval_queries
          # HEALPix-SO(3) is defined only on 72 * 8^N points; we find the closest
          # valid grid size (in log space) to the requested size.
          # The largest grid size we consider has 19M points.
          grid_sizes = 72*8**np.arange(7)
          size = grid_sizes[
               np.argmin(np.abs(np.log(number_queries) - np.log(grid_sizes)))
               ]
          if self.grids.get(size) is not None:
               return self.grids[size]
          else:
               grid_created = False
               if not grid_created:
                    self.grids[size] = np.float32(generate_healpix_grid(size=size))
               self.grids[size] = torch.tensor(self.grids[size], dtype=torch.float32)
               return self.grids[size]
          
     def generate_queries(self, number_queries, mode='random'):
          if mode == 'random':
               return self.generate_queries_random(number_queries)
          elif mode == 'grid':
               return self.get_closest_available_grid(number_queries)

     def generate_queries_random(self, number_queries):
          """
          Generates rotation matrices from SO(3) uniformly at random.

          Args:
               number_queries: Number of queries.

          Returns:
               A tensor of shape [number_queries, 3, 3].
          """
          # Generate random rotation matrices using scipy
          random_rotations = R.random(number_queries).as_matrix()  # Shape: (number_queries, 3, 3)
          random_rotations = torch.tensor(random_rotations, dtype=torch.float32)
          return random_rotations


def generate_healpix_grid(recursion_level=None, size=None):
     """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

     Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
     along the 'tilt' direction 6*2**recursion_level times over 2pi.

     Args:
     recursion_level: An integer which determines the level of resolution of the
          grid. The final number of points will be 72*8**recursion_level. A
          recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
          for evaluation.
     size: A number of rotations to be included in the grid. The nearest grid
          size in log space is returned.

     Returns:
     (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
     """
     import healpy as hp 
     assert not (recursion_level is None and size is None)
     if size:
          recursion_level = max(int(np.round(np.log(size / 72.) / np.log(8.))), 0)
     number_per_side = 2**recursion_level
     number_pix = hp.nside2npix(number_per_side)
     s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
     s2_points = np.stack([*s2_points], 1)

     # Take these points on the sphere and
     azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
     tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)
     polars = np.arccos(s2_points[:, 2])
     grid_rots_mats = []

     for tilt in tilts:
          # Build up the rotations from Euler angles, zyz format
          rot_mats = R.from_euler('zyz', np.stack([azimuths,
                                                  np.zeros(number_pix),
                                                  np.zeros(number_pix)], 1))
          rot_mats = rot_mats.as_matrix() @ R.from_euler('zyz', np.stack([np.zeros(number_pix),
                                                                           np.zeros(number_pix),
                                                                           polars], 1)).as_matrix()
          rot_mats = rot_mats @ R.from_euler('zyz', [[tilt, 0., 0.]]).as_matrix().reshape(1, 3, 3)
          grid_rots_mats.append(rot_mats)

     grid_rots_mats = np.concatenate(grid_rots_mats, 0)
     grid_rots_mats = torch.tensor(grid_rots_mats, dtype=torch.float32)
     return grid_rots_mats

