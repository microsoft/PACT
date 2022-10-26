# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Implement modality transform utilities for mushr dataset."""

import torch


def create_batch_dict(states, actions):
    """Create a batch dictionary from states, actions, and timesteps."""
    batch = {}
    batch["state"] = states
    batch["action"] = actions
    return batch


def insert_in_tensor_and_shift(tensor, val, pos, shift):
    """Insert val in tensor's last position and shift the rest of the tensor to the left."""
    if shift:
        tensor = torch.roll(tensor, shifts=-1, dims=1)
    tensor[0, pos] = val
    return tensor


def norm_angle(angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return 2.0 * (angle - act_min) / (act_max - act_min) - 1.0


def denorm_angle(norm_angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return (norm_angle + 1.0) / 2 * (act_max - act_min) + act_min


# def get_embeddings(states, actions, timesteps, is_state_embed, state_tokenizer_type):
#     B = states.shape[0]
#     if is_state_embed:
#         state_embeddings = states
#     else:
#         if state_tokenizer_type == "resnet18":
#             state_embeddings = self.state_tokenizer(
#                 states.reshape(-1, 1, 200, 200).type(torch.float32).contiguous()
#             )  # (batch * block_size, n_embd)
#             state_embeddings = state_embeddings.reshape(
#                 states.shape[0], states.shape[1], self.args.n_embd
#             )  # (batch, block_size, n_embd)
#         elif state_tokenizer_type == "conv2D":
#             state_embeddings = self.state_tokenizer(
#                 states.reshape(-1, 1, 244, 244).type(torch.float32).contiguous()
#             )  # (batch * block_size, n_embd)
#             state_embeddings = state_embeddings.reshape(
#                 states.shape[0], states.shape[1], self.args.n_embd
#             )  # (batch, block_size, n_embd)
#         elif state_tokenizer_type == "pointnet":
#             if self.args.use_mat_embed:
#                 state_embeddings = self.state_tokenizer(states.reshape(states.shape[0], states.shape[1], -1).type(torch.float32).contiguous())
#             else:
#                 state_embeddings = self.state_tokenizer(
#                     states.transpose(-2, -1).reshape(-1, 2, 720).type(torch.float32).contiguous()
#                 )  # (batch * block_size, n_embd)

#                 state_embeddings = state_embeddings.reshape(
#                     states.shape[0], states.shape[1], self.args.n_embd
#                 )  # (batch, block_size, n_embd)

#         # TODO(yue)
#         elif state_tokenizer_type == "oned":
#             # print("DEBUG state.shape", states.shape, self.state_tokenizer)
#             state_embeddings = self.state_tokenizer(
#                 states.type(torch.float32).contiguous()
#             )
#             state_embeddings = state_embeddings.reshape(
#                 states.shape[0], states.shape[1], self.args.n_embd
#             )
#             # print("DEBUG state.shape", state_embeddings.shape)

#         else:
#             print("Not supported!")

#     if actions is not None:
#         actions_reshaped = rearrange(actions, "b n c -> (b n) c")
#         action_embeddings = self.action_tokenizer(actions_reshaped)
#         action_embeddings = rearrange(
#             action_embeddings, "(b n) c -> b n c", b=actions.shape[0]
#         )  # (batch, block_size, n_embd)
#     else:
#         raise NotImplementedError()

#     if self.args.arch == "simp":
#         position_embeddings_global, position_embeddings_local = torch.zeros_like(action_embeddings), torch.zeros_like(action_embeddings)
#     else:
#         # calculate global positional embedding (unique for each token)
#         timesteps_global = torch.arange(
#             timesteps.min(),
#             timesteps.min() + 2 * timesteps.shape[1],
#             device=self.device,
#         ).view(1, -1)
#         position_embeddings_global = self.embed_timestep_global(timesteps_global)
#         position_embeddings_global = position_embeddings_global.repeat(B, 1, 1)

#         # calculate local positional embedding (unique for each state-action pair)
#         position_embeddings_local = self.embed_timestep_local(timesteps)
#         position_embeddings_local = torch.repeat_interleave(
#             position_embeddings_local, 2, dim=1
#         )
#         # position_embeddings_local = position_embeddings_local.repeat(B,1,1)

#     return state_embeddings, action_embeddings, position_embeddings_global, position_embeddings_local


# class InputTransform:
#     def __init__(self, modal_config: Dict[str, Any]) -> None:
#         self.inputs = {}
#         self.feature_map = {}
#         for key, modal in modal_config.items():
#             modal["kwargs"]["feature"] = key
#             if modal["kwargs"].get("feature_in_batch", None) is not None:
#                 # this feature is not being sent to model
#                 self.feature_map[key] = modal["kwargs"]["feature_in_batch"]
#             self.inputs[key] = instantiate_modal_class(modal)
#         return

#     def pre_transform(self, x: Dict[str, np.ndarray]):
#         """apply pre-transform on frame data.
#         Args:
#             x (Dict[str, np.ndarray]): each value of the dict is one single frame data
#         """
#         res = {}
#         for key, value in x.items():
#             # static methods can be called from instances:
#             # https://docs.python.org/3/library/functions.html#staticmethod
#             res[key] = self.inputs[key].pre_transform(value)
#         return res

#     def transform(
#         self, transform_type: str, x: Dict[str, np.ndarray]
#     ) -> Dict[str, torch.Tensor]:

#         res = {}
#         for _, input in self.inputs.items():
#             key = input.feature
#             if key not in self.feature_map:
#                 # this input is not being sent to model in batch
#                 continue

#             res[self.feature_map[key]] = self.inputs[key].transform(
#                 transform_type, x[key]
#             )
#         return res
