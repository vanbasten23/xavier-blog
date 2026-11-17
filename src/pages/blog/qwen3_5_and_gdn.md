---
layout: ../../layouts/Layout.astro
title: "Qwen 3.5 and GDN explained"
description: "An intuitive explanation of the qwen 3.5 and GDN Pallas kernel"
pubDate: 2026-11-07
---

# Understand Qwen 3.5 and GDN (Gated DeltaNet)

- checkpoint [config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json?utm_source=chatgpt.com)
- [configuration_qwen3_5_moe.py](https://github.com/huggingface/transformers/blob/e20e0787c9863f4b78781e603adfe6648415a854/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py)
- [modeling_qwen3_5_moe.py](https://github.com/huggingface/transformers/blob/e20e0787c9863f4b78781e603adfe6648415a854/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)

# Decoder layer

## Overall forward computation

Overall forward computation: [Qwen3_5MoeDecoderLayer](https://github.com/huggingface/transformers/blob/e20e0787c9863f4b78781e603adfe6648415a854/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L934-L992)

```
class Qwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states
```


## Q/K/V, Conv1D, recurrent state, gates

[Qwen3_5MoeGatedDeltaNet](https://github.com/huggingface/transformers/blob/e20e0787c9863f4b78781e603adfe6648415a854/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L491-L654)

Qwen3_5MoeGatedDeltaNet.forward with annotations. There are 5 stages:
1. Linear projections (Generate Q, K, V, output gate z, update gate b, decay gate a)
2. Causal Conv1D + SiLU (Mix neighboring tokens in QKV)
3. Gated Delta Rule (Read/write recurrent state with Q, K, V, β, g)
4. Gated RMSNorm (Normalize the GDN output and apply gate z)
5. Output projection ([B, T, 4096])
```
@force_accelerate_hooks("conv1d")
def forward(
    self,
    hidden_states: torch.Tensor,     # [B,T,H] = [B,T,4096]
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,  # typically [B,T] for GDN padding mask
    **kwargs: Unpack[TransformersKwargs],
):
    # Part A — Input and cache
    # Zero out padded tokens, if any; the shape is unchanged.
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)  # [B,T,4096]

    batch_size, seq_len, _ = hidden_states.shape  # B, T, H=4096

    # Load TWO kinds of persistent per-layer state, when available.
    if cache_params is not None:
        conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
        # conv_state: [B,C,W-1] = [B,12288,3], if already initialized
        recurrent_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
        # recurrent_state: [B,Nv,Dk,Dv] = [B,64,128,128], if initialized
    else:
        conv_state, recurrent_state = None, None

    # Part B — Four linear projections
    # Four independent projections from the same hidden_states.
    mixed_qkv = self.in_proj_qkv(hidden_states)  # [B,T,2*Kdim+Vdim] = [B,T,12288]
    mixed_qkv = mixed_qkv.transpose(1, 2)       # [B,12288,T] (Conv1D channel-first)

    z = self.in_proj_z(hidden_states)            # [B,T,Vdim] = [B,T,8192], produces the output gate.
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [B,T,Nv,Dv]=[B,T,64,128]

    b = self.in_proj_b(hidden_states)            # [B,T,Nv] = [B,T,64] (write gate logits). Produces the delta-rule write gate.
    a = self.in_proj_a(hidden_states)            # [B,T,Nv] = [B,T,64] (decay gate logits). Produces the decay/forget gate.

    # Part C — Causal Conv1D
    # DECODE CONV PATH: one token, existing conv cache, not recording past.
    if conv_state is not None and seq_len == 1 and not cache_params.is_recording_past(self.layer_idx):
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,                      # [B,12288,1]
            conv_state,                     # [B,12288,3]; mutated to hold newest history
            self.conv1d.weight.squeeze(1), # [C,W] = [12288,4]; depthwise weights
            self.conv1d.bias,              # None (bias=False)
            self.activation,               # 'silu'
        )  # [B,12288,1]
    else:
        # PREFILL / GENERAL CONV PATH: cache update may prepend W-1 past tokens.
        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(
                mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
            )  # typically [B,12288,T+3] when prior history is prepended;
               # confirm exact behavior in the cache implementation.

        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,                      # [B,12288,L], L=T or possibly T+3
            self.conv1d.weight.squeeze(1), # [12288,4]
            self.conv1d.bias,              # None
            activation=self.activation,    # 'silu'
            **kwargs,
        )  # [B,12288,L] for causal Conv1D helper

        # Keep only the T outputs corresponding to current tokens.
        if cache_params is not None:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]  # [B,12288,T]

    # Part D — Split Q, K, V
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [B,T,12288]
    query, key, value = torch.split(
        mixed_qkv,               # [B,T,12288]
        [self.key_dim,           # 2048
         self.key_dim,           # 2048
         self.value_dim],        # 8192
        dim=-1,
    )  # query [B,T,2048], key [B,T,2048], value [B,T,8192]

    # Notice how this differs from standard GQA: Qwen's GDN uses fewer 
    # Q/K heads than value heads. The GDN helper expands the Q/K head
    # mapping logically to match the 64 V heads.
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim) # [B,T,16,128], 16 is the num_q_heads
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)     # [B,T,16,128], 16 is the num_k_heads
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim) # [B,T,64,128], 64 is the num_v_heads

    # Part E — Compute update and decay gates.
    # - `beta` determines how strongly the state is updated.
    # - `g` determines forgetting.
    # Gate controlling write/correction amount for each V head.
    beta = b.sigmoid()  # [B,T,64], values in (0,1)

    # g is LOG decay; exp(g) is a per-head forget multiplier in (0,1].
    # A_log [64], dt_bias [64], a [B,T,64] -> broadcast result [B,T,64].
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B,T,64], float32

    # Part F — Execute the Gated Delta Rule
    # DECODE GDN: recurrent algorithm when past state exists and T=1.
    if recurrent_state is not None and seq_len == 1:
        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query,             # [B,1,16,128]
            key,               # [B,1,16,128]
            value,             # [B,1,64,128]
            g=g,               # [B,1,64]
            beta=beta,         # [B,1,64]
            initial_state=recurrent_state,  # [B,64,128,128]
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            **kwargs,
        )  # core_attn_out [B,1,64,128]; final state [B,64,128,128] (or None)
    else:
        # PREFILL GDN: chunk-wise algorithm, optionally with an initial state.
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query,             # [B,T,16,128]
            key,               # [B,T,16,128]
            value,             # [B,T,64,128]
            g=g,               # [B,T,64]
            beta=beta,         # [B,T,64]
            initial_state=recurrent_state,  # [B,64,128,128] or None
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            **kwargs,
        )  # core_attn_out [B,T,64,128]; final state [B,64,128,128] (or None)

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        # Persist recurrent state for subsequent decode calls: [B,64,128,128]

    # Part G — Normalize, gate, and project
    # Gated RMSNorm is applied independently to each (token, V-head) vector.
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)  # [B*T*64,128]
    z = z.reshape(-1, self.head_v_dim)                          # [B*T*64,128]
    core_attn_out = self.norm(core_attn_out, z)                # [B*T*64,128]
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1) # [B,T,8192]

    output = self.out_proj(core_attn_out)  # [B,T,4096]
    return output                          # [B,T,4096]
```

Note:
1. Conv1D handles local context. Its cache only remembers the previous three inputs per channel.
2. GDN handles accumulated context. Its recurrent state has constant size regardless of sequence length.
3. Prefill and decode implement the same mathematics differently. Prefill exploits chunked matrix operations; decode updates the recurrent state token by token.


## torch_recurrent_gated_delta_rule

It is the decode algorithm. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L423-L490)

### Math

## 1.1 What is the recurrent state?

For each value head, Gated DeltaNet maintains a matrix:

$$
S_t \in \mathbb{R}^{D_k \times D_v}
$$

Where:
- $D_k$: Dimension of each key/query head.
- $D_v$: Dimension of each value head.

For Qwen 3.5-397B-A17B:
- $D_k = 128$
- $D_v = 128$
- Number of value heads = 64

Therefore, the recurrent state for one sequence has shape:

$$
S_t \in \mathbb{R}^{64 \times 128 \times 128}
$$

Think of this matrix as **memory that maps keys to values**.

Given a key $k$, the expression

$$
\hat{v} = S^\top k
$$

asks: Based on the memory accumulated so far, what value does this key predict?

Unlike a conventional KV cache, this memory has a fixed size independent of sequence length.

## 1.2 The Five Equations

For one token and one head, let:

- $q_t, k_t \in \mathbb{R}^{D_k}$
- $v_t \in \mathbb{R}^{D_v}$
- $\beta_t \in (0,1)$: Update gate
- $g_t \leq 0$: Log-decay gate

### Step 1: Forget old memory

$$
\widetilde{S}_t = e^{g_t} S_{t-1}
$$

Scale down the previous memory. A decay factor near 1 retains most of it; near 0 forgets most of it.

### Step 2: Predict from memory

$$
\hat{v}_t = \widetilde{S}_t^\top k_t
$$

Look up what the existing memory associates with the current key.

### Step 3: Compute prediction error

$$
\delta_t = \beta_t(v_t - \hat{v}_t)
$$

Write only the correction, scaled by the update gate.

### Step 4: Update memory

$$
S_t = \widetilde{S}_t + k_t \delta_t^\top
$$

The outer product creates a rank-one matrix update.

### Step 5: Read the output

$$
o_t = S_t^\top q_t
$$

Query the newly updated memory to produce the attention output.

**Note:** The implementation optionally L2-normalizes Q and K, and always scales Q by $1/\sqrt{D_k}$. The equations above assume these operations have already been applied.

## 1.3 Why the Delta Rule Is Different from Ordinary Linear Attention

Ordinary additive linear attention can be written as:

$$
S_t = S_{t-1} + k_t v_t^\top
$$

It simply adds a new key-value association to memory.

The gated delta rule instead performs:

$$
S_t = \widetilde{S}_t +
\beta_t k_t(v_t - \widetilde{S}_t^\top k_t)^\top
$$

The key difference is the **prediction-error term**.

If memory already predicts the correct value, the update is small. If its prediction is wrong, the update corrects it.

This resembles online learning: use the current key as an input, predict its target value, then update memory based on the prediction error.

### Connection to Gradient Descent

Consider the loss:

$$
L(S) = \frac{1}{2}\|S^\top k - v\|^2
$$

Its gradient is:

$$
\nabla_S L = k(S^\top k-v)^\top
$$

A gradient descent step gives:

$$
S_{\text{new}} = S - \beta \nabla_S L
$$

Substituting the gradient:

$$
S_{\text{new}} = S + \beta k(v-S^\top k)^\top
$$

This is exactly the delta-rule update when applied to the decayed state $\widetilde{S}_t$.

**Key insight:** The delta rule updates memory based on prediction error, rather than blindly accumulating key-value associations.


## torch_chunk_gated_delta_rule

It is the prefill algorithm. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L281-L422)

## Qwen3_5MoeSparseMoeBlock

The routing and expert computation. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L832-L912)

## Qwen3_5MoeTextModel.forward

The layer loop, masks, cache management. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1310-L1380)
