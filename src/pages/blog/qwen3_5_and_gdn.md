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

#### 1.1 What is the recurrent state?

For each value head, Gated DeltaNet maintains a matrix:

$$
S_t \in \mathbb{R}^{D_k \times D_v}
$$

Where:
- $D_k$: Dimension of each key/query head.
- $D_v$: Dimension of each value head.

Therefore, the recurrent state for one sequence has shape:

$$
S_t \in = \mathbb{R}^{64 \times 128 \times 128}
$$

where 64 is the num_value_heads. Think of this matrix as **memory that maps keys to values**.

Given a key $k$, the expression

$$
\hat{v} = S^\top k
$$

asks: Based on the memory accumulated so far, what value does this key predict?

Unlike a conventional KV cache, this memory has a fixed size independent of sequence length.

#### 1.2 The Five Equations

For one token and one head, let:

- $q_t, k_t \in \mathbb{R}^{D_k}$
- $v_t \in \mathbb{R}^{D_v}$
- $\beta_t \in (0,1)$: Update gate
- $g_t \leq 0$: Log-decay gate

##### Step 1: Forget old memory

<div>
$$
\widetilde{S}_t = e^{g_t} S_{t-1}
$$
</div>

Scale down the previous memory. A decay factor near 1 retains most of it; near 0 forgets most of it.

##### Step 2: Predict from memory

$$
\hat{v}_t = \widetilde{S}_t^\top k_t
$$

Look up what the existing memory associates with the current key.

##### Step 3: Compute prediction error

$$
\delta_t = \beta_t(v_t - \hat{v}_t)
$$

Write only the correction, scaled by the update gate.

##### Step 4: Update memory

$$
S_t = \widetilde{S}_t + k_t \delta_t^\top
$$

The outer product creates a rank-one matrix update.

##### Step 5: Read the output

$$
o_t = S_t^\top q_t
$$

Query the newly updated memory to produce the attention output.

**Note:** The implementation optionally L2-normalizes Q and K, and always scales Q by $1/\sqrt{D_k}$. The equations above assume these operations have already been applied.

#### 1.3 Why the Delta Rule Is Different from Ordinary Linear Attention

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

### [Code](https://github.com/huggingface/transformers/blob/fc00b5b3c9f9306c55908bb546fa3b033a4bfb1d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L423-L490)

```
@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")
def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,                   # [B, T, Nk, Dk] = [B, T, 16, 128]
    key: torch.Tensor,                     # [B, T, Nk, Dk] = [B, T, 16, 128]
    value: torch.Tensor,                   # [B, T, Nv, Dv] = [B, T, 64, 128]
    g: torch.Tensor,                       # [B, T, Nv]     = [B, T, 64] (log decay <= 0)
    beta: torch.Tensor,                    # [B, T, Nv]     = [B, T, 64] (write gate)
    initial_state: torch.Tensor | None = None, # [B, Nv, Dk, Dv] = [B, 64, 128, 128]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the gated delta rule, by iterating over each token in the sequence dimension.
    Same args and return value as torch_chunk_gated_delta_rule, except for `chunk_size` because the sequence dim is not
    chunked."""
    initial_dtype = query.dtype               # Remember original dtype (e.g. BF16)
    batch_size, sequence_length, num_k_heads, k_head_dim = key.shape  # B, T, Nk=16, Dk=128
    num_v_heads, v_head_dim = value.shape[-2:]                      # Nv=64, Dv=128
    decay = g                                 # [B, T, Nv] log-decay, NOT exp(g) yet

    # One Q/K head is shared by 64/16=4 value heads in this checkpoint.
    if num_k_heads != num_v_heads:
        query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)  # [B, T, Nv, Dk]
        key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)      # [B, T, Nv, Dk]

    # Move head before sequence; convert to FP32 for stable accumulation.
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # query/key: [B, Nv, T, Dk]; value: [B, Nv, T, Dv]
    # beta/decay: [B, Nv, T]; all five tensors are FP32.
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)  # [B, Nv, T, Dk], approximately unit-length
        key = l2norm(key, dim=-1, eps=1e-6)      # [B, Nv, T, Dk]

    query = query / (query.shape[-1] ** 0.5)     # [B, Nv, T, Dk], /sqrt(128)

    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        # [B, Nv, Dk, Dv] = [B, 64, 128, 128]
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)  # [B, Nv, Dk, Dv], FP32
    core_attn_out = torch.zeros_like(value)             # [B, Nv, T, Dv], FP32

    for i in range(sequence_length):                    # Usually T=1 for decode
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # q_t, k_t: [B, Nv, Dk]; v_t: [B, Nv, Dv]

        # Step 1: Forget old memory
        decay_t = decay[:, :, i].exp()[..., None, None]  # [B, Nv, 1, 1]
        last_recurrent_state = last_recurrent_state * decay_t  # [B, Nv, Dk, Dv]
        # S_bar = exp(g_t) * S_prev

        beta_t = beta[:, :, i].unsqueeze(-1)             # [B, Nv, 1]
        # Step 2: Predict from memory
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        # [B,Nv,Dk,Dv] * [B,Nv,Dk,1] -> sum over Dk -> [B,Nv,Dv]
        # kv_mem = S_bar^T @ k_t

        # Step 3: Compute prediction error
        delta = (v_t - kv_mem) * beta_t                  # [B, Nv, Dv]
        # delta = beta_t * (v_t - kv_mem)

        # Step 4: Update memory
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # S = S_bar + outer(k_t, delta): [B,Nv,Dk,1] * [B,Nv,1,Dv]
        # resulting state: [B, Nv, Dk, Dv]

        # Step 5: Read the output
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        # [B,Nv,Dk,Dv] * [B,Nv,Dk,1] -> sum Dk -> [B,Nv,Dv]
        # o_t = S_t^T @ q_t (after update, not before)

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # [B, Nv, Dk, Dv] FP32, or None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # [B, Nv, T, Dv] -> [B, T, Nv, Dv]; output dtype restored
    return core_attn_out, last_recurrent_state
    # return: ([B, T, 64, 128], [B, 64, 128, 128] or None)
```


## torch_chunk_gated_delta_rule

It is the prefill algorithm. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L281-L422)

Key idea:
- The recurrent implementation processes tokens one by one.
- The chunked implementation processes groups of tokens (64 by default), using matrix multiplications.
- Both the recurrent implementation and the chunked implementation are mathematically equivalent.
- The tricky part is understanding how it replaces sequential updates with a triangular matrix solve.

### 1. The math

#### 1.1 Start with the recurrent Gated Delta Rule

For a single token and a single value head:

$$
\begin{aligned}
\widetilde S_t &= \alpha_t S_{t-1},\\
\hat v_t &= \widetilde S_t^\top k_t,\\
\delta_t &= \beta_t(v_t-\hat v_t),\\
S_t &= \widetilde S_t+k_t\delta_t^\top,\\
o_t &= S_t^\top q_t
\end{aligned}
$$

where:

- $q_t,k_t\in\mathbb R^{D_k}$ are query and key vectors.
- $v_t,\delta_t,o_t\in\mathbb R^{D_v}$ are value, correction, and output vectors.
- $S_t\in\mathbb R^{D_k\times D_v}$ is the recurrent state.
- $\beta_t\in(0,1)$ is the update gate.
- $g_t\le0$ is the log-decay and $\alpha_t=e^{g_t}$ is the decay factor.

The recurrence is sequential because $S_t$ depends on $S_{t-1}$.

#### 1.2 Split the sequence into chunks

Let chunk size be $C$ (default: 64).

For one chunk, stack the token vectors into:

$$
Q,K\in\mathbb R^{C\times D_k},\qquad
V\in\mathbb R^{C\times D_v}.
$$

Let $S_{\mathrm{in}}$ be the state before the chunk and $S_{\mathrm{out}}$ the state after it.

For example, a 128-token prefill becomes two chunks of 64 tokens. Each chunk performs parallel matrix operations, while the second chunk receives the state from the first.

<img
  src="/images/qwen3_5_and_gdn/devide_tokens_into_chunks.png"
  width="700"
/>

#### 1.3 Expand the recurrent state

Define cumulative log-decay inside the chunk:

$$
G_i=\sum_{r=1}^{i}g_r.
$$

The decay from token $j$ to token $i$ is:

$$
d_{ij}=\exp(G_i-G_j),\qquad j\le i.
$$

Expanding the Gated Delta Rule recurrence

$$\boxed{S_t = e^{g_t} S_{t-1} + k_t \delta_t^\top}$$

yields:

$$
S_i=e^{G_i}S_{\mathrm{in}}
+\sum_{j=1}^{i}d_{ij}k_j\delta_j^\top.
$$

Recall the original recurrent delta rule:

$$\delta_i = \beta_i(v_i - \widetilde{S}_i^\top k_i)$$

where

$$
\widetilde{S}\_i = e^{g_i} S\_{i-1}
$$

Substituting into the prediction-error update gives:

$$
\boxed{
\delta_i=
\beta_i v_i
-\beta_i e^{G_i}S_{\mathrm{in}}^\top k_i
-\sum_{j<i}\beta_i d_{ij}(k_i^\top k_j)\delta_j
}
$$

This still contains dependencies on earlier corrections, but they are **linear** dependencies.

#### 1.4 Turn the dependencies into a triangular solve

Define the strictly lower-triangular matrix $L\in\mathbb R^{C\times C}$:

$$
L_{ij}=
\begin{cases}
\beta_i(k_i^\top k_j)e^{G_i-G_j},&j<i,\\\\
0,&j\ge i.
\end{cases}
$$

where $k_i\in\mathbb R^{D_k}$ is the key vector, $\beta_t\in(0,1)$ is the update gate.

Define $B=\operatorname{diag}(\beta_1,\dots,\beta_C)$.

Let $\Delta\in\mathbb R^{C\times D_v}$ contain the correction vectors as rows.

Let $Z\in\mathbb R^{C\times D_k}$ have row $i$ equal to $\beta_i e^{G_i}k_i^\top$.

Then (TODO: how do we get the formula below?):

$$
(I+L)\Delta=BV-ZS_{\mathrm{in}}.
$$

Therefore:

$$
\Delta=(I+L)^{-1}BV-(I+L)^{-1}ZS_{\mathrm{in}}.
$$

Define:

$$
U=(I+L)^{-1}BV,\qquad W=(I+L)^{-1}Z.
$$

So the key identity is:

$$
\boxed{\Delta=U-WS_{\mathrm{in}}}
$$

That's what these three lines are doing:

```
new_values = torch.linalg.solve_triangular(
    ut_system, v_beta,
    upper=False, unitriangular=True,
)

k_cumdecay = torch.linalg.solve_triangular(
    ut_system, decayed_k_beta,
    upper=False, unitriangular=True,
)

v_new = new_values[:, :, i] - (
    k_cumdecay[:, :, i] @ last_recurrent_state
)
```

`v_new` is really the matrix of corrected value updates, $\Delta$, not simply the original value tensor.

#### 1.5 Compute all outputs in a chunk

Once we have $\Delta$, the output for token $i$ is:

$$
o_i=e^{G_i}q_i^\top S_{\mathrm{in}}
+\sum_{j\le i}e^{G_i-G_j}(q_i^\top k_j)\delta_j.
$$

This separates natually into two pieces.

**Inter-chunk attention**: The contribution from the state that existed before this chunk.

$$
O_{\mathrm{inter}}=Q_{\mathrm{scaled}}S_{\mathrm{in}}
$$

where
$$
(Q_{\mathrm{scaled}})_i=e^{G_i}q_i^\top.
$$

**Intra-chunk attention**: The contribution from tokens in the current chunk.

Define the causal matrix A:

$$
A_{ij}=
\begin{cases}
(q_i^\top k_j)e^{G_i-G_j},&j\le i,\\
0,&j>i.
\end{cases}
$$

Then:

$$
O_{\mathrm{intra}}=A\Delta
$$

And finally:

$$
\boxed{O_{\mathrm{chunk}}=Q_{\mathrm{scaled}}S_{\mathrm{in}}+A\Delta}
$$

The diagonal of $A$ is included: a token can read its own newly updated state.

Notice that A is lower triangular. It includes the diagonal because each token is allowed to read its own state update.

#### 1.6 Compute the final state of the chunk

We also need the recurrent state to process the next chunk.

Using the expanded recurrence:

$$
S_{\mathrm{out}}=e^{G_C}S_{\mathrm{in}}
+\sum_{j=1}^{C}e^{G_C-G_j}k_j\delta_j^\top.
$$

Define a scaled key matrix with rows:

$$
(K_{\mathrm{scaled}})_j=e^{G_C-G_j}k_j^\top.
$$

Then:

$$
\boxed{S_{\mathrm{out}}=e^{G_C}S_{\mathrm{in}}+K_{\mathrm{scaled}}^\top\Delta}
$$

This is the state passed to the next chunk.

```
last_recurrent_state = (
    last_recurrent_state * chunk_decay[:, :, i]
    + key[:, :, i].transpose(-1, -2) @ v_new
)
```

### The code.

On a high level, it does a few steps:

- A. Prepare heads, precision, and padding
- B. Build chunks
- C. Build the causal decay matrix
- D. Construct the two key matrices
- E. Solve the triangular system
- F. Initialize the incoming recurrent state
- G. Prepare the three decay-scaled tensors
- H. Sequential scan over chunks
- I. Restore the original output layout

```
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,               # [B, T, Nk, Dk]
    key: torch.Tensor,                 # [B, T, Nk, Dk]
    value: torch.Tensor,               # [B, T, Nv, Dv]
    g: torch.Tensor,                   # [B, T, Nv] log decay, g<=0
    beta: torch.Tensor,                # [B, T, Nv] update gate
    chunk_size: int = 64,             # C
    initial_state: torch.Tensor | None = None,  # [B, Nv, Dk, Dv] or None
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes the gated delta rule, by chunking along the sequence dimension.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]. num_v_heads can be equal
            to num_k_heads, same for v_head_dim and k_head_dim.
        g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
            multiplied by exp(g) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_v_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
        use_qk_l2norm_in_kernel: If this flag is set to True, query and key vectors are L2-normalized.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
    """
    # Step A: Prepare heads, precision, and padding
    initial_dtype = query.dtype
    batch_size, sequence_length, num_k_heads, k_head_dim = key.shape  # B,T,16,128
    num_v_heads, v_head_dim = value.shape[-2:]  # 64,128
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)  # [B,Nv,Dk,Dv]
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # [B,Nv,P,Dv]
    decay = g  # [B,T,Nv], log space; name required by FLA API

    # Since there are 16 Q/K heads and 64 V heads, each Q/K head is repeated four times.
    # Expand Nk Q/K heads to Nv logical heads: Nv/Nk copies per Q/K head.
    if num_k_heads != num_v_heads:
        query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)  # [B,T,Nv,Dv]
        key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)      # [B,T,Nv,Dv]

    # Permute token/head axes and cast to FP32 (including beta and decay).
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # query,key,value: [B,Nv,T,Dv]; beta,decay: [B,Nv,T]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)  # [B,Nv,T,Dv]
        key = l2norm(key, dim=-1, eps=1e-6)      # [B,Nv,T,Dv]
    scaling = query.shape[-1] ** -0.5            # scalar = 1 / sqrt(Dk)
    query = query * scaling                      # [B,Nv,T,Dv]

    # Zero-pad only the token dimension to an integer number of chunks.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size  # P - T
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    # query,key: [B,Nv,P,Dk]; value: [B,Nv,P,Dv]
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    # beta,decay: [B,Nv,P]; padded beta=0, padded log-decay=0

    total_sequence_length = sequence_length + pad_size  # P
    num_chunks = total_sequence_length // chunk_size    # N

    # Step B: Build chunks
    v_beta = value * beta.unsqueeze(-1)  # [B,Nv,P,Dv]; row i = beta_i * v_i
    k_beta = key * beta.unsqueeze(-1)    # [B,Nv,P,Dv]; row i = beta_i * k_i
    # Both v_beta and k_beta are needed in the triangular system.

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    # query,key,k_beta: [B,Nv,N,C,Dk]; v_beta: [B,Nv,N,C,Dv]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    # decay: [B,Nv,N,C]
    # where C is the chunk_size.
    # Now the tensor has a separate dimension for chunk index and token index inside the chunk.

    # Step C: Build the causal decay matrix
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    # [C,C], True when column j > row i (future tokens)

    cum_decay = decay.cumsum(dim=3)  # [B,Nv,N,C], G_i = sum_{r<=i} g_r

    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    # [B,Nv,N,C,C]; (i,j) = G_i - G_j
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    # Future positions j>i become -inf before exp.
    pairwise_decay = pairwise_decay.exp()
    # [B,Nv,N,C,C]; lower-triangular exp(G_i-G_j), upper=0
    # Entry (i, j) becomes: exp(G_i - G_j) for j <= i and zero otherwise.


    # Step D: Construct the two key matrices
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    # [B,Nv,N,C,C]; L_ij = beta_i*(k_i dot k_j)*exp(G_i-G_j), j<=i
    # solve_triangular(..., unitriangular=True) treats the diagonal as exactly 1.
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    # [B,Nv,N,C,C]; A_ij = (q_i dot k_j)*exp(G_i-G_j), j<=i
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    # [B,Nv,N,C,Dk]; row i = beta_i*exp(G_i)*k_i
    # These three statements are the heart of the first phase.
    # - `ut_system` means: Coefficients for solving the dependencies between delta updates.
    # - `intra_chunk_attn` means: Causal QK interaction matrix inside the chunk.
    # - `decayed_k_beta` means: Coefficients describing how the incoming state affects the delta updates.


    # Step E: Solve the triangular system
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        # [B,Nv,N,C,Dv]; U = (I+strict_lower(L))^{-1} (beta V)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
        # [B,Nv,N,C,Dk]; Z = (I+strict_lower(L))^{-1} (beta*exp(G)*K)
    else:
        # The else branch manually builds the triangular inverse using forward
        # substitution for export environments where the solver is unsupported.
        # It computes the same algebraic transformation as the `if` branch.
        ut_system = -ut_system.tril(-1)  # [B,Nv,N,C,C], strictly lower, negated
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()  # [B,Nv,N,i]
            sub = ut_system[..., :i, :i].clone()  # [B,Nv,N,i,i]
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            # Construct the strictly lower portion of inverse(I+L) via forward substitution.
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        # [B,Nv,N,C,C] now represents inverse(I+strict_lower(L)).
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta
        # new_values: [B,Nv,N,C,Dv]; k_cumdecay: [B,Nv,N,C,Dk]
    # At this point, the expensive within-chunk dependency analysis has already
    # been performed.


    # Step F: Initialize the incoming recurrent state
    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    # The recurrent state is either initialized to zeros or loaded from the 
    # caller's previous state.
    # last_recurrent_state: [B,Nv,Dk,Dv], FP32
    core_attn_out = torch.zeros_like(new_values)
    # [B,Nv,N,C,Dv], FP32
    # core_attn_out holds every token's output, grouped into chunks.


    # Step G: Prepare the three decay-scaled tensors.
    query = query * cum_decay.exp().unsqueeze(-1)
    # [B,Nv,N,C,Dk]; row i: q_i*exp(G_i), for reading S_in
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    # [B,Nv,N,C,Dk]; row i: k_i*exp(G_C-G_i), for updating S_out
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    # [B,Nv,N,1,1], exp(G_C) decay of the entire chunk
    # Each of query, key, and chunk_decay has a different scaling and purpose:
    # - query, scaled by exp(G_i), is intended to read the incoming state.
    # - key, scaled by exp(G_C-G_i), is intended to update the outgoing state.
    # - chunk-decay, scaled by exp(G_C), is intended to decay the incoming state across the entire chunk.
    # Why different scalings? 
    # The incoming state is older than all tokens in the chunk, but each
    # token's update needs to be decayed only over the remaining tokens
    # before the chunk ends.

    # Step H: Sequential scan over chunks
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        # [B,Nv,C,Dv] - ([B,Nv,C,Dk] @ [B,Nv,Dk,Dv]) -> [B,Nv,C,Dv]
        # Corrected token values account for prior state and earlier tokens in the chunk.
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        # [B,Nv,C,Dk] @ [B,Nv,Dk,Dv] -> [B,Nv,C,Dv]
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # [B,Nv,C,Dv] + ([B,Nv,C,C] @ [B,Nv,C,Dv]) -> [B,Nv,C,Dv]
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
        # [B,Nv,Dk,Dv] * [B,Nv,1,1] + ([B,Nv,Dk,C] @ [B,Nv,C,Dv])
        # -> [B,Nv,Dk,Dv], state after chunk i
    # Each iteration does four things:
    # 1. v_new: Calculate all corrected value updates for this chunk, accounting for the incoming state.
    # 2. inter_chunk_attn: Read information from before the chunk.
    # 3. core_attn_out: Add within-chunk and previous-state contributions to produce the outputs.
    # 4. last_recurrent_state: Update the state once for the entire chunk.
    # The loop is sequential across chunks, but its matrix operations process many tokens together.


    # Step I. Restore the original output layout
    # The final steps flatten the chunk dimension, remove padding, restore the
    # original tensor layout and dtype, and optionally return the recurrent
    # state for subsequent calls.
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # [B,Nv,Dk,Dv] or None
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    # [B,Nv,P,Dv]; flatten N and C
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # [B,Nv,T,Dv]; remove padding
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    # [B,T,Nv,Dv], original input dtype
    return core_attn_out, last_recurrent_state
```

### Why do we need torch.linalg.solve_triangular

Remember we have

$$
\boxed{
\delta_i=
\beta_i v_i
-\beta_i e^{G_i}S_{\mathrm{in}}^\top k_i
-\sum_{j<i}\beta_i d_{ij}(k_i^\top k_j)\delta_j
}
$$

Consider a chunk with just three tokens. Define:

$$l_{ij} = \beta_i d_{ij} (k_i^\top k_j)$$

and

$$r_i = \beta_i(v_i - e^{G_i} S_{\text{in}}^\top k_i)$$

Then our equation becomes:

$$\delta_i = r_i - \sum_{j<i} l_{ij} \delta_j$$

Expand it for the three tokens:

$$\begin{aligned}
\delta_1 &= r_1 \\\\
\delta_2 &= r_2 - l_{21}\delta_1 \\\\
\delta_3 &= r_3 - l_{31}\delta_1 - l_{32}\delta_2
\end{aligned}$$

Move all the unknown $\delta$ terms to the left:

$$\begin{aligned}
\delta_1 &= r_1 \\\\
l_{21}\delta_1 + \delta_2 &= r_2 \\\\
l_{31}\delta_1 + l_{32}\delta_2 + \delta_3 &= r_3
\end{aligned}$$

Now write it as a matrix equation:

$$
\underbrace{\begin{bmatrix} 1 & 0 & 0 \\\\ l_{21} & 1 & 0 \\\\ l_{31} & l_{32} & 1 \end{bmatrix}}\_{I + L} \underbrace{\begin{bmatrix} \delta_1^\top \\\\ \delta_2^\top \\\\ \delta_3^\top \end{bmatrix}}\_{\Delta} = \underbrace{\begin{bmatrix} r_1^\top \\\\ r_2^\top \\\\ r_3^\top \end{bmatrix}}\_{R}
$$

Or simply:

$$(I + L)\Delta = R$$

Therefore:

$$\boxed{\Delta = (I + L)^{-1}R}$$

This is precisely why the chunked implementation uses:

```
torch.linalg.solve_triangular(
    ut_system,
    ...,
    upper=False,
    unitriangular=True,
)
```

## Qwen3_5MoeSparseMoeBlock

The routing and expert computation. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L832-L912)

## Qwen3_5MoeTextModel.forward

The layer loop, masks, cache management. [Code](https://github.com/huggingface/transformers/blob/109a7d19bf667168837f32929e00b791e195d14d/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1310-L1380)
