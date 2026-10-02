---
layout: ../../layouts/Layout.astro
title: "Understanding fused_moe_v2 Pallas kernel"
description: "An intuitive explanation of the fused_moe_v2 Pallas kernel"
pubDate: 2026-09-22
---

# Understanding the fused_moe_v2 Pallas kernel

I will use this post to explain the [fused_moe_v2 Pallas TPU kernel](https://github.com/vllm-project/vllm-torchtpu/tree/862433c874223c2983bd93f2b716cd16d52b78ec/src/vllm_torchtpu/kernels/fused_moe/v2). You can find the unit tests [here](https://github.com/vllm-project/vllm-torchtpu/tree/862433c874223c2983bd93f2b716cd16d52b78ec/tests/kernels/fused_moe).

The code reference uses the commit https://github.com/vllm-project/vllm-torchtpu/commit/22286cfcbfdd8983028475d5959ef274a2f4fd09

## 1. Reference Implementation (What does the kernel do?)

You can find the reference implementation in this [test](https://github.com/vllm-project/vllm-torchtpu/blob/862433c874223c2983bd93f2b716cd16d52b78ec/tests/kernels/fused_moe/test_fused_ep_moe_v2.py#L220)

`_dense_reference` (`tests/kernels/fused_moe/test_fused_ep_moe_v2.py:176`) computes the same MoE layer the simple way: **every expert runs on every token**, and tokens that didn't pick an expert get a weight of 0.

**Symbols** (with the `SMALL` test values): `T` = tokens (256), `H` = hidden (512), `I` = inter (256), `E` = total experts (32), `ep` = chips (8), `G = E/ep` = experts per chip (4), `K` = topk (4).

### Inputs (global arrays)

| Name | Shape | dtype | Notes |
|---|---|---|---|
| `x` | `[T, H]` | bf16 | |
| `gating` | `[T, E]` | f32 | router logits |
| `w1` | `[E, H, 2I]` | fp8 or bf16 | gate and up weights side by side |
| `w2` | `[E, I, H]` | fp8 or bf16 | down projection |
| `w1_scale` | `[E, 2I]` | f32 | one scale per output column; `None` for bf16 |
| `w2_scale` | `[E, H]` | f32 | one scale per output column; `None` for bf16 |

Note: In the above table w.shape[0] should be `E` (global) not `G` (local). This is because when it prepare the inputs

```
def _make_weights(mesh, seed, *, hidden, inter, e_total, weight_format):
    ...

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        weights = []
        for key, shape in (
            (k1, (g_local, hidden, 2 * inter)),
            (k2, (g_local, inter, hidden)),
        ):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            if not quantized:
                weights.append((w.astype(jnp.bfloat16), None))
                continue
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
            weights.append(
                (
                    (w / scale).astype(FP8),
                    scale.astype(jnp.float32).reshape(shape[0], shape[2]),
                )
            )
        (w1, w1s), (w2, w2s) = weights
        return (w1, w2) if not quantized else (w1, w2, w1s, w2s)

    out_specs = (P(axis),) * (4 if quantized else 2)
    built = jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=(P(axis),), out_specs=out_specs, check_vma=False
        )
    )(jnp.zeros((EP,), jnp.float32))
    return built if quantized else (built[0], built[1], None, None)
``` 

- `local` runs once per device, so inside it each shard's `w1` is `[g_local, hidden, 2*inter]`.
- But `out_specs=(P(axis),)` joins those shards back along dim 0. So `_make_weights` returns one global w1 and that global array is what gets passed to `_dense_reference`.

### The setup (runs once, outside the shard)

```python
(axis, ) = mesh.axis_names                 # "d"
g_local = w1.shape[0] // mesh.shape[axis]  # G = 32 // 8 = 4   (python int)
inter = w1.shape[2] // 2                   # I = 2I // 2 = 256 (python int)
scaled = w1_scale is not None              # True for fp8, False for bf16
```

`shard_map` runs `local` once per chip, and `in_specs` decides what each chip sees:

```python
in_specs = (P(), P(), P(axis), P(axis)) + ((P(axis),) * 2 if scaled else ())
#           x     gating  w1       w2         w1_scale, w2_scale
```

### Inside `local` (per chip)

```python
def local(x_g, gating_g, w1_l, w2_l, *scales):
    # x_g      [T, H]      bf16   full batch (replicated)
    # gating_g [T, E]      f32    full batch
    # w1_l     [G, H, 2I]  fp8    this chip's experts only
    # w2_l     [G, I, H]   fp8
    # scales = (s1 [G, 2I], s2 [G, H])  or ()
    s1, s2 = scales if scaled else (None, None)
    # s1 [G, 2I] f32, s2 [G, H] f32   (or None, None)
    scores = jax.nn.softmax(gating_g.astype(jnp.float32), axis=-1)
    # scores [T, E] f32      each row sums to 1 over all 32 experts
    topk_weights, topk_idx = lax.top_k(scores, topk)
    # topk_weights [T, K] f32   the K largest scores per token
    # topk_idx     [T, K] i32   their global expert ids, in 0..E-1
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    # .sum(..., keepdims=True) -> [T, 1];  division broadcasts
    # topk_weights [T, K] f32   now each row sums to 1 over its K picks
    #                           (this is renormalize=True)
    first = lax.axis_index(axis) * g_local
    # first  scalar i32   global id of this chip's expert 0
    #                     chip 0 -> 0, chip 1 -> 4, ..., chip 7 -> 28
    x32 = x_g.astype(jnp.float32)
    # x32 [T, H] f32
```

### The per-expert loop body

```python
    def expert(e, acc):
        # e    scalar i32 in 0..G-1   LOCAL expert index on this chip
        # acc  [T, H] f32             running sum over experts so far
        # dynamic_index_in_dim(operand, index, axis=0, keepdims=True, *, ...),
        w1e = lax.dynamic_index_in_dim(w1_l, e, 0, keepdims=False).astype(jnp.float32)
        # w1_l[e]  -> w1e [H, 2I] f32
        w2e = lax.dynamic_index_in_dim(w2_l, e, 0, keepdims=False).astype(jnp.float32)
        # w2_l[e]  -> w2e [I, H] f32
        if scaled:
            w1e = w1e * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
            # s1[e:e+1] -> [1, 2I];  [H, 2I] * [1, 2I] -> w1e [H, 2I]
            # each output COLUMN j is multiplied by its scale s1[e, j]
            w2e = w2e * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            # s2[e:e+1] -> [1, H];   [I, H] * [1, H]   -> w2e [I, H]
            # after this, w1e and w2e are the true (dequantized) weights
        weight = jnp.sum(jnp.where(topk_idx == first + e, topk_weights, 0.0),
                         axis=-1)[:, None]
        #   topk_idx == first + e             -> [T, K] bool
        #   where(..., topk_weights, 0.0)     -> [T, K] f32
        #   sum(axis=-1)                      -> [T]
        #   [:, None]                         -> weight [T, 1] f32
        # For each token: its router weight for THIS expert, or 0 if it
        # didn't pick it. top_k never returns the same expert twice, so at
        # most one of the K slots matches.
        acc1 = x32 @ w1e
        # [T, H] @ [H, 2I] -> acc1 [T, 2I] f32
        row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
        #   acc1[:, :inter]   gate  [T, I]
        #   acc1[:, inter:]   up    [T, I]
        #   silu(gate) * up         [T, I]
        #   @ w2e [I, H]      -> row [T, H] f32   this expert's FFN for every token
        return acc + weight * row
        # [T, 1] * [T, H] -> [T, H]   (tokens that skipped e contribute 0)
        # N.B. The ref impl never shards the number of tokens dimension.
        # acc + ...       -> [T, H]
```

### Summing the experts, then across chips

```python
    mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
    # start: zeros [T, H] f32; runs e = 0..G-1
    # mine [T, H] f32   = Σ over THIS chip's 4 experts of weight_e * FFN_e(x)
    return lax.psum(mine, axis)
    # adds `mine` from all 8 chips
    # -> [T, H] f32   = Σ over all 32 experts; identical on every chip
```

`out_specs=P()` says the result is replicated, so the function returns one global `[T, H]` f32 array.

### Why this matches the formula

For token `t`:

```
out[t] = Σ over chips  Σ over e in that chip  weight[t, e] · FFN_e(x[t])
       = Σ over all 32 experts                weight[t, e] · FFN_e(x[t])
       = Σ over its K picks                   w_k · FFN_{e_k}(x[t])     # the other 28 weights are 0
```

### How it differs from the kernel (only the important ones)

| | Reference | Kernel |
|---|---|---|
| Work | All `T × E` (token, expert) pairs; 28 of every 32 are multiplied by 0 | Only the `T × K` pairs that were picked |
| Output | f32, replicated | `x.dtype` (bf16), sharded like `x` |

N.B. About the `Work`
- reference (256 × 32 = 8,192 FFN evaluations): Each chip loops over its 4 experts, and for each one runs x32 @ w1e on all 256 tokens (after the all-gather).
- kernel (about 256 × 4 = 1,024 FFN evaluations): The slab has one row per (token, expert) pair that was actually picked. A slab belongs to one chip. It's the list of rows that chip will compute. Each row stands for one (token, expert) pair where the expert lives on this chip and the token picked it.

## 2. How does the vllm-torchtpu use the kernel.

The framework calls this kernel through one wrapper, `layers/adapter/fused_moe_ep.py`. Its interface matches what `test_fused_ep_moe_v2.py` exercises on shapes, dtypes and sharding.

### 2.1. At weight load (once per MoE layer).

Each quant method (fp8.py:1005, [nvfp4.py](https://github.com/vllm-project/vllm-torchtpu/blob/b2be5ce4d87292e4d12663bde47202d0aeb9d765/src/vllm_torchtpu/layers/adapter/quantization/nvfp4.py#L222), mxfp4.py:415) calls `prebuild_fused_moe_ep`. That function checks whether this layer can use the kernel and returns None if any check fails:
- the USE_MOE_FUSED_EP_KERNEL env var is set
- etc.

If all checks pass, `_build_op` wraps `fused_ep_moe_v2` as a torch custom op named `pallas::fused_moe_ep`, using `sharded_jax_op`. The op is stored on the layer's quant method.

### 2.2 In the forward pass. [apply_monolithic](https://github.com/vllm-project/vllm-torchtpu/blob/b2be5ce4d87292e4d12663bde47202d0aeb9d765/src/vllm_torchtpu/layers/adapter/quantization/nvfp4.py#L496) calls fused_moe_ep(...), which runs:
```
op(x, w1, w2, s1, s2, logits, rank[, score_bias | hash_ids])
```
[here](https://github.com/vllm-project/vllm-torchtpu/blob/b2be5ce4d87292e4d12663bde47202d0aeb9d765/src/vllm_torchtpu/layers/adapter/fused_moe_ep.py#L379).
The op runs `moe` / `moe_with_score_bias` / `moe_with_hash_ids` (around [here](https://github.com/vllm-project/vllm-torchtpu/blob/b2be5ce4d87292e4d12663bde47202d0aeb9d765/src/vllm_torchtpu/layers/adapter/fused_moe_ep.py#L241-L292)), which call `call()`. That in turn calls `fused_ep_moe_v2(...)` [here](https://github.com/vllm-project/vllm-torchtpu/blob/b2be5ce4d87292e4d12663bde47202d0aeb9d765/src/vllm_torchtpu/layers/adapter/fused_moe_ep.py#L215).

### 2.3 vLLM's own collectives are skipped. 

`__init__.py` patches `MoERunner` so the kernel's output counts as already reduced. For an armed layer, vLLM therefore does no all-gather before the layer and no reduce-scatter after it: each chip passes in its own tokens and gets its own tokens back.

N.B. Armed layer
An armed layer is an MoE layer where `prebuild_fused_moe_ep` passed every check at weight load and returned an op. That op is stored on the layer's quant method under `_fused_moe_ep_op`, and `fused_moe_ep_supported(quant_method)` then returns `True`.
- Armed: every forward step of that layer runs `fused_ep_moe_v2`, decode included. vLLM skips its own all-gather and reduce-scatter around the layer.
- Not armed (refused): any check failed, the attribute is `None`, and the layer uses the normal grouped-matmul path with vLLM's collectives.


## 3. How does the kernel work?

<img
  src="/images/fused_moe_v2/data_path_for_token_x.png"
  width="700"
/>

### Overview

- (1) Route: Each chip runs softmax and top-k on its own tokens, then all-gathers every chip's picks and builds the routing plan: the slab and address tables that tell the kernel which (token, expert) rows to compute and which chip each result goes to.
- (2) All-gather: Each chip quantizes its tokens to fp8 and all-gathers them, so every chip holds a copy of every token.
- (3) Fused kernel: Each chip runs its own experts' FFN on only the tokens routed to them, and sends each finished row by remote DMA to the chip that owns the token while the next tile computes.
- (4) Combine: Each chip sums the k rows that came back for each of its tokens, weighted by the router scores (times the fp8 row scale), to get the final output.

The driver code in `layer.py → fused_ep_moe_v2 → local_fn` reads almost like the picture. Two things to watch in the shapes: the all-gather at ② is the only place a full-batch `T` appears, and the kernel's output is `t·K` unweighted rows, which ④ folds back down to `t` tokens.

```
# This body runs once per chip. Shapes use:
#   T = tokens      t = T/ep tokens this chip owns     H = hidden
#   E = experts     G = E/ep experts this chip owns    I = inter
#   K = topk        LB = H/128 lane blocks
#
# Inputs (this chip's shard):
#   x_l      [t, H]       bf16
#   rank_l   [1, 1]       i32
#   w1_l     [G, H, 2I]   fp8 e4m3
#   w2_l     [G, I, H]    fp8 e4m3
#   w1s_l    [G, 2I]      f32
#   w2s_l    [G, H]       f32
#   gate_l   [t, E]       f32
#
# Optional:
#   hash_l        [t, K]  i32        (hash-routed layers)
#   score_bias_l  [E]     f32        (replicated, not sharded)

x_l                                          # my tokens
                                             #   [t, H] bf16

q_l, row_scale = rowquant_fp8(x_l)           # per-row fp8 quantize
                                             #   [t, H] fp8  +  [t, 1] f32
q_g = lax.all_gather(q_l.reshape(t, LB, 128),
                     axis=0, tiled=True)     # ② every chip gets all tokens
                                             #   [T, LB, 128] fp8

scores = jax.nn.softmax(gate_l)              # router scores
                                             #   [t, E] f32
w, idx = pallas_select(scores, topk=K)       # ① top-k (router_ops.py)
                                             #   [t, K] f32  +  [t, K] i32

exchange = lax.all_gather(pack(idx, row_scale_bits, counts))
                                             # ① everyone's picks + row scales
                                             #   [ep, width] i32
routing = build_routing_tables(pairs.reshape(T, K), ...)
                                             # ① "the plan" (host.py)
token_gather = shard_token_gather(routing, me, ...)
                                             # slab row -> token id
                                             #   [ragged_stride] i32
scale_slab = scatter(row scale bits onto slab rows)
                                             # each slab row's fp8 scale
                                             #   [slab rows/128, 128] f32

arrivals, arrival_scales = kfn(tables, token_gather, q_g, w1_l, w2_l, ...)
                                             # ③ THE kernel
                                             #   [recv_rows, LB, 128] fp8
                                             #     recv_rows ≈ t·K result rows
                                             #   [mirror rows, 128] f32
                                             #     one f32 per arrival row

pos = routing.arrival_row[my tokens]         # where my results came back
                                             #   [t, K] i32  rows of `arrivals`
return _combine_arrivals(arrivals, arrival_scales, pos, ..., w)
                                             # ④ weighted sum per token
                                             #   [t, H] bf16  sharded like x
```

### The running example

Every lesson uses the same toy setup, small enough to follow each number. The real code ran on it to produce the tables below.

| Symbol | Meaning | Toy value | Qwen3.5-397B (served) |
|---|---|---|---|
| `ep` | chips on the expert-parallel axis | 2 | 8 |
| `E` | experts in total | 4 | 512 |
| `G = E/ep` | experts per chip | 2 | 64 |
| `T` | tokens in total | 8 | 16,384 |
| `t = T/ep` | tokens each chip owns | 4 | 2,048 |
| `K` | experts picked per token (top-k) | 2 | 10 |
| `H` | hidden size | (not needed in step 1) | 4,096 |

**Who owns what:**

```
chip 0: tokens t0 t1 t2 t3   experts 0, 1
chip 1: tokens t4 t5 t6 t7   experts 2, 3
```

Tokens are split by row (`x` and `gating` are sharded `P("d")`), and experts are split by expert id (`w1`, `w2` are sharded `P("d")` on dim 0).

### Step 1 · Route

**In one sentence:** each chip runs softmax and top-k on its own tokens, then all-gathers every chip's picks and builds the routing plan: the slab and address tables that tell the kernel which (token, expert) rows to compute and which chip each result goes to.

Step 1 does no matmuls and moves no token data. It only produces small integer tables. Everything the kernel does later is driven by these tables.

#### Shapes going in and coming out (per chip)

**Input**
- `gate_l` `[t, E]` f32: router logits for my tokens. (vLLM's gate gives bf16; the adapter casts it to f32.) Toy: `[4, 4]` · Qwen: `[2048, 512]`
- Config: `K`, `renormalize`, `scoring_fn` (softmax by default).
- Optional: `score_bias` `[E]` f32, or `hash_topk_ids` `[t, K]` i32.

**Output**
- `topk_weights` `[t, K]` f32, kept on this chip for step 4.
- `token_gather` `[stride]` i32: slab row → token id.
- `expert_rows`, `slab_base` `[G]` i32 and `visit` `[G]` i32.
- Transport tables and `counts` `[8, ep]` i32.
- `pos` = `arrival_row` slice `[t, K]` i32, used in step 4.

#### Files and functions to read

Read them in this order. The first row is the driver; each other row is one function it calls.

| Where | What it does |
|---|---|
| `layer.py:446` `local_fn`, lines 484–651 | The per-chip driver. Read these lines top to bottom; they are all of step 1. |
| `layer.py:67` `_apply_scoring_fn` | Logits → scores. Softmax by default; sigmoid and sqrt-softplus also exist. |
| `router_ops.py:69` `pallas_select` | Top-k as a tiny Pallas kernel: K times, take the row max, record it, mask it out. |
| `layer.py:193` `_relabel_expert_ids_to_mesh_order` | Only when device order differs from EP-rank order: renames the picked expert ids. |
| `layer.py:139–191` `_exchange_layout`, `_pack_…`, `_unpack_routing_exchange` | Pack my picks, row scales and per-expert counts into one flat i32 vector, so a single all-gather shares them. |
| `host.py:522` `build_routing_tables` | The plan. From all `T×K` picks, decide the slab row every pair computes on and the arrival row it comes back to. Returns `RoutingTables` (`host.py:423`). |
| `host.py:976–1258` `shard_*`, `expert_visit_list` | Cut the parts this chip needs from the plan: its token gather list, its expert slab sizes, the visit order, the send/receive tables and counts. |

> `build_routing_tables_sharded` (`host.py:820`) builds the same tables but only this chip's slices. It is off by default (`MOE_FUSED_EP_V2_SHARDED_PLAN=0`), and a test holds the two to bit-exact agreement, so you can skip it on a first read.

Step 1 has four parts: 1a scores, 1b shares, 1c plans and 1d cuts.

### 1a · Score and pick (each chip, its own tokens)

```python
scores = _apply_scoring_fn(gate_l, "softmax")      # [t, E] f32
topk_weights, topk_idx = pallas_select(scores, topk=K)
                                                   # [t, K] f32, [t, K] i32
topk_weights /= topk_weights.sum(-1, keepdims=True) # if renormalize
```

In the toy, each chip scores its 4 tokens over 4 experts and keeps the top 2. Picks are in **bold**. The weights on the right are the two picked scores divided by their sum.

| token | e0 | e1 | e2 | e3 | topk_idx | topk_weights |
|---|---|---|---|---|---|---|
| t0 (chip 0) | **.53** | .08 | **.32** | .07 | [0, 2] | [.62, .38] |
| t1 (chip 0) | .08 | .10 | **.60** | **.22** | [2, 3] | [.73, .27] |
| t2 (chip 0) | .12 | **.54** | **.24** | .09 | [1, 2] | [.69, .31] |
| t3 (chip 0) | **.23** | .07 | .08 | **.62** | [3, 0] | [.73, .27] |
| t4 (chip 1) | **.16** | .06 | **.71** | .07 | [2, 0] | [.82, .18] |
| t5 (chip 1) | .08 | **.25** | .10 | **.56** | [3, 1] | [.69, .31] |
| t6 (chip 1) | **.45** | .10 | **.36** | .09 | [0, 2] | [.55, .45] |
| t7 (chip 1) | .07 | **.58** | .10 | **.26** | [1, 3] | [.69, .31] |

Chip 0 computes the top four rows and chip 1 the bottom four, in parallel. `topk_idx` is ordered by score, so t3's list is `[3, 0]`, not `[0, 3]`. `topk_weights` never leaves the chip: step 4 applies it on the token's own chip.

#### 1b · Share the picks (one small all-gather)

To build the plan, every chip needs to know every token's picks, not just its own. `local_fn` packs three things into one flat int32 vector and all-gathers it once:

<img
  src="/images/fused_moe_v2/step1_1b.png"
  width="700"
/>

Each section starts on a 128-lane boundary: `width = align(t·K) + align(t) + E`. Qwen: 20480 + 2048 + 512 = 23040.

- **picks** are the `topk_idx` ids, flattened row by row. Chip 0's are `[0, 2, 2, 3, 1, 2, 3, 0]`, chip 1's are `[2, 0, 3, 1, 0, 2, 1, 3]`. Entry `i` belongs to token `i // K`, so no token ids need to be sent.
- **row scales** are the fp8 quantization scale of each token (made in step 2, stored as raw int32 bits).
- **counts** say how many picks this chip sent to each expert.
- Weights are *not* sent.

After unpacking, every chip holds the same picks table for all 8 tokens:

| token | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
|---|---|---|---|---|---|---|---|---|
| owner | chip 0 | chip 0 | chip 0 | chip 0 | chip 1 | chip 1 | chip 1 | chip 1 |
| experts | 0, 2 | 2, 3 | 1, 2 | 3, 0 | 2, 0 | 3, 1 | 0, 2 | 1, 3 |


**Example**

1. Each chip builds its payload

Chip 0's picks, from 1a:

`t0: [0, 2]   t1: [2, 3]   t2: [1, 2]   t3: [3, 0]`

The payload has three sections, each padded to a 128-entry boundary (`_exchange_layout`):

| Section | Content (chip 0) | Length |
| :--- | :--- | :--- |
| picks | [0,2, 2,3, 1,2, 3,0] + 120 zeros | align(t·K = 8) = 128 |
| row scales | [s0, s1, s2, s3] + 124 zeros | align(t = 4) = 128 |
| counts | [2, 1, 3, 2] | E = 4 |

That makes width = 128 + 128 + 4 = 260 int32 values.

- picks is `topk_idx` flattened row by row.
- row scales holds each token's fp8 scale as raw int32 bits. For example, a scale of `0.002` (f32) is stored as `990057071`.
- counts is how many of this chip's picks went to each expert. Among chip 0's picks, expert 0 appears 2 times, expert 1 once, expert 2 three times and expert 3 twice, so it's [2, 1, 3, 2].

Chip 1 does the same with its own tokens:

`picks  = [2,0, 3,1, 0,2, 1,3]      (t4..t7)`
`counts = [2, 2, 2, 2]`

2. One all-gather

exchange = lax.all_gather(payload, axis=0, tiled=True).reshape(ep, width)   # [2, 260] i32

Both chips now hold the same [2, 260] array:

```
row 0 (from chip 0): [0,2,2,3,1,2,3,0, 0…] [s0,s1,s2,s3, 0…] [2,1,3,2]
row 1 (from chip 1): [2,0,3,1,0,2,1,3, 0…] [s4,s5,s6,s7, 0…] [2,2,2,2]
```

3. Unpack (_unpack_routing_exchange)

```
Output: pairs_g
Shape: [2, 8] → reshape [T, K] = [8, 2]
Value: [[0,2],[2,3],[1,2],[3,0],[2,0],[3,1],[0,2],[1,3]]
```

────────────────────────────────────────
```
Output: scale_bits_g
Shape: [T] = [8]
Value: [s0 … s7]
```
────────────────────────────────────────
```
Output: rows_by_dest (counts)
Shape: [ep, E] = [2, 4]
Value: [[2,1,3,2], [2,2,2,2]]
```


#### 1c · Build the plan: a slab of rows per chip

`build_routing_tables` takes those `T×K = 16` (token, expert) pairs and gives each one a **slab row**. A chip's slab lists, for each of its experts, one row per pair routed to that expert. Two rules decide the order:

1. Group by **expert**. Expert 0's rows first, then expert 1's.
2. Inside an expert, group by **the chip that owns the token**. Each group is a "run". A run is padded up to a multiple of `ROWBLK = 2` rows.

Rule 2 is what makes step 3 cheap: each run is one contiguous chunk of results headed to one chip, so it ships as a single DMA.

<img
  src="/images/fused_moe_v2/step1_1c_1.png"
  width="700"
/>

The same function also decides where each result **comes back**. Each chip gets a receive buffer (`arrivals`), grouped by (source chip, expert). `arrival_row[token, k]` records which row of that buffer will hold the token's k-th result. Chip 0's receive buffer in the toy:

<img
  src="/images/fused_moe_v2/step1_1c_2.png"
  width="700"
/>

**Example**: The plan table (`RoutingTables`), toy example

The plan is one `RoutingTables` object: a handful of small integer tables. Every chip builds the same copy. Below are the real values `build_routing_tables` produced for the toy (2 chips, 4 experts, 8 tokens, top-2), grouped by the question each one answers.

##### A. How long is each expert's list, and where does it start?

| expert | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| lives on | chip 0 | chip 0 | chip 1 | chip 1 |
| `expert_rows_aligned` (rows, pad included) | 4 | 4 | 6 | 4 |
| `expert_base` (first global slab row) | 0 | 4 | 256 | 262 |

Chip 1's slab starts at global row 256 (`stride`), which is why expert 2 starts there.

##### B. Inside each expert, how many rows come from each chip's tokens?

Each cell is one run: one expert, one owner chip. Values are listed as (chip 0, chip 1).

| expert | `run_rows` (real) | `run_rows_aligned` (padded) | `run_start_aligned` (start inside the expert) |
|---|---|---|---|
| 0 | 2, 2 | 2, 2 | 0, 2 |
| 1 | **1**, 2 | **2**, 2 | 0, 2 |
| 2 | **3**, 2 | **4**, 2 | 0, **4** |
| 3 | 2, 2 | 2, 2 | 0, 2 |

The runs with 1 and 3 real rows each get a pad row to reach an even count. That's also why chip 1's run in expert 2 starts at 4.

##### C. Which row does each (token, pick) job go to?

`slab_row`, one entry per pair, in the order t0k0, t0k1, t1k0, …:

```
[0, 256, 257, 262,  4, 258, 263, 1,  260, 2, 264, 6,  3, 261, 7, 265]
```

For example, t6's second pick (expert 2) is at 261, which is chip 1's row 5. It's built from A and B: `expert_base[2] + run_start_aligned[2, chip 1] + position in run = 256 + 4 + 1`.

##### D. Where does each result land when it comes back?

| table | value | meaning |
|---|---|---|
| `recv_rows` | `[10, 8]` | Chip 0 receives 10 rows and chip 1 receives 8, pad included. |
| `recv_base[dest][src][expert]` | chip 0: from chip 0 `[0, 2]`, from chip 1 `[4, 8]` | In chip 0's arrivals, chip 0's e0 run lands at row 0, its e1 run at row 2, chip 1's e2 run at row 4, and its e3 run at row 8. |
| | chip 1: from chip 0 `[0, 2]`, from chip 1 `[4, 6]` | The same, for chip 1's arrivals. |
| `arrival_row` `[T, K]` | t0 `[0, 4]`, t1 `[5, 8]`, t2 `[2, 6]`, t3 `[9, 1]`, t4 `[4, 0]`, t5 `[6, 2]`, t6 `[1, 5]`, t7 `[3, 7]` | The row each token's k-th result lands on, in its own chip's arrivals. |

`region_rows` is table B's padded run lengths regrouped by (source chip, expert, destination). The remaining fields (`mirror_row`, `run_tiles`) do the same bookkeeping for the fp8 scales that travel alongside the result rows.

**In short:** A and B lay out the to-do lists, C puts each job on a row, and D says where each result will come back. Step 1d then cuts out each chip's own share of these tables.

#### 1d · Cut this chip's tables

1c builds one set of tables, `RoutingTables`, that describes all chips. Every chip builds the same copy. 1d is each chip keeping only the parts that concern itself, using its own index me. Those parts are what the kernel receives.

| Table (function) | Shape, dtype | Chip 0 in the toy | Chip 1 in the toy |
|---|---|---|---|
| `token_gather` (`shard_token_gather`) | `[stride]` i32 | `[0,3,4,6,2,0,5,7, …]` | `[0,1,2,0,4,6,1,3,5,7, …]` |
| `expert_rows`, `slab_base` (`shard_expert_slabs`) | `[G]` i32 each | `[4,4]`, `[0,4]` | `[6,4]`, `[0,6]` |
| `visit` (`expert_visit_list`) | `[G]` i32 | `[0,1]` (tie, lower id first) | `[0,1]` (expert 2 has more rows) |
| push and receive offsets (`shard_transport_tables_in_blocks`, `shard_push_tables_in_rows`) | small `[G, ep]` i32 | For each (my expert, destination chip): where the run starts in my slab, how many real rows it has, and which row of the destination's arrivals it lands on. | (same) |
| `counts` (`shard_count_vector`) | `[8, ep]` i32 | Totals the kernel checks at the end: rows sent, rows received, experts visited. | (same) |
| `pos` (slice of `arrival_row`) | `[t, K]` i32 | `[[0,4],[5,8],[2,6],[9,1]]` | `[[4,0],[6,2],[1,5],[3,7]]` |

`stride` is the per-chip slab allocation, sized for the worst case where every pair lands on one chip (`ragged_stride_bound`). The toy's stride is 256; only the first 8 or 10 rows are live. `visit` puts busy experts first and drops empty ones, so an unused expert's weights are never loaded.

#### What is a slab row really?

A slab is a chip's to-do list. Each slab row is one job on that list: "run token X through expert Y."

Chip 1 owns experts 2 and 3, so its list is:
```
row 0: t0 → expert 2
row 1: t1 → expert 2
row 2: t2 → expert 2
row 3: (empty filler)
row 4: t4 → expert 2
row 5: t6 → expert 2
row 6: t1 → expert 3
row 7: t3 → expert 3
row 8: t5 → expert 3
row 9: t7 → expert 3
```
That's the whole idea.

#### Check yourself

1. Why is chip 1's slab 10 rows but chip 0's only 8? *(Expert 2 is popular: 5 picks, padded to 6.)*
2. t0's result from expert 2 is computed on which chip, at which slab row, and lands on which row of chip 0's arrivals? *(Chip 1, slab row 0, arrival row 4.)*
3. Which step-1 output is *not* shared with other chips? *(`topk_weights`.)*


### Step 2 · All-gather

**In one sentence:** each chip quantizes its tokens to fp8 row by row and all-gathers them, so every chip holds a copy of every token.

Step 1 decided *which* token each slab row needs. Think of step 1 as writing a to-do list that holds only **token names**, not token data.
After step 1, chip 1's list says:
```
row 0: run t0 through expert 2
row 1: run t1 through expert 2
...
row 5: run t6 through expert 2
...
```
In code, that list is `token_gather = [0, 1, 2, 0, 4, 6, 1, 3, 5, 7]`. It's just token numbers.

The problem: row 0 says "t0", but t0's actual data (its 256 numbers) sits on chip 0. Chip 1 has only the name, not the numbers.

Step 2 makes sure the token's data is already sitting in every chip's own memory, so the kernel in step 3 can fetch any row locally without asking another chip. Specifically, after the all-gather, chip 1 holds a copy of every token's data:
```
q_g[0] = t0's data   ← came from chip 0
q_g[1] = t1's data   ← came from chip 0
...
q_g[6] = t6's data   ← chip 1's own
```
Now, when the kernel processes row 0 on chip 1, it reads `token_gather[0] = 0` and fetches `q_g[0]`, which is t0's data, already in chip 1's own memory:
```
step 1:  row 0 needs token 0        (the name)
step 2:  q_g[0] is token 0's data   (the numbers)
step 3:  row 0 = expert 2 applied to q_g[0]
```

#### Shapes going in and coming out (per chip)

**Input**
- `x_l` `[t, H]` bf16: my tokens' hidden states. Toy: `[4, 256]` · Qwen: `[2048, 4096]`
- The weight format, which decides whether rows are quantized (fp8, fp4 weights) or sent as bf16 (bf16, int8 weights).
- From 1b: `scale_bits_g` `[T]` i32 and this chip's `slab_row`.

**Output**
- `q_g` `[T, LB, 128]` fp8: every token, on every chip. Toy: `[8, 2, 128]` · Qwen: `[16384, 32, 128]` (64 MB)
- `scale_slab` `[rows/128 + 1, 128]` f32: the fp8 scale of each of *my* slab rows.

#### Files and functions to read

| Where | What it does |
|---|---|
| `layer.py:470–482` in `local_fn` | The whole of step 2's data path: cast to bf16, quantize, all-gather. |
| `kernel.py:188` `rowquant_fp8` | Per-row fp8 quantization. Returns the fp8 row and its f32 scale. |
| `kernel.py:168` `_row_scale` | `scale = max\|row\| / 448`. A row of all zeros, or with an inf or NaN, gets scale 0 and is zeroed whole. |
| `host.py:278` `WeightForm`, `host.py:331` `WEIGHT_FORMS` | One row per weight format. `act_dtype` says whether rows travel as fp8 or bf16. |
| `layer.py:625–635` in `local_fn` | Builds `scale_slab`: puts each token's row scale onto every slab row that computes that token. |

#### 2a · Quantize each row to fp8

```python
rows_bf16 = x_l.astype(jnp.bfloat16)               # [t, H] bf16
q_l, row_scale_l = rowquant_fp8(rows_bf16)         # [t, H] fp8, [t, 1] f32
#   scale = max(|row|) / 448       (448 is the largest fp8 e4m3 value)
#   q     = fp8(row / scale)       (the biggest element lands on ±448)
```


#### 2b · All-gather the fp8 rows

```python
q_g = lax.all_gather(
    q_l.reshape(t, LB, 128),   # [4, 2, 128] fp8 on each chip
    axis_name, axis=0, tiled=True)
                               # [8, 2, 128] fp8 on every chip: q_g[i] is token i
```

#### 2c · Put each row's scale on the slab

The kernel computes slab rows, not tokens, so it needs the scale per slab row. `local_fn` copies each token's scale onto every slab row that computes that token, using the same `slab_row` index as `token_gather`:

```python
scale_bits = jnp.repeat(scale_bits_g, K)         # [T·K]: one scale per (token, pick)
scale_slab = zeros[...].at[slab_row].add(scale_bits, mode="drop")
                                                  # rows of other chips are dropped
scale_slab = bitcast(scale_slab, f32).reshape(rows/128 + 1, 128)
```

For chip 1 in the toy, the two tables line up row for row:

| slab row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `token_gather` | t0 | t1 | t2 | pad | t4 | t6 | t1 | t3 | t5 | t7 |
| `scale_slab` | .004987 | .000894 | .003348 | 0 | .006906 | .003122 | .000894 | .000089 | .001962 | .004464 |

So when the kernel computes slab row 6 (expert 3 on token t1), it fetches `q_g[1]` and multiplies by `0.000894`. The pad row gets scale 0.

### Lesson 3 · The fused kernel

**In one sentence:** each chip runs its own experts' FFN on only the tokens routed to them, and sends each finished row by DMA to the chip that owns the token while the next tile computes.

This is the one Pallas call. Everything before it built tables and copied tokens; everything after it adds rows up. In terms of the to-do list: the kernel works through its slab top to bottom, and for each row it fetches the token, runs the expert, and mails the result to the token's owner.

#### Shapes going in and coming out (per chip)

**Input**
- From step 1: the small tables (`visit`, `expert_rows`, `slab_base`, run and push offsets, `counts`), prefetched into scalar memory, and `token_gather` `[stride]` i32.
- From step 2: `q_g` `[T, LB, 128]` fp8 and `scale_slab` f32.
- My experts' weights: `w1_l` `[G, H, 2I]` fp8, `w2_l` `[G, I, H]` fp8, `w1s` `[G, 2I]` f32, `w2s` `[G, H]` f32. Toy (I = 128): `w1_l [2, 256, 256]` · Qwen: `[64, 4096, 2048]`

Table: `visit [G]` i32
What it contains: The order in which to process my experts. Busiest first;
experts with no rows are skipped.
Chip 1 in the toy: `[0, 1]`: expert 2 (6 rows), then expert 3 (4 rows)

────────────────────────────────────────

Table: `expert_rows [G]` i32
What it contains: How many slab rows each of my experts has, padding included.
Chip 1 in the toy: `[6, 4]`

────────────────────────────────────────

Table: `slab_base [G]` i32
What it contains: Where each of my experts' rows start in my slab.
Chip 1 in the toy: `[0, 6]`: expert 2 is rows 0–5, expert 3 is rows 6–9

────────────────────────────────────────

Table: counts `[8, ep]` i32
What it contains: Totals the kernel waits for at the end: rows I send, rows I
receive, experts I visit, and so on. Each row of the table is summed to get
one number.
Chip 1 in the toy: send 5 rows (t0, t1, t2 from e2; t1, t3 from e3), receive 4
rows (from chip 0), visit 2 experts

────────────────────────────────────────

Table: `q_g [T, LB, 128]` fp8
What it contains: Every token's fp8 data, all chips' tokens. Row i is token i.
Chip 1 in the toy: 8 rows, t0–t7

────────────────────────────────────────

Table: `scale_slab` f32
What it contains: The fp8 scale for each of my slab rows, to undo the
quantization.
Chip 1 in the toy: `[.004987, .000894, .003348, 0, .006906, .003122, .000894,
.000089, .001962, .004464]`

How they work together, for chip 1:
1. `visit` says to start with expert 2.
2. `expert_rows` and `slab_base` say expert 2 is rows 0–5 of the slab.
3. For each of those rows, the kernel fetches `q_g[token_gather[r]]` and its scale `scale_slab[r]`.
4. At the end, `counts` says to wait for 5 rows sent and 4 received.

**Output**
- `arrivals` `[recv_rows, LB, 128]` fp8: result rows *for my tokens*, written into my memory by every chip (me included).
- `arrival_scales` `[mirror rows, 128]` f32: one scale per arrival row.
- Rows are **unweighted**: the router weight is applied in step 4.
- Toy: recv_rows = t·K + E = 12 (chip 0 fills 10, chip 1 fills 8).


#### Files and functions to read

`kernel.py` is about 2,300 lines, mostly buffer bookkeeping. Read only the docstrings of these functions first. Most are nested inside `kernel(*refs)` (`kernel.py:751`).

| Where | What it does |
|---|---|
| `kernel.py:1986` `fn` | The entry `local_fn` calls as `kfn(...)`. Its argument list is the kernel's full input list. |
| `kernel.py:936` `prologue` | Start loading the first experts' weights, then a barrier with all chips. |
| `kernel.py:1795` `visit_step` | The outer loop body: one expert from `visit`. |
| `kernel.py:1239` `expert_tiles` | For one expert: wait for its weights, start loading a later expert's, then loop over its tiles. |
| `kernel.py:1434` `tile_body` | The inner loop body: one tile of up to 128 slab rows. Gather, compute, ship. |
| `kernel.py:1106` `lhs_issue_tile` | Gather: one small DMA per row, `q_g[token_gather[r]] → lhs_vm`. |
| `kernel.py:1507` `compute_tile` → `:322` `expert_ffn_fp8`, `:204` `wire_row_and_scale` | The FFN math on one tile, then quantize the result rows to fp8 for sending. These two are plain jnp, easy to read. |
| `kernel.py:1337` `ship_tile` | Small batches: send each run of the tile straight to its destination chip. |
| `kernel.py:1293` `commit_tile` + `:1732` `push_expert` | Large batches: stage rows in local HBM, then push each run once the expert is done. |
| `kernel.py:1809` `drain_transport` | Wait until every expected row has been sent and received, then a final barrier. |

#### 3a · The loop

```
prologue()                               # start weight loads; barrier
for e in visit:                          # visit_step: busiest expert first
    expert_tiles(e):
        wait for e's weights; start a later expert's
        for t in tiles of e's slab rows: # tile_body, 128 rows per tile
            gather rows      q_g[token_gather[r]] → lhs_vm
            compute          FFN on the tile      → out_vm
            ship             each run → its owner chip's arrivals
drain_transport()                        # wait for all sends/receives; barrier
```

In the toy, chip 1 visits expert 2 (6 rows) then expert 3 (4 rows). Each fits in one tile. At the Qwen shape, an expert averages `T·K / E = 16384 × 10 / 512 = 320` rows, so about 3 tiles.

#### 3b · Gather the tile's rows

The kernel doesn't reorder `q_g`. For each slab row it reads the token number from `token_gather` and DMAs that one row into a VMEM tile buffer (`lhs_vm`). Chip 1, expert 2:

<img
  src="/images/fused_moe_v2/3b.png"
  width="700"
/>

#### 3c · Compute the tile

`expert_ffn_fp8` then `wire_row_and_scale`. The toy shapes for chip 1's expert 2 tile (6 live rows, H = 256, I = 128):

<img
  src="/images/fused_moe_v2/3c.png"
  width="700"
/>

#### 3d · Ship each run home

The finished tile sits in `out_vm`. Each run in it goes to the chip that owns its tokens, landing at the row `recv_base` picked in 1c. A run for the chip's own tokens is a plain local copy; the others are remote DMAs. All of chip 1's shipping in the toy:

<img
  src="/images/fused_moe_v2/3d.png"
  width="700"
/>

There are two shipping routes, picked when the kernel is built (`push_from_vmem`):

| Route | How | Used when |
|---|---|---|
| `ship_tile` (one hop) | Each tile sends its runs straight from `out_vm`. | Every expert is expected to fit in one tile. The toy, and Qwen at up to 512 local tokens. |
| `commit_tile` + `push_expert` (two hops) | Each tile copies its rows to a local HBM buffer (`contrib`). After the expert's last tile, each run is pushed once. | Experts span many tiles. Qwen at 2,048 local tokens. |

The fp8 scale of each result row travels next to it, in a separate small f32 buffer (the "scale mirror"), with its own DMA.

#### 3e · What runs at the same time

The kernel is fast because the matrix unit rarely waits. While tile *t* computes, three kinds of copies run in the background:

- **Next rows:** tile *t+1*'s token rows are gathered into the other half of a double buffer.
- **Next weights:** the weights of an expert up to two visits ahead stream in (3 weight buffers).
- **Previous results:** tile *t−1*'s rows are still on their way to other chips.

At the end, `drain_transport` waits until the row counts in `counts` (from lesson 1) have all been sent and received, then all chips meet at a barrier. Only then is `arrivals` complete.

#### Check yourself

1. Which chip computes t3 → expert 3, and which row of whose arrivals does the result land on? *(Chip 1 computes it at slab row 7; it lands on chip 0's arrival row 9.)*
2. Why does chip 1 never send anything about t4 to chip 0? *(t4 is owned by chip 1, so its results go to chip 1's own arrivals.)*
3. Is the result row already multiplied by the router weight when it is sent? *(No. Step 4 applies it on the owner chip.)*


### Lesson 4 · Combine

**In one sentence:** each chip sums the K rows that came back for each of its tokens, weighted by the router scores (times the fp8 row scale), to get the final output.

After step 3, chip 0's `arrivals` holds 2 result rows for each of t0–t3, but they are scattered across the buffer, still in fp8 and not yet weighted. Step 4 collects them into one output row per token.

Why `2 result rows`? They are the outputs of the two experts each token picked (K = 2). For chip 0's tokens:

| token | picked experts | result row 1 | result row 2 |
| :--- | :--- | :--- | :--- |
| t0 | 0, 2 | FFN₀(t0), at arrival row 0 | FFN₂(t0), at arrival row 4 |
| t1 | 2, 3 | FFN₂(t1), at row 5 | FFN₃(t1), at row 8 |
| t2 | 1, 2 | FFN₁(t2), at row 2 | FFN₂(t2), at row 6 |
| t3 | 3, 0 | FFN₃(t3), at row 9 | FFN₀(t3), at row 1 |

#### Shapes going in and coming out (per chip)

**Input**
- From step 3: `arrivals` `[recv_rows, LB, 128]` fp8 and `arrival_scales` `[mirror rows, 128]`. `arrival_scales` holds one f32 number per result row in `arrivals`: the fp8 scale that result row was quantized with. f32. Toy: arrivals `[12, 2, 128]` · Qwen: `[20992, 32, 128]`
- From step 1: `pos` `[t, K]` i32 (which arrival row), `mirror_pos` `[t, K]` i32 (where its scale is), `topk_weights` `[t, K]` f32.

**Output**
- `out` `[t, H]` in `x`'s dtype (bf16): one row per token this chip owns. Toy: `[4, 256]` · Qwen: `[2048, 4096]`
- Sharded like `x`. Nothing is gathered or reduced across chips afterwards.

#### Files and functions to read

| Where | What it does |
|---|---|
| `layer.py:682–699` in `local_fn` | All of step 4: slice `pos` for my tokens, combine, restore row order. |
| `layer.py:81` `_combine_arrivals` | Builds the coefficients, then picks the path. Its XLA path (the last 10 lines) is the clearest statement of the math. |
| `kernel.py:2156` `combine_step_tokens` | Decides the path: 0 means plain XLA; a positive value means use the Pallas combine kernel. |
| `kernel.py:2180` `_build_combine_kernel`, `:2217` `reduce_token` | The Pallas version: DMA each token's K rows into VMEM, weighted sum, write bf16. |
| `token_parallel.py:80` `TokenReplicaLayout.restore` | Only with tensor parallelism: gathers the TP group's slices back into the original rows. Otherwise it does nothing. |

#### 4a · Find each token's rows

`pos[token, k]` is the arrival row holding the token's k-th result, decided back in lesson 1. For chip 0:

<img
  src="/images/fused_moe_v2/4a.png"
  width="700"
/>

### 4b · Weight and sum

Each arrival row needs two multipliers: its fp8 scale (to undo step 3's wire quantization) and the token's router weight. `_combine_arrivals` multiplies them together first, into one coefficient per (token, k):

```python
row_scales = arrival_scales[mirror_pos]            # [t, K] f32: each row's fp8 scale
coef       = row_scales * topk_weights             # [t, K] f32
out[i]     = sum over k of  coef[i, k] * arrivals[pos[i, k]]   # [H], summed in f32
out        = out.astype(x.dtype)                   # [t, H] bf16
```

### 4c · Two ways to run it

| Path | How | Used when |
|---|---|---|
| XLA | `arrivals[pos]` gather, then K multiply-adds. The last 10 lines of `_combine_arrivals`. | Fewer than 256 local tokens. The toy (t = 4). |
| Pallas combine kernel | For 64 tokens at a time: DMA their K rows each into VMEM, then `reduce_token` does the weighted sum and writes bf16. | `combine_step_tokens(t) > 0`, from 256 local tokens. Qwen at t = 2048. |

Both compute the same thing. The kernel exists because, at large batches, XLA's gather writes the gathered rows out to HBM and reads them straight back before summing. The kernel keeps them in VMEM and overlaps the next 64 tokens' DMAs with the current sum.

### 4d · Back to the framework

`local_fn` returns `out` `[t, H]`. The `shard_map` declares its output as `P("d")`, so globally it is `[T, H]` sharded exactly like `x`. The torch op returns each chip's `[t, H]` slice, and vLLM skips its own reduce-scatter because `vllm_torchtpu/__init__.py` patches `MoERunner._fused_output_is_reduced` to say this output is already reduced.

### The whole trip for t0

| Step | What happens to t0 |
|---|---|
| ① route | Chip 0 picks experts `[0, 2]` with weights `[.62, .38]`. The plan puts t0·e0 at chip 0 slab row 0 and t0·e2 at chip 1 slab row 0, and says their results come back to chip 0's arrival rows 0 and 4. |
| ② all-gather | Chip 0 quantizes t0 (scale 0.004987) and all-gathers it. Chip 1 now has `q_g[0]`. |
| ③ kernel | Chip 0 runs expert 0 on it and copies the result to its own arrival row 0. Chip 1 runs expert 2 on it and DMAs the result to chip 0's arrival row 4. |
| ④ combine | Chip 0 computes `out[t0] = coef[0,0] · arrivals[0] + coef[0,1] · arrivals[4]`. |
