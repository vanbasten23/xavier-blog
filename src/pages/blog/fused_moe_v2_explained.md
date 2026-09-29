---
layout: ../../layouts/Layout.astro
title: "Understanding fused_moe_v2 Pallas kernel"
description: "An intuitive explanation of the fused_moe_v2 Pallas kernel"
pubDate: 2026-09-22
---

# Understanding the fused_moe_v2 Pallas kernel

I will use this post to explain the [fused_moe_v2 Pallas TPU kernel](https://github.com/vllm-project/vllm-torchtpu/tree/862433c874223c2983bd93f2b716cd16d52b78ec/src/vllm_torchtpu/kernels/fused_moe/v2). You can find the unit tests [here](https://github.com/vllm-project/vllm-torchtpu/tree/862433c874223c2983bd93f2b716cd16d52b78ec/tests/kernels/fused_moe).

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

![Fused MoE overview](/images/fused_moe_v2/data_path_for_token_x.png)

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

![1b](/images/fused_moe_v2/step1_1b.png)

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

┌────────────┬──────────────────────────────────┬──────────────────────┐
│  Section   │         Content (chip 0)         │        Length        │
├────────────┼──────────────────────────────────┼──────────────────────┤
│ picks      │ `[0,2, 2,3, 1,2, 3,0]` + 120 zeros │ align(t·K = 8) = 128 │
├────────────┼──────────────────────────────────┼──────────────────────┤
│ row scales │ `[s0, s1, s2, s3]` + 124 zeros     │ align(t = 4) = 128   │
├────────────┼──────────────────────────────────┼──────────────────────┤
│ counts     │ `[2, 1, 3, 2]`                     │ E = 4                │
└────────────┴──────────────────────────────────┴──────────────────────┘

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

Output: pairs_g
Shape: [2, 8] → reshape [T, K] = [8, 2]
Value: [[0,2],[2,3],[1,2],[3,0],[2,0],[3,1],[0,2],[1,3]]

────────────────────────────────────────

Output: scale_bits_g
Shape: [T] = [8]
Value: [s0 … s7]

────────────────────────────────────────

Output: rows_by_dest (counts)
Shape: [ep, E] = [2, 4]
Value: [[2,1,3,2], [2,2,2,2]]

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
