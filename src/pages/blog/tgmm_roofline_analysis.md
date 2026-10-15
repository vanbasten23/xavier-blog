# Roofline Analysis: tgmm_v2 on TPU v7x (ghostfish), test_tgmm_perf_baseline

Measured kernel runtime: 6529.78 us

xw32 tldr: Step 1 and step 2 are mostly enough to determine the roofline.

---

## Step 0 - What does a roofline answer?

The roofline model asks: given the work this kernel MUST do, what is the
fastest it could possibly run on this hardware? Two resources set the limit:

```
T_compute  = FLOPs / peak_FLOPs_per_second
T_memory   = HBM_bytes / peak_HBM_bytes_per_second
T_roofline = max(T_compute, T_memory)
```

You can also look at it through arithmetic intensity (AI):

```
AI          = FLOPs / HBM_bytes
ridge_point = peak_FLOPs_per_second / peak_HBM_bytes_per_second

AI > ridge_point  -> compute-bound
AI < ridge_point  -> memory-bound
```

Roofline efficiency compares the best possible time with the measured time:

```
roofline_efficiency = T_roofline / T_measured
```

---

## Step 1 - Hardware numbers (per device = per TensorCore)

Source: https://docs.cloud.google.com/tpu/docs/tpu7x

Each chip has 2 TensorCores, and each TensorCore is its own JAX device. So the
per-chip numbers are divided by 2:

```
Spec           Per chip        Per device (TC)
-------------  --------------  ---------------
Peak BF16      2307 TFLOP/s    1153.5 TFLOP/s
HBM bandwidth  7380 GB/s       3690 GB/s
Ridge point    -               312.6 FLOP/byte   (= 1153.5e12 / 3690e9)
VMEM           -               64 MiB            (from pltpu.get_tpu_info())
```

Note: the trace confirms that only /device:TPU:0 runs the kernel:

- The xprof roofline tool reports megacore = 0.
- The "HBM Rd+Wr - core 1" counter is 0.

So the TensorCoreMesh("core") axis has size 1 here, and all of the work lands
on one TC. Using per-TC numbers is correct.

---

## Step 2 - Algorithmic (ideal) roofline

Workload:

- m=262144, k=7168, n=1024, G=256.
- All inputs and the output are bf16 (2 bytes).
- The math (reference_tgmm) is out[g] = lhs[rows_g].T @ grad[rows_g].
- Each of the m rows belongs to exactly one group.

```
Quantity           Formula            Value
-----------------  -----------------  ---------------
FLOPs              2 * m * k * n      3.848e12 FLOP
LHS read           m * k * 2          3.758 GB
RHS (grad) read    m * n * 2          0.537 GB
Out write          G * k * n * 2      3.758 GB
Total bytes (min)  sum of the above   8.053 GB

T_compute          3.848e12 / 1153.5e12   = 3336 us
T_memory           8.053e9  / 3690e9      = 2182 us
AI                 3.848e12 / 8.053e9     = 477.9 FLOP/byte  (> ridge 312.6)
```

Result: the kernel is COMPUTE-BOUND. Roofline lower bound = 3336 us.

Roofline efficiency:

```
efficiency = T_roofline / T_measured
           = 3336 us / 6530 us
           = 51%
```

Because the kernel is compute-bound, this is the same as achieved FLOP/s
divided by peak FLOP/s:

```
achieved FLOP/s = 3.848e12 FLOP / 6529.78 us = 589.3 TFLOP/s
efficiency      = 589.3 / 1153.5             = 51.1%
```

---

## Step 3 - Implementation-aware roofline (what THIS kernel design must do)

The ideal roofline assumes perfect data reuse and no wasted MXU work. Next,
look at what the chosen tiling actually costs.

### 3a. Tile sizes (from calculate_tgmm_tiling)

The VMEM limit is 0.9 * 64 MiB = 60.4 MB.

- tile_n = 1024 needs a 77.1 MB budget, which does not fit. So tile_n = 512
  (budget 46.9 MB, which fits).
- tile_k = 7168 (full K).
- tile_m = 512 would need 62.4 MB, which does not fit. So tile_m = 256.
- Result: tm=256, tk=7168, tn=512. This matches the op name
  "...-tm_256-tk_7168-tn_512".
- Grid = (num_n=2, num_k=1, num_gm).

### 3b. Extra HBM traffic: LHS is read twice

The LHS index map does not depend on n_id, and n is the OUTER grid axis. So
the whole LHS (3.76 GB) is streamed once per N-tile, which means 2 times.

```
                                              Bytes      T_memory
--------------------------------------------  ---------  --------
Predicted: 2*LHS + RHS + Out                  11.81 GB   3201 us
Measured (HW counters): 8.17 GB read
                        + 3.758 GB write      11.93 GB   3233 us
```

Notes on the table:

- The extra ~0.12 GB of reads comes from sublane-aligned (16-row) boundary
  rows that are re-read at group edges.
- The AI of this implementation drops to about 326 FLOP/byte, which is only
  just above the ridge point (312.6).
- Memory time and compute time are now close (3.2 ms vs 3.3 ms), so DMA must
  overlap with compute almost perfectly.

### 3c. Extra MXU work: padding at group boundaries

Every gm tile runs a full 256-row matmul, even when the group fills only part
of it. (bucket_base == tile_m, so the bucket switch never shrinks it.)

Computed with the actual get_group_sizes output (min 15, max 1992, no empty
groups):

```
num_gm = sum over groups g of ceil( (size_g + (start_g mod 16)) / 256 )
       = 1157        (vs m / 256 = 1024 with no padding)
```

- Padded FLOPs = 1157 * 256 * k * n * 2 = 4.348e12 FLOP (+13.0%).
- The trace agrees: it shows 2314 = 2 * 1157 pipeline steps.
- T_compute_padded = 4.348e12 / 1153.5e12 = 3769 us.
- Cross-check: padded FLOPs / MXU-busy cycles (from the HW counters) is
  exactly 524,288 FLOP/cycle. So whenever the MXU is busy, it is doing exactly
  this padded work at full rate.

Implementation roofline = max(3769, 3233) = 3769 us.
The measured 6530 us is 57.7% of this bound.

---

## Step 4 - Where the remaining time goes (profile breakdown)

Sources: trace_viewer.json (Tensor Core Sync Flag track) and the
utilization_viewer HW counters.

```
Bucket                                      Time (us)  Cumulative (us)  Evidence
------------------------------------------  ---------  ---------------  ------------------------------------------
Ideal compute                               3336       3336             roofline
+ group padding (+13%)                      +434       3770             num_gm = 1157
+ MXU idle inside compute steps             +~1780     ~5550            MXU busy only 69.4% of cycles;
                                                                        median step 2.4-2.5 us vs 1.63 us ideal
+ DMA waits: output writeback               +631       ~6181            SyncWait:58/59, 512 times;
                                                                        406 of 512 group changes stall ~1 us
+ DMA waits: LHS/RHS input                  +321       ~6502            SyncWait:54-57, mostly right after
                                                                        group changes
+ prologue / epilogue                       +~29       6530             zero-init + fill_metadata + first DMA;
                                                                        first step starts at 21.6 us
```

### Interpreting each gap

1. MXU idle inside steps (largest gap, about 1.8 ms). Each step does more than
   the matmul, and on average that extra work is not fully hidden behind the
   MXU:
   - Two jnp.where masks over the full 256x7168 LHS tile and 256x512 RHS tile.
   - The contraction over dim 0 (lhs.T @ rhs), which needs XLU transposes.
     The XLUs are busy 43.5% of cycles (1.04 M instructions per XLU).
   - "acc += acc_ref[...]" and "acc_ref[...] = acc" on a 7168x512 f32
     accumulator (14.7 MB) every step. (VMEM loads: 8.43 M. VMEM stores:
     3.69 M.)
   - This overhead happens inside the core. In percale/pacchetto it shows up
     as gaps on the vmatmul track.

2. Output writeback stalls (631 us).
   - In a step where the group changes, HBM has to move LHS + RHS (3.93 MB)
     plus a 7.34 MB output tile.
   - That takes at least 3.05 us at 3690 GB/s, but the step's compute is only
     about 2.4 us.
   - So these steps are briefly memory-bound, even though the kernel as a
     whole is compute-bound. The output writes come in bursts instead of
     being spread out evenly.

3. Input waits (321 us). These come mostly right after the output bursts,
   because LHS prefetch is competing with the writeback for HBM bandwidth.

4. HBM utilization = 11.93 GB / 6.53 ms = 1.83 TB/s, which is 49.5% of
   3690 GB/s. The kernel is not saturating HBM on average, which fits the
   compute-bound diagnosis.

---

## Step 5 - Summary

```
Level                                            Time (us)  Efficiency (roofline / measured)
-----------------------------------------------  ---------  --------------------------------
Algorithmic roofline (compute-bound)             3336       51%
Implementation roofline (padding, 2x LHS read)   3769       58%
Measured                                         6530       100%
```

Bottleneck classification:

- The roofline says the kernel is compute-bound.
- The biggest real loss is MXU under-utilization inside each step (VPU
  masking, XLU transposes, accumulator VMEM round-trips).
- Next come bursty output writeback and group-boundary padding.

---

## Open questions (not guessed)

1. Which peak FLOP/s should we use?
   - xprof's built-in roofline tool uses peak_flop_rate = 1028.75 TFLOP/s and
     peak_hbm_bw = 3433 GiB/s (about 3686 GB/s) for GhostFish.
   - The public spec and tpu_info give 1153.5 / 1155 TFLOP/s.
   - This analysis uses the public spec (1153.5).
   - Against xprof's peak, the roofline is 3741 us, which is 57% efficiency.
     That matches xprof's reported roofline_efficiency = 57.3%.

2. Clock.
   - The HW counters total 11.95 M cycles over the 6.53 ms kernel, which is
     about 1.83 G cycles/s. There were also 451 K "LDIDT droop clocks
     skipped".
   - Combined with 524,288 FLOP per MXU-busy cycle, that implies about
     959 TFLOP/s even if the MXU were busy 100% of the time. That is below
     both peak numbers.
   - The nominal v7x TC clock, and whether this run was droop-throttled, are
     not confirmed.

---

## Appendix - How to reproduce the profile numbers

- Kernel time and step timeline: trace_viewer.json, track
  "Tensor Core Sync Flag".
  - SyncWait:54/55 = LHS buffer slots.
  - SyncWait:56/57 = RHS buffer slots.
  - SyncWait:58/59 = output buffer slots.
- HW counters (MXU/XLU busy, VMEM loads/stores, HBM bytes):
  utilization_viewer.json.
- xprof's own roofline: roofline_model.json.
- Fetch method: XprofAnalysisClient().get_profile_data(<tool_name>, session_id)
  from google3.perftools.accelerators.xprof.api.python.xprof_analysis_client.
