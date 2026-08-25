/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM90 bf16 block-sparse attention for OSA's replicated frame-to-frame plans.
//
// STATUS (2026-08-25): experimental — correct (max |diff| ~5e-4 vs the
// Triton reference at real shapes) but at ~350 TFLOP/s vs the Triton
// kernel's ~480 (0.73x). The Triton kernel remains the production path.
//
// Iteration log (720p Self-Forcing shapes, kv 27..81 frames):
//   v1 serial GMMA loop                            265 TF/s
//   v2 frame-interleaved QK/PV                     265
//   v3 producer WG + mbarriers (no syncthreads)    343
//   v4/v5 split K/V-free barriers, counted waits   309-317 (reverted)
//   v6 = v3 + RS PV (P in regs) + kStages=5        353   <- current
//   v7 Triton-shaped independent CTAs, 2 CTA/SM    252 (launch_bounds spills)
//   v8 cross-block QK prefetch                     163 (register blowup)
//
// Why the gap: at production grid sizes all same-head CTAs hit L2 (DRAM ~5%
// busy), so this kernel's halved K/V traffic buys little, while its
// pipeline coordination (barriers, producer) costs what Triton's fully
// independent programs never pay. Closing the remaining 1.35x needs TMA +
// FA3-grade warp-specialized ping-pong scheduling.
//
// The plan gives every (head, query tile) the same number of full 64-token
// key blocks (absolute token starts, `starts[h, q_tile, n]`), and the same
// plan row serves every query frame of the chunk. One CTA folds TWO query
// frames over a single K/V stream (Q resident in smem, one accumulator set
// per frame); both matmuls run as SM90 GMMA.
//
// Layout: one CTA = (query-frame pair, 128-row query tile, head).
//   * 2 consumer warpgroups; warpgroup w owns query rows [64w, 64w+64) of
//     both folded frames.
//   * K/V blocks stream through a cp.async ring (kStages stages); each stage
//     holds one 64x128 bf16 K tile and V tile.
//   * QK: GMMA SS (Q and K in smem, both K-major).  PV: GMMA RS (P in
//     registers, V in smem MN-major -- its natural [tokens, dim] layout).
//   * Online softmax per frame with exp2-domain running max/sum, exactly the
//     flash recurrence.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, bf16_t
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>



#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace osa_sm90 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------
static constexpr int kBlockM = 64;    // query rows per warpgroup
static constexpr int kBlockN = 64;    // key tokens per plan block
static constexpr int kHeadDim = 128;  // model head dim (fixed)
static constexpr int kQTile = 128;    // plan query tile (2 warpgroups)
static constexpr int kStages = 5;     // cp.async ring depth
static constexpr int kNumThreads = 384;  // 2 consumer warpgroups + 1 producer
static constexpr int kNumConsumers = 256;

// GMMA tiled MMAs.
// QK: A = Q (M=64, K=128) K-major smem; B = Kblk (N=64, K=128) K-major smem.
using TiledMmaQK = decltype(make_tiled_mma(
    GMMA::ss_op_selector<bf16, bf16, float, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                         GMMA::Major::K, GMMA::Major::K>()));
// PV: A = P (M=64, K=64) in registers (softmaxed QK accumulator,
// re-laid out); B = V (N=128, K=64) MN-major smem (natural [tokens, dim]
// storage: dim contiguous).
using TiledMmaPV = decltype(make_tiled_mma(
    GMMA::rs_op_selector<bf16, bf16, float, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>,
                         GMMA::Major::K, GMMA::Major::MN>()));

// Reinterpret a QK accumulator ((2,2,V),MMA_M,MMA_N) as the A-operand
// fragment layout ((2,2,2),MMA_M,(V/2,MMA_N)) for the PV RS GMMA (the
// standard SM90 flash-attention conversion).
template <typename Layout>
__forceinline__ __device__ auto acc_to_a_regs(Layout acc_layout) {
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});
  return make_layout(
      make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
      coalesce(make_layout(get<2, 1>(l), get<2>(acc_layout))));
}

// Smem layouts.
// Non-swizzled INTER atoms: the bundled cutlass double-applies swizzles on
// this write-through-functor + read-through-descriptor path (one-hot probe),
// so the swizzle-free canonical layouts are used. Step orders per GMMA
// convention: Step<_1,_2> K-major, Step<_2,_1> MN-major.
using SmemLayoutQ = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<bf16>{},
                                           Shape<Int<kQTile>, Int<kHeadDim>>{},
                                           Step<_1, _2>{}));
using SmemLayoutK = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<bf16>{},
                                           Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                           Step<_1, _2>{}));
// V viewed as the PV B operand (N=dim, K=tokens), MN-major over the same
// [tokens, dim] bytes: dim runs fastest.
using SmemLayoutV = decltype(tile_to_shape(GMMA::Layout_MN_INTER_Atom<bf16>{},
                                           Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                           Step<_2, _1>{}));

struct SharedStorage {
  alignas(1024) bf16 q[2][kQTile * kHeadDim];         // both folded frames
  alignas(1024) bf16 k[kStages][kBlockN * kHeadDim];  // K ring
  alignas(1024) bf16 v[kStages][kBlockN * kHeadDim];  // V ring
  cutlass::arch::ClusterTransactionBarrier bar_ready[kStages];  // producer -> consumers
  cutlass::arch::ClusterBarrier bar_free_k[kStages];  // consumers done with K
  cutlass::arch::ClusterBarrier bar_free_v[kStages];  // consumers done with V
};

// ---------------------------------------------------------------------------
// cp.async helpers (16B, cache-global)
// ---------------------------------------------------------------------------
__forceinline__ __device__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}
__forceinline__ __device__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__forceinline__ __device__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Producer-warpgroup 64x128 bf16 tile load: 128 threads move eight 16B
// vectors each. Gmem rows are tokens (dim contiguous); kTransposed selects
// whether the smem layout is (token, dim) or (dim, token).
template <typename SmemLayout, bool kTransposed>
__forceinline__ __device__ void load_tile_async_wg(bf16* smem, const bf16* gmem,
                                                   int64_t gmem_row_stride, int lane) {
  SmemLayout layout{};
#pragma unroll
  for (int it = 0; it < (kBlockN * kHeadDim / 8) / 128; ++it) {
    const int vec = lane + it * 128;          // 16B vector index
    const int row = vec / (kHeadDim / 8);     // token row
    const int col = (vec % (kHeadDim / 8)) * 8;
    bf16* dst = smem + (kTransposed ? layout(col, row) : layout(row, col));
    cp_async_16(dst, gmem + int64_t(row) * gmem_row_stride + col);
  }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
struct OsaParams {
  const bf16* __restrict__ q;  // [q_len, heads, dim]
  const bf16* __restrict__ k;  // [kv_len, heads, dim]
  const bf16* __restrict__ v;  // [kv_len, heads, dim]
  bf16* __restrict__ out;      // [q_len, heads, dim]
  const int32_t* __restrict__ starts;  // [heads, q_tiles, n_blocks]
  int num_heads;
  int q_tiles_per_frame;
  int num_q_frames;
  int frame_seqlen;
  int n_blocks;
  float scale_log2e;  // softmax_scale * log2(e)
};

__global__ __launch_bounds__(kNumThreads) void osa_block_sparse_kernel(
    __grid_constant__ const OsaParams p) {
  extern __shared__ char smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int frame_pairs = (p.num_q_frames + 1) / 2;
  const int q_tile = blockIdx.x / frame_pairs;
  const int pair = blockIdx.x % frame_pairs;
  const int head = blockIdx.y;
  const int frame0 = 2 * pair;
  const int frame1 = 2 * pair + 1;
  const bool has_f1 = frame1 < p.num_q_frames;

  const int wg = threadIdx.x / 128;        // 0,1 consumers (query-row half); 2 producer
  const int lane_in_wg = threadIdx.x % 128;

  // Stage barriers: ready counts the producer's 128 cp.async-completing
  // threads; free counts the 256 consumer threads done with a stage.
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < kStages; ++i) {
      smem.bar_ready[i].init(128);
      smem.bar_free_k[i].init(kNumConsumers);
      smem.bar_free_v[i].init(kNumConsumers);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  const int64_t row_stride = int64_t(p.num_heads) * kHeadDim;  // tokens are rows
  const int tile_row0 = q_tile * kQTile;
  // The frame's last query tile can be short; masked rows load frame row 0
  // (harmless: their outputs are never stored).
  auto q_gmem_row = [&](int frame, int r) {
    const int local = tile_row0 + r < p.frame_seqlen ? tile_row0 + r : 0;
    return int64_t(frame) * p.frame_seqlen + local;
  };

  const int32_t* plan = p.starts + (int64_t(head) * p.q_tiles_per_frame + q_tile) * p.n_blocks;

  // ---- producer warpgroup: stream K/V blocks through the stage ring ----
  if (wg == 2) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    int phase_free[kStages];
#pragma unroll
    for (int i = 0; i < kStages; ++i) phase_free[i] = 0;  // first wait: phase-0 completion
    for (int block = 0; block < p.n_blocks; ++block) {
      const int stage = block % kStages;
      const int32_t start = __ldg(plan + block);
      const bf16* k_src = p.k + (int64_t(start) * p.num_heads + head) * kHeadDim;
      const bf16* v_src = p.v + (int64_t(start) * p.num_heads + head) * kHeadDim;
      if (block >= kStages) {
        smem.bar_free_k[stage].wait(phase_free[stage]);
      }
      load_tile_async_wg<SmemLayoutK, false>(smem.k[stage], k_src, row_stride, lane_in_wg);
      if (block >= kStages) {
        smem.bar_free_v[stage].wait(phase_free[stage]);
        phase_free[stage] ^= 1;
      }
      load_tile_async_wg<SmemLayoutV, true>(smem.v[stage], v_src, row_stride, lane_in_wg);
      // Arrival fires when this thread's cp.asyncs complete; 128 arrivals
      // flip the barrier.
      cutlass::arch::cpasync_barrier_arrive_noinc(
          reinterpret_cast<uint64_t*>(&smem.bar_ready[stage]));
    }
    return;
  }

  // ---- consumers: load Q for both frames (cp.async, 256 threads) ----
  {
    SmemLayoutQ layout{};
    const int tid = threadIdx.x;
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      if (f == 1 && !has_f1) break;
      const int frame = f == 0 ? frame0 : frame1;
#pragma unroll
      for (int it = 0; it < (kQTile * kHeadDim / 8) / kNumConsumers; ++it) {
        const int vec = tid + it * kNumConsumers;
        const int row = vec / (kHeadDim / 8);
        const int col = (vec % (kHeadDim / 8)) * 8;
        bf16* dst = smem.q[f] + layout(row, col);
        const bf16* src = p.q + (q_gmem_row(frame, row) * p.num_heads + head) * kHeadDim + col;
        cp_async_16(dst, src);
      }
    }
    cp_async_commit();
    cp_async_wait<0>();
    cutlass::arch::fence_view_async_shared();
  }
  cutlass::arch::warpgroup_reg_alloc<224>();

  // ---- register state ----
  TiledMmaQK mma_qk;
  TiledMmaPV mma_pv;
  auto thr_qk = mma_qk.get_slice(lane_in_wg);
  auto thr_pv = mma_pv.get_slice(lane_in_wg);

  Tensor rO0 = partition_fragment_C(mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor rO1 = partition_fragment_C(mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(rO0);
  clear(rO1);
  float rM[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
  float rL[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

  Tensor sQ0 = make_tensor(make_smem_ptr(smem.q[0]), SmemLayoutQ{});
  Tensor sQ1 = make_tensor(make_smem_ptr(smem.q[1]), SmemLayoutQ{});
  // Each warpgroup's 64-row half of the 128-row query tile.
  Tensor sQ0_wg = local_tile(sQ0, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(wg, 0));
  Tensor sQ1_wg = local_tile(sQ1, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(wg, 0));

  // Wait for Q (it was in the first commit group; K/V primes follow).
  // wait_group<prime> would be dynamic; conservatively wait for all issued
  // groups down to the first K/V stage being ready before the loop instead.

  // Frame-staged compute pieces: QK commit (async), then softmax + P staging
  // + PV commit. Interleaving the two frames overlaps QK(f1) with
  // softmax(f0) and PV(f0) with softmax(f1).
  Tensor rS0 = partition_fragment_C(mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor rS1 = partition_fragment_C(mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});

  auto commit_qk = [&](auto frame_tag, int stage) {
    constexpr int F = decltype(frame_tag)::value;
    Tensor sK = make_tensor(make_smem_ptr(smem.k[stage]), SmemLayoutK{});
    auto& sQ_wg = F == 0 ? sQ0_wg : sQ1_wg;
    auto& rS = F == 0 ? rS0 : rS1;
    Tensor sA = thr_qk.partition_fragment_A(sQ_wg);
    Tensor sB = thr_qk.partition_fragment_B(sK);
    clear(rS);
    warpgroup_fence_operand(rS);
    warpgroup_arrive();
    gemm(mma_qk, sA, sB, rS);
    warpgroup_commit_batch();
  };

  auto softmax_and_pv = [&](auto frame_tag, int stage) {
    constexpr int F = decltype(frame_tag)::value;
    Tensor sV = make_tensor(make_smem_ptr(smem.v[stage]), SmemLayoutV{});
    auto& rO = F == 0 ? rO0 : rO1;
    auto& rS = F == 0 ? rS0 : rS1;
    warpgroup_fence_operand(rS);

    // Per-element (row, col) coordinates straight from the tiled MMA — no
    // hand-derived fragment model. Each thread's fragment covers exactly two
    // distinct rows; `row_slot(i)` maps element i to its running-state slot.
    Tensor cS = thr_qk.partition_C(
        make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
    const int row0 = get<0>(cS(0));
    int row1 = row0;
#pragma unroll
    for (int i = 1; i < size(cS); ++i) {
      const int r = get<0>(cS(i));
      if (r != row0) row1 = r;
    }
    auto row_slot = [&](int i) { return get<0>(cS(i)) == row0 ? 0 : 1; };

    // Online softmax (exp2 domain).
    float cur[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int i = 0; i < size(rS); ++i) cur[row_slot(i)] = fmaxf(cur[row_slot(i)], rS(i));
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      cur[slot] = fmaxf(cur[slot], __shfl_xor_sync(0xffffffff, cur[slot], 1));
      cur[slot] = fmaxf(cur[slot], __shfl_xor_sync(0xffffffff, cur[slot], 2));
      cur[slot] *= p.scale_log2e;
    }
    float new_max[2], rescale[2], sum[2] = {0.f, 0.f};
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      new_max[slot] = fmaxf(rM[F][slot], cur[slot]);
      rescale[slot] = exp2f(rM[F][slot] - new_max[slot]);
      rM[F][slot] = new_max[slot];
    }
#pragma unroll
    for (int i = 0; i < size(rS); ++i) {
      const float e = exp2f(rS(i) * p.scale_log2e - new_max[row_slot(i)]);
      rS(i) = e;
      sum[row_slot(i)] += e;
    }
#pragma unroll
    for (int slot = 0; slot < 2; ++slot)
      rL[F][slot] = rL[F][slot] * rescale[slot] + sum[slot];

    // rO rows follow the PV mma's coordinates (same m64 layout family, but
    // derived independently to stay assumption-free).
    Tensor cO = thr_pv.partition_C(
        make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
#pragma unroll
    for (int i = 0; i < size(rO); ++i)
      rO(i) *= rescale[get<0>(cO(i)) == row0 ? 0 : 1];

    // P straight from registers: reinterpret the (softmaxed) QK accumulator
    // as the RS A-operand fragment and convert to bf16 in place.
    {
      Tensor rP_acc = make_tensor(rS.data(), acc_to_a_regs(rS.layout()));
      Tensor rP = make_tensor<bf16>(rP_acc.layout());
#pragma unroll
      for (int i = 0; i < size(rP_acc); ++i) rP(i) = bf16(rP_acc(i));
      Tensor sB = thr_pv.partition_fragment_B(sV);
      warpgroup_fence_operand(rP);
      warpgroup_fence_operand(rO);
      warpgroup_arrive();
      gemm(mma_pv, rP, sB, rO);
      warpgroup_commit_batch();
    }
  };

  // ---- consumer main loop: barrier-paced, no CTA-wide syncs ----
  // wgmma groups retire in commit order, so counted waits give exact
  // lifetimes: after `wait<2>` the previous block's two PVs are done (only
  // this block's two QKs remain), after the first `wait<1>` QK(f0) is done,
  // after the second QK(f1). PV(f0)/PV(f1) of block b drain during block
  // b+1's barrier wait and QK commits.
  int phase_ready[kStages];
#pragma unroll
  for (int i = 0; i < kStages; ++i) phase_ready[i] = 0;
  int pending_v_stage = -1;
  for (int block = 0; block < p.n_blocks; ++block) {
    const int stage = block % kStages;
    smem.bar_ready[stage].wait(phase_ready[stage]);
    phase_ready[stage] ^= 1;

    if (has_f1) {
      commit_qk(cute::C<0>{}, stage);
      commit_qk(cute::C<1>{}, stage);
      warpgroup_wait<2>();  // prior block's PVs retired
      warpgroup_fence_operand(rO0);
      warpgroup_fence_operand(rO1);
      if (pending_v_stage >= 0) smem.bar_free_v[pending_v_stage].arrive();
      warpgroup_wait<1>();                  // QK(f0) done
      softmax_and_pv(cute::C<0>{}, stage);  // commits PV(f0)
      warpgroup_wait<1>();                  // QK(f1) done; PV(f0) in flight
      softmax_and_pv(cute::C<1>{}, stage);  // commits PV(f1)
      smem.bar_free_k[stage].arrive();
    } else {
      commit_qk(cute::C<0>{}, stage);
      warpgroup_wait<1>();  // prior PV retired
      warpgroup_fence_operand(rO0);
      if (pending_v_stage >= 0) smem.bar_free_v[pending_v_stage].arrive();
      warpgroup_wait<0>();  // QK done
      softmax_and_pv(cute::C<0>{}, stage);
      smem.bar_free_k[stage].arrive();
    }
    pending_v_stage = stage;
  }
  // Drain the last block's PVs.
  warpgroup_wait<0>();
  warpgroup_fence_operand(rO0);
  warpgroup_fence_operand(rO1);
  if (pending_v_stage >= 0) smem.bar_free_v[pending_v_stage].arrive();

  // ---- epilogue: normalise and store (coordinates from the PV mma) ----
  // rL held per-thread partial sums (each row's 64 columns spread over the
  // 4 quad lanes); the rescale factors were quad-uniform, so one reduction
  // here yields the full row sums.
#pragma unroll
  for (int f = 0; f < 2; ++f) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      rL[f][slot] += __shfl_xor_sync(0xffffffff, rL[f][slot], 1);
      rL[f][slot] += __shfl_xor_sync(0xffffffff, rL[f][slot], 2);
    }
  }
  Tensor cO = thr_pv.partition_C(
      make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  const int erow0 = get<0>(cO(0));
  int erow1 = erow0;
  for (int i = 1; i < size(cO); ++i) {
    const int r = get<0>(cO(i));
    if (r != erow0) erow1 = r;
  }
#pragma unroll
  for (int f = 0; f < 2; ++f) {
    if (f == 1 && !has_f1) break;
    const int frame = f == 0 ? frame0 : frame1;
    auto& rO = f == 0 ? rO0 : rO1;
#pragma unroll
    for (int i = 0; i < size(rO); i += 2) {
      const int r_local = get<0>(cO(i));
      const int slot = r_local == erow0 ? 0 : 1;
      const float inv = rL[f][slot] > 0.f ? 1.f / rL[f][slot] : 0.f;
      const int global_row_in_frame = tile_row0 + wg * kBlockM + r_local;
      if (global_row_in_frame >= p.frame_seqlen) continue;
      bf16* out_row =
          p.out + ((int64_t(frame) * p.frame_seqlen + global_row_in_frame) * p.num_heads + head) * kHeadDim;
      const int col = get<1>(cO(i));
      bf16 packed[2] = {bf16(rO(i) * inv), bf16(rO(i + 1) * inv)};
      *reinterpret_cast<uint32_t*>(out_row + col) = *reinterpret_cast<uint32_t*>(packed);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
inline void osa_block_sparse(tvm::ffi::TensorView out, tvm::ffi::TensorView q,
                             tvm::ffi::TensorView k, tvm::ffi::TensorView v,
                             tvm::ffi::TensorView starts, int64_t q_tiles_per_frame,
                             int64_t num_q_frames, int64_t frame_seqlen, double scale) {
  using namespace host;
  SymbolicSize QLen{"q_len"}, KvLen{"kv_len"}, Heads{"heads"}, Dim{"dim"};
  SymbolicSize PHeads{"plan_heads"}, PTiles{"plan_q_tiles"}, PBlocks{"n_blocks"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({QLen, Heads, Dim})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(q)
      .verify(out);
  TensorMatcher({KvLen, Heads, Dim})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(k)
      .verify(v);
  TensorMatcher({PHeads, PTiles, PBlocks}).with_device<kDLCUDA>(device).verify(starts);

  RuntimeCheck(Dim.unwrap() == kHeadDim, "osa_block_sparse: head_dim must be ", kHeadDim);
  RuntimeCheck(PHeads.unwrap() == Heads.unwrap(), "plan heads mismatch");
  RuntimeCheck(PTiles.unwrap() == (size_t)q_tiles_per_frame, "plan q_tiles mismatch");
  RuntimeCheck(QLen.unwrap() == (size_t)(num_q_frames * frame_seqlen),
               "q_len must equal num_q_frames * frame_seqlen");

  OsaParams p;
  p.q = static_cast<const bf16*>(q.data_ptr());
  p.k = static_cast<const bf16*>(k.data_ptr());
  p.v = static_cast<const bf16*>(v.data_ptr());
  p.out = static_cast<bf16*>(out.data_ptr());
  p.starts = static_cast<const int32_t*>(starts.data_ptr());
  p.num_heads = (int)Heads.unwrap();
  p.q_tiles_per_frame = (int)q_tiles_per_frame;
  p.num_q_frames = (int)num_q_frames;
  p.frame_seqlen = (int)frame_seqlen;
  p.n_blocks = (int)PBlocks.unwrap();
  p.scale_log2e = (float)(scale * 1.4426950408889634);

  const int frame_pairs = (p.num_q_frames + 1) / 2;
  const dim3 grid(q_tiles_per_frame * frame_pairs, p.num_heads);
  const size_t smem = sizeof(SharedStorage);
  static bool configured = false;
  if (!configured) {
    RuntimeDeviceCheck(cudaFuncSetAttribute(osa_block_sparse_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  LaunchKernel(grid, kNumThreads, device.unwrap(), smem)(osa_block_sparse_kernel, p);
}

}  // namespace osa_sm90
