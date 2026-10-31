#define MALI_SCORE_OFFSET 176104
#define MALI_TRACE_OFFSET 191104
#ifndef MALI_SCORE_TRACE
#define MALI_SCORE_TRACE 0
#endif
#define SCORE_PTR(out) ((__global float *)((__global uchar *)(out)+MALI_SCORE_OFFSET))
#define TRACE_PTR(out) ((__global float *)((__global uchar *)(out)+MALI_TRACE_OFFSET))
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL FP_CONTRACT OFF
#ifndef WG
#define WG 256
#endif
#define H 640
#define G 2560
#define MAX_TOKENS 3750
#ifndef TDT_KERNEL
#define TDT_KERNEL tdt_persistent
#define TDT_RESUME 0
#endif
typedef struct {
  int frame, frames, token, update, count, steps, on_frame, done;
  float h0[H], h1[H], c0[H], c1[H], pred[H], g0[G], g1[G], logits[8200];
  char4 q0[160], q1[160], qj[160];
  float scale0, scale1, scalej;
  int zp0, zp1, zpj;
} State;

inline float pf(__global const uchar *restrict w, ulong off, int i) {
  return ((__global const float *)(w + off))[i];
}
inline int pi(__global const uchar *restrict w, ulong off, int i) {
  return ((__global const int *)(w + off))[i];
}

// One work-group owns all recurrent state. No cross-group spin/barrier.
inline void quant(__local float *x, __local char4 *q, __local float *scale,
                  __local int *zp, __local float *lo, __local float *hi) {
  int l = get_local_id(0);
  float a = 0, b = 0;
  for (int i = l; i < H; i += WG) {
    a = fmin(a, x[i]);
    b = fmax(b, x[i]);
  }
  lo[l] = a;
  hi[l] = b;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG / 2; s; s /= 2) {
    if (l < s) {
      lo[l] = fmin(lo[l], lo[l + s]);
      hi[l] = fmax(hi[l], hi[l + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) {
    float v = (hi[0] - lo[0]) / 255.0f;
    scale[0] = v == 0 ? 1.0f : v;
    zp[0] = clamp(convert_int_rte(-lo[0] / scale[0]), 0, 255);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = l; i < H / 4; i += WG) {
    int4 v = convert_int4_rte(vload4(i, x) / scale[0]) + (int4)(zp[0]);
    v = clamp(v, (int4)(0), (int4)(255)) - (int4)(128);
    q[i] = convert_char4(v);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

inline int dot640(__global const uchar *restrict w, ulong off, int n, int out,
                  __local char4 *q, int zp, ulong sum) {
  __global const char16 *matrix = (__global const char16 *)(w + off);
  int4 a = (int4)(0);
#pragma unroll 4
  for (int k = 0; k < 40; ++k) {
    char16 x = vload16(k, (__local char *)q), b = matrix[k * n + out];
    a.x = arm_dot_acc(x.s0123, b.s0123, a.x);
    a.y = arm_dot_acc(x.s4567, b.s4567, a.y);
    a.z = arm_dot_acc(x.s89ab, b.s89ab, a.z);
    a.w = arm_dot_acc(x.scdef, b.scdef, a.w);
  }
  return a.x + a.y + a.z + a.w + (128 - zp) * pi(w, sum, out);
}
#if NATIVE_MATH
inline float sigmoid(float x) { return 1.0f / (1.0f + native_exp(-x)); }
inline float fast_tanh(float x) {
  float a = fabs(x);
  if (a < 0.0001f)
    return x;
  float e = native_exp(-2.0f * a);
  return copysign((1.0f - e) / (1.0f + e), x);
}
#define tanh fast_tanh
#else
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
#endif
inline int4 gates_dot(__global const uchar *restrict w, ulong off, int i,
                      __local char4 *q, int zp, ulong sum) {
  __global const char4 *matrix = (__global const char4 *)(w + off);
  int4 a = (int4)(0);
  for (int k = 0; k < 160; ++k) {
    char4 x = q[k];
    int p = k * G + i;
    a.x = arm_dot_acc(x, matrix[p], a.x);
    a.y = arm_dot_acc(x, matrix[p + H], a.y);
    a.z = arm_dot_acc(x, matrix[p + 2 * H], a.z);
    a.w = arm_dot_acc(x, matrix[p + 3 * H], a.w);
  }
  return a + (128 - zp) * (int4)(pi(w, sum, i), pi(w, sum, i + H),
                                 pi(w, sum, i + 2 * H), pi(w, sum, i + 3 * H));
}
inline float4 gate_params(__global const uchar *restrict w, ulong off, int i) {
  return (float4)(pf(w, off, i), pf(w, off, i + H), pf(w, off, i + 2 * H),
                  pf(w, off, i + 3 * H));
}

inline void affine(__global const uchar *restrict w, ulong W, ulong S, ulong B,
                   ulong SUM, int N, __local char4 *q, float scale, int zp,
                   __local float *dst, int add, int embedding) {
  int l = get_local_id(0);
  for (int o = l; o < N; o += WG) {
    int v = dot640(w, W, N, o, q, zp, SUM);
    {
      float f = convert_float(v) * (scale * pf(w, S, o)) + pf(w, B, o);
      if (embedding >= 0)
        f = pf(w, EMBED_GATES, embedding * G + o) + f;
      if (add)
        dst[o] += f;
      else
        dst[o] = f;
    }
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1))) __kernel void
TDT_KERNEL(__global const half *restrict encoder, int frames,
           __global const uchar *restrict w, __global int *restrict out,
           __global float *restrict debug, int debug_mode,
           __global State *restrict state, int resume_mode,
           __global const int *meta, int gpu_front) {
  if (gpu_front)
    frames = meta[2];
  const int resume = TDT_RESUME;
  const int l = get_local_id(0);
  __local float h0[H], h1[H], c0[H], c1[H], pred[H], joint[H], gates[G];
  __local char4 q0[160], q1[160];
  __local float s0[1], s1[1], lo[WG], hi[WG], duration_logits[5], best[WG], sums[WG];
  __local int z0[1], z1[1], indices[WG];
  __local int token, frame, steps, count, on_frame, update, done;
  for (int i = l; i < H; i += WG) {
    h0[i] = resume ? state->h0[i] : 0;
    h1[i] = resume ? state->h1[i] : 0;
    c0[i] = resume ? state->c0[i] : 0;
    c1[i] = resume ? state->c1[i] : 0;
    pred[i] = resume ? state->pred[i] : 0;
  }
  if (l == 0) {
    token = resume ? state->token : 8192;
    frame = resume ? state->frame : 0;
    steps = resume ? state->steps : 0;
    count = resume ? state->count : 0;
    on_frame = resume ? state->on_frame : 0;
    update = resume ? state->update : 1;
    done = resume ? state->done : 0;
    if (!resume)
      out[0] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  for (int i = l; i < frames * H; i += WG)
    if (!isfinite(convert_float(encoder[i])))
      atomic_or(out, 2);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  while (frame < frames && steps < 4128) {
    if (update) {
      quant(h0, q0, s0, z0, lo, hi);
      for (int o = l; o < G; o += WG) {
        int a = dot640(w, H0_W, G, o, q0, z0[0], H0_SUM);
        gates[o] =
            pf(w, EMBED_GATES, token * G + o) +
            (convert_float(a) * (s0[0] * pf(w, H0_S, o)) + pf(w, H0_B, o));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int i = l; i < H; i += WG) {
        float v = sigmoid(gates[H + i]) * c0[i] +
                  sigmoid(gates[i]) * tanh(gates[2 * H + i]);
        c0[i] = v;
        h0[i] = sigmoid(gates[3 * H + i]) * tanh(v);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      quant(h0, q0, s0, z0, lo, hi);
      quant(h1, q1, s1, z1, lo, hi);
      for (int o = l; o < G; o += WG) {
        int a = dot640(w, X1_W, G, o, q0, z0[0], X1_SUM),
            b = dot640(w, H1_W, G, o, q1, z1[0], H1_SUM);
        gates[o] =
            (convert_float(a) * (s0[0] * pf(w, X1_S, o)) + pf(w, X1_B, o)) +
            (convert_float(b) * (s1[0] * pf(w, H1_S, o)) + pf(w, H1_B, o));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int i = l; i < H; i += WG) {
        float v = sigmoid(gates[H + i]) * c1[i] +
                  sigmoid(gates[i]) * tanh(gates[2 * H + i]);
        c1[i] = v;
        h1[i] = sigmoid(gates[3 * H + i]) * tanh(v);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      quant(h1, q0, s0, z0, lo, hi);
      for (int o = l; o < H; o += WG) {
        int v = dot640(w, PRED_W, H, o, q0, z0[0], PRED_SUM);
        pred[o] =
            convert_float(v) * (s0[0] * pf(w, PRED_S, o)) + pf(w, PRED_B, o);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (debug_mode && steps == 0) {
        for (int i = l; i < H; i += WG) {
          debug[i] = h0[i];
          debug[H + i] = c0[i];
          debug[2 * H + i] = h1[i];
          debug[3 * H + i] = c1[i];
          debug[4 * H + i] = pred[i];
        }
      }
    }
    for (int i = l; i < H; i += WG)
      joint[i] = fmax(convert_float(encoder[frame * H + i]) + pred[i], 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    quant(joint, q0, s0, z0, lo, hi);
    float v = -INFINITY;
    float lane_sum = 0.0f;
    int index = 2147483647;
    for (int o = l; o < 8198; o += WG) {
      int value = dot640(w, OUT_W, 8200, o, q0, z0[0], OUT_SUM);
      float r = convert_float(value) * (s0[0] * pf(w, OUT_S, o)) + pf(w, OUT_B, o);
      if (!isfinite(r)) atomic_or(out, 4);
      if (debug_mode && steps == 0) debug[5 * H + o] = r;
      if (o < 8193) {
        float m = fmax(v, r);
        lane_sum = lane_sum * exp(v-m) + exp(r-m);
#if MALI_SCORE_TRACE
        if (count == 0) TRACE_PTR(out)[o] = r;
#endif
        if (r > v || (r == v && o < index)) { v = r; index = o; }
      } else duration_logits[o - 8193] = r;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    best[l] = v;
    indices[l] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n = WG / 2; n; n /= 2) {
      if (l < n && (best[l + n] > best[l] ||
                    (best[l + n] == best[l] && indices[l + n] < indices[l]))) {
        best[l] = best[l + n];
        indices[l] = indices[l + n];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    sums[l] = lane_sum * exp(v-best[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = WG / 2; k; k /= 2) {
      if (l < k) sums[l] += sums[l + k];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
      int y = indices[0], skip = 0;
      for (int i = 1; i < 5; ++i)
        if (duration_logits[i] > duration_logits[skip])
          skip = i;
      if (y != 8192) {
        if (count >= MAX_TOKENS) {
          atomic_or(out, 8);
          done = 1;
        } else {
          out[8 + count * 3] = y;
          out[8 + count * 3 + 1] = frame;
          out[8 + count * 3 + 2] = skip;
          SCORE_PTR(out)[count] = -log(sums[0]);
          ++count;
        }
        token = y;
        on_frame++;
        update = 1;
      } else
        update = 0;
      if (skip > 0)
        on_frame = 0;
      if (on_frame >= 10) {
        on_frame = 0;
        skip = 1;
      }
      if (y == 8192 && skip == 0) {
        on_frame = 0;
        skip = 1;
      }
      frame += skip;
      ++steps;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (done)
      break;
  }
  if (l == 0) {
    if (frame < frames)
      atomic_or(out, 16);
    out[1] = count;
    out[2] = steps;
    out[3] = frame;
  }
}
