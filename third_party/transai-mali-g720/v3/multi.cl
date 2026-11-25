// Kernel boundaries provide device-wide ordering. No cross-workgroup spins.
inline int dot_global(__global const uchar *restrict w, ulong off, int n,
                      int out, __global char4 *q, int zp, ulong sum) {
  __global const char16 *matrix = (__global const char16 *)(w + off);
  int4 a = (int4)(0);
#pragma unroll 4
  for (int k = 0; k < 40; ++k) {
    char16 x = vload16(k, (__global char *)q), b = matrix[k * n + out];
    a.x = arm_dot_acc(x.s0123, b.s0123, a.x);
    a.y = arm_dot_acc(x.s4567, b.s4567, a.y);
    a.z = arm_dot_acc(x.s89ab, b.s89ab, a.z);
    a.w = arm_dot_acc(x.scdef, b.scdef, a.w);
  }
  return a.x + a.y + a.z + a.w + (128 - zp) * pi(w, sum, out);
}
#define ARGS                                                                   \
  __global State *restrict s, __global const half *restrict encoder,           \
      __global const uchar *restrict w, __global int *restrict out
#define ACTIVE                                                                 \
  if (s->frame >= s->frames || s->done)                                        \
  return
#define UPDATE                                                                 \
  ACTIVE;                                                                      \
  if (!s->update)                                                              \
  return
__kernel void tdt_init(ARGS, int frames, __global const int *meta) {
  if (frames < 0)
    frames = meta[2];
  int l = get_local_id(0);
  for (int i = l; i < sizeof(State) / 4; i += WG)
    ((__global int *)s)[i] = 0;
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = l; i < 160; i += WG)
    s->q0[i] = s->q1[i] = (char4)(-128);
  if (l == 0) {
    s->frames = frames;
    s->token = 8192;
    s->update = 1;
    s->scale0 = s->scale1 = 1;
    out[0] = 0;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = l; i < frames * H; i += WG)
    if (!isfinite(convert_float(encoder[i])))
      atomic_or(out, 2);
}
__kernel void tdt_rec(ARGS) {
  UPDATE;
  int id = get_global_id(0), o = id % G;
  if (id >= 2 * G)
    return;
  int second = id >= G;
  ulong W = second ? H1_W : H0_W, S = second ? H1_S : H0_S,
        B = second ? H1_B : H0_B, SUM = second ? H1_SUM : H0_SUM;
  int v = dot_global(w, W, G, o, second ? s->q1 : s->q0,
                     second ? s->zp1 : s->zp0, SUM);
  float f =
      convert_float(v) * ((second ? s->scale1 : s->scale0) * pf(w, S, o)) +
      pf(w, B, o);
  if (second)
    s->g1[o] = f;
  else
    s->g0[o] = pf(w, EMBED_GATES, s->token * G + o) + f;
}
__kernel void tdt_l0(ARGS) {
  UPDATE;
  int l = get_local_id(0);
  __local float x[H], lo[WG], hi[WG], scale[1];
  __local int zp[1];
  __local char4 q[160];
  for (int i = l; i < H; i += WG) {
    float c = sigmoid(s->g0[H + i]) * s->c0[i] +
              sigmoid(s->g0[i]) * tanh(s->g0[2 * H + i]);
    s->c0[i] = c;
    x[i] = sigmoid(s->g0[3 * H + i]) * tanh(c);
    s->h0[i] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  quant(x, q, scale, zp, lo, hi);
  for (int i = l; i < 160; i += WG)
    s->q0[i] = q[i];
  if (l == 0) {
    s->scale0 = scale[0];
    s->zp0 = zp[0];
  }
}
__kernel void tdt_x1(ARGS) {
  UPDATE;
  int o = get_global_id(0);
  if (o >= G)
    return;
  int v = dot_global(w, X1_W, G, o, s->q0, s->zp0, X1_SUM);
  float f = convert_float(v) * (s->scale0 * pf(w, X1_S, o)) + pf(w, X1_B, o);
  s->g1[o] = f + s->g1[o];
}
__kernel void tdt_l1(ARGS) {
  UPDATE;
  int l = get_local_id(0);
  __local float x[H], lo[WG], hi[WG], scale[1];
  __local int zp[1];
  __local char4 q[160];
  for (int i = l; i < H; i += WG) {
    float c = sigmoid(s->g1[H + i]) * s->c1[i] +
              sigmoid(s->g1[i]) * tanh(s->g1[2 * H + i]);
    s->c1[i] = c;
    x[i] = sigmoid(s->g1[3 * H + i]) * tanh(c);
    s->h1[i] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  quant(x, q, scale, zp, lo, hi);
  for (int i = l; i < 160; i += WG)
    s->q1[i] = q[i];
  if (l == 0) {
    s->scale1 = scale[0];
    s->zp1 = zp[0];
  }
}
__kernel void tdt_pred(ARGS) {
  UPDATE;
  int o = get_global_id(0);
  if (o >= H)
    return;
  int v = dot_global(w, PRED_W, H, o, s->q1, s->zp1, PRED_SUM);
  s->pred[o] =
      convert_float(v) * (s->scale1 * pf(w, PRED_S, o)) + pf(w, PRED_B, o);
}
__kernel void tdt_joint(ARGS) {
  ACTIVE;
  int l = get_local_id(0);
  __local float x[H], lo[WG], hi[WG], scale[1];
  __local int zp[1];
  __local char4 q[160];
  for (int i = l; i < H; i += WG)
    x[i] = fmax(convert_float(encoder[s->frame * H + i]) + s->pred[i], 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  quant(x, q, scale, zp, lo, hi);
  for (int i = l; i < 160; i += WG)
    s->qj[i] = q[i];
  if (l == 0) {
    s->scalej = scale[0];
    s->zpj = zp[0];
  }
}
__kernel void tdt_out(ARGS) {
  ACTIVE;
  int o = get_global_id(0);
  if (o >= 8198)
    return;
  int v = dot_global(w, OUT_W, 8200, o, s->qj, s->zpj, OUT_SUM);
  float f = convert_float(v) * (s->scalej * pf(w, OUT_S, o)) + pf(w, OUT_B, o);
  s->logits[o] = f;
  if (!isfinite(f))
    atomic_or(out, 4);
}
__kernel void tdt_select(ARGS) {
  ACTIVE;
  int l = get_local_id(0);
  __local float best[WG], sums[WG];
  __local int indices[WG];
  float v = -INFINITY;
  int index = 2147483647;
  for (int i = l; i < 8193; i += WG)
    if (s->logits[i] > v || (s->logits[i] == v && i < index)) {
      v = s->logits[i];
      index = i;
    }
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
  float score_sum = 0.0f;
  for (int i=l; i<8193; i+=WG) score_sum += exp(s->logits[i]-best[0]);
  sums[l] = score_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = WG / 2; k; k /= 2) {
    if (l < k) sums[l] += sums[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#if MALI_SCORE_TRACE
  if (s->count == 0) for (int i=l; i<8193; i+=WG) TRACE_PTR(out)[i] = s->logits[i];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
  if (l == 0) {
    int y = indices[0], skip = 0;
    for (int i = 1; i < 5; ++i)
      if (s->logits[8193 + i] > s->logits[8193 + skip])
        skip = i;
    if (y != 8192) {
      if (s->count >= MAX_TOKENS) {
        atomic_or(out, 8);
        s->done = 1;
      } else {
        out[8 + s->count * 3] = y;
        out[9 + s->count * 3] = s->frame;
        out[10 + s->count * 3] = skip;
        SCORE_PTR(out)[s->count] = -log(sums[0]);
        s->count++;
      }
      s->token = y;
      s->on_frame++;
      s->update = 1;
    } else
      s->update = 0;
    if (skip > 0)
      s->on_frame = 0;
    if (s->on_frame >= 10) {
      s->on_frame = 0;
      skip = 1;
    }
    if (y == 8192 && skip == 0) {
      s->on_frame = 0;
      skip = 1;
    }
    s->frame += skip;
    s->steps++;
  }
}

// Only two dispatches per decision: parallel old-state recurrent GEMVs above,
// then this fused input-affine/LSTM/projection/joiner/selection group.
__attribute__((reqd_work_group_size(WG, 1, 1))) __kernel void
tdt_fused_step(ARGS) {
  ACTIVE;
  int l = get_local_id(0);
  __local float x[H], pred[H], joint[H], gates[G], logits[8200], lo[WG], hi[WG],
      best[WG], sums[WG], scale[1];
  __local int zp[1], indices[WG];
  __local char4 q[160];
  if (s->update) {
    for (int i = l; i < H; i += WG) {
      float c = sigmoid(s->g0[H + i]) * s->c0[i] +
                sigmoid(s->g0[i]) * tanh(s->g0[2 * H + i]);
      s->c0[i] = c;
      x[i] = sigmoid(s->g0[3 * H + i]) * tanh(c);
      s->h0[i] = x[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    quant(x, q, scale, zp, lo, hi);
    for (int i = l; i < 160; i += WG)
      s->q0[i] = q[i];
    if (l == 0) {
      s->scale0 = scale[0];
      s->zp0 = zp[0];
    }
    for (int o = l; o < G; o += WG) {
      int v = dot640(w, X1_W, G, o, q, zp[0], X1_SUM);
      gates[o] =
          (convert_float(v) * (scale[0] * pf(w, X1_S, o)) + pf(w, X1_B, o)) +
          s->g1[o];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = l; i < H; i += WG) {
      float c = sigmoid(gates[H + i]) * s->c1[i] +
                sigmoid(gates[i]) * tanh(gates[2 * H + i]);
      s->c1[i] = c;
      x[i] = sigmoid(gates[3 * H + i]) * tanh(c);
      s->h1[i] = x[i];
      if (!isfinite(x[i]) || !isfinite(c))
        atomic_or(out, 32);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    quant(x, q, scale, zp, lo, hi);
    for (int i = l; i < 160; i += WG)
      s->q1[i] = q[i];
    if (l == 0) {
      s->scale1 = scale[0];
      s->zp1 = zp[0];
    }
    for (int o = l; o < H; o += WG) {
      int v = dot640(w, PRED_W, H, o, q, zp[0], PRED_SUM);
      pred[o] =
          convert_float(v) * (scale[0] * pf(w, PRED_S, o)) + pf(w, PRED_B, o);
      s->pred[o] = pred[o];
    }
  } else
    for (int i = l; i < H; i += WG)
      pred[i] = s->pred[i];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  for (int i = l; i < H; i += WG)
    joint[i] = fmax(convert_float(encoder[s->frame * H + i]) + pred[i], 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  quant(joint, q, scale, zp, lo, hi);
  for (int o = l; o < 8198; o += WG) {
    int v = dot640(w, OUT_W, 8200, o, q, zp[0], OUT_SUM);
    logits[o] =
        convert_float(v) * (scale[0] * pf(w, OUT_S, o)) + pf(w, OUT_B, o);
    if (!isfinite(logits[o]))
      atomic_or(out, 4);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float v = -INFINITY;
  int index = 2147483647;
  for (int i = l; i < 8193; i += WG)
    if (logits[i] > v || (logits[i] == v && i < index)) {
      v = logits[i];
      index = i;
    }
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
  float score_sum = 0.0f;
  for (int i=l; i<8193; i+=WG) score_sum += exp(logits[i]-best[0]);
  sums[l] = score_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = WG / 2; k; k /= 2) {
    if (l < k) sums[l] += sums[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#if MALI_SCORE_TRACE
  if (s->count == 0) for (int i=l; i<8193; i+=WG) TRACE_PTR(out)[i] = logits[i];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
  if (l == 0) {
    int y = indices[0], skip = 0;
    for (int i = 1; i < 5; ++i)
      if (logits[8193 + i] > logits[8193 + skip])
        skip = i;
    if (y != 8192) {
      if (s->count >= MAX_TOKENS) {
        atomic_or(out, 8);
        s->done = 1;
      } else {
        out[8 + s->count * 3] = y;
        out[9 + s->count * 3] = s->frame;
        out[10 + s->count * 3] = skip;
        SCORE_PTR(out)[s->count] = -log(sums[0]);
        s->count++;
      }
      s->token = y;
      s->on_frame++;
      s->update = 1;
    } else
      s->update = 0;
    if (skip > 0)
      s->on_frame = 0;
    if (s->on_frame >= 10) {
      s->on_frame = 0;
      skip = 1;
    }
    if (y == 8192 && skip == 0) {
      s->on_frame = 0;
      skip = 1;
    }
    s->frame += skip;
    s->steps++;
  }
}
