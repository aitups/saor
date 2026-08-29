// Decodificador CPPN (especificación v4) en OpenCL C 1.2 (límite del driver
// NVIDIA, D9). Un work-item por CONEXIÓN (máximo paralelismo para bloques
// grandes: 89M–201M conexiones en los modelos objetivo):
//   * escribe la matriz densa enmascarada `w_out` (disjunta por conexión);
//   * acumula la adyacencia con `atomic_or` sobre palabras u32 (32 conexiones),
//     evitando carreras entre work-items del mismo byte;
//   * `pack_adjacency` convierte las palabras u32 al bit-tensor u8 LSB-first
//     `ffn_dag_adjacency` (espejo de `saor-streamer`).

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define CPPN_INPUT_DIM 9
#define CPPN_HIDDEN 16
#define PI 3.14159265358979323846f

// Diseño del genoma aplanado (debe coincidir con saor_domain::cppn::flatten):
//   w0[64*9] | b0[64] | w1[64*64] | b1[64] | w2[2*64] | b2[2]
#define OFF_B0 (CPPN_HIDDEN * CPPN_INPUT_DIM)
#define OFF_W1 (OFF_B0 + CPPN_HIDDEN)
#define OFF_B1 (OFF_W1 + CPPN_HIDDEN * CPPN_HIDDEN)
#define OFF_W2 (OFF_B1 + CPPN_HIDDEN)
#define OFF_B2 (OFF_W2 + 2 * CPPN_HIDDEN)

// Evalúa la CPPN para un vector de entrada v[8]; devuelve w_ij y deja l_ij
// (sigmoide) en *out_l.
//
// Optimización de registros: se acumula la salida (w, l_raw) AL VUELO mientras
// se computa h1, sin materializar el array h1[64] en registros privados (que
// forzaba spilling a memoria y hundía el rendimiento en bloques grandes).
float eval_cppn(__global const float* genome, float v[CPPN_INPUT_DIM], float* out_l) {
    float h0[CPPN_HIDDEN];
    for (int o = 0; o < CPPN_HIDDEN; o++) {
        float acc = genome[OFF_B0 + o];
        for (int k = 0; k < CPPN_INPUT_DIM; k++) {
            acc += genome[o * CPPN_INPUT_DIM + k] * v[k];
        }
        h0[o] = tanh(acc);
    }
    float acc_w = genome[OFF_B2 + 0];
    float acc_l = genome[OFF_B2 + 1];
    for (int o = 0; o < CPPN_HIDDEN; o++) {
        float h1o = genome[OFF_B1 + o];
        for (int k = 0; k < CPPN_HIDDEN; k++) {
            h1o += genome[OFF_W1 + o * CPPN_HIDDEN + k] * h0[k];
        }
        h1o = sin(h1o);
        acc_w += genome[OFF_W2 + 0 * CPPN_HIDDEN + o] * h1o;
        acc_l += genome[OFF_W2 + 1 * CPPN_HIDDEN + o] * h1o;
    }
    *out_l = 1.0f / (1.0f + exp(-acc_l));
    return acc_w;
}

// Decodifica una conexión: evalua la CPPN, aplica τ, escribe w_out y acumula
// el bit de adyacencia en `adj_words[conn >> 5]` (bit `conn & 31`).
__kernel void cppn_decode(
    __global const float* genome, // [466]
    const int d_in,
    const int d_out,
    const float tau,
    const int layer,
    const int n_layers,
    __global float* w_out,        // d_out * d_in (enmascarada con ceros)
    __global uint* adj_words,     // ceil(total/32)
    __global uint* active_out)    // [1]
{
    const int conn = get_global_id(0);
    const int total = d_in * d_out;
    if (conn >= total) return;
    const int i = conn / d_out;
    const int j = conn % d_out;
    const float y_i = (d_in > 1) ? -1.0f + 2.0f * (float)i / (float)(d_in - 1) : 0.0f;
    const float y_j = (d_out > 1) ? -1.0f + 2.0f * (float)j / (float)(d_out - 1) : 0.0f;
    const float y_layer = (n_layers > 1) ? -1.0f + 2.0f * ((float)layer + 0.5f) / (float)n_layers : 0.0f;
    float v[CPPN_INPUT_DIM];
    v[0] = -1.0f;
    v[1] = y_i;
    v[2] = 1.0f;
    v[3] = y_j;
    v[4] = 2.0f;
    v[5] = y_j - y_i;
    v[6] = sin(PI * y_i);
    v[7] = cos(PI * y_j);
    v[8] = y_layer;
    float l;
    float w = eval_cppn(genome, v, &l);
    if (l > tau) {
        w_out[j * d_in + i] = w;
        atomic_or(&adj_words[conn >> 5], 1u << (conn & 31));
        atomic_inc(active_out);
    } else {
        w_out[j * d_in + i] = 0.0f;
    }
}

// Variante solo-adyacencia (warm-start teacher-copy): no materializa w_out
// (evita 805 MB de escritura GPU + lectura host en bloques ALIA/Qwen). El
// fitness teacher-copy solo necesita el bit-tensor + el conteo de activos.
__kernel void cppn_decode_adj(
    __global const float* genome,
    const int d_in,
    const int d_out,
    const float tau,
    const int layer,
    const int n_layers,
    __global uint* adj_words,     // ceil(total/32)
    __global uint* active_out)    // [1]
{
    const int conn = get_global_id(0);
    const int total = d_in * d_out;
    if (conn >= total) return;
    const int i = conn / d_out;
    const int j = conn % d_out;
    const float y_i = (d_in > 1) ? -1.0f + 2.0f * (float)i / (float)(d_in - 1) : 0.0f;
    const float y_j = (d_out > 1) ? -1.0f + 2.0f * (float)j / (float)(d_out - 1) : 0.0f;
    const float y_layer = (n_layers > 1) ? -1.0f + 2.0f * ((float)layer + 0.5f) / (float)n_layers : 0.0f;
    float v[CPPN_INPUT_DIM];
    v[0] = -1.0f;
    v[1] = y_i;
    v[2] = 1.0f;
    v[3] = y_j;
    v[4] = 2.0f;
    v[5] = y_j - y_i;
    v[6] = sin(PI * y_i);
    v[7] = cos(PI * y_j);
    v[8] = y_layer;
    float l;
    eval_cppn(genome, v, &l);
    if (l > tau) {
        atomic_or(&adj_words[conn >> 5], 1u << (conn & 31));
        atomic_inc(active_out);
    }
}

// Convierte las palabras u32 de adyacencia al bit-tensor u8 (LSB-first).
__kernel void pack_adjacency(
    __global const uint* adj_words, // ceil(total/32)
    __global uchar* adj_out)        // ceil(total/8)
{
    const int w = get_global_id(0);
    const uint word = adj_words[w];
    adj_out[4 * w + 0] = (uchar)(word & 0xFFu);
    adj_out[4 * w + 1] = (uchar)((word >> 8) & 0xFFu);
    adj_out[4 * w + 2] = (uchar)((word >> 16) & 0xFFu);
    adj_out[4 * w + 3] = (uchar)((word >> 24) & 0xFFu);
}

// ─── Tensorización de la población (Vía B, optimización GPU) ────────────────
// Genoma CPPN real: 466 f32 (16*9 + 16 + 16*16 + 16 + 2*16 + 2).
#define GENOME_LEN (CPPN_HIDDEN * CPPN_INPUT_DIM + CPPN_HIDDEN + \
                    CPPN_HIDDEN * CPPN_HIDDEN + CPPN_HIDDEN + 2 * CPPN_HIDDEN + 2)

// Variante batcheada de `cppn_decode_adj` (solo-adyacencia, warm-start): decodifica
// la topología de N candidatos en un único dispatch. Rejilla 2D [conexiones, N].
// NO materializa pesos densos `[N, d_in*d_out]` (no caben en 6 GB en ALIA/Qwen).
__kernel void cppn_decode_adj_batched(
    __global const float* genome,     // [N * 466]
    const int d_in,
    const int d_out,
    const float tau,
    const int layer,
    const int n_layers,
    const int n_candidates,
    __global uint* adj_words,         // [N * ceil(total/32)]
    __global uint* active_out)        // [N]
{
    const int conn = get_global_id(0);
    const int cand = get_global_id(1);
    const int total = d_in * d_out;
    if (conn >= total || cand >= n_candidates) return;
    const int i = conn / d_out;
    const int j = conn % d_out;
    const float y_i = (d_in > 1) ? -1.0f + 2.0f * (float)i / (float)(d_in - 1) : 0.0f;
    const float y_j = (d_out > 1) ? -1.0f + 2.0f * (float)j / (float)(d_out - 1) : 0.0f;
    const float y_layer = (n_layers > 1) ? -1.0f + 2.0f * ((float)layer + 0.5f) / (float)n_layers : 0.0f;
    float v[CPPN_INPUT_DIM];
    v[0] = -1.0f;
    v[1] = y_i;
    v[2] = 1.0f;
    v[3] = y_j;
    v[4] = 2.0f;
    v[5] = y_j - y_i;
    v[6] = sin(PI * y_i);
    v[7] = cos(PI * y_j);
    v[8] = y_layer;
    float l;
    const __global float* g = genome + cand * GENOME_LEN;
    eval_cppn(g, v, &l);
    if (l > tau) {
        const int off = cand * ((total + 31) >> 5);
        atomic_or(&adj_words[off + (conn >> 5)], 1u << (conn & 31));
        atomic_inc(&active_out[cand]);
    }
}

// Empaquetado u32 -> u8 batcheado (N candidatos).
__kernel void pack_adjacency_batched(
    __global const uint* adj_words, // [N * n_words]
    const int n_words,
    const int n_candidates,
    __global uchar* adj_out)        // [N * n_bytes]
{
    const int w = get_global_id(0);
    const int cand = get_global_id(1);
    if (w >= n_words || cand >= n_candidates) return;
    const uint word = adj_words[cand * n_words + w];
    uchar* o = &adj_out[cand * (n_words * 4) + 4 * w];
    o[0] = (uchar)(word & 0xFFu);
    o[1] = (uchar)((word >> 8) & 0xFFu);
    o[2] = (uchar)((word >> 16) & 0xFFu);
    o[3] = (uchar)((word >> 24) & 0xFFu);
}


