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

#define CPPN_INPUT_DIM 8
#define CPPN_HIDDEN 16
#define PI 3.14159265358979323846f

// Diseño del genoma aplanado (debe coincidir con saor_domain::cppn::flatten):
//   w0[64*8] | b0[64] | w1[64*64] | b1[64] | w2[2*64] | b2[2]
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
    __global const float* genome, // [4866]
    const int d_in,
    const int d_out,
    const float tau,
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
    float v[CPPN_INPUT_DIM];
    v[0] = -1.0f;
    v[1] = y_i;
    v[2] = 1.0f;
    v[3] = y_j;
    v[4] = 2.0f;
    v[5] = y_j - y_i;
    v[6] = sin(PI * y_i);
    v[7] = cos(PI * y_j);
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

