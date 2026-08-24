// Decodificador CPPN (especificación v4) en OpenCL C 1.2 (límite del driver
// NVIDIA, D9). Un work-item por BYTE de adyacencia (8 conexiones), evaluando la
// CPPN localmente y escribiendo la matriz densa enmascarada + el bit-tensor
// `ffn_dag_adjacency` + el conteo de conexiones activas (sin carreras).

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define CPPN_INPUT_DIM 8
#define CPPN_HIDDEN 64
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
float eval_cppn(__global const float* genome, float v[CPPN_INPUT_DIM], float* out_l) {
    float h0[CPPN_HIDDEN];
    for (int o = 0; o < CPPN_HIDDEN; o++) {
        float acc = genome[OFF_B0 + o];
        for (int k = 0; k < CPPN_INPUT_DIM; k++) {
            acc += genome[o * CPPN_INPUT_DIM + k] * v[k];
        }
        h0[o] = tanh(acc);
    }
    float h1[CPPN_HIDDEN];
    for (int o = 0; o < CPPN_HIDDEN; o++) {
        float acc = genome[OFF_B1 + o];
        for (int k = 0; k < CPPN_HIDDEN; k++) {
            acc += genome[OFF_W1 + o * CPPN_HIDDEN + k] * h0[k];
        }
        h1[o] = sin(acc);
    }
    float w = genome[OFF_B2 + 0];
    float l_raw = genome[OFF_B2 + 1];
    for (int k = 0; k < CPPN_HIDDEN; k++) {
        w += genome[OFF_W2 + 0 * CPPN_HIDDEN + k] * h1[k];
        l_raw += genome[OFF_W2 + 1 * CPPN_HIDDEN + k] * h1[k];
    }
    *out_l = 1.0f / (1.0f + exp(-l_raw));
    return w;
}

// Fija los offsets de offset para mantener legibilidad en la firma.
__kernel void cppn_decode(
    __global const float* genome, // [4866]
    const int d_in,
    const int d_out,
    const float tau,
    __global float* w_out,        // d_out * d_in (enmascarada con ceros)
    __global uchar* adj_out,      // ceil(d_in*d_out/8)
    __global uint* active_out)    // [1]
{
    const int total = d_in * d_out;
    const int n_bytes = (total + 7) / 8;
    const int byte_idx = get_global_id(0);
    if (byte_idx >= n_bytes) return;

    uchar bits = 0;
    for (int bit = 0; bit < 8; bit++) {
        const int conn = byte_idx * 8 + bit;
        if (conn >= total) break;
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
            bits |= (uchar)(1 << bit);
            w_out[j * d_in + i] = w;
            atomic_inc(active_out);
        } else {
            w_out[j * d_in + i] = 0.0f;
        }
    }
    adj_out[byte_idx] = bits;
}
