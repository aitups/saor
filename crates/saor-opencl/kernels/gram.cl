// Matriz de Gram K = H H^T (B x B) para CKA, en OpenCL C 1.2.
// Un work-item por par (b1, b2); el bucle interno recorre D.

__kernel void gram(
    __global const float* h, // B * d
    const int d,
    const int batch,
    __global float* k) // B * B
{
    const int b1 = get_global_id(0);
    const int b2 = get_global_id(1);
    if (b1 >= batch || b2 >= batch) return;
    float acc = 0.0f;
    for (int dd = 0; dd < d; dd++) {
        acc += h[b1 * d + dd] * h[b2 * d + dd];
    }
    k[b1 * batch + b2] = acc;
}
