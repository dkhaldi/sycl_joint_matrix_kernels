#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define BF16_EPSILON 0.00781250

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename KernelName> size_t get_sg_size(queue q) {
  auto KernelID = get_kernel_id<KernelName>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  auto kernel = KB.get_kernel(KernelID);

  return kernel
      .template get_info<info::kernel_device_specific::max_sub_group_size>(
          q.get_device());
}

void fill_matrix(bfloat16 *M, size_t Rows, size_t Cols) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-1.0, 1.0);
  for (unsigned int i = 0; i < Rows; i++) {
    for (unsigned int j = 0; j < Cols; j++) {
      M[i * Cols + j] = bfloat16(fdistr(dev));
    }
  }
}

void native_matmul(bfloat16 *A, bfloat16 *B, float *C, size_t M, size_t N,
                   size_t K) {
  memset(C, 0, sizeof(float) * M * N);
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int k = 0; k < K; k++) {
      for (unsigned int j = 0; j < N; j++) {
        C[i * N + j] += make_fp32(A[i * K + k]) * make_fp32(B[k * N + j]);
      }
    }
  }
}

void verify_result(float *result, float *ref, size_t M, size_t N, size_t K,
                   float floatTol = BF16_EPSILON) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      float a = result[i * N + j];
      float b = ref[i * N + j];
      if ((fabs(a - b)) > floatTol) {
        std::cout << "failed at index " << i << ", " << j << ", res " << a
                  << " != ref " << b << " difference is " << a - b << "\n";
        return;
      }
      // assert((fabs(a) - fabs(b)) <= floatTol);
    }
  }
}

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor = 2) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}