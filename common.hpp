#include <random>
#include <sycl/sycl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/kernel_bundle.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace syclintelex = sycl::ext::intel::experimental;
using bfloat16 = sycl::ext::oneapi::bfloat16;

// Most of the time, failures related to floating-point calculations (both float
// and bfloat16) are caused by accumulation errors rather than the algorithm
// itself. If it is an algorithm issue, the calculated result gap from the
// reference would be much bigger. To avoid flaky test results while catching
// algorithm errors, we are increasing the accuracy threshold.
// Something like this should be good enough to catch algorithm errors:
// fabs(ref[i] - val[i])/max(fabs(ref)) < 10e-2
constexpr float FLOAT_EPSILON = 10e-2;

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

template <typename T>
void matrix_rand(unsigned int rows, unsigned int cols, T *src, T val) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-val, val);
  std::uniform_int_distribution idistr((int)-val, (int)val);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      if constexpr (std::is_same_v<T, sycl::half> ||
                    std::is_same_v<T, bfloat16> || std::is_same_v<T, float> ||
                    std::is_same_v<T, double>) {
        src[i * cols + j] = T(fdistr(dev));
      } else if constexpr (std::is_integral_v<T>) {
        src[i * cols + j] = T(idistr(dev));
      } else {
        assert(false && "Unsupported type in matrix_rand.");
      }
    }
  }
}


template <typename Ta, typename Tb, typename Tc, unsigned int VF = 1,
          typename F = std::nullptr_t>
void matrix_multiply_ref(Ta *A, Tb *B, Tc *C, int M, int N, int K,
                         bool transpose_c = false, bool colmajor_a = false,
                         bool colmajor_b = false, F &&lambda = {}) {
  for (unsigned int m = 0; m < M; m++) {
    for (unsigned int n = 0; n < N; n++) {
      int c_ind = transpose_c ? (n * M + m) : m * N + n;
      Tc acc = *(C + c_ind);

      for (unsigned int k = 0; k < K; k++) {
        int a_ind = colmajor_a ? (k * M + m) : m * K + k;
        int b_ind = colmajor_b ? (n * K + k) : k * N + n;
        Ta *va = (Ta *)(A + a_ind * VF);
        Tb *vb = (Tb *)(B + b_ind * VF);

        for (unsigned int i = 0; i < VF; i++) {
          if constexpr (std::is_same_v<Ta, bfloat16> &&
                        std::is_same_v<Tc, float>)
            acc += make_fp32(va[i]) * make_fp32(vb[i]);
          else if constexpr (std::is_same_v<Ta, sycl::half>)
            acc += (float)va[i] * (float)vb[i];
          else if constexpr (std::is_same_v<Ta, float> &&
                                 std::is_same_v<Tc, float> ||
                             std::is_integral_v<Ta> && std::is_integral_v<Tc> ||
                             (std::is_same_v<Ta, bfloat16> ||
                              std::is_same_v<Ta, sycl::half>) ||
                             (std::is_same_v<Ta, double> &&
                              std::is_same_v<Tc, double>))
            acc += va[i] * vb[i];
          else
            assert(false && "Unsupported type in matrix_multiply_ref.");
        }
      }

      if constexpr (!std::is_same_v<F, std::nullptr_t>) {
        lambda(acc);
      }
      *(C + c_ind) = acc;
    }
  }
}

// A: M * K BF16 or FP16
// B: N * (K//4) INT4 data but as BF16 or FP16
// C: M * N BF16 or FP16
template <typename Tsrc, typename Tdst>
void matrix_multiply_int4_ref(Tsrc *A, Tsrc *B, Tdst *C, int M, int N, int K) {
  for (unsigned int m = 0; m < M; m++) {
    for (unsigned int n = 0; n < N; n++) {
      int c_ind = m * N + n;
      Tdst acc = *(C + m * N + n);

      for (unsigned int k = 0; k < K; k+=4) {
        int a_ind = m * K + k;
        int b_ind = k/4 + K / 4 * n;
        Tsrc vb = *(B + b_ind);

        uint16_t src_int = sycl::bit_cast<uint16_t>(vb);
        auto f1 = (float)((src_int & 0x000f));
        auto f2 = (float)((src_int & 0x00f0) >> 4);
        auto f3 = (float)((src_int & 0x0f00) >> 8);
        auto f4 = (float)((src_int & 0xf000) >> 12);
        acc += (Tdst)((float)A[a_ind] * f1 + (float)A[a_ind + 1] * f2 +
                      (float)A[a_ind + 2] * f3 + (float)A[a_ind + 3] * f4);
      }

      *(C + c_ind) = (Tdst)acc;
    }
  }
}

template <typename T1, typename T2, bool exact = false>
bool matrix_compare(unsigned int rows, unsigned int cols, T1 *src, T2 *ref) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if constexpr (!exact && (std::is_same_v<T1, float> ||
                               std::is_same_v<T1, bfloat16> ||
                               (std::is_same_v<T1, double> &&
                                std::is_same_v<T2, double>))) {
        float diff = std::fabs(src[i * cols + j] - (T1)ref[i * cols + j]);
        if (diff > FLOAT_EPSILON || std::isnan(src[i * cols + j])) {
          std::cout << "Incorrect result in matrix. " << "i: " << i
                    << ", j: " << j << ", Ref: " << (T1)ref[i * cols + j]
                    << ", Val: " << src[i * cols + j] << ", Diff: " << diff
                    << ", Epsilon: " << FLOAT_EPSILON << "\n";
          return false;
        }
      } else if constexpr (exact || std::is_same_v<T1, int32_t>) {
        if (src[i * cols + j] != ref[i * cols + j]) {
          std::cout << "Incorrect result in matrix." << "i: " << i
                    << ", j: " << j << ", Ref: " << ref[i * cols + j]
                    << ", Val: " << src[i * cols + j] << "\n";
          return false;
        }
      } else {
        std::cout << "Unsupported type in matrix_compare\n";
        return false;
      }
    }
  }
  return true;
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
