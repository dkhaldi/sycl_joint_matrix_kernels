#include <random>
#include <sycl/sycl.hpp>

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

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int vnniFactor, typename TOperand,
          typename TResult, size_t MCache1, size_t NCache1, size_t KCache1,
          size_t MCache2, size_t NCache2, size_t KCache2, size_t tM, size_t tN,
          size_t tK>
inline void
load_mad(TOperand *A, TOperand *B,
         multi_ptr<TOperand, sycl::access::address_space::global_space,
                   sycl::access::decorated::no> &pA,
         multi_ptr<TOperand, sycl::access::address_space::global_space,
                   sycl::access::decorated::no> &pB,
         sub_group sg, size_t m2, size_t n2, size_t m1, size_t n1,
         size_t SG_SIZE,
         joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
             tC[MCache1 / tM][NCache1 / tN]) {
#ifdef PREFETCH
  size_t sgId = sg.get_group_id()[0];
  // There are MCache2/MCache1 x NCache2/NCache1 subgroups: NumSGs
  // PVC case: this is 8x4 subgroups
  // BKM for PVC is to use prefetch of 8x32 for each subgroup
  constexpr size_t prefRow = 8;
  constexpr size_t prefCol = 32;
  // All the SGs of one workgroup prefetch MCache2xKCache2 of A
  // All the SGs of one workgroup prefetch KCache2xNCache2 of B
  // PVC case: 256x32 of A and 32x256 of B
  // For both A and B: each subgroup performs a prefetch of
  // prefRow rows and prefCol cols at a time
  // For A, the subgroups are distributed along the row dimension:
  // PVC: A layed as MCache2/prefRow (256/32)
  // For B: the subgroups are distributed along the column dimension
  // PVC: NCache2/prefCol = 256/32 = 8 SGs on the column dimension and
  // KCache2/prefRow = 32/8 = 4 SGs on the row dimension
#ifdef VNNI
  // In the VNNI case, each subgroup still gets prefRow x prefCol
  // In the PVC case: subgroups distribution become
  // (NCache2*2)/prefCol = 512/32 = 16 SGs on the column dimension and
  // (KCache2/2)/prefRow = 16/8 = 2 SGs on the row dimension
  // pm1B and pn1B are used to identify the distribution of subgroups
  // along the workgroup prefetch for B matrix. For A matrix, sgId is
  // enough.
  size_t pm1B = sgId / 16;   // prefetch m1 (sgId/16)
  size_t pn1B = sgId & 0x15; // prefetch n1 (sgId%16)
#else                        // VNNI
  size_t pm1B = sgId / 8;   // prefetch m1 (sgId/8)
  size_t pn1B = sgId & 0x7; // prefetch n1 (sgId%8)
#endif                       // VNNI
  constexpr size_t prefDistance = 3;
  for (int p = 0; p < prefDistance; p++)
    joint_matrix_prefetch<prefRow, prefCol>(
        sg, A + (m2 * MCache2 + sgId * prefRow) * colsA + p * prefCol, colsA,
        layout::row_major, syclex::properties{syclex::prefetch_hint_L1});

#ifdef VNNI
  for (int p = 0; p < prefDistance; p++)
    joint_matrix_prefetch<prefRow, prefCol>(
        sg,
        B + (p * (KCache2 / vnniFactor) + pm1B * prefRow) * colsB * vnniFactor +
            (n2 * NCache2 * vnniFactor + pn1B * prefCol),
        colsB * vnniFactor, layout::row_major,
        syclex::properties{syclex::prefetch_hint_L1});
#else  // VNNI
  for (int p = 0; p < prefDistance; p++)
    joint_matrix_prefetch<prefRow, prefCol>(
        sg,
        B + (p * KCache2 + pm1B * prefRow) * colsB + n2 * NCache2 +
            pn1B * prefCol,
        colsB, layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
#endif // VNNI
#endif // PREFETCH

  for (unsigned int m = 0; m < MCache1 / tM; m++) {
    for (unsigned int n = 0; n < NCache1 / tN; n++) {
      joint_matrix_fill(sg, tC[m][n], 0);
    }
  }

  for (unsigned int k2 = 0; k2 < colsA / KCache2; k2++) {
    joint_matrix<sub_group, TOperand, use::a, tM, tK, layout::row_major>
        tA[MCache1 / tM][KCache2 / KCache1];
#ifdef VNNI
    joint_matrix<sub_group, TOperand, use::b, tK, tN, layout::ext_intel_packed>

        tB[NCache1 / tN][KCache2 / KCache1];
#else
    joint_matrix<sub_group, TOperand, use::b, tK, tN, layout::row_major>

        tB[NCache1 / tN][KCache2 / KCache1];
#endif
    for (unsigned int k1 = 0; k1 < KCache2 / KCache1; k1++) {
      //  physical layer
      unsigned int k = (k2 * KCache2 + k1 * KCache1) / tK;
      for (unsigned int m = 0; m < MCache1 / tM; m++) {
#ifdef OOB
        ext::intel::experimental::matrix::joint_matrix_load_checked(
            sg, tA[m][k1], pA, colsA, rowsA, colsA,
            m2 * MCache2 + m1 * MCache1 + m * tM, k * tK);
#else  // OOB
        joint_matrix_load(sg, tA[m][k1],
                          pA + (m2 * MCache2 + m1 * MCache1 + m * tM) * colsA +
                              k * tK,
                          colsA);
#endif // OOB
      }
      for (unsigned int n = 0; n < NCache1 / tN; n++) {
#ifdef OOB
#ifdef VNNI
        ext::intel::experimental::matrix::joint_matrix_load_checked(
            sg, tB[n][k1], pB, colsB * vnniFactor, rowsB / vnniFactor,
            colsB * vnniFactor, k * tK / vnniFactor,
            (n2 * NCache2 + n1 * NCache1 + n * tN) * vnniFactor);
#else // VNNI
        ext::intel::experimental::matrix::joint_matrix_load_checked(
            sg, tB[n][k1], pB, colsB, rowsB, colsB, k * tK,
            n2 * NCache2 + n1 * NCache1 + n * tN);

#endif // VNNI
#else  // OOB
#ifdef VNNI
        joint_matrix_load(sg, tB[n][k1],
                          pB + (k * tK / vnniFactor) * (colsB * vnniFactor) +
                              (n2 * NCache2 + n1 * NCache1 + n * tN) *
                                  vnniFactor,
                          colsB * vnniFactor);
#else  // VNNI
        joint_matrix_load(sg, tB[n][k1],
                          pB + (k * tK) * (colsB) +
                              (n2 * NCache2 + n1 * NCache1 + n * tN),
                          colsB);
#endif // VNNI
#endif // OOB
      }
      for (unsigned int m = 0; m < MCache1 / tM; m++) {
        for (unsigned int n = 0; n < NCache1 / tN; n++) {
          joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1], tC[m][n]);
        }
      }
    } // for k1
#ifdef PREFETCH
    auto prefetch_offsetA =
        (m2 * MCache2 + sgId * prefRow) * colsA + (k2 + prefDistance) * prefCol;
    if ((prefetch_offsetA + (prefRow * colsA) + prefCol) < (rowsA * colsA))
      joint_matrix_prefetch<prefRow, prefCol>(
          sg, A + prefetch_offsetA, colsA, layout::row_major,
          syclex::properties{syclex::prefetch_hint_L1});

#ifdef VNNI
    auto prefetch_offsetB =
        ((k2 + prefDistance) * (KCache2 / vnniFactor) + pm1B * prefRow) *
            (colsB)*vnniFactor +
        (n2 * NCache2 * vnniFactor + pn1B * prefCol);
    if ((prefetch_offsetB + (prefRow * colsB * vnniFactor) + prefCol) <
        (rowsB * colsB))
      joint_matrix_prefetch<prefRow, prefCol>(
          sg, B + prefetch_offsetB, colsB * vnniFactor, layout::row_major,
          syclex::properties{syclex::prefetch_hint_L1});
#else  // VNNI
    auto prefetch_offsetB =
        ((k2 + prefDistance) * KCache2 + pm1B * prefRow) * (colsB) +
        (n2 * NCache2 + pn1B * prefCol);
    if ((prefetch_offsetB + (prefRow * colsB) + prefCol) < (rowsB * colsB))
      joint_matrix_prefetch<prefRow, prefCol>(
          sg, B + prefetch_offsetB, colsB, layout::row_major,
          syclex::properties{syclex::prefetch_hint_L1});
#endif // VNNI
#endif // PREFETCH
  } // for k2
  return;
}
