// Optimizations:
// GEMM uses joint matrix
// GELU is fused with GEMM in one kernel using joint matrix object

#include "common.hpp"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 2048
#endif

#ifndef MATRIX_M
#define MATRIX_M MATRIX_SIZE
#endif
#ifndef MATRIX_N
#define MATRIX_N MATRIX_SIZE
#endif
#ifndef MATRIX_K
#define MATRIX_K MATRIX_SIZE
#endif

#ifndef MCACHE1
#define MCACHE1 32
#endif
#ifndef NCACHE1
#define NCACHE1 64
#endif
#ifndef KCACHE1
#define KCACHE1 16
#endif

#ifndef MCACHE2
#define MCACHE2 256
#endif
#ifndef NCACHE2
#define NCACHE2 256
#endif
#ifndef KCACHE2
#define KCACHE2 32
#endif

#define BF16_EPSILON 0.00781250

inline float gelu(float val) {
  return val *
         (0.5f + 0.5f * sycl::tanh(val * (0.7978845608028654f +
                                          0.035677408136300125f * val * val)));
}

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int vnniFactor, typename TOperand,
          typename TResult, size_t tM, size_t tN, size_t tK, class kernel_name>
double joint_matmul_gelu(TOperand *A, TOperand *B, TResult *C, queue &q,
                         int testIterations) {
  size_t sgSize = get_sg_size<kernel_name>(q);
  range<2> global{MATRIX_M / MCACHE1, (MATRIX_N / NCACHE1) * sgSize};
  range<2> cachelocal{MCACHE2 / MCACHE1, NCACHE2 / NCACHE1 * sgSize};

  // throw error if padding needed
  assert(MATRIX_M % tM == 0);
  assert(MATRIX_K % tK == 0);
  assert(MATRIX_N % tN == 0);

  // submit main kernel

  auto start = std::chrono::steady_clock::now();

  for (unsigned int i = 0; i < testIterations; i++) {

    auto mk = q.submit([&](handler &h) {
      h.parallel_for<kernel_name>(
          nd_range<2>{global, cachelocal}, [=](nd_item<2> it) {
            auto pA =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::yes>(A);
            auto pB =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::yes>(B);
            auto pC =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::yes>(C);
            auto m2 = it.get_group(0);
            auto n2 = it.get_group(1);
            auto m1 = it.get_local_id(0);
            auto n1 = it.get_local_id(1) / sgSize;
            auto sg = it.get_sub_group();
            joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
                tC[MCACHE1 / tM][NCACHE1 / tN];

            load_mad<rowsA, colsA, rowsB, colsB, 2, bfloat16, float, MCACHE1,
                     NCACHE1, KCACHE1, MCACHE2, NCACHE2, KCACHE2, tM, tN, tK>(
                A, B, pA, pB, sg, m2, n2, m1, n1, tC);

            for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                joint_matrix_apply(sg, tC[m][n],
                                   [=](float &x) { x = gelu(x); });
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, tC[m][n], pC, colsB, layout::row_major, rowsA, colsB,
                    m2 * MCACHE2 + m1 * MCACHE1 + m * tM,
                    n2 * NCACHE2 + n1 * NCACHE1 + n * tN);
#else  // OOB
            joint_matrix_store(
                sg, tC[m][n],
                pC + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * MATRIX_N +
                    (n2 * NCACHE2 + n1 * NCACHE1 + n * tN),
                MATRIX_N, layout::row_major);
#endif // OOB
              }
            }
          });
    });
  } // end testIterations
  q.wait();
  std::chrono::duration<double, std::milli> duration =
      std::chrono::steady_clock::now() - start;

  return duration.count();
}

void native_matmul(bfloat16 *A, bfloat16 *B, float *C) {
  memset(C, 0, sizeof(float) * MATRIX_M * MATRIX_N);
  for (unsigned int i = 0; i < MATRIX_M; i++) {
    for (unsigned int j = 0; j < MATRIX_N; j++) {
      for (unsigned int k = 0; k < MATRIX_K; k++) {
        C[i * MATRIX_N + j] +=
            make_fp32(A[i * MATRIX_K + k]) * make_fp32(B[k * MATRIX_N + j]);
      }
      C[i * MATRIX_N + j] = gelu(C[i * MATRIX_N + j]);
    }
  }
}

template <typename T1, typename T2, size_t tM, size_t tN, size_t tK,
          class kernel_name>
int gemm_gelu(void) {
  // number of test iterations
  constexpr unsigned int testIterations = 100;

  queue q;

  T1 *A = malloc_shared<T1>(MATRIX_M * MATRIX_K, q);
  T1 *B = malloc_shared<T1>(MATRIX_K * MATRIX_N, q);
#ifdef VNNI
  T1 *vnniB = malloc_shared<T1>(MATRIX_K * MATRIX_N, q);
#endif
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *refC = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  matrix_rand(MATRIX_M, MATRIX_K, A, T1(1));
  matrix_rand(MATRIX_K, MATRIX_N, B, T1(1));
  matrix_multiply_ref(A, B, refC, MATRIX_M, MATRIX_N, MATRIX_K, false, false,
                      false, [](float &x) { x = gelu(x); });

#ifdef VNNI
  matrix_vnni<T1>(MATRIX_K, MATRIX_N, B, vnniB, 2);
  B = vnniB;
#endif

  std::cerr << "Running tests...";

  double duration_first =
      joint_matmul_gelu<MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N, 2, T1, float,
                        tM, tN, tK, kernel_name>(A, B, C, q,
                                                 1); // first time run
  double totalDuration =
      joint_matmul_gelu<MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N, 2, T1, float,
                        tM, tN, tK, kernel_name>(A, B, C, q, testIterations);
  verify_result(C, refC, MATRIX_M, MATRIX_N, MATRIX_K);

  double msecPerMatrixMul = totalDuration / static_cast<double>(testIterations);
  std::cerr << "DONE for GEMM size " << MATRIX_M << "x" << MATRIX_N << "x"
            << MATRIX_K << " Matrix Combination is " << tM << "x" << tN << "x"
            << tK << std::endl;
  std::cerr << "Average test time is " << msecPerMatrixMul << " ms"
            << std::endl;

  free(A, q);
  free(B, q);
  free(C, q);

  return 0;
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::bf16) {
      if (combinations[i].nsize == 0) {
        gemm_gelu<bfloat16, float, 16 /*tM*/, 16 /*tN*/, 32 /*tK*/,
                  class amx>(); // AMX
        break;
      }
      if (combinations[i].nsize == 16) { // PVC
        gemm_gelu<bfloat16, float, 8, 16, 16, class pvc_8x16x16>();
        if constexpr (NCACHE1 >= 64)
          gemm_gelu<bfloat16, float, 32, 64, 16, class pvc_32x64x16>();
        break;
      }
      if (combinations[i].nsize == 8) { // DG2
        gemm_gelu<bfloat16, float, 8, 8, 16, class dg2>();
        break;
      }
    }
  }
  return 0;
}
