// Optimizations:
// cache tiling of i and j
// cache tiling on k as well (so no reordering is needed)
// data reuse of A and B in physical layer

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

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int vnniFactor, typename TOperand,
          typename TResult, size_t tM, size_t tN, size_t tK, class kernel_name>
double joint_matmul(TOperand *A, TOperand *B, TResult *C, queue &q,
                    int testIterations) {

  size_t SG_SIZE = get_sg_size<kernel_name>(q);
  range<2> global{rowsA / MCACHE1, (colsB / NCACHE1) * SG_SIZE};
  range<2> cachelocal{MCACHE2 / MCACHE1, NCACHE2 / NCACHE1 * SG_SIZE};

  // throw error if padding needed
  assert(colsA == rowsB);
  assert(rowsA >= MCACHE2 && rowsA % tM == 0);
  assert(colsA >= KCACHE2 && colsA % tK == 0);
  assert(colsB >= NCACHE2 && colsB % tN == 0);

  // submit main kernel

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (unsigned int i = 0; i < testIterations; i++) {

    auto mk = q.submit([&](handler &h) {
      h.parallel_for<kernel_name>( // cache layer#1
          nd_range<2>{global, cachelocal}, [=](nd_item<2> it) {
            auto pA =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(A);
            auto pB =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(B);
            auto pC =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(C);

            auto m2 = it.get_group(0);
            auto n2 = it.get_group(1);
            auto m1 = it.get_local_id(0);
            auto n1 = it.get_local_id(1) / SG_SIZE;
            auto sg = it.get_sub_group();
            joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
                tC[MCACHE1 / tM][NCACHE1 / tN];
            for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                joint_matrix_fill(sg, tC[m][n], 0);
              }
            }

            for (unsigned int k2 = 0; k2 < colsA / KCACHE2; k2++) {
              joint_matrix<sub_group, TOperand, use::a, tM, tK,
                           layout::row_major>
                  tA[MCACHE1 / tM][KCACHE2 / KCACHE1];
#ifdef VNNI
              joint_matrix<sub_group, TOperand, use::b, tK, tN,
                           layout::ext_intel_packed>

                  tB[NCACHE1 / tN][KCACHE2 / KCACHE1];
#else
              joint_matrix<sub_group, TOperand, use::b, tK, tN,
                           layout::row_major>

                  tB[NCACHE1 / tN][KCACHE2 / KCACHE1];
#endif
              for (unsigned int k1 = 0; k1 < KCACHE2 / KCACHE1; k1++) {
                //  physical layer
                unsigned int k = (k2 * KCACHE2 + k1 * KCACHE1) / tK;

                for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
                  joint_matrix_load(
                      sg, tA[m][k1],
                      pA + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsA +
                          k * tK,
                      colsA);
                }
                for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#ifdef VNNI
                  joint_matrix_load(
                      sg, tB[n][k1],
                      pB + (k * tK / vnniFactor) * (colsB * vnniFactor) +
                          (n2 * NCACHE2 + n1 * NCACHE1 + n * tN) * vnniFactor,
                      colsB * vnniFactor);
#else
                  joint_matrix_load(
                      sg, tB[n][k1],
                      pB + (k * tK) * (colsB) +
                          (n2 * NCACHE2 + n1 * NCACHE1 + n * tN), colsB);
#endif
                }
                for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
                  for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                    joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                     tC[m][n]); // 32 DPAS
                  }
                }
              } // for k1
            } // for k2
            for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                joint_matrix_store(
                    sg, tC[m][n],
                    pC + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsB +
                        (n2 * NCACHE2 + n1 * NCACHE1 + n * tN),
                    colsB, layout::row_major);
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

template <size_t tM, size_t tN, size_t tK, class kernel_name> int gemm(void) {
  // number of test iterations
  constexpr unsigned int testIterations = 100;

  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_M * MATRIX_K, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
#ifdef VNNI
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
#endif
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *refC = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  // Initialize; fill matrices
  fill_matrix(A, MATRIX_M, MATRIX_K);
  fill_matrix(B, MATRIX_K, MATRIX_N);
  native_matmul(A, B, refC, MATRIX_M, MATRIX_N, MATRIX_K);
#ifdef VNNI
  matrix_vnni<bfloat16>(MATRIX_K, MATRIX_N, B, vnniB, 2);
  B = vnniB;
#endif

  std::cerr << "Running tests...";

  // warm up
  joint_matmul<MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N, 2, bfloat16, float, tM,
               tN, tK, kernel_name>(A, B, C, q, 1);

  // run testIterations time, aggregate and calculate average run time
  double duration =
      joint_matmul<MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N, 2, bfloat16, float,
                   tM, tN, tK, kernel_name>(A, B, C, q, testIterations);

  verify_result(C, refC, MATRIX_M, MATRIX_N, MATRIX_K);

  double msecPerMatrixMul = duration / static_cast<double>(testIterations);
  double gflops = (2.f * MATRIX_M * MATRIX_N * MATRIX_K * 1.0e-9f) /
                  (msecPerMatrixMul / 1000.f);

  std::cerr << "DONE for GEMM size " << MATRIX_M << "x" << MATRIX_N << "x"
            << MATRIX_K << " Matrix Combination is " << tM << "x" << tN << "x"
            << tK << std::endl;
  std::cerr << "Average test time is " << msecPerMatrixMul << " ms"
            << std::endl;

  std::cerr << "GOPS is " << gflops << " Gop/s" << std::endl;

  free(A, q);
  free(B, q);
  free(C, q);

  return 0;
}

int main() {
#ifdef NVIDIA
  // Use -DMCACHE1=64 -DNCACHE1=64 -DMCACHE2=128 -DNCACHE2=128
  gemm<16, 16, 16, class nvidia_16x16x16>();
#else  
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::bf16) {
      if (combinations[i].nsize == 0) {
        gemm<16 /*tM*/, 16 /*tN*/, 32 /*tK*/, class amx>(); // AMX
        break;
      }
      if (combinations[i].nsize == 16) { // PVC
        gemm<8, 16, 16, class pvc_8x16x16>();
        // This case will be added once agama is updated to latest (2025.0)
	gemm<32, 64, 16, class pvc_32x64x16>();
        break;
      }
      if (combinations[i].nsize == 8) { // DG2
        gemm<8, 8, 16, class dg2>();
        break;
      }
    }
  }
#endif  
  return 0;
}