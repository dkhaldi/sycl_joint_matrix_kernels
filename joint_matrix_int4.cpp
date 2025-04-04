#include "common.hpp"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 2048
#endif

#define BF16_INT4 4

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

#ifndef NUM_B_MATRICES
#define NUM_B_MATRICES 10
#endif

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, typename TOperand,
          typename TResult, size_t tM, size_t tN, size_t tK, size_t TMCACHE1,
          size_t TNCACHE1, size_t TKCACHE1, size_t TMCACHE2, size_t TNCACHE2,
          size_t TKCACHE2, class kernel_name>
double joint_matmul_int4(TOperand *A, TOperand *B_[NUM_B_MATRICES], TResult *C, queue &q,
                    int testIterations) {

  size_t SG_SIZE = get_sg_size<kernel_name>(q);
  range<2> global{rowsA / TMCACHE1, (colsB / TNCACHE1) * SG_SIZE};
  range<2> cachelocal{TMCACHE2 / TMCACHE1, TNCACHE2 / TNCACHE1 * SG_SIZE};

  // throw error if padding or different tuning parameters are needed
  static_assert(colsA == rowsB);
  static_assert(rowsA >= TMCACHE2 && rowsA % tM == 0);
  static_assert(colsA >= TKCACHE2 && colsA % tK == 0);
  static_assert(colsB >= TNCACHE2 && colsB % tN == 0);
  static_assert(colsB >= TNCACHE2 && colsB % tN == 0);
  static_assert((colsB % TNCACHE2 == 0) &&
                "NCACHE2 does not multiply MATRIX_N, use a different NCACHE2 "
                "in the command line for instance -DNCACHE2=128 or pad "
                "MATRIX_N to be multiple of NCACHE2");

  // submit main kernel

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (unsigned int i = 0; i < testIterations; i++) {

    auto B = B_[i % NUM_B_MATRICES];
    auto mk = q.submit([&](handler &h) {
      local_accessor<TOperand, 2> tileB{
        {TKCACHE2 * BF16_INT4, TNCACHE2}, h};
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

            constexpr unsigned int SGs = (TKCACHE2 / TKCACHE1) * (TNCACHE2 / TNCACHE1);
            size_t local_offset_B_row = sg.get_group_id() / (TNCACHE2 / TNCACHE1) * TKCACHE1 * BF16_INT4;
            size_t local_offset_B_col = sg.get_group_id() % (TNCACHE2 / TNCACHE1) * TKCACHE1;
            size_t local_offset_B = local_offset_B_row * TNCACHE2 + local_offset_B_col;
            auto pTmp = tileB.template get_multi_ptr<sycl::access::decorated::no>() + local_offset_B;

            joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
                tC[TMCACHE1 / tM][TNCACHE1 / tN];
            for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                    joint_matrix_fill(sg, tC[m][n], 0);
              }
            }

            for (unsigned int k2 = 0; k2 <  colsA / BF16_INT4 / TKCACHE2; k2++) {
              joint_matrix<sub_group, TOperand, use::a, tM, tK,
                           layout::row_major>
                  tA[TMCACHE1 / tM][TKCACHE2 / TKCACHE1];
              joint_matrix<sub_group, TOperand, use::b, tK, tN,
                           layout::row_major>
                  tB_int4[TNCACHE1 / tN][TKCACHE2 / TKCACHE1];
              joint_matrix<sub_group, TOperand, use::b, tK, tN,
                           layout::row_major>
                  tB_bf16[TNCACHE1 / tN][TKCACHE2 / TKCACHE1][BF16_INT4];
              for (unsigned int k1 = 0; k1 < TKCACHE2 / TKCACHE1; k1++) {
                //  physical layer
                unsigned int k = (k2 * TKCACHE2 + k1 * TKCACHE1) / tK;
                for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
                  ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, tA[m][k1], pA, colsA, rowsA, colsA,
                      m2 * TMCACHE2 + m1 * TMCACHE1 + m * tM, k * tK);
                }
                for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                auto pTmp_n = pTmp + n * tN;
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tB_int4[n][k1], pB, colsB, rowsB, colsB, k * tK,
                    n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN);
                ext::intel::experimental::matrix::joint_matrix_apply(sg, tB_int4[n][k1], [&](TOperand &src, size_t row, size_t col) {
                  uint16_t src_int = sycl::bit_cast<uint16_t>(src);
                  pTmp_n[(col * BF16_INT4 + 0) * tN + row] = TOperand((src_int & 0x000f));
                  pTmp_n[(col * BF16_INT4 + 1) * tN + row] = TOperand((src_int & 0x00f0) >> 4);
                  pTmp_n[(col * BF16_INT4 + 2) * tN + row] = TOperand((src_int & 0x0f00) >> 8);
                  pTmp_n[(col * BF16_INT4 + 3) * tN + row] = TOperand((src_int & 0xf000) >> 12);
                });
                for (int i = 0; i < BF16_INT4; i++) {
                  joint_matrix_load(sg, tB_bf16[n][k1][i], pTmp_n + i * tK * tN, tN);
                }
                }
                for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
                  for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                      for (int i = 0; i < BF16_INT4; i++) {
                        joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB_bf16[n][k1][i],
                                         tC[m][n]); // 32 DPAS
                      }
                  }
                }
              } // for k1
              it.barrier(access::fence_space::local_space);
            } // for k2
            for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                  joint_matrix_store(
                      sg, tC[m][n],
                      pC + (m2 * TMCACHE2 + m1 * TMCACHE1 + m * tM) * colsB +
                          (n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN),
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


template <typename T1, typename T2, size_t tM, size_t tN, size_t tK,
          size_t MCache1, size_t NCache1, size_t KCache1, size_t MCache2,
          size_t NCache2, size_t KCache2,
          class kernel_name, bool reduce = false>
int gemm(void) {
  // number of test iterations
  constexpr unsigned int testIterations = 100;

  queue q;
  T1 *A = malloc_shared<T1>(MATRIX_M * MATRIX_K, q);
  T1 *B[NUM_B_MATRICES];
  for (int i = 0; i < NUM_B_MATRICES; ++i){
    B[i] = malloc_shared<T1>(MATRIX_K * MATRIX_N, q);
  }
  T2 *C = malloc_shared<T2>(MATRIX_M * MATRIX_N, q);
  T2 *refC = malloc_shared<T2>(MATRIX_M * MATRIX_N, q);
  // Initialize; fill matrices
  matrix_rand(MATRIX_M, MATRIX_K, A, T1(1));
  matrix_rand(MATRIX_K, MATRIX_N, B[0], T1(1));
  matrix_multiply_int4_ref(A, B[0], refC, MATRIX_M, MATRIX_N, MATRIX_K);
  joint_matmul_int4 < MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N,
  T1, T2, tM, tN, tK, (MATRIX_M >= MCache1) ? MCache1 : MATRIX_M,
  (MATRIX_N >= NCache1) ? NCache1 : MATRIX_N, KCache1, (MATRIX_M >= MCache2) ? MCache2 : MATRIX_M,
  (MATRIX_N >= NCache1) ? NCache2 : MATRIX_N,
  KCache2, kernel_name > (A, B, C, q, 1);
  matrix_compare(MATRIX_M, MATRIX_N, C, refC);

  std::cerr << "Running tests..." << std::endl;
  double duration = 0;

  // run testIterations time, aggregate and calculate average run time
  duration = joint_matmul_int4 < MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N,
  T1, T2, tM, tN, tK, (MATRIX_M >= MCache1) ? MCache1 : MATRIX_M,
  (MATRIX_N >= NCache1) ? NCache1 : MATRIX_N, KCache1, (MATRIX_M >= MCache2) ? MCache2 : MATRIX_M,
  (MATRIX_N >= NCache1) ? NCache2 : MATRIX_N,
  KCache2, kernel_name > (A, B, C, q, testIterations);


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
  for (int i = 0; i < NUM_B_MATRICES; ++i){
    free(B[i], q);
  }
  // free(B, q);
  free(C, q);

  return 0;
}

int main() {
  constexpr size_t MCache1 = MCACHE1;
  constexpr size_t MCache2 = MCACHE2;
  constexpr size_t NCache1 = NCACHE1;
  constexpr size_t NCache2 = NCACHE2;
  constexpr size_t KCache1 = KCACHE1;
  constexpr size_t KCache2 = KCACHE2;

  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::bf16) {
      if (combinations[i].nsize == 16) { // PVC
        std::cerr << "PVC bf16 \n";
        gemm<bfloat16, float, (MATRIX_M >= 8) ? 8 : MATRIX_M, (MATRIX_N >= 16) ? 16 : MATRIX_N, 16, MCache1,
             NCache1, KCache1, MCache2, NCache2, KCache2,
             class pvc_bf16_8x16x16>();
        // gemm<bfloat16, float, (MATRIX_M >= 8) ? 8 : MATRIX_M, (MATRIX_N >= 64) ? 64 : MATRIX_N, 32, MCache1,
        //      NCache1, KCache1, MCache2, NCache2, KCache2, 2,
        //      class pvc_bf16_8x32x16>();
        // gemm<bfloat16, float, (MATRIX_M >= 8) ? 8 : MATRIX_M, (MATRIX_N >= 64) ? 64 : MATRIX_N, 16, MCache1,
        //      NCache1, KCache1, MCache2, NCache2, KCache2,
        //      class pvc_bf16_8x64x16>();
        break;
      }
    }
  }
  return 0;
}
