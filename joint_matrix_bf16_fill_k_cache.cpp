// Portable Optimizations:
// - cache tiling of i and j
// - cache tiling on k as well (so no reordering is needed)
// - data reuse of A and B in physical layer
// Specific Optimizations for PVC:
// - Out of Bounds checking is used for PVC using -DOOB
// - Prefetch for PVC is enabled under -DPREFETCH

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
#ifndef ANNOT
            auto pA =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(A);
            auto pB =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(B);
#endif
            auto pC =
                address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(C);

            auto m2 = it.get_group(0);
            auto n2 = it.get_group(1);
            auto m1 = it.get_local_id(0);
            auto n1 = it.get_local_id(1) / SG_SIZE;
            auto sg = it.get_sub_group();
#ifdef PREFETCH
            size_t sgId = sg.get_group_id()[0];
            // There are MCACHE2/MCACHE1 x NCACHE2/NCACHE1 subgroups: NumSGs
            // PVC case: this is 8x4 subgroups
            // BKM for PVC is to use prefetch of 8x32 for each subgroup
            constexpr size_t prefRow = 8;
            constexpr size_t prefCol = 32;
        // All the SGs of one workgroup prefetch MCACHE2xKCACHE2 of A
        // All the SGs of one workgroup prefetch KCACHE2xNCACHE2 of B
        // PVC case: 256x32 of A and 32x256 of B
        // For both A and B: each subgroup performs a prefetch of
        // prefRow rows and prefCol cols at a time
        // For A, the subgroups are distributed along the row dimension:
        // PVC: A layed as MCACHE2/prefRow (256/32)
        // For B: the subgroups are distributed along the column dimension
        // PVC: NCACHE2/prefCol = 256/32 = 8 SGs on the column dimension and
        // KCACHE2/prefRow = 32/8 = 4 SGs on the row dimension
#ifdef VNNI
            // In the VNNI case, each subgroup still gets prefRow x prefCol
            // In the PVC case: subgroups distribution become
            // NCACHE2*2/prefCol = 16 subgroups on the col dimension and
            // (KCACHE2/2)/prefRow = 2 on the row dimension
            // (NCACHE2*2)/prefCol = 512/32 = 16 SGs on the column dimension and
            // (KCACHE2/2)/prefRow = 16/8 = 2 SGs on the row dimension
            // pm1B and pn1B are used to identify the distribution of subgroups
            // along the workgroup prefetch for B matrix. For A matrix, sgId is
            // enough.
            size_t pm1B = sgId / 16;   // prefetch m1 (sgId/16)
            size_t pn1B = sgId & 0x15; // prefetch n1 (sgId%16)
#else                                  // VNNI
          size_t pm1B = sgId / 8;   // prefetch m1 (sgId/8)
          size_t pn1B = sgId & 0x7; // prefetch n1 (sgId%8)
#endif                                 // VNNI
            constexpr size_t prefDistance = 3;
            for (int p = 0; p < prefDistance; p++)
              joint_matrix_prefetch<prefRow, prefCol>(
                  sg, A + (m2 * MCACHE2 + sgId * prefRow) * colsA + p * prefCol,
                  colsA, layout::row_major,
                  syclex::properties{syclex::prefetch_hint_L1});

#ifdef VNNI
            for (int p = 0; p < prefDistance; p++)
              joint_matrix_prefetch<prefRow, prefCol>(
                  sg,
                  B +
                      (p * (KCACHE2 / vnniFactor) + pm1B * prefRow) * colsB *
                          vnniFactor +
                      (n2 * NCACHE2 * vnniFactor + pn1B * prefCol),
                  colsB * vnniFactor, layout::row_major,
                  syclex::properties{syclex::prefetch_hint_L1});
#else  // VNNI
          for (int p = 0; p < prefDistance; p++)
            joint_matrix_prefetch<prefRow, prefCol>(
                sg,
                B + (p * KCACHE2 + pm1B * prefRow) * colsB + n2 * NCACHE2 +
                    pn1B * prefCol,
                colsB, layout::row_major,
                syclex::properties{syclex::prefetch_hint_L1});
#endif // VNNI
#endif // PREFETCH

            joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
                tC[MCACHE1 / tM][NCACHE1 / tN];
            for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                joint_matrix_fill(sg, tC[m][n], 0);
              }
            }
#ifdef ANNOT
            auto pA = syclex::annotated_ptr{
                A, syclex::properties{
                       syclintelex::read_hint<syclintelex::cache_control<
                           syclintelex::cache_mode::uncached,
                           syclex::cache_level::L2, syclex::cache_level::L3>>}};
            auto pB = syclex::annotated_ptr{
                B, syclex::properties{
                       syclintelex::read_hint<syclintelex::cache_control<
                           syclintelex::cache_mode::cached,
                           syclex::cache_level::L2, syclex::cache_level::L3>>}};
#endif

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
#ifdef OOB
                  ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, tA[m][k1], pA, colsA, rowsA, colsA,
                      m2 * MCACHE2 + m1 * MCACHE1 + m * tM, k * tK);
#else  // OOB
                  joint_matrix_load(
                      sg, tA[m][k1],
                      pA + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsA +
                          k * tK,
                      colsA);
#endif // OOB
                }
                for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#ifdef OOB
#ifdef VNNI
                  ext::intel::experimental::matrix::joint_matrix_load_checked(
                      sg, tB[n][k1], pB, colsB * vnniFactor, rowsB / vnniFactor,
                      colsB * vnniFactor, k * tK / vnniFactor,
                      (n2 * NCACHE2 + n1 * NCACHE1 + n * tN) * vnniFactor);
#else // VNNI
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tB[n][k1], pB, colsB, rowsB, colsB, k * tK,
                    n2 * NCACHE2 + n1 * NCACHE1 + n * tN);

#endif // VNNI
#else  // OOB
#ifdef VNNI
                  joint_matrix_load(
                      sg, tB[n][k1],
                      pB + (k * tK / vnniFactor) * (colsB * vnniFactor) +
                          (n2 * NCACHE2 + n1 * NCACHE1 + n * tN) * vnniFactor,
                      colsB * vnniFactor);
#else  // VNNI
                  joint_matrix_load(
                      sg, tB[n][k1],
                      pB + (k * tK) * (colsB) +
                          (n2 * NCACHE2 + n1 * NCACHE1 + n * tN), colsB);
#endif // VNNI
#endif // OOB
                }
                for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
                  for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
                    joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                     tC[m][n]); // 32 DPAS
                  }
                }
              } // for k1
#ifdef PREFETCH
              auto prefetch_offsetA = (m2 * MCACHE2 + sgId * prefRow) * colsA +
                                      (k2 + prefDistance) * prefCol;
              if ((prefetch_offsetA + (prefRow * MATRIX_K) + prefCol) <
                  (MATRIX_M * MATRIX_K))
                joint_matrix_prefetch<prefRow, prefCol>(
                    sg, A + prefetch_offsetA, colsA, layout::row_major,
                    syclex::properties{syclex::prefetch_hint_L1});

#ifdef VNNI
              auto prefetch_offsetB =
                  ((k2 + prefDistance) * (KCACHE2 / vnniFactor) +
                   pm1B * prefRow) *
                      (colsB)*vnniFactor +
                  (n2 * NCACHE2 * vnniFactor + pn1B * prefCol);
              if ((prefetch_offsetB + (prefRow * MATRIX_N * vnniFactor) +
                   prefCol) < (MATRIX_K * MATRIX_N))
                joint_matrix_prefetch<prefRow, prefCol>(
                    sg, B + prefetch_offsetB, colsB * vnniFactor,
                    layout::row_major,
                    syclex::properties{syclex::prefetch_hint_L1});
#else  // VNNI
            auto prefetch_offsetB =
                ((k2 + prefDistance) * KCACHE2 + pm1B * prefRow) * (colsB) +
                (n2 * NCACHE2 + pn1B * prefCol);
            if ((prefetch_offsetB + (prefRow * MATRIX_N) + prefCol) <
                (MATRIX_K * MATRIX_N))
              joint_matrix_prefetch<prefRow, prefCol>(
                  sg, B + prefetch_offsetB, colsB, layout::row_major,
                  syclex::properties{syclex::prefetch_hint_L1});
#endif // VNNI
#endif // PREFETCH
            } // for k2
            for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_store_checked(
                    sg, tC[m][n], pC, colsB, layout::row_major, rowsA, colsB,
                    m2 * MCACHE2 + m1 * MCACHE1 + m * tM,
                    n2 * NCACHE2 + n1 * NCACHE1 + n * tN);
#else  // OOB
                joint_matrix_store(
                    sg, tC[m][n],
                    pC + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsB +
                        (n2 * NCACHE2 + n1 * NCACHE1 + n * tN),
                    colsB, layout::row_major);
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
