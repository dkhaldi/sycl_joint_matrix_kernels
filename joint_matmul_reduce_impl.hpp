// Optimizations:
// cache tiling of k and j
// Reduction along the k dimension using atomics
// data reuse of A and B in physical layer
// TODO: add prefetch and OOB optimization

template <unsigned int M, unsigned int N, unsigned int K,
          unsigned int vnniFactor, typename TOperand, typename TResult,
          size_t tM, size_t tN, size_t tK, size_t TMCACHE1, size_t TNCACHE1,
          size_t TKCACHE1, size_t TMCACHE2, size_t TNCACHE2, size_t TKCACHE2,
          class kernel_name>
double joint_matmul_reduce(TOperand *A, TOperand *B, TResult *C, queue &q,
                           int testIterations) {
  size_t SG_SIZE = get_sg_size<kernel_name>(q);
  range<2> global{K / TKCACHE1, (N / TNCACHE1) * SG_SIZE};
  range<2> cachelocal{TKCACHE2 / TKCACHE1, TNCACHE2 / TNCACHE1 * SG_SIZE};

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (unsigned int i = 0; i < testIterations; i++) {
    q.memset(C, 0, sizeof(float) * M * N);
    q.wait();

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
                tC[TMCACHE1 / tM][TNCACHE1 / tN];
            for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                joint_matrix_fill(sg, tC[m][n], 0);
              }
            }

            joint_matrix<sub_group, TOperand, use::a, tM, tK, layout::row_major>
                tA[TMCACHE1 / tM][TKCACHE2 / TKCACHE1];
#ifdef VNNI
            joint_matrix<sub_group, TOperand, use::b, tK, tN,
                         layout::ext_intel_packed>
                tB[TNCACHE1 / tN][TKCACHE2 / TKCACHE1];
#else
            joint_matrix<sub_group, TOperand, use::b, tK, tN, layout::row_major>
                tB[TNCACHE1 / tN][TKCACHE2 / TKCACHE1];
#endif
            for (unsigned int k1 = 0; k1 < TKCACHE2 / TKCACHE1; k1++) {
              //  physical layer
              unsigned int k = (0 * TKCACHE2 + k1 * TKCACHE1) / tK;

              for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
                joint_matrix_load(sg, tA[m][k1],
                                  pA + (m * tM) * K + k * tK + m2 * TKCACHE2 +
                                      m1 * TKCACHE1,
                                  K);
              }
              for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
#ifdef VNNI
                joint_matrix_load(sg, tB[n][k1],
                                  pB + (m2 * TKCACHE2 + m1 * TKCACHE1) * (N) +

                                      (n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN) *
                                          vnniFactor,
                                  N * vnniFactor);

#else
                joint_matrix_load(sg, tB[n][k1],
                                  pB + (m2 * TKCACHE2 + m1 * TKCACHE1) * (N) +
                                      (n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN),
                                  N);
#endif
              }
              for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
                for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
                  joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                   tC[m][n]); // 4 DPAS
                }
              }
            } // for k1

            for (unsigned int m = 0; m < TMCACHE1 / tM; m++) {
              for (unsigned int n = 0; n < TNCACHE1 / tN; n++) {
#ifdef LEGACY
                auto wi_slice = ext::oneapi::detail::get_wi_data(sg, tC[m][n]);

                for (int i = 0; i < wi_slice.length(); i++) {
                  auto atm = atomic_ref<TResult, memory_order::relaxed,
                                        memory_scope::device,
                                        access::address_space::global_space>(
                      C[n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN +
                        sg.get_local_id()]);
                  atm.fetch_add(wi_slice[i], memory_order::relaxed);
                }
#else
              joint_matrix_apply(sg, tC[m][n], [=](TResult &x) {
                auto atm = atomic_ref<TResult, memory_order::relaxed,
                                      memory_scope::device,
                                      access::address_space::global_space>(
                    C[n2 * TNCACHE2 + n1 * TNCACHE1 + n * tN +
                      sg.get_local_id()]);
                atm.fetch_add(x, memory_order::relaxed);
              });

#endif
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
