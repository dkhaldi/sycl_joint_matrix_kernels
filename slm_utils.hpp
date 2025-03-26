//==------------------ slm_utils.hpp  - DPC++ joint_matrix------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int MCache2, unsigned int NCache2,
          unsigned int KCache2, unsigned int vnniFactor, size_t SGs,
          typename TOperand, access::address_space Space>
inline void
slm_read_write(multi_ptr<TOperand, Space, access::decorated::yes> pA,
               multi_ptr<TOperand, Space, access::decorated::yes> pB,
               multi_ptr<TOperand, access::address_space::local_space,
                         access::decorated::yes>
                   tileA,
               multi_ptr<TOperand, access::address_space::local_space,
                         access::decorated::yes>
                   tileB,
               sub_group sg, unsigned int k2, size_t m2, size_t n2,
               size_t sgSize) {
  auto striped = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::data_placement_striped,
      sycl::ext::oneapi::experimental::full_group,
      sycl::ext::oneapi::experimental::detail::native_local_block_io,
      // Alignment is assumed here
      syclex::alignment<16>};
  // SLM read/write depend on SG size, when this is a constexpr, performance is
  // a improved (2x)
  // An another way is to pass the SG size in the template after right dispatch.
  // The dispatch seems to cause regression when number of branches > 1
#ifdef SG_SIZE
  constexpr unsigned int sgSizeConst = SG_SIZE;
#else
  unsigned int sgSizeConst = sgSize;
#endif
  // A further optimization is to make these SLM read elements constexpr to get
  // rid of the branches early (6%)
#ifdef SG_SIZE
  constexpr size_t slmReadA = KCache2 / sgSizeConst;
#else
  size_t slmReadA = KCache2 / sgSizeConst;
#endif

  for (int i = 0; i < (MCache2 / SGs); i++) {
#ifdef SG_SIZE
    if constexpr (slmReadA == 2) {
#else
    if (slmReadA == 2) {
#endif
      vec<TOperand, 2> slmVecA;
      sycl::ext::oneapi::experimental::group_load(
          sg,
          pA.get() +

              (m2 * MCache2 + sg.get_group_id() * (MCache2 / SGs) + i) * colsA +
              k2 * KCache2,
          slmVecA, striped);
      sycl::ext::oneapi::experimental::group_store(
          sg, slmVecA,
          tileA.get() + (sg.get_group_id() * (MCache2 / SGs) + i) * KCache2,
          striped);
#ifdef SG_SIZE
    } else if constexpr (slmReadA == 4) {
#else
    } else if (slmReadA == 4) {
#endif
      vec<TOperand, 4> slmVecA;
      sycl::ext::oneapi::experimental::group_load(
          sg,
          pA.get() +
              (m2 * MCache2 + sg.get_group_id() * (MCache2 / SGs) + i) * colsA +
              k2 * KCache2,
          slmVecA, striped);
      sycl::ext::oneapi::experimental::group_store(
          sg, slmVecA,
          tileA.get() + ((sg.get_group_id()) * (MCache2 / SGs) + i) * KCache2,
          striped);
#ifdef SG_SIZE
    } else if constexpr (slmReadA == 1) {
#else
    } else if (slmReadA == 1) {
#endif
      TOperand slmS;
      sycl::ext::oneapi::experimental::group_load(
          sg,
          pA.get() +
              (m2 * MCache2 + sg.get_group_id() * (MCache2 / SGs) + i) * colsA +
              k2 * KCache2,
          slmS, striped);
      sycl::ext::oneapi::experimental::group_store(
          sg, slmS,
          tileA.get() + (sg.get_group_id() * (MCache2 / SGs) + i) * KCache2,
          striped);
    } else
      assert(slmReadA == 1 || slmReadA == 2 || slmReadA == 4);
  } // end i
  // how much each SG will load to SLM --> has to be contiguous
  // NCache2*KCache2/(SGs*SG_SIZE) = 16
#ifdef SG_SIZE
  constexpr size_t slmRead = NCache2 * KCache2 / (SGs * sgSizeConst);
  constexpr size_t sgsPerRow = (NCache2 * vnniFactor) / (slmRead * sgSizeConst);
#else
  size_t slmRead = NCache2 * KCache2 / (SGs * sgSizeConst);
  size_t sgsPerRow = (NCache2 * vnniFactor) / (slmRead * sgSizeConst);
#endif
  // Bug: This dummy loop seems to trigger some LICM in IGC
  for (unsigned int b = 0; b < NCache2 / (16 * sgSize); b++) {
#ifdef SG_SIZE
    if constexpr (slmRead == 16) {
#else  // SG_SIZE
    if (slmRead == 16) {
#endif // SG_SIZE
      vec<TOperand, 16> slmVecB;
      sycl::ext::oneapi::experimental::group_load(
          sg,
          pB.get() +
              (k2 * (KCache2 / vnniFactor) +
               (uint)(sg.get_group_id() / sgsPerRow)) *
                  (colsB * vnniFactor) +
              n2 * NCache2 * vnniFactor +
              (sg.get_group_id() % sgsPerRow) * (slmRead * sgSizeConst),
          slmVecB, striped);
      sycl::ext::oneapi::experimental::group_store(
          sg, slmVecB,
          tileB.get() +
              ((uint)sg.get_group_id() / sgsPerRow) * NCache2 * vnniFactor +
              (sg.get_group_id() % sgsPerRow) * (slmRead * sgSizeConst),
          striped);
#ifdef SG_SIZE
    } else if constexpr (slmRead == 8) {
#else  // SG_SIZE
    } else if (slmRead == 8) {
#endif // SG_SIZE
      vec<TOperand, 8> slmVecB;
      sycl::ext::oneapi::experimental::group_load(
          sg,
          pB.get() +
              (k2 * (KCache2 / vnniFactor) +
               ((uint)sg.get_group_id() / sgsPerRow)) *
                  (colsB * vnniFactor) +
              n2 * NCache2 * vnniFactor +
              (sg.get_group_id() % sgsPerRow) * (slmRead * sgSizeConst),
          slmVecB, striped);
      sycl::ext::oneapi::experimental::group_store(
          sg, slmVecB,
          tileB.get() +
              ((uint)sg.get_group_id() / sgsPerRow) * NCache2 * vnniFactor +
              (sg.get_group_id() % sgsPerRow) * slmRead * sgSizeConst,
          striped);
    } else
      assert(slmRead == 8 || slmRead == 16);
  } // end dummy loop
}
