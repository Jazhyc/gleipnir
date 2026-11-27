// Isolated CUTLASS native S4 x S4 -> S32 GEMM, not a serving backend.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

template<int BM, int BN>
int launch(const void* a, const void* b, void* c, int m, int n, int k,
           cudaStream_t stream) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor,
        cutlass::int4b_t, cutlass::layout::ColumnMajor,
        int, cutlass::layout::RowMajor, int,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<BM, BN, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::epilogue::thread::LinearCombination<int, 4, int, int>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    typename Gemm::Arguments args(
        {m, n, k}, {static_cast<const cutlass::int4b_t*>(a), k},
        {static_cast<const cutlass::int4b_t*>(b), k},
        {static_cast<int*>(c), n}, {static_cast<int*>(c), n}, {1, 0});
    Gemm op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
    return static_cast<int>(op(args, nullptr, stream));
}

extern "C" int int4_gemm(const void* a, const void* b, void* c,
                         int m, int n, int k, int tile, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (tile == 0) return launch<128, 128>(a, b, c, m, n, k, s);
    if (tile == 1) return launch<128, 64>(a, b, c, m, n, k, s);
    if (tile == 2) return launch<64, 128>(a, b, c, m, n, k, s);
    return -1;
}
