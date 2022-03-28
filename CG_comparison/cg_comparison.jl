using CUDA
using Random

# include("level_2_multigrid_new.jl")
include("../split_matrix_free.jl")

evel=6,nu=3,ω=2/3,SBPp=2

(A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level,p=SBPp);

A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
x_GPU_sparse = CuArray(zeros(Nx*Ny))
b_GPU_sparse = CuArray(b)


x_GPU = CuArray(zeros(Nx,Ny))
b_GPU = CuArray(reshape(b,Nx,Ny))




cg(A_GPU_sparse,b_GPU_sparse;abstol=1e-6,maxiter=100)
CG_full_GPU(b_GPU,x_GPU;abstol=1e-6)



