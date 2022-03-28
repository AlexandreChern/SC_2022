using CUDA
using Random

include("../preconditioned_CG/level_2_multigrid_new.jl")
include("../split_matrix_free.jl")

level= 11
nu=3
ω=2/3
SBPp=2

reltol = sqrt(eps(Float64))

(A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level,p=SBPp);

A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
x_GPU_sparse = CuArray(zeros(Nx*Ny))
b_GPU_sparse = CuArray(b)
b_reshaped_GPU = reshape(b_GPU_sparse,Nx,Ny)

x_GPU = CuArray(zeros(Nx,Ny))
b_GPU = CuArray(reshape(b,Nx,Ny))



x,history = cg!(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol*norm(b),log=true)
history.iters
CG_full_GPU(b_GPU,x_GPU;abstol=norm(b)*reltol)

b_reshaped_GPU = reshape(reverse(b_GPU_sparse),Nx,Ny)

Ap_GPU = CuArray(zeros(Nx,Ny));

function CG_Matrix_Free_GPU(x_GPU,Ap_GPU,b_reshaped_GPU,Nx,Ny;abstol=reltol)
    matrix_free_A_full_GPU(x_GPU,Ap_GPU)
    r_GPU = b_reshaped_GPU - Ap_GPU
    p_GPU = copy(r_GPU)
    # rsold_GPU = sum(r_GPU .* r_GPU)
    rsold_GPU = dot(r_GPU,r_GPU)
    num_iter_steps = 0
    # for i in 1:Nx*Ny
    norms = [sqrt(rsold_GPU)]
    for i in 1:Nx*Ny
        num_iter_steps += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        # alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        alpha_GPU = rsold_GPU / dot(p_GPU,Ap_GPU)
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        # rsnew_GPU = sum(r_GPU .* r_GPU)
        rsnew_GPU = dot(r_GPU,r_GPU)
        if rsnew_GPU < abstol^2
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) .* p_GPU
        rsold_GPU = rsnew_GPU
        push!(norms,sqrt(rsnew_GPU))
    end
    return num_iter_steps,abstol,norms[end]
end


function CG_GPU_sparse(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol)
    Ap_GPU_sparse = A_GPU_sparse * x_GPU_sparse
    r_GPU_sparse = b_GPU_sparse - Ap_GPU_sparse
    p_GPU_sparse = copy(r_GPU_sparse)
    rsold_GPU = sum(r_GPU_sparse .* r_GPU_sparse)
    num_iter_steps = 0
    norms = [sqrt(rsold_GPU)]
    for i in 1:Nx*Ny
        num_iter_steps += 1
        # Ap_GPU_sparse .= A_GPU_sparse * p_GPU_sparse
        mul!(Ap_GPU_sparse,A_GPU_sparse,p_GPU_sparse)
        # alpha_GPU = rsold_GPU / (sum(p_GPU_sparse .* Ap_GPU_sparse))
        alpha_GPU = rsold_GPU / dot(p_GPU_sparse,Ap_GPU_sparse)
        r_GPU_sparse .-= alpha_GPU .* Ap_GPU_sparse
        x_GPU_sparse .+= alpha_GPU .* p_GPU_sparse
        # rsnew_GPU = sum(r_GPU_sparse .* r_GPU_sparse)
        rsnew_GPU = dot(r_GPU_sparse,r_GPU_sparse)
        if rsnew_GPU < abstol^2
            break
        end
        p_GPU_sparse .= r_GPU_sparse .+ (rsnew_GPU / rsold_GPU) .* p_GPU_sparse
        rsold_GPU = rsnew_GPU
        push!(norms,sqrt(rsnew_GPU))
    end
    return num_iter_steps,abstol,norms[end]
end

x_GPU_sparse .= 0
CG_GPU_sparse(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol*norm(b))

x_GPU .= 0
num_iters_steps,tolerance,last_norm = CG_Matrix_Free_GPU(x_GPU,Ap_GPU,b_reshaped_GPU,Nx,Ny;abstol=reltol*norm(b))

x_GPU .= 0
CG_Matrix_Free_GPU(x_GPU,Ap_GPU,b_reshaped_GPU,Nx,Ny;abstol=reltol*norm(b))

REPEAT = 3

time_CG_GPU_sparse = @elapsed for _ in 1:REPEAT
    x_GPU_sparse  .= 0
    cg!(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol*norm(b),log=true)
end

time_CG_GPU_sparse_diy = @elapsed for _ in 1:REPEAT
    x_GPU_sparse  .= 0
    CG_GPU_sparse(x_GPU_sparse,A_GPU_sparse,b_GPU_sparse;abstol=reltol*norm(b))
end

time_CG_full_GPU = @elapsed for _ in 1:REPEAT
    x_GPU .= 0
    CG_full_GPU(b_GPU,x_GPU;abstol=norm(b)*reltol)
end

time_CG_Matrix_Free_GPU  = @elapsed for _ in 1:REPEAT
    x_GPU .= 0
    CG_Matrix_Free_GPU(x_GPU,Ap_GPU,b_reshaped_GPU,Nx,Ny;abstol=reltol*norm(b))
end

@show time_CG_full_GPU
@show time_CG_Matrix_Free_GPU