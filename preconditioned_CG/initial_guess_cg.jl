using CUDA
using Random

include("level_2_multigrid_new.jl")
include("../split_matrix_free.jl")

Random.seed!(123)

if length(ARGS) != 0
    level = parse(Int,ARGS[1])
    # Iterations = parse(Int,ARGS[2])
else
    level = 10
    # Iterations = 10000
end

function matrix_free_prolongation_2d(idata,odata)
    size_idata = size(idata)
    odata_tmp = zeros(size_idata .* 2)
    for i in 1:size_idata[1]-1
        for j in 1:size_idata[2]-1
            odata[2*i-1,2*j-1] = idata[i,j]
            odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
            odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
            odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
        end
    end
    for j in 1:size_idata[2]-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end
    for i in 1:size_idata[1]-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end
    odata[end,end] = idata[end,end]
    return nothing
end


function prolongation_2D_kernel(idata,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy


    if 1 <= i <= Nx-1 && 1 <= j <= Ny-1
        odata[2*i-1,2*j-1] = idata[i,j]
        odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
        odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
        odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
    end 

    if 1 <= j <= Ny-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end

    if 1 <= i <= Nx-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end

    odata[end,end] = idata[end,end]
    return nothing
end

function matrix_free_prolongation_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim prolongation_2D_kernel(idata,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing
end


function matrix_free_restriction_2d(idata,odata)
    size_idata = size(idata)
    size_odata = div.(size_idata .+ 1,2)
    idata_tmp = zeros(size_idata .+ 2)
    idata_tmp[2:end-1,2:end-1] .= idata

    for i in 1:size_odata[1]
        for j in 1:size_odata[2]
            odata[i,j] = (4*idata_tmp[2*i,2*j] + 
            2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
             (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        end
    end
    return nothing
end



function restriction_2D_kernel(idata_tmp,odata,Nx,Ny,::Val{TILE_DIM_1},::Val{TILE_DIM_2}) where {TILE_DIM_1,TILE_DIM_2}
    tidx = threadIdx().x
    tidy = threadIdx().y
    i = (blockIdx().x - 1) * TILE_DIM_1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM_2 + tidy

    # idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    # idata_tmp[2:end-1,2:end-1] .= idata

    size_odata = (div(Nx+1,2),div(Ny+1,2))

    if 1 <= i <= size_odata[1] && 1 <= j <= size_odata[2]
        odata[i,j] = (4*idata_tmp[2*i,2*j] + 
        2 * (idata_tmp[2*i,2*j-1] + idata_tmp[2*i,2*j+1] + idata_tmp[2*i-1,2*j] + idata_tmp[2*i+1,2*j]) +
         (idata_tmp[2*i-1,2*j-1] + idata_tmp[2*i-1,2*j+1] + idata_tmp[2*i+1,2*j-1]) + idata_tmp[2*i+1,2*j+1]) / 16
        # odata[i,j] = idata_tmp[2*i,2*j]
        # odata[i,j] = 1
    end
   
    return nothing
end

function matrix_free_restriction_2d_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    idata_tmp = CuArray(zeros(Nx+2,Ny+2))
    copyto!(view(idata_tmp,2:Nx+1,2:Ny+1),idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim restriction_2D_kernel(idata_tmp,odata,Nx,Ny,Val(TILE_DIM_1),Val(TILE_DIM_2))
    nothing

end


function matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=3,ω=0.15)
    if maxiter==0
        odata_GPU .= idata_GPU
    else
        for _ in 1:maxiter
            matrix_free_A_full_GPU(idata_GPU,odata_GPU) # matrix_free_A_full_GPU is -A here, becareful
            odata_GPU .= idata_GPU .+ ω * (b_GPU .+ odata_GPU)
            idata_GPU .= odata_GPU
        end
    end
end


function matrix_free_Two_level_multigrid(b_GPU,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    (Nx_2h,Ny_2h) = div.((Nx,Ny) .+ 1,2)
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    v_values_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    v_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    Av_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    rhs_values_GPU = Dict(1=>b_GPU)
    N_values = Dict(1=>Nx)
    N_values[2] = div(Nx+1,2)
    f_GPU = Dict(1=>CuArray(zeros(Nx_2h,Ny_2h)))
    e_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    
    for cycle_number in 1:NUM_V_CYCLES
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
        r_GPU = b_GPU + Av_values_out_GPU[1]
        matrix_free_restriction_2d_GPU(r_GPU,f_GPU[1])
        # v_values_GPU[2] = reshape(CuArray(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        # v_values_GPU[2] = reshape(CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h) \ f_GPU[1][:],Nx_2h,Ny_2h)
        v_values_GPU[2] = reshape(CuArray(A_2h \ Array(f_GPU[1][:])),Nx_2h,Ny_2h)

        # matrix_free_richardson(v_values_out_GPU[2],v_values_GPU[2],f_GPU[1];maxiter=20)

        matrix_free_prolongation_2d_GPU(v_values_GPU[2],e_GPU[1])
        v_values_GPU[1] .+= e_GPU[1]
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end

function matrix_free_Two_level_multigrid_simple(b_GPU,A_2h;nu=3,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    (Nx_2h,Ny_2h) = div.((Nx,Ny) .+ 1,2)

    v_value = CuArray(zeros(Nx,Ny))
    v_out = CuArray(zeros(Nx,Ny))
    Av_value = CuArray(zeros(Nx,Ny))
    f_GPU = CuArray(zeros(Nx_2h,Ny_2h))
    e_GPU = CuArray(zeros(Nx,Ny))
    matrix_free_richardson(v_value,v_out,b_GPU;maxiter=nu)
    matrix_free_A_full_GPU(v_out,Av_value)
    r_GPU = b_GPU + Av_value
    matrix_free_restriction_2d_GPU(r_GPU,f_GPU)
    v_value_2 =  reshape(CuArray(A_2h \ Array(f_GPU[:])),Nx_2h,Ny_2h)
    matrix_free_prolongation_2d_GPU(v_value_2,e_GPU)
    v_value .+= e_GPU
    matrix_free_richardson(v_value,v_out,b_GPU;maxiter=nu)
    return v_out
end

function Three_level_multigrid(A,b,A_2h,b_2h,A_4h,b_4h,Nx,Ny;nu=3,NUM_V_CYCLES=1,SBPp=2)
    v_values = Dict(1=>zeros(Nx*Ny))
    Nx_2h = Ny_2h = div(Nx+1,2)
    Nx_4h = Ny_4h = div(Nx_2h+1,2)

    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = Nx_2h
    N_values[3] = Nx_4h

    x = zeros(length(b));
    v_values[1] = x
    v_values[2] = zeros(Nx_2h*Ny_2h)
    
    for cycle_number in 1:NUM_V_CYCLES
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[1],A,b,maxiter=nu)
        r_h = b - A*v_values[1];

        rhs_values[2] = restriction_2d(Nx) * r_h;

        modified_richardson!(v_values[2],A_2h,rhs_values[2],maxiter=nu)
        # r_2h = b_2h - A_2h * v_values[2]
        r_2h = - A_2h * v_values[2]
        rhs_values[3] = restriction_2d(Nx_2h) * r_2h

        v_values[3] = A_4h \ rhs_values[3]

        # println("Pass first part")

        e_2 = prolongation_2d(N_values[3]) * v_values[3];
        v_values[2] = v_values[2] + e_2;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        # jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        modified_richardson!(v_values[2],A_2h,rhs_values[2],maxiter=nu)
        e_1 = prolongation_2d(N_values[2]) * v_values[2]
        v_values[1] = v_values[1] + e_1
        modified_richardson!(v_values[1],A,b,maxiter=nu)
    end
    return (v_values[1],norm(A * v_values[1] - b))
end

function matrix_free_Three_level_multigrid(b_GPU,A_4h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    (Nx_2h,Ny_2h) = div.((Nx,Ny) .+ 1,2)
    (Nx_4h,Ny_4h) = div.((Nx_2h,Ny_2h) .+ 1,2)
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    v_values_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))

    v_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    v_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))

    Av_values_out_GPU = Dict(1=>CuArray(zeros(Nx,Ny)))
    Av_values_out_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    rhs_values_GPU = Dict(1=>b_GPU)
    rhs_values_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h))
    rhs_values_GPU[3] = CuArray(zeros(Nx_4h,Ny_4h))

    N_values = Dict(1=>Nx)
    N_values[2] = Nx_2h
    N_values[3] = Nx_4h
    f_GPU = Dict(1=>CuArray(zeros(Nx_2h,Ny_2h)))
    f_GPU[2] = CuArray(zeros(Nx_4h,Ny_4h))

    e_GPU = Dict(1=>CuArray(zeros(Nx,Ny)));
    e_GPU[2] = CuArray(zeros(Nx_2h,Ny_2h));
    
    for cycle_number in 1:NUM_V_CYCLES
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
        r_h_GPU = b_GPU + Av_values_out_GPU[1]

        matrix_free_restriction_2d_GPU(r_h_GPU,rhs_values_GPU[2])

        matrix_free_richardson(v_values_GPU[2],v_values_out_GPU[2],rhs_values_GPU[2];maxiter=nu)
        matrix_free_A_full_GPU(v_values_out_GPU[2],Av_values_out_GPU[2])

        # r_2h_GPU = b_2h_GPU + Av_values_out_GPU[2]
        r_2h_GPU = Av_values_out_GPU[2]
        matrix_free_restriction_2d_GPU(r_2h_GPU,rhs_values_GPU[3])

        v_values_GPU[3] = reshape(CuArray(A_4h \ Array(rhs_values_GPU[3][:])),Nx_4h,Ny_4h)

        # matrix_free_richardson(v_values_out_GPU[2],v_values_GPU[2],f_GPU[1];maxiter=20)

        matrix_free_prolongation_2d_GPU(v_values_GPU[3],e_GPU[2])
        v_values_out_GPU[2] += e_GPU[2]
        v_values_GPU[2] .= v_values_out_GPU[2]
        matrix_free_richardson(v_values_GPU[2],v_values_out_GPU[2],rhs_values_GPU[2];maxiter=nu)
        matrix_free_prolongation_2d_GPU(v_values_out_GPU[2],e_GPU[1])
        v_values_out_GPU[1] += e_GPU[1]
        v_values_GPU[1] .= v_values_out_GPU[1]
        matrix_free_richardson(v_values_out_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
        matrix_free_richardson(v_values_GPU[1],v_values_out_GPU[1],rhs_values_GPU[1];maxiter=nu)
    end
    matrix_free_A_full_GPU(v_values_out_GPU[1],Av_values_out_GPU[1])
    return (v_values_out_GPU[1],norm(-Av_values_out_GPU[1]-b_GPU))
end

function matrix_free_MGCG(b_GPU,x_GPU;A_2h = A_2h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    # A_2h = lu(A_2h)
    Ax_GPU = CuArray(zeros(size(x_GPU)))
    matrix_free_A_full_GPU(x_GPU,Ax_GPU)
    r_GPU = b_GPU + Ax_GPU
    z_GPU = matrix_free_Two_level_multigrid(r_GPU,A_2h;nu=nu)[1]
    p_GPU = copy(z_GPU)
    Ap_GPU = copy(p_GPU)
    num_iter_steps_GPU = 0
    norms_GPU = [norm(r_GPU)]
    errors_GPU = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol)) # need to rewrite
    end

    rzold_GPU = sum(r_GPU .* z_GPU)

    for step = 1:maxiter
        num_iter_steps_GPU += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        alpha_GPU = - rzold_GPU / sum(p_GPU .* Ap_GPU)
        x_GPU .+= alpha_GPU .* p_GPU
        r_GPU .+= alpha_GPU .* Ap_GPU
        rs_GPU = sum(r_GPU .* r_GPU)
        append!(norms_GPU,sqrt(rs_GPU))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol)) # need to rewrite
            # @show error
            append!(errors,error)
        end
        if sqrt(rs_GPU) < abstol
            break
        end
        z_GPU .=  matrix_free_Two_level_multigrid(r_GPU,A_2h;nu=nu)[1]
        rznew_GPU = sum(r_GPU .* z_GPU)
        beta_GPU = rznew_GPU / rzold_GPU
        p_GPU .= z_GPU .+ beta_GPU .* p_GPU
        rzold_GPU = rznew_GPU
    end
    return num_iter_steps_GPU, norms_GPU, x_GPU
end

function matrix_free_MGCG_Three_level(b_GPU,x_GPU;A_4h = A_4h_lu,maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)
    (Nx,Ny) = size(b_GPU)
    level = Int(log(2,Nx-1))
    # (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp)
    # A_2h = lu(A_2h)
    Ax_GPU = CuArray(zeros(size(x_GPU)))
    matrix_free_A_full_GPU(x_GPU,Ax_GPU)
    r_GPU = b_GPU + Ax_GPU
    z_GPU = matrix_free_Three_level_multigrid(r_GPU,A_4h;nu=nu)[1]
    p_GPU = copy(z_GPU)
    Ap_GPU = copy(p_GPU)
    num_iter_steps_GPU = 0
    norms_GPU = [norm(r_GPU)]
    errors_GPU = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol)) # need to rewrite
    end

    rzold_GPU = sum(r_GPU .* z_GPU)

    for step = 1:maxiter
        num_iter_steps_GPU += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU)
        alpha_GPU = - rzold_GPU / sum(p_GPU .* Ap_GPU)
        x_GPU .+= alpha_GPU .* p_GPU
        r_GPU .+= alpha_GPU .* Ap_GPU
        rs_GPU = sum(r_GPU .* r_GPU)
        append!(norms_GPU,sqrt(rs_GPU))
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol)) # need to rewrite
            # @show error
            append!(errors,error)
        end
        if sqrt(rs_GPU) < abstol
            break
        end
        z_GPU .=  matrix_free_Three_level_multigrid(r_GPU,A_4h;nu=nu)[1]
        rznew_GPU = sum(r_GPU .* z_GPU)
        beta_GPU = rznew_GPU / rzold_GPU
        p_GPU .= z_GPU .+ beta_GPU .* p_GPU
        rzold_GPU = rznew_GPU
    end
    return num_iter_steps_GPU, norms_GPU
end


function initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    x_interpolated = prolongation_2d(Nx_2h) * x_2h
    x,history = cg!(x_interpolated,A,b;abstol = abstol,log=true)
    return x,history.iters, history.data[:resnorm]
end

function initial_guess_interpolation_CG_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h)
    x,history = cg!(x_interpolated,A_GPU,b_GPU;abstol = abstol,log=true)
    return x, history.iters, history.data[:resnorm]
end

function initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    x_2h = A_2h \ b_2h
    # x_2h_flip = zeros(length(x_2h))
    # for i in 1:length(x_2h)
    #     x_2h_flip[i] = x_2h[end+1-i]
    # end
    # x_2h_flip = reverse(x_2h)
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h)
    x_interpolated = reverse(x_interpolated)
    x_interpolated_reshaped = reshape(x_interpolated,size(b_GPU))
    # x,history = cg!(x_interpolated,A_GPU,b_GPU;abstol = abstol,log=true)
    Ap_GPU = similar(b_GPU)
    nums_CG_Matrix_Free_GPU, CG_Matrix_Free_tol, final_norm =  CG_Matrix_Free_GPU_v2(x_interpolated_reshaped,Ap_GPU,b_GPU,Nx,Ny;abstol=sqrt(eps(real(eltype(b_GPU))))) 
    # x = Array(x_interpolated_reshaped)
    # x_flip = similar(x)
    # for i in 1:length(x_flip)
    #     x_flip[i] = x[end+1-i]
    # end
    # x_flip = reverse(x)
    x = reverse(x_interpolated_reshaped[:])
    # return x[:], history.iters, history.data[:resnorm]
    return x, nums_CG_Matrix_Free_GPU, final_norm
end

function MG_interpolation_CG_Matrix_Free_GPU(A_GPU,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    # x_MG_initial_guess = CuArray(zeros(Nx,Ny))
    # Ap_GPU = similar(b_GPU)
    # # x_MG_initial_guess_reverse, _ = matrix_free_Two_level_multigrid(b_GPU,A_2h;nu=0,NUM_V_CYCLES=1,SBPp=2)
    # x_MG_initial_guess_reverse = matrix_free_Two_level_multigrid_simple(b_GPU,A_2h;nu=50,SBPp=2)
    
    x_MG_initial_guess_reverse = Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=100,NUM_V_CYCLES=1,SBPp=2)[1]
    x_MG_initial_guess_reverse_reshaped = reshape(x_MG_initial_guess_reverse,Nx,Ny)
    
    x_MG_initial_guess = CuArray(reverse(x_MG_initial_guess_reverse_reshaped,dims=2))
    nums_CG_Matrix_Free_GPU, CG_Matrix_Free_tol, final_norm = CG_Matrix_Free_GPU_v2(x_MG_initial_guess,Ap_GPU,b_GPU,Nx,Ny;abstol=sqrt(eps(real(eltype(b_GPU))))) 
    x = reverse(x_MG_initial_guess[:])
    return x, nums_CG_Matrix_Free_GPU, final_norm
end

function initial_guess_interpolation_three_level_CG_GPU(A_GPU,A_2h_GPU,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    x_4h = A_4h \ b_4h
    x_2h_interpolated  = prolongation_2d_GPU(Nx_4h) * CuArray(x_4h)
    x_2h_interpolated,history_2h = cg!(x_2h_interpolated,A_2h_GPU,b_2h_GPU;abstol=abstol,log=true)
    x_interpolated = prolongation_2d_GPU(Nx_2h) * CuArray(x_2h_interpolated)
    x,history_h = cg!(x_interpolated,A_GPU,b_GPU;abstol=abstol,log=true)
    # @show history_2h.iters, history_2h.data[:resnorm]
    # @show history_h.iters, history_h.data[:resnorm]
    return x, history_2h.iters,history_h.iters,history_2h.data[:resnorm],history_h.data[:resnorm]
end

function precond_matrix(A, b; m=3, solver="jacobi",ω_richardson=ω_richardson,h=h,SBPp=SBPp)
    #pre and post smoothing 
    N = length(b)
    IN = sparse(Matrix(I, N, N))
    P = Diagonal(diag(A))
    Pinv = Diagonal(1 ./ diag(A))
    Q = P-A
    L = A - triu(A)
    U = A - tril(A)

    if solver == "jacobi"
       ω = 2/3
        H = ω*Pinv*Q + (1-ω)*IN 
        R = ω*Pinv 
        R0 = ω*Pinv 
    elseif solver == "ssor"
        ω = 1.4  #this is just a guess. Need to compute ω_optimal (from jacobi method)
        B1 = (P + ω*U)\Matrix(-ω*L + (1-ω)*P)
        B2 = (P + ω*L)\Matrix(-ω*U + (1-ω)*P) 
        H = B1*B2
        X = (P+ω*L)\Matrix(IN)
   
        R = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
        R0 = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
    elseif solver == "richardson"
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        R0 = ω*IN
    elseif solver == "richardson_chebyshev" #TODO: FIX ME FOR CHEB
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        R0 = ω*IN
    elseif solver == "chebyshev" #TODO: FIX ME FOR CHEB
        ω =ω_richardson
        H = IN - ω*A
        R = ω*IN
        R0 = ω*IN
    else   
    end

    for i = 1:m-1
        R += H^i * R0
    end

    (A_2h, b_2h, x_2h, H1_2h) = get_operators(SBPp, 2*h);
    I_r = standard_restriction_matrix_2D(N)
    
    I_p = standard_prolongation_matrix_2D(length(b_2h))
    M = H^m * (R + I_p * (A_2h\Matrix(I_r*(IN - A * R)))) + R
   
    return (M, R, H, I_p, A_2h, I_r, IN)
end

function test_matrix_free_MGCG(;level=6,nu=3,ω=2/3,SBPp=2)
    (A,b,H_tilde,Nx,Ny,analy_sol) = Assembling_matrix(level,p=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h,analy_sol_2h) = Assembling_matrix(level-1,p=SBPp);
    (A_4h,b_4h,H_tilde_4h,Nx_4h,Ny_4h,analy_sol_4h) = Assembling_matrix(level-2,p=SBPp);

    A_2h_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A_2h)
    b_2h_GPU = CuArray(b_2h)

    A_2h_lu = lu(A_2h)
    A_4h_lu = lu(A_4h)
  
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    # abstol = norm(A*x-b) * reltol
    abstol = reltol

    x_GPU = CuArray(zeros(Nx,Ny))
    b_GPU = CuArray(reshape(b,Nx,Ny))
    b_GPU_v2 = CuArray(reshape(reverse(b),Nx,Ny)) # Remember to reverse b_GPU when using matrix_free

    ω_richardson = 0.15
    h = 1/(Nx-1)
    # (M,R,H,I_p,_,I_r,IN) = precond_matrix(A,b;m=3,solver="richardson",ω_richardson=ω_richardson,h=h,SBPp=SBPp)


    A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    x_GPU_sparse = CuArray(zeros(Nx*Ny))
    b_GPU_sparse = CuArray(b)

  

    x_GPU = CuArray(zeros(Nx,Ny))
    x_GPU_flat = x_GPU[:]
    b_GPU_flat = CuArray(b)
    # @show norms_GPU
    # @show norm_mg_cg
    # @show norm_mg_cg_GPU

    # x_CG_GPU, history = cg(A_GPU_sparse,b_GPU_sparse,abstol=abstol,maxiter=length(b),log=true)
    x_CG_GPU, nums_CG_GPU_step = CG_CPU(A_GPU_sparse,b_GPU_flat,x_GPU_flat)
    error_CG_GPU = sqrt((Array(x_CG_GPU)-analy_sol)'*H_tilde*(Array(x_CG_GPU)-analy_sol))

    nums_CG_Matrix_Free_GPU, CG_Matrix_Free_tol, final_norm = CG_full_GPU(b_GPU_v2,x_GPU;abstol=sqrt(eps(real(eltype(b_GPU)))))
    x_GPU_to_CPU = Array(x_GPU)
    x_GPU_flip = zeros(Nx,Ny)
    for i in 1:Nx*Ny
        x_GPU_flip[i] = x_GPU_to_CPU[end+1-i]
    end
    error_CG_Matrix_Free_GPU = sqrt((x_GPU_flip[:]-analy_sol)'*H_tilde*(x_GPU_flip[:]-analy_sol))

    Ax_GPU = CuArray(zeros(Nx,Ny))
    x_GPU .= 0
    iters_matrix_free_CG = CG_Matrix_Free_GPU_v2(x_GPU,Ax_GPU,b_GPU,Nx,Ny;abstol=reltol) 
    x_GPU_reverse = reverse(x_GPU[:])
    error_CG_Matrix_Free_v2_GPU = sqrt((Array(x_GPU_reverse)-analy_sol)'*H_tilde*(Array(x_GPU_reverse)-analy_sol))


    # @show reshape(x_CG_GPU,Nx,Ny)
    # @show x_GPU
    # x_GPU_flip = CuArray(zeros(Nx,Ny))
    # for i in 1:Nx*Ny
    #     x_GPU_flip[i] = x_GPU[end+1-i]
    # end
    # @show norm(x_GPU_flip - reshape(x_CG_GPU,Nx,Ny))

    x_initial_guess, iter_initial_guess_cg, norm_initial_guess_cg = initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    
    initial_guess_cg_error = sqrt((x_initial_guess - analy_sol)'*H_tilde*(x_initial_guess-analy_sol))

    x_initial_guess_GPU, iter_initial_guess_cg_GPU, norm_initial_guess_cg_GPU = initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_cg_GPU_error = sqrt((Array(x_initial_guess_GPU) - analy_sol)'*H_tilde*(Array(x_initial_guess_GPU)-analy_sol))


    
    x_initial_guess_Matrix_Free_GPU, iter_initial_guess_cg_Matrix_Free_GPU, norm_initial_guess_cg_Matrix_Free_GPU = initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b_GPU))
    initial_guess_cg_Matrix_Free_GPU_error = sqrt((Array(x_initial_guess_Matrix_Free_GPU[:])-analy_sol)'*H_tilde*(Array(x_initial_guess_Matrix_Free_GPU[:])-analy_sol))


    x_MG_initial_guess_Matrix_Free_GPU, iter_MG_initial_guess_cg_Matrix_Free_GPU, norm_MG_initial_guess_cg_Matrix_Free_GPU = MG_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b_GPU))
    MG_initial_guess_cg_Matrix_Free_GPU_error = sqrt((Array(x_MG_initial_guess_Matrix_Free_GPU[:])-analy_sol)'*H_tilde*(Array(x_MG_initial_guess_Matrix_Free_GPU[:])-analy_sol))

    # 3-level interpolation 
    x_initial_guess_three_level_GPU,iter_initial_guess_three_level_cg_GPU_2h,iter_initial_guess_three_level_cg_GPU_h, norm_initial_guess_three_level_cg_GPU_2h,norm_initial_guess_three_level_cg_GPU_h = initial_guess_interpolation_three_level_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    initial_guess_three_level_cg_GPU_error = sqrt((Array(x_initial_guess_three_level_GPU) - analy_sol)'*H_tilde*(Array(x_initial_guess_three_level_GPU) - analy_sol))

    # # 3-level multigrid
    # x_3mg, norm_3mg = Three_level_multigrid(A,b,A_2h,b_2h,A_4h,b_4h,Nx,Ny;nu=3,NUM_V_CYCLES=1,SBPp=2)

    # # 2-level multigrid
    # x_2mg, norm_2mg = Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)



    # # 3-level multigrid GPU 
    # matrix_free_Three_level_multigrid(b_GPU,A_4h;nu=3,NUM_V_CYCLES=1,SBPp=2)

    REPEAT = 1

    # t_matrix_free_MGCG_Three_level_GPU = @elapsed for _ in 1:REPEAT
    #     x_GPU = CuArray(zeros(Nx,Ny))
    #     matrix_free_MGCG_Three_level(b_GPU,x_GPU;A_4h=A_4h_lu,maxiter=length(b_GPU),abstol=abstol,nu=nu)
    # end



    x_GPU_flat .= 0
    t_CG_GPU_sparse = @elapsed for _ in 1:REPEAT
        # cg(A_GPU_sparse,b_GPU_sparse,abstol=abstol,log=true)
        x_GPU_flat .= 0
        # CG_CPU(A_GPU_sparse,b_GPU_flat,x_GPU_flat)
        CG_GPU_sparse(x_GPU_flat,A_GPU_sparse,b_GPU_sparse;abstol=reltol)
    end

    x_GPU .= 0
    x_MG_initial_guess = x_GPU

    t_CG_Matrix_Free_GPU = @elapsed for _ in 1:REPEAT
        # CG_full_GPU(b_GPU,x_GPU;abstol=sqrt(eps(real(eltype(b_GPU)))))
        x_GPU .= 0
        CG_Matrix_Free_GPU_v2(x_GPU,Ax_GPU,b_GPU,Nx,Ny;abstol=reltol) 
    end

    t_CG_CPU_initial_guess = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_CG(A,b,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    end

    t_CG_GPU_initial_guess = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_CG_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))
    end

    t_CG_Matrix_Free_GPU_initial_guess = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b))    
    end

    t_CG_Matrix_Free_GPU_MG_initial_guess = @elapsed for _ in 1:REPEAT
        MG_interpolation_CG_Matrix_Free_GPU(A_GPU_sparse,b_GPU,b_2h,x,Nx_2h;Nx=Nx,Ny=Ny,A_2h = A_2h_lu,abstol=abstol,maxiter=length(b)) 
        # matrix_free_MGCG(b_GPU,x_MG_initial_guess;A_2h = A_2h,maxiter=length(b_GPU),abstol=sqrt(eps(real(eltype(b_GPU)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0,SBPp=2)  
        # matrix_free_Two_level_multigrid(b_GPU,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)
    end

    t_CG_GPU_initial_guess_three_level = @elapsed for _ in 1:REPEAT
        initial_guess_interpolation_three_level_CG_GPU(A_GPU_sparse,A_2h_GPU_sparse,b_GPU,b_2h_GPU,b_2h,b_4h,x,Nx_2h,Nx_4h;A_2h = A_2h_lu, A_4h = A_4h_lu,abstol=abstol,maxiter=length(b))
    end

    println()

    # t_matrix_free_MGCG_Three_level_GPU = t_matrix_free_MGCG_Three_level_GPU / REPEAT
    t_CG_GPU_sparse /= REPEAT
    t_CG_CPU_initial_guess /= REPEAT
    t_CG_GPU_initial_guess /= REPEAT
    t_CG_Matrix_Free_GPU_initial_guess /= REPEAT
    t_CG_GPU_initial_guess_three_level /= REPEAT

    @show Nx, Ny

  
    # @show t_CG_GPU_sparse, length(history.data[:resnorm])
    @show t_CG_GPU_sparse, nums_CG_GPU_step
    @show t_CG_Matrix_Free_GPU, nums_CG_Matrix_Free_GPU
    @show t_CG_CPU_initial_guess, iter_initial_guess_cg
    @show t_CG_GPU_initial_guess, iter_initial_guess_cg
    @show t_CG_Matrix_Free_GPU_initial_guess, iter_initial_guess_cg_Matrix_Free_GPU
    @show t_CG_Matrix_Free_GPU_MG_initial_guess, iter_MG_initial_guess_cg_Matrix_Free_GPU
    @show t_CG_GPU_initial_guess_three_level, iter_initial_guess_three_level_cg_GPU_2h,iter_initial_guess_three_level_cg_GPU_h


    println("###### Show errors ########")
    @show error_CG_GPU
    @show error_CG_Matrix_Free_GPU
    @show initial_guess_cg_error
    @show initial_guess_cg_GPU_error
    @show initial_guess_cg_Matrix_Free_GPU_error
    @show MG_initial_guess_cg_Matrix_Free_GPU_error
    @show initial_guess_three_level_cg_GPU_error
    println()


    return nothing
end


# test_matrix_free_MGCG(level=3)

test_matrix_free_MGCG(level=6)
test_matrix_free_MGCG(level=7)
test_matrix_free_MGCG(level=8)
test_matrix_free_MGCG(level=9)

test_matrix_free_MGCG(level=10)
test_matrix_free_MGCG(level=11)
test_matrix_free_MGCG(level=12)
test_matrix_free_MGCG(level=13)
