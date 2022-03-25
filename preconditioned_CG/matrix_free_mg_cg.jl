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
    for _ in 1:maxiter
        matrix_free_A_full_GPU(idata_GPU,odata_GPU) # matrix_free_A_full_GPU is -A here, becareful
        odata_GPU .= idata_GPU .+ ω * (b_GPU .+ odata_GPU)
        idata_GPU .= odata_GPU
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
    return num_iter_steps_GPU, norms_GPU
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


function precond_matrix(A, b; m=3, solver="jacobi")
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
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level,p=SBPp);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1,p=SBPp);
    (A_4h,b_4h,H_tilde_4h,Nx_4h,Ny_4h) = Assembling_matrix(level-2,p=SBPp);
    A_2h_lu = lu(A_2h)
    A_4h_lu = lu(A_4h)
    direct_sol = A\b
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    abstol = norm(A*x-b) * reltol

    x_GPU = CuArray(zeros(Nx,Ny))
    b_GPU = CuArray(reshape(b,Nx,Ny))

    ω_richardson = 0.15
    h = 1/(Nx-1)
    (M,R,H,I_p,_,I_r,IN) = precond_matrix(A,b,m=3,solver="richardson")

    num_iter_steps_matrix_free_GPU, norms_matrix_free_GPU = matrix_free_MGCG(b_GPU,x_GPU;A_2h = A_2h_lu,maxiter=length(b_GPU),abstol=abstol,nu=nu)


    A_GPU_sparse = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    x_GPU_sparse = CuArray(zeros(Nx*Ny))
    b_GPU_sparse = CuArray(b)

  

    x_GPU = CuArray(zeros(Nx,Ny))
    num_iter_steps_matrix_free_GPU_Three_level, norms_matrix_free_GPU_Three_level = matrix_free_MGCG_Three_level(b_GPU,x_GPU;A_4h = A_4h_lu,maxiter=length(b_GPU),abstol=abstol,nu=nu)
    
    iter_mg_cg, norm_mg_cg, error_mg_cg = mg_preconditioned_CG(A,b,x;maxiter=length(b),A_2h = A_2h_lu, abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,SBPp=SBPp)

    iter_mg_cg_GPU, norm_mg_cg_GPU, error_mg_cg_GPU = mg_preconditioned_CG_GPU(A_GPU_sparse,b_GPU_sparse,x_GPU_sparse;maxiter=length(b_GPU_sparse),A_2h = A_2h_lu, abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,H_tilde=H_tilde,SBPp=SBPp)
    # @show norms_GPU
    # @show norm_mg_cg
    # @show norm_mg_cg_GPU

    norms_CG_GPU, history = cg(A_GPU_sparse,b_GPU_sparse,abstol=abstol,log=true)

    # # 3-level multigrid
    # x_3mg, norm_3mg = Three_level_multigrid(A,b,A_2h,b_2h,A_4h,b_4h,Nx,Ny;nu=3,NUM_V_CYCLES=1,SBPp=2)

    # # 2-level multigrid
    # x_2mg, norm_2mg = Two_level_multigrid(A,b,Nx,Ny,A_2h;nu=3,NUM_V_CYCLES=1,SBPp=2)



    # # 3-level multigrid GPU 
    # matrix_free_Three_level_multigrid(b_GPU,A_4h;nu=3,NUM_V_CYCLES=1,SBPp=2)

    REPEAT = 1

    t_matrix_free_MGCG_GPU = @elapsed for _ in 1:REPEAT
        x_GPU = CuArray(zeros(Nx,Ny))
        matrix_free_MGCG(b_GPU,x_GPU;A_2h=A_2h_lu,maxiter=length(b_GPU),abstol=abstol,nu=nu)
    end

    t_matrix_free_MGCG_Three_level_GPU = @elapsed for _ in 1:REPEAT
        x_GPU = CuArray(zeros(Nx,Ny))
        matrix_free_MGCG_Three_level(b_GPU,x_GPU;A_4h=A_4h_lu,maxiter=length(b_GPU),abstol=abstol,nu=nu)
    end

    t_MGCG_CPU = @elapsed for _ in 1:REPEAT
        x = zeros(Nx*Ny)
        mg_preconditioned_CG(A,b,x;A_2h=A_2h_lu,maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde,SBPp=SBPp)
    end

    t_MGCG_GPU_sparse = @elapsed for _ in 1:REPEAT
        x_GPU_sparse = CuArray(zeros(Nx*Ny))
        mg_preconditioned_CG_GPU(A_GPU_sparse,b_GPU_sparse,x_GPU_sparse;maxiter=length(b_GPU),A_2h = A_2h_lu, abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,H_tilde=H_tilde,SBPp=SBPp)
    end

    t_CG_GPU_sparse = @elapsed for _ in 1:REPEAT
        cg(A_GPU_sparse,b_GPU_sparse,abstol=abstol,log=true)
    end

    println()


    t_matrix_free_MGCG_GPU = t_matrix_free_MGCG_GPU / REPEAT
    t_matrix_free_MGCG_Three_level_GPU = t_matrix_free_MGCG_Three_level_GPU / REPEAT
    t_MGCG_CPU /= REPEAT
    t_MGCG_GPU_sparse /= REPEAT
    t_CG_GPU_sparse /= REPEAT

    @show Nx, Ny

    @show t_matrix_free_MGCG_GPU, num_iter_steps_matrix_free_GPU
    @show  t_matrix_free_MGCG_Three_level_GPU, num_iter_steps_matrix_free_GPU_Three_level
    @show t_MGCG_CPU, iter_mg_cg
    @show t_MGCG_GPU_sparse, iter_mg_cg_GPU
    @show t_CG_GPU_sparse, length(history.data[:resnorm])

    return nothing
end


test_matrix_free_MGCG(level=6)
test_matrix_free_MGCG(level=7)
test_matrix_free_MGCG(level=8)
test_matrix_free_MGCG(level=9)
test_matrix_free_MGCG(level=10)
test_matrix_free_MGCG(level=11)
# test_matrix_free_MGCG(level=11)
# test_matrix_free_MGCG(level=12)

# let
#     level = 6
#     N = 2^level + 1
#     Random.seed!(0)
#     idata = randn(N,N)
#     idata_flat = idata[:]
#     idata_GPU = CuArray(idata)
#     odata_GPU = CuArray(zeros(N,N))
    
#     x = zeros(length(idata_flat))
#     x_GPU_flat = CuArray(x)
#     odata_reshaped = reshape(prolongation_2d(N)*idata_flat,2*N-1,2*N-1)

#     size_idata = size(idata)
#     odata_prolongation = zeros(2*size_idata[1]-1,2*size_idata[2]-1)
#     odata_restriction = zeros(div.(size_idata .+ 1,2))

#     odata_prolongation_GPU = CuArray(odata_prolongation)
#     odata_restriction_GPU = CuArray(odata_restriction)

#     matrix_free_restriction_2d(idata,odata_restriction)
#     matrix_free_prolongation_2d(idata,odata_prolongation)

#     @assert odata_restriction[:] ≈ restriction_2d(N) * idata_flat
#     @assert odata_prolongation[:] ≈ prolongation_2d(N) * idata_flat

#     matrix_free_prolongation_2d(idata_GPU,odata_prolongation_GPU)
#     matrix_free_prolongation_2d_GPU(idata_GPU,odata_prolongation_GPU)

#     matrix_free_restriction_2d(idata_GPU,odata_restriction_GPU)
#     matrix_free_restriction_2d_GPU(idata_GPU,odata_restriction_GPU)

#     @assert odata_restriction ≈ Array(odata_restriction_GPU)
#     @assert odata_prolongation ≈ Array(odata_prolongation_GPU)

    

#     (A,b,H,Nx,Ny) = Assembling_matrix(level,p=2)

#     maxiter=2
#     modified_richardson!(idata_flat,A,b;maxiter=maxiter)
#     b_GPU = CuArray(reshape(b,N,N))
#     matrix_free_richardson(idata_GPU,odata_GPU,b_GPU;maxiter=maxiter)

#     richardson_out_CPU = idata_flat + 0.15 * (b - A*idata_flat)
#     matrix_free_A_full_GPU(idata_GPU,odata_GPU) # Be careful, matrix_free_A_GPU is -A here, with minus sign
#     richardson_out_GPU = idata_GPU + 0.15 * (b_GPU + odata_GPU) #
# end
