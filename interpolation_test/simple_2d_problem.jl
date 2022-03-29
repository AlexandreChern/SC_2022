using LinearAlgebra
using Plots
using IterativeSolvers
Δx = 1/8
x_list = Δx:Δx:1-Δx
N = length(x)

B = sin.(2π*x)

A = zeros(N,N)
for i in 1:N, j in 1:N
    abs(i-j) <= 1 && (A[i,j]+=1)
    i == j && (A[i,j] -=3)
end

A = A/(Δx^2)

U = A\B


function assemble_A_b(level)
    Δx = 1/2^level
    x_list = Δx:Δx:1-Δx
    N = length(x_list)
    b = sin.(2π*x_list)

    A = zeros(N,N)
    for i in 1:N, j in 1:N
        abs(i-j) <= 1 && (A[i,j]+=1)
        i == j && (A[i,j] -=3)
    end

    A = A/(Δx^2)

    return (A,b,x_list)
end


function linear_interpolation(x)
    len = length(x)
    interpolated_x = zeros(2*len+1)
    interpolated_x[1] = x[1]/2
    interpolated_x[end] = x[end]/2
    for i in 2:2*len
        if i % 2 == 0
            interpolated_x[i] = x[div(i,2)]
        else
            interpolated_x[i] = (x[div(i,2)] + x[div(i,2)+1])/2
        end
    end
    return interpolated_x
end

f(x) = -sin(2π*x) / (4*π^2)

function PDE_interpolation(x,f,Δx,x_list)
    len = length(x)
    interpolated_x = zeros(2*len+1)
    interpolated_x[1] = x[1]/2
    interpolated_x[end] = x[end]/2
    for i in 2:2*len
        if i % 2 == 0
            interpolated_x[i] = x[div(i,2)]
        else
            interpolated_x[i] = (x[div(i,2)] + x[div(i,2)+1])/2 + Δx^2*(f(x_list[i])/(4*π^2)) /2
        end
    end
    return interpolated_x
end
test_level = 4
(A_2h,b_2h,x_list_2h) = assemble_A_b(test_level-1)


(A,b,x_list) = assemble_A_b(test_level)


x_2h = A_2h \ b_2h
x = A\b

# plot(x_2h)

x_initial_guess = linear_interpolation(x_2h)
norm(x_initial_guess - f.(x_list))

x_initial_guess_PDE = PDE_interpolation(x_2h,f,Δx,x_list)
norm(x_initial_guess_PDE - f.(x_list))

x,history_zero_initial_guess = cg(A,b;abstol=1e-8,log=true)


function CG_CPU(A,b,x)
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = spzeros(length(b))
    Ap = similar(b);

    num_iter_steps = 0
    # @show rsold
    for _ = 1:length(b)
    # for _ in 1:20
        num_iter_steps += 1
        mul!(Ap,A,p);
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        if sqrt(rsnew) < sqrt(eps(real(eltype(b))))
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
        # @show rsold
    end
    # @show num_iter_steps
    return x, num_iter_steps
end


x_zero_initial_guess = zeros(length(b))

CG_CPU(A,b,x)