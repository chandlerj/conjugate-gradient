#include <array>
#include <iostream>
#include <cmath>

template <typename T, std::size_t N>
T dot_product(const std::array<T,N> &a, const std::array<T,N> &b){
    T res = 0;
    for (int i = 0; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

template <typename T, std::size_t M, std::size_t N>
std::array<T,M> vec_matrix_prod(const std::array<std::array<T,N>, M> &A, const std::array<T,N> &x){

   std::array<T, M> prod_arr;
   // take the dot product component wise to build
   // the product result
   for (int i = 0; i < M; ++i){
        T res = dot_product(A[i], x);
        prod_arr[i] = res;
   }
   return prod_arr;
}

template <typename T, std::size_t N>
std::array<T, N> vec_add(const std::array<T,N> &a, const std::array<T,N> &b) {
    std::array<T, N> res;

    for (int i = 0; i < N; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

template <typename T, std::size_t N>
std::array<T, N> vec_sub(const std::array<T,N> &a, const std::array<T,N> &b) {
     std::array<T, N> res;

    for (int i = 0; i < N; ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
   
}

template <typename T, std::size_t N>
std::array<T, N> vec_scale(const std::array<T,N> &a, T b) {
    std::array<T, N> res;

    for (int i = 0; i < N; ++i) {
        res[i] = a[i] * b;
    }
    return res;
}



template <typename T, std::size_t N>
T l2_norm(const std::array<T, N> &x) {
    T res = 0;
    for (const T &item : x) {
        res += item * item;
    }
    return sqrt(res);
}

template <typename T, std::size_t M, std::size_t N>
std::array<T,M> conjugate_gradient(
        const std::array<std::array<T,N>, M> &A, 
        const std::array<T,M> &b, 
        const std::array<T,M> &x0,
        T &tolerance){
    // initialize the residual
    std::array<T, M> Ax = vec_matrix_prod(A, x0);
    std::array<T, M> residual = vec_sub(b, Ax);
    // initialize the search direction
    std::array<T, M> search_direction = residual;
    // compute initial squared residual norm
    T old_residual_norm = dot_product(residual, residual);
    // initialize x
    std::array<T, M> x = x0;
    
    while (old_residual_norm > tolerance) {
        std::array<T, M> A_search_direction = vec_matrix_prod(A, search_direction);
        T step_size = old_residual_norm / dot_product(search_direction, A_search_direction);
        // update solution
        x = vec_add(x, vec_scale(search_direction ,step_size));
        // update residual
        residual = vec_sub(residual, vec_scale(A_search_direction, step_size));
        T new_residual_norm = dot_product(residual, residual);
        T beta = new_residual_norm / old_residual_norm;

        // update search direction
        search_direction = vec_add(residual, vec_scale(search_direction, beta));
        
        // update old residual for next iteration
        old_residual_norm = new_residual_norm;
        std::cout << old_residual_norm << std::endl;
    }
    return x;
}

template <typename T, std::size_t N>
void print_arr(std::array<T,N> x){
    for (const auto &item : x) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

int main() {
    using T = double;

    constexpr std::size_t N = 2;

    std::array<std::array<T, N>, N> A = {{
        {{4.0, 1.0}},
        {{1.0, 3.0}}
    }};

    std::array<T, N> b = {1.0, 2.0};
    std::array<T, N> x0 = {0.0, 0.0};

    T tolerance = 1e-6;

    std::array<T, N> x = conjugate_gradient(A, b, x0, tolerance);

    std::cout << "Computed solution:\n";
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    }

    std::cout << "\nExpected solution:\n";
    std::cout << "x[0] ≈ 0.090909\nx[1] ≈ 0.636364\n";

    // Verify correctness within tolerance
    bool correct = std::abs(x[0] - 0.090909) < 1e-5 && std::abs(x[1] - 0.636364) < 1e-5;
    if (correct) {
        std::cout << "\n✅ Test passed.\n";
    } else {
        std::cout << "\n❌ Test failed.\n";
    }

    return 0;
}
