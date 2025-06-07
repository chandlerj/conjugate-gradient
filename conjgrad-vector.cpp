#include <vector>
#include <ranges>
#include <expected>
#include <iostream>
#include <cmath>

template <typename T>
std::expected<T, std::string> dot_product(const std::vector<T> &a, const std::vector<T> &b){

    // get the size of each vector
    const std::size_t a_size = a.size();
    const std::size_t b_size = b.size();
    if (a_size != b_size) {
        return std::unexpected("Vectors must be the same size");
    }

    T res = 0;
    for (int i = 0; i < a_size; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

template <typename T>
std::expected<std::vector<T>, std::string> vec_matrix_prod(const std::vector<std::vector<T>> &A, const std::vector<T> &x){
    
    // verify size of vector is equal to number of columns
    const std::size_t A_col_size = A[0].size();
    const std::size_t A_row_size = A.size();
    const std::size_t x_size = x.size();
    
    if (A_col_size != x_size) {
        return std::unexpected("The length of the vector must be equal to the number of columns in A");
    }

    std::vector<T> prod_vec(A_row_size);
    // take the dot product component wise to build
    // the product result
    for (int i = 0; i < A_row_size; ++i) {
         auto res = dot_product(A[i], x);
         if (res.has_value()) {
             prod_vec[i] = res.value();
         }
         else {
             // propogate the error forward 
             return std::unexpected(res.error());
         }
    }
    return prod_vec;
}

template <typename T>
std::expected<std::vector<T>, std::string> vec_add(const std::vector<T> &a, const std::vector<T> &b) {
    const size_t a_size = a.size();
    const size_t b_size = b.size();
    if (a_size != b_size) {
        return std::unexpected("vectors must be of equal length");
    }
    std::vector<T> res(a.size());
    for(int i = 0; i < a_size; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

template <typename T>
std::expected<std::vector<T>, std::string> vec_sub(const std::vector<T> &a, const std::vector<T> &b) {
    const size_t a_size = a.size();
    const size_t b_size = b.size();
    if (a_size != b_size) {
        return std::unexpected("vectors must be of equal length");
    }
    std::vector<T> res(a.size());
     for(int i = 0; i < a_size; ++i) {
        res[i] = a[i] - b[i];
    }   
    return res; 
}

template <typename T>
std::vector<T> vec_scale(const std::vector<T> &a, T b) {

    std::vector<T> res(a.size());
    for (int i = 0; i < a.size(); ++i) {
        res[i] = a[i] * b;
    }
    return res;
}



template <typename T, std::size_t N>
std::expected<T, std::string> l2_norm(const std::vector<T> &x) {

    if (x.size() == 0) return std::unexpected("vector is empty");

    T res = 0;
    for (const T &item : x) {
        res += item * item;
    }
    return sqrt(res);
}

template <typename T>
std::expected<std::vector<T>, std::string> conjugate_gradient(
        const std::vector<std::vector<T>> &A, 
        const std::vector<T> &b, 
        const std::vector<T> &x0,
        T &tolerance){
    // initialize the residual
    auto Ax_try = vec_matrix_prod(A, x0);
    if (!Ax_try.has_value()){
        return std::unexpected(Ax_try.error());
    }
    std::vector<T> Ax = Ax_try.value();

    auto residual_try = vec_sub(b, Ax);
    if (!residual_try.has_value()) {
        return std::unexpected(residual_try.error());
    }
    std::vector<T> residual = residual_try.value();
    // initialize the search direction
    std::vector<T> search_direction = residual;
    // compute initial squared residual norm
    auto old_residual_norm_try = dot_product(residual, residual);
    if (!old_residual_norm_try.has_value()) {
        return std::unexpected(old_residual_norm_try.error());
    }
    T old_residual_norm = old_residual_norm_try.value();
    // initialize x
    std::vector<T> x = x0;
    
    while (old_residual_norm > tolerance) {
        std::vector<T> A_search_direction = vec_matrix_prod(A, search_direction).value();
        T step_size = old_residual_norm / dot_product(search_direction, A_search_direction).value();
        // update solution
        x = vec_add(x, vec_scale(search_direction ,step_size)).value();
        // update residual
        residual = vec_sub(residual, vec_scale(A_search_direction, step_size)).value();
        T new_residual_norm = dot_product(residual, residual).value();
        T beta = new_residual_norm / old_residual_norm;

        // update search direction
        search_direction = vec_add(residual, vec_scale(search_direction, beta)).value();
        
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

    std::vector<std::vector<T>> A = {{
        {{4.0, 1.0}},
        {{1.0, 3.0}}
    }};

    std::vector<T> b = {1.0, 2.0};
    std::vector<T> x0 = {0.0, 0.0};

    T tolerance = 1e-6;

    std::vector<T> x = conjugate_gradient(A, b, x0, tolerance).value();

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
