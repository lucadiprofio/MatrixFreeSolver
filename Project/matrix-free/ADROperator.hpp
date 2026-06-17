#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include "../test/test1.hpp"
#include "../test/manufactured.hpp"


namespace MtxFree {
using namespace manufactured;

// Derives from MatrixFreeOperators::Base, which already implements vmult(),
// Tvmult(), the constrained-DOF handling and the interface to the iterative
// solvers. We only need to provide the local kernels: the contract is that a
// derived class implements apply_add() (the actual matrix-vector product, built
// on a cell_loop) and, since we use it as a smoother, compute_diagonal().

template <int dim, int fe_degree, typename number>
class ADR_Operator : public dealii::MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
  ADR_Operator() = default;

  void clear() override;

  // Pre-evaluates the PDE coefficients (mu, beta, gamma) at every quadrature
  // point and caches them, so the hot matrix-vector loop only does table
  // look-ups instead of re-evaluating Function objects.
  void setup_coefficients(const DiffusionCoefficient<dim> &mu_function, const AdvectionCoefficient<dim> &beta_function, const ReactionCoefficient<dim> &gamma_function);
  void compute_diagonal() override;

private:
  void apply_add(VectorType &dst, const VectorType &src) const override;
  void local_apply(const dealii::MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const;
  void local_compute_diagonal(dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const;
  
  dealii::VectorizedArray<number> compute_tau(const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &inv_jac, const dealii::VectorizedArray<number> &mu_val, const Tensor<1, dim, dealii::VectorizedArray<number>> &beta_val, const dealii::VectorizedArray<number> &gamma_val) const;

  // Cached coefficient values, one entry per (cell-batch, quadrature point).
  // AlignedVector<VectorizedArray> gives SIMD-aligned storage: each entry holds
  // the coefficient for all cells in a vectorized batch at once.
  dealii::AlignedVector<dealii::VectorizedArray<number>> mu_values;
  dealii::AlignedVector<dealii::VectorizedArray<number>> gamma_values;
  dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> beta_values;
};

} // namespace MtxFree
