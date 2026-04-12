#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include "../test/test1.hpp"
#include "../test/manufactured.hpp"


namespace MtxFree {
using namespace manufactured;

template <int dim, int fe_degree, typename number>
class ADR_Operator : public dealii::MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
  ADR_Operator() = default;

  void clear() override;
  void setup_coefficients(const DiffusionCoefficient<dim> &mu_function, const AdvectionCoefficient<dim> &beta_function, const ReactionCoefficient<dim> &gamma_function);
  void compute_diagonal() override;

private:
  void apply_add(VectorType &dst, const VectorType &src) const override;
  void local_apply(const dealii::MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const;
  void local_compute_diagonal(dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const;
  
  dealii::VectorizedArray<number> compute_tau(const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &inv_jac, const dealii::VectorizedArray<number> &mu_val, const Tensor<1, dim, dealii::VectorizedArray<number>> &beta_val, const dealii::VectorizedArray<number> &gamma_val) const;

  dealii::AlignedVector<dealii::VectorizedArray<number>> mu_values;
  dealii::AlignedVector<dealii::VectorizedArray<number>> gamma_values;
  dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> beta_values;
};

} // namespace MtxFree
