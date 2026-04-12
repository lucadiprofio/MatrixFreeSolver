#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include "../test/test1.hpp"
#include "../test/manufactured.hpp"


namespace MtxFree {
using namespace dealii;
using namespace manufactured;

template <int dim, int fe_degree, typename number>
class ADR_Operator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  ADR_Operator() = default;

  void clear() override;
  void setup_coefficients(const DiffusionCoefficient<dim> &mu_function, const AdvectionCoefficient<dim> &beta_function, const ReactionCoefficient<dim> &gamma_function);
  void compute_diagonal() override;

private:
  void apply_add(VectorType &dst, const VectorType &src) const override;
  void local_apply(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &cell_range) const;
  void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const;
  
  VectorizedArray<number> compute_tau(const Tensor<2, dim, VectorizedArray<number>> &inv_jac, const VectorizedArray<number> &mu_val, const Tensor<1, dim, VectorizedArray<number>> &beta_val, const VectorizedArray<number> &gamma_val) const;

  AlignedVector<VectorizedArray<number>> mu_values;
  AlignedVector<VectorizedArray<number>> gamma_values;
  AlignedVector<Tensor<1, dim, VectorizedArray<number>>> beta_values;
};


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::clear() {
  MatrixFreeOperators::Base<dim, VectorType>::clear();
}


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::setup_coefficients(
  const DiffusionCoefficient<dim> &mu_function,
  const AdvectionCoefficient<dim> &beta_function,
  const ReactionCoefficient<dim> &gamma_function
) {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);
  const unsigned int n_batches = this->data->n_cell_batches();
  const unsigned int n_q_points = phi.n_q_points;

  // Resize vectors
  const unsigned int total_q_points = n_batches * n_q_points;
  mu_values.resize(total_q_points);
  gamma_values.resize(total_q_points);
  beta_values.resize(total_q_points);

  for (unsigned int cell = 0; cell < n_batches; ++cell) {
    phi.reinit(cell);
    for (unsigned int q = 0; q < n_q_points; ++q) {
      const auto x_q = phi.quadrature_point(q);
      const unsigned int index = cell * n_q_points + q;

      mu_values[index] = mu_function.value(x_q);
      beta_values[index] = beta_function.value(x_q);
      gamma_values[index] = gamma_function.value(x_q);
    }
  }
}


template <int dim, int fe_degree, typename number>
VectorizedArray<number> ADR_Operator<dim, fe_degree, number>::compute_tau(const Tensor<2, dim, VectorizedArray<number>> &inv_jac, const VectorizedArray<number> &mu_val, const Tensor<1, dim, VectorizedArray<number>> &beta_val, const VectorizedArray<number> &gamma_val) const {
  VectorizedArray<number> max_inv_jac = number(0.0);
  for (unsigned int i = 0; i < dim; ++i)
    max_inv_jac = std::max(max_inv_jac, inv_jac[i].norm());

  const auto h = number(1.0) / max_inv_jac;
  const auto vel_norm = beta_val.norm();

  const auto tau_inv_sq =
    (number(2.0) * vel_norm / h) * (number(2.0) * vel_norm / h) +
    number(9.0) * (number(4.0) * mu_val / (h * h)) *
    (number(4.0) * mu_val / (h * h)) +
    (gamma_val) * (gamma_val);
  
  // return number(1.0) / std::sqrt(tau_inv_sq); SUPG on
  (void) tau_inv_sq;
  return number(0.0); // SUPG off
}


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::local_apply(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &range) const {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  for (unsigned int cell = range.first; cell < range.second; ++cell) {
    phi.reinit(cell);
    phi.read_dof_values(src);
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const unsigned int index = cell * phi.n_q_points + q;
      
      const auto mu_val = mu_values[index];
      const auto gamma_val = gamma_values[index];
      const auto beta_val = beta_values[index];

      const auto u = phi.get_value(q);
      const auto grad_u = phi.get_gradient(q);

      // SUPG parameter
      const auto tau = compute_tau(phi.inverse_jacobian(q), mu_val, beta_val, gamma_val);

      const auto strong_residual = beta_val * grad_u + gamma_val * u;

      phi.submit_gradient(
        mu_val * grad_u // Diffusion
        + (tau * strong_residual) * beta_val, // SUPG
      q);
      phi.submit_value(strong_residual, q); // Advection and reaction
    }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi.distribute_local_to_global(dst);
  }
}


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::apply_add(VectorType &dst, const VectorType &src) const {
  this->data->cell_loop(&ADR_Operator::local_apply, this, dst, src);
}


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const {
  const unsigned int cell = phi.get_current_cell_index();

  phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

  for (unsigned int q = 0; q < phi.n_q_points; ++q) {
    const unsigned int index = cell * phi.n_q_points + q;
    const auto mu_val = mu_values[index];
    const auto gamma_val = gamma_values[index];
    const auto beta_val = beta_values[index];

    const auto u = phi.get_value(q);
    const auto grad_u = phi.get_gradient(q);

    // SUPG
    const auto tau = compute_tau(phi.inverse_jacobian(q), mu_val, beta_val, gamma_val);

    // Reaction and advection
    const auto strong_residual = beta_val * grad_u + gamma_val * u;

    phi.submit_gradient(mu_val * grad_u + (tau * strong_residual) * beta_val, q);
    phi.submit_value(strong_residual, q);
  }

  phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
}


template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::compute_diagonal() {
  this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
  LinearAlgebra::distributed::Vector<number> &inv_diag = this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inv_diag);

  MatrixFreeTools::compute_diagonal(*this->data, inv_diag, &ADR_Operator::local_compute_diagonal, this);
  this->set_constrained_entries_to_one(inv_diag);

  for (unsigned int i = 0; i < inv_diag.locally_owned_size(); ++i)
    inv_diag.local_element(i) = (std::abs(inv_diag.local_element(i)) > 1e-14) ? number(1.0) / inv_diag.local_element(i) : number(1.0);
}


} // namespace MtxFree
