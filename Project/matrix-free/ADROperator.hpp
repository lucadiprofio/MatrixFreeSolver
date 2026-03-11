#pragma once

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../matrix-based/ProblemData.hpp"


namespace Project {
  using namespace dealii;
  using namespace ProblemData;

  template <int dim, int fe_degree, typename number>
  class ADR_Operator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
    public:
      ADR_Operator() = default;

      void clear() override;
      void setup_coefficients(const DiffusionCoefficient<dim> &mu_function, const AdvectionCoefficient<dim> &beta_function, const ReactionCoefficient<dim> &gamma_function);
      void compute_diagonal() override;


    private:
      void apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const override;
      void local_apply(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int,unsigned int> &cell_range) const;
      void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> &phi) const;

      const DiffusionCoefficient<dim> *mu_function = nullptr;
      const AdvectionCoefficient<dim> *beta_function = nullptr;
      const ReactionCoefficient<dim> *gamma_function = nullptr;
  };


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::clear() {
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
  }


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::setup_coefficients(const DiffusionCoefficient<dim> &mu_function, const AdvectionCoefficient<dim> &beta_function, const ReactionCoefficient<dim> &gamma_function) {  
    this->mu_function = &mu_function;
    this->beta_function = &beta_function;
    this->gamma_function = &gamma_function;
  }


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::local_apply(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int,unsigned int> &range) const {
    FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> phi(data);

    for (unsigned int cell=range.first; cell<range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto x_q = phi.quadrature_point(q);

        const auto mu_val = mu_function->value(x_q);
        const auto gamma_val = gamma_function->value(x_q);
        const auto beta_val = beta_function->value(x_q);

        const auto u = phi.get_value(q);
        const auto grad_u = phi.get_gradient(q);
        
        // SUPG parameter
        // TODO is it a good etstimatior of the diameter of h? 
        const auto inv_jac = phi.inverse_jacobian(q); 
        VectorizedArray<number> max_inv_jac = 0.0;
        for (unsigned int i=0; i<dim; ++i)
          max_inv_jac = std::max(max_inv_jac, inv_jac[i].norm());
        const auto h = 1.0 / max_inv_jac;
        const auto vel_norm = beta_val.norm();

        const auto tau_inv_sq = (2.0 * vel_norm / h) * (2.0 * vel_norm / h)
          + 9.0 * (4.0 * mu_val / (h * h)) * (4.0 * mu_val / (h * h))
          + (gamma_val) * (gamma_val);

        // const auto tau = 1.0 / std::sqrt(tau_inv_sq);
        const auto tau = 0.0 / std::sqrt(tau_inv_sq); // SUPG off

        // Reaction and advection
        const auto strong_residual = beta_val * grad_u + gamma_val * u;

        phi.submit_gradient(
          mu_val * grad_u // Diffusion
          + (tau * strong_residual) * beta_val // SUPG
        , q);
        phi.submit_value(
          strong_residual // Advection and reaction
        ,q);
      }

      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
  }


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const {
    this->data->cell_loop(&ADR_Operator::local_apply, this, dst, src);
  }


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> &phi) const {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const auto x_q = phi.quadrature_point(q);
      auto mu_val = mu_function->value(x_q);
      const auto beta_val = beta_function->value(x_q);
      auto gamma_val = gamma_function->value(x_q);
      
      const auto u = phi.get_value(q);
      const auto grad_u = phi.get_gradient(q);

      // SUPG
      const auto inv_jac = phi.inverse_jacobian(q); 
      VectorizedArray<number> max_inv_jac = number(0.0);
      for (unsigned int i=0; i<dim; ++i)
        max_inv_jac = std::max(max_inv_jac, inv_jac[i].norm());
      
      const auto h = number(1.0) / max_inv_jac;
      const auto vel_norm = beta_val.norm();

      const auto tau_inv_sq = (number(2.0) * vel_norm / h) * (number(2.0) * vel_norm / h)
        + number(9.0) * (number(4.0) * mu_val / (h * h)) * (number(4.0) * mu_val / (h * h))
        + (gamma_val) * (gamma_val);

      // const auto tau = number(1.0) / std::sqrt(tau_inv_sq);
      const auto tau = number(0.0) / std::sqrt(tau_inv_sq); // SUPG off

      // Reaction and advection
      const auto strong_residual = beta_val * grad_u + gamma_val * u;

      phi.submit_gradient(mu_val * grad_u + (tau * strong_residual) * beta_val, q);
      phi.submit_value(strong_residual, q);
      // phi.submit_gradient(mu_val*grad_u, q);
      // phi.submit_value(gamma_val*u, q);
    }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }


  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::compute_diagonal() {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inv_diag = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inv_diag);

    MatrixFreeTools::compute_diagonal(*this->data, inv_diag, &ADR_Operator::local_compute_diagonal, this);

    this->set_constrained_entries_to_one(inv_diag);

    for (unsigned int i=0; i<inv_diag.locally_owned_size(); ++i)
      inv_diag.local_element(i) = number(1.0) / inv_diag.local_element(i);
  }

}