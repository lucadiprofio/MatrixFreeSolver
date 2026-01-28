#pragma once

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../../Coefficients.hpp"

namespace Project
{
  using namespace dealii;

  template <int dim, int fe_degree, typename number>
  class ADR_Operator
    : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    ADR_Operator() = default;
    void clear() override;

    void evaluate_coefficients(const Mu<dim> &mu_fun,
                               const Beta<dim> &beta_fun,
                               const Gamma<dim> &gamma_fun);

    void compute_diagonal() override;

  private:
    void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
                   const LinearAlgebra::distributed::Vector<number> &src) const override;

    void local_apply(const MatrixFree<dim, number> &data,
                     LinearAlgebra::distributed::Vector<number> &dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     const std::pair<unsigned int,unsigned int> &cell_range) const;

    void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> &phi) const;

    Table<2, VectorizedArray<number>> mu;
    Table<2, Tensor<1,dim,VectorizedArray<number>>> beta;
    Table<2, VectorizedArray<number>> gamma;
  };

  // ===== implementazioni template (devono stare qui) =====

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::clear()
  {
    mu.reinit(0,0); beta.reinit(0,0); gamma.reinit(0,0);
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
  }

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::evaluate_coefficients(const Mu<dim> &mu_fun,
                                                                 const Beta<dim> &beta_fun,
                                                                 const Gamma<dim> &gamma_fun)
  {
    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> phi(*this->data);

    mu.reinit(n_cells, phi.n_q_points);
    beta.reinit(n_cells, phi.n_q_points);
    gamma.reinit(n_cells, phi.n_q_points);

    for (unsigned int cell=0; cell<n_cells; ++cell)
    {
      phi.reinit(cell);
      for (const unsigned int q : phi.quadrature_point_indices())
      {
        const auto xq = phi.quadrature_point(q);
        mu(cell,q)    = mu_fun.template value_t<VectorizedArray<number>>(xq);
        gamma(cell,q) = gamma_fun.template value_t<VectorizedArray<number>>(xq);

        const auto b = beta_fun.template value_t<VectorizedArray<number>>(xq);
        beta(cell,q) = b;
      }
    }
  }

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::local_apply(const MatrixFree<dim, number> &data,
                                                       LinearAlgebra::distributed::Vector<number> &dst,
                                                       const LinearAlgebra::distributed::Vector<number> &src,
                                                       const std::pair<unsigned int,unsigned int> &range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> phi(data);

    for (unsigned int cell=range.first; cell<range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (const unsigned int q : phi.quadrature_point_indices())
      {
        const auto u      = phi.get_value(q);
        const auto grad_u = phi.get_gradient(q);

        Tensor<1,dim,VectorizedArray<number>> flux = mu(cell,q)*grad_u - beta(cell,q)*u;

        phi.submit_gradient(flux, q);
        phi.submit_value(gamma(cell,q)*u, q);
      }

      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
  }

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst,
                                                     const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&ADR_Operator::local_apply, this, dst, src);
  }

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::local_compute_diagonal(
      FEEvaluation<dim, fe_degree, fe_degree+1, 1, number> &phi) const
  {
    const unsigned int cell = phi.get_current_cell_index();

    // ATTENZIONE: per un operatore non SPD, il "diagonal" è per precondizionamento;
    // spesso si usa la diagonale della parte diffusiva/reazione, non includendo beta.
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (const unsigned int q : phi.quadrature_point_indices())
    {
      const auto u      = phi.get_value(q);
      const auto grad_u = phi.get_gradient(q);

      phi.submit_gradient(mu(cell,q)*grad_u, q);      // solo diffusione
      phi.submit_value(gamma(cell,q)*u, q);           // reazione
    }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template<int dim,int fe_degree,typename number>
  void ADR_Operator<dim,fe_degree,number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());

    auto &inv_diag = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inv_diag);

    MatrixFreeTools::compute_diagonal(*this->data, inv_diag,
                                      &ADR_Operator::local_compute_diagonal, this);

    this->set_constrained_entries_to_one(inv_diag);

    for (unsigned int i=0; i<inv_diag.locally_owned_size(); ++i)
      inv_diag.local_element(i) = 1.0 / inv_diag.local_element(i);
  }

}
