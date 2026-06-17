#include <deal.II/matrix_free/fe_evaluation.h>

#include "ADROperator.hpp"


namespace MtxFree {
using namespace dealii;

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
  //   <dim, fe_degree, n_q_points_1d = fe_degree+1, n_components = 1, number>.
  // n_q_points_1d = fe_degree+1 is a Gauss rule that integrates the bilinear
  // form exactly on affine (MappingQ1) cells.
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

  // Matrix-free does NOT loop over single cells: it groups several cells into a
  // "batch" and processes them simultaneously in the SIMD lanes of a
  // VectorizedArray (e.g. 8 cells at once with AVX-512 doubles). So we iterate
  // over batches, and each cached value is a VectorizedArray holding the
  // coefficient for every cell in the batch.
  const unsigned int n_batches = this->data->n_cell_batches();
  const unsigned int n_q_points = phi.n_q_points;

  // Flat storage indexed as cell_batch * n_q_points + q.
  const unsigned int total_q_points = n_batches * n_q_points;
  mu_values.resize(total_q_points);
  gamma_values.resize(total_q_points);
  beta_values.resize(total_q_points);

  for (unsigned int cell = 0; cell < n_batches; ++cell) {
    phi.reinit(cell);
    for (unsigned int q = 0; q < n_q_points; ++q) {
      // quadrature_point(q) returns a Point whose coordinates are
      // VectorizedArrays: one physical point per lane (per cell in the batch).
      // The Function objects are evaluated lane-wise.
      const auto x_q = phi.quadrature_point(q);
      const unsigned int index = cell * n_q_points + q;

      mu_values[index] = mu_function.value(x_q);
      beta_values[index] = beta_function.value(x_q);
      gamma_values[index] = gamma_function.value(x_q);
    }
  }
}

// SUPG (Streamline-Upwind/Petrov-Galerkin) stabilization parameter tau.
// Standard advection-diffusion-reaction design: tau is built so that 1/tau^2
// blends the three regime time scales, advective (2|beta|/h), diffusive
// (mu/h^2) and reactive (gamma), so that tau adapts to whichever term
// dominates locally. h is a local element size recovered from the inverse
// Jacobian of the mapping.
// NOTE: Galerkin alone is fine when
// the problem is diffusion-dominated; for advection-dominated regimes SUPG is
// needed to avoid spurious oscillations.
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

// local_apply is the heart of the matrix-free operator: it computes the
// matrix-vector product y = A*x cell-batch by cell-batch, WITHOUT ever forming
// A. For each batch it reads the local DOFs of src, evaluates the FE function
// and its gradient at the quadrature points, applies the bilinear form
// pointwise, and integrates back to the test space. This is the discrete weak
// form of the ADR operator:
//   a(u,v) = (mu*grad u, grad v) + (beta.grad u + gamma*u, v)   [+ SUPG]
template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::local_apply(const MatrixFree<dim, number> &data, VectorType &dst, const VectorType &src, const std::pair<unsigned int, unsigned int> &range) const {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  for (unsigned int cell = range.first; cell < range.second; ++cell) {
    phi.reinit(cell);
    // gather local DOFs from src
    phi.read_dof_values(src);
    // evaluate the values and gradients of the unknown at all quadrature points
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const unsigned int index = cell * phi.n_q_points + q;
      
      const auto mu_val = mu_values[index];
      const auto gamma_val = gamma_values[index];
      const auto beta_val = beta_values[index];

      const auto u = phi.get_value(q);
      const auto grad_u = phi.get_gradient(q);

      // SUPG parameter (if = 0 -> the SUPG terms below vanish)
      const auto tau = compute_tau(phi.inverse_jacobian(q), mu_val, beta_val, gamma_val);

      // strong residual of the differential operator (the diffusive part
      // -div(mu grad u) is dropped here; it is exactly 0 for P1 but NOT for the
      // P2 used by default).
      const auto strong_residual = beta_val * grad_u + gamma_val * u;

      // submit_gradient => term tested against grad(v): diffusion (mu grad u)
      // plus the SUPG term (tau * R) * beta which acts along the streamline.
      phi.submit_gradient(
        mu_val * grad_u // Diffusion
        + (tau * strong_residual) * beta_val, // SUPG
      q);

      // submit_value => term tested against v: advection + reaction.
      phi.submit_value(strong_residual, q); // Advection and reaction
    }

    // integrate the submitted (gradient, value) contributions against the test
    // functions, then scatter the local result back into the global dst vector.
    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi.distribute_local_to_global(dst);
  }
}

// apply_add is the override required by MatrixFreeOperators::Base. It just
// dispatches local_apply over all cell batches via cell_loop, which also
// handles MPI ghost exchange and (optional) thread parallelism. Base::vmult()
// calls this.
template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::apply_add(VectorType &dst, const VectorType &src) const {
  this->data->cell_loop(&ADR_Operator::local_apply, this, dst, src);
}

// Same pointwise kernel as local_apply, but written for a SINGLE cell so that
// MatrixFreeTools::compute_diagonal can extract the matrix diagonal cheaply (by
// probing the operator with unit basis vectors, all matrix-free). We need the
// diagonal because the Chebyshev smoother used in the multigrid is a Jacobi-
// type (point) smoother: it only requires matrix-vector products and the
// (inverse) diagonal.
template <int dim, int fe_degree, typename number>
void ADR_Operator<dim, fe_degree, number>::local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const {
  const unsigned int cell = phi.get_current_cell_index();

  phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

  for (unsigned int q = 0; q < phi.n_q_points; ++q) {
    const unsigned int index = cell * phi.n_q_points + q;

    // cached coefficients
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
  // Base stores the *inverse* diagonal as a DiagonalMatrix, which the smoother
  // consumes directly.
  this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
  LinearAlgebra::distributed::Vector<number> &inv_diag = this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inv_diag);

  // Compute the diagonal of A using the local kernel above.
  MatrixFreeTools::compute_diagonal(*this->data, inv_diag, &ADR_Operator::local_compute_diagonal, this);
  // Constrained DOFs (Dirichlet/hanging) sit on the identity in the operator,
  // so their diagonal must be 1.
  this->set_constrained_entries_to_one(inv_diag);

  // Invert entry-wise, guarding against (near-)zero diagonal entries to avoid
  // division blow-ups; such entries are left at 1.
  for (unsigned int i = 0; i < inv_diag.locally_owned_size(); ++i)
    inv_diag.local_element(i) = (std::abs(inv_diag.local_element(i)) > 1e-14) ? number(1.0) / inv_diag.local_element(i) : number(1.0);
}



// Explicit instantiations. Because the templates live in the .cpp, every
// (dim, degree, number) combination actually used must be instantiated here.
// number = double is the active (fine) level; number = float is used on the
// multigrid levels (mixed precision: the preconditioner only needs to be
// approximate, so single precision halves its memory traffic).
template class ADR_Operator<3, 1, double>;
template class ADR_Operator<3, 2, double>;
template class ADR_Operator<3, 3, double>;
template class ADR_Operator<3, 4, double>;
template class ADR_Operator<3, 5, double>;
template class ADR_Operator<3, 6, double>;
template class ADR_Operator<3, 7, double>;

template class ADR_Operator<3, 1, float>;
template class ADR_Operator<3, 2, float>;
template class ADR_Operator<3, 3, float>;
template class ADR_Operator<3, 4, float>;
template class ADR_Operator<3, 5, float>;
template class ADR_Operator<3, 6, float>;
template class ADR_Operator<3, 7, float>;

} // namespace MtxFree
