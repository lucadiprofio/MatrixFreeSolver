#ifndef TEST_MMS_VAR_NEU_HPP
#define TEST_MMS_VAR_NEU_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace manufactured3 {
using namespace dealii;

// 1. Soluzione Esatta: u = x^2*y + z^2
template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    const double x = p[0];
    const double y = (dim > 1) ? p[1] : 0.0;
    const double z = (dim > 2) ? p[2] : 0.0;
    return x * x * y + z * z;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    const VectorizedArray<number> x = p[0];
    const VectorizedArray<number> y = (dim > 1) ? p[1] : VectorizedArray<number>(0.0);
    const VectorizedArray<number> z = (dim > 2) ? p[2] : VectorizedArray<number>(0.0);
    return x * x * y + z * z;
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
    Tensor<1, dim> grad;
    grad[0] = 2.0 * p[0] * ((dim > 1) ? p[1] : 0.0);
    if (dim > 1) grad[1] = p[0] * p[0];
    if (dim > 2) grad[2] = 2.0 * p[2];
    return grad;
  }
};

// 2. Diffusione: \nu = 1 + x
template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
  DiffusionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return 1.0 + p[0];
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    return number(1.0) + p[0];
  }
};

// 3. Advezione: \beta = (1, 0, 0)^T
template <int dim> class AdvectionCoefficient : public Function<dim> {
public:
  AdvectionCoefficient() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override {
    values[0] = 1.0;
    if (dim > 1) values[1] = 0.0;
    if (dim > 2) values[2] = 0.0;
  }

  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &) const {
    Tensor<1, dim, VectorizedArray<number>> result;
    result[0] = VectorizedArray<number>(number(1.0));
    if (dim > 1) result[1] = VectorizedArray<number>(number(0.0));
    if (dim > 2) result[2] = VectorizedArray<number>(number(0.0));
    return result;
  }

      void tensor_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const {
    for (unsigned int p = 0; p < points.size(); ++p) {        
      values[p][0] = 1.0;
      values[p][1] = 0.0;
      values[p][2] = 0.0;
    }
  }
};

// 4. Reazione: \sigma = 1.0
template <int dim> class ReactionCoefficient : public Function<dim> {
public:
  ReactionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override { 
    return 1.0; 
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(1.0));
  }
};

// 5. Termine Noto (RHS) calcolato analiticamente
template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    const double x = p[0];
    const double y = (dim > 1) ? p[1] : 0.0;
    const double z = (dim > 2) ? p[2] : 0.0;
    return x * x * y + z * z - 2.0 * x * y - 2.0 * x - 2.0 * y - 2.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    const VectorizedArray<number> x = p[0];
    const VectorizedArray<number> y = (dim > 1) ? p[1] : VectorizedArray<number>(0.0);
    const VectorizedArray<number> z = (dim > 2) ? p[2] : VectorizedArray<number>(0.0);
    return x * x * y + z * z - number(2.0) * x * y - number(2.0) * x - number(2.0) * y - number(2.0);
  }
};

// 6. Condizioni di Dirichlet (Usano la soluzione esatta)
template <int dim> class DirichletBoundaryInlet : public ExactSolution<dim> {};
template <int dim> class DirichletBoundaryWalls : public ExactSolution<dim> {};

// 7. Condizione di Neumann sulla faccia z=1 (Faccia 5)
// Flusso g = \nu * (\nabla u \cdot n) = (1+x) * (2z) valutato in z=1 -> 2 + 2x
template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
  NeumannBoundaryValues() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return 2.0 + 2.0 * p[0];
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    return number(2.0) + number(2.0) * p[0];
  }
};

} // namespace test_mms_var_neu

#endif