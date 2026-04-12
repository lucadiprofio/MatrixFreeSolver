#ifndef TEST_MMS_POLY_HPP
#define TEST_MMS_POLY_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace manufactured2 {
using namespace dealii;

template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    double x = p[0];
    double y = (dim > 1) ? p[1] : 0.0;
    double z = (dim > 2) ? p[2] : 0.0;
    return x * x + 2.0 * y * y + z;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    VectorizedArray<number> x = p[0];
    VectorizedArray<number> y = (dim > 1) ? p[1] : VectorizedArray<number>(0.0);
    VectorizedArray<number> z = (dim > 2) ? p[2] : VectorizedArray<number>(0.0);
    return x * x + number(2.0) * y * y + z;
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
    Tensor<1, dim> grad;
    grad[0] = 2.0 * p[0];
    if (dim > 1) grad[1] = 4.0 * p[1];
    if (dim > 2) grad[2] = 1.0;
    return grad;
  }
};

template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    double x = p[0];
    double y = (dim > 1) ? p[1] : 0.0;
    double z = (dim > 2) ? p[2] : 0.0;
    return x * x + 2.0 * y * y + z + 2.0 * x + 4.0 * y - 5.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    VectorizedArray<number> x = p[0];
    VectorizedArray<number> y = (dim > 1) ? p[1] : VectorizedArray<number>(0.0);
    VectorizedArray<number> z = (dim > 2) ? p[2] : VectorizedArray<number>(0.0);
    return x * x + number(2.0) * y * y + z + number(2.0) * x + number(4.0) * y - number(5.0);
  }
};

template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
  DiffusionCoefficient() : Function<dim>() {}
  virtual double value(const Point<dim> &, const unsigned int = 0) const override { return 1.0; }
  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(1.0));
  }
};

template <int dim> class AdvectionCoefficient : public Function<dim> {
public:
  AdvectionCoefficient() : Function<dim>(dim) {}
  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override {
    values[0] = 1.0;
    if (dim > 1) values[1] = 1.0;
    if (dim > 2) values[2] = 1.0;
  }
  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &) const {
    Tensor<1, dim, VectorizedArray<number>> result;
    result[0] = VectorizedArray<number>(number(1.0));
    if (dim > 1) result[1] = VectorizedArray<number>(number(1.0));
    if (dim > 2) result[2] = VectorizedArray<number>(number(1.0));
    return result;
  }

    void tensor_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const {
    for (unsigned int p = 0; p < points.size(); ++p) {        
      values[p][0] = 1.0;
      values[p][1] = 1.0;
      values[p][2] = 1.0;
    }
  }
};

template <int dim> class ReactionCoefficient : public Function<dim> {
public:
  ReactionCoefficient() : Function<dim>() {}
  virtual double value(const Point<dim> &, const unsigned int = 0) const override { return 1.0; }
  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(1.0));
  }
};

// Usa la soluzione esatta per i bordi di Dirichlet!
template <int dim> class DirichletBoundaryInlet : public ExactSolution<dim> {};
template <int dim> class DirichletBoundaryWalls : public ExactSolution<dim> {};

template <int dim>
class NeumannBoundaryValues : public Function<dim> {
public:
  NeumannBoundaryValues() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};

} // namespace

#endif