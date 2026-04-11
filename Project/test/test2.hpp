#ifndef TEST2_HPP
#define TEST2_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace test2 {
using namespace dealii;

template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};

// adr
template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
  DiffusionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.1;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.1));
  }
};

template <int dim> class AdvectionCoefficient : public Function<dim> {
public:
  AdvectionCoefficient() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override {
    const double speed_factor = 0.2;

    values[0] = (0.5 - p[1]) * speed_factor;
    values[1] = (p[0] - 0.5) * speed_factor;
    values[2] = 1 * speed_factor;
  }

  template <typename number>
  Tensor<1, dim, VectorizedArray<number>>
  value(const Point<dim, VectorizedArray<number>> &p) const {
    VectorizedArray<number> one(number(1.0));
    VectorizedArray<number> half(number(0.5));

    const number speed_factor = number(0.2);

    Tensor<1, dim, VectorizedArray<number>> result;
    result[0] = (half - p[1]) * speed_factor;
    result[1] = (p[0] - half) * speed_factor;
    result[2] = one * speed_factor;
    return result;
  }

  void tensor_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const {
    for (unsigned int p = 0; p < points.size(); ++p) {
      const double speed_factor = 0.2;
        
      values[p][0] = (0.5 - points[p][1]) * speed_factor;
      values[p][1] = (points[p][0] - 0.5) * speed_factor;
      values[p][2] = 1.0 * speed_factor;
    }
  }
};

template <int dim> class ReactionCoefficient : public Function<dim> {
public:
  ReactionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.01;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.01));
  }
};

// boundary conditions
template <int dim> class DirichletBoundaryInlet : public Function<dim> {
public:
  DirichletBoundaryInlet() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int = 0) const override {
    const double x_c = 0.75;
    const double y_c = 0.5;
    const double sigma_sq = 0.15 * 0.15;

    return std::exp(
        -((p[0] - x_c) * (p[0] - x_c) + (p[1] - y_c) * (p[1] - y_c)) /
        (2.0 * sigma_sq));
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                                const unsigned int = 0) const {
    VectorizedArray<number> point_seventyfive(0.75);
    VectorizedArray<number> point_five(0.5);
    VectorizedArray<number> sigma(0.15 * 0.15);
    return std::exp(-((p[0] - point_seventyfive) * (p[0] - point_seventyfive) +
                      (p[1] - point_five) * (p[1] - point_five)) /
                    (2.0 * sigma));
  }
};

template <int dim> class DirichletBoundaryWalls : public Function<dim> {
public:
  DirichletBoundaryWalls() : Function<dim>() {}

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};

template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
  NeumannBoundaryValues() : Function<dim>() {}

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};

// not useful exact solution...
template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>() {};

  virtual double value(const Point<dim> &,
                       const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,
                                const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};

} // namespace test1

#endif
