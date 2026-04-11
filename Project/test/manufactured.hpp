#ifndef PROBLEM_DATA_HPP
#define PROBLEM_DATA_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>


namespace manufactured {
using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double speed = 0.2;

    return 3.0 * 0.1 * numbers::PI * numbers::PI * std::sin(numbers::PI * x) *
               std::sin(numbers::PI * y) * std::sin(numbers::PI * z) +
           ( // beta * grad_u_ex
               (0.5 - y) * speed * numbers::PI * std::cos(numbers::PI * x) *
                   std::sin(numbers::PI * y) * std::sin(numbers::PI * z) +
               (x - 0.5) * speed * numbers::PI * std::sin(numbers::PI * x) *
                   std::cos(numbers::PI * y) * std::sin(numbers::PI * z) +
               1 * speed * numbers::PI * std::sin(numbers::PI * x) *
                   std::sin(numbers::PI * y) * std::cos(numbers::PI * z)) +
           0.01 * std::sin(numbers::PI * x) * std::sin(numbers::PI * y) *
               std::sin(numbers::PI * z);
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    const VectorizedArray<number> x = p[0];
    const VectorizedArray<number> y = p[1];
    const VectorizedArray<number> z = p[2];

    const number speed = number(0.2);

    return number(3.0 * 0.1) * numbers::PI * numbers::PI *
               std::sin(numbers::PI * x) * std::sin(numbers::PI * y) *
               std::sin(numbers::PI * z) +
           ( // beta * grad_u_ex
               (number(0.5) - y) * speed * numbers::PI *
                   std::cos(numbers::PI * x) * std::sin(numbers::PI * y) *
                   std::sin(numbers::PI * z) +
               (x - number(0.5)) * speed * numbers::PI *
                   std::sin(numbers::PI * x) * std::cos(numbers::PI * y) *
                   std::sin(numbers::PI * z) +
               number(1.0) * speed * numbers::PI * std::sin(numbers::PI * x) *
                   std::sin(numbers::PI * y) * std::cos(numbers::PI * z)) +
           number(0.01) * std::sin(numbers::PI * x) *
               std::sin(numbers::PI * y) * std::sin(numbers::PI * z);
  }
};


template <int dim>
class DiffusionCoefficient : public Function<dim> {
public:
  DiffusionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.1;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.1));
  }
};


template <int dim>
class AdvectionCoefficient : public Function<dim> {
public:
  AdvectionCoefficient() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {
    const double speed_factor = 0.2;

    values[0] = (0.5 - p[1]) * speed_factor;
    values[1] = (p[0] - 0.5) * speed_factor;
    values[2] = 1 * speed_factor;
  }

  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &p) const {
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



template <int dim>
class ReactionCoefficient : public Function<dim> {
public:
  ReactionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.01;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.01));
  }
};


template <int dim>
class DirichletBoundaryInlet : public Function<dim> {
public:
  DirichletBoundaryInlet() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};


template <int dim>
class DirichletBoundaryWalls : public Function<dim> {
public:
  DirichletBoundaryWalls() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};


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


// ------
// exact solution
// ------
template <int dim>
class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]) * std::sin(numbers::PI * p[2]);
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
    Tensor<1, dim> grad;
    grad[0] = numbers::PI * std::cos(numbers::PI * p[0]) *
              std::sin(numbers::PI * p[1]) * std::sin(numbers::PI * p[2]);
    grad[1] = numbers::PI * std::sin(numbers::PI * p[0]) *
              std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[2]);
    grad[2] = numbers::PI * std::sin(numbers::PI * p[0]) *
              std::sin(numbers::PI * p[1]) * std::cos(numbers::PI * p[2]);
    return grad;
  }
};

} // namespace manufactured

#endif