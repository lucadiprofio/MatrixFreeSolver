#ifndef PROBLEM_DATA_HPP
#define PROBLEM_DATA_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/numbers.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace ProblemData {
  using namespace dealii;

  template <int dim>
  class RightHandSide : public Function<dim> {
    public:
      RightHandSide() : Function<dim>() {}

      virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
        const double x = p[0];
        const double y = p[1];
        const double z = p[2];
        
        return
          (3.0 * (numbers::PI / 2.0) * (numbers::PI / 2.0) + 1.0) * std::sin((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2])
          + (numbers::PI / 2.0) * (
            std::cos((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2]) +
            std::sin((numbers::PI / 2.0) * p[0]) * std::cos((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2]) +
            std::sin((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::cos((numbers::PI / 2.0) * p[2])
          );
      }

      template <typename number>
      VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
        return (VectorizedArray<number>(numbers::PI / 2.0) * VectorizedArray<number>(numbers::PI / 2.0) * 3.0 + 1.0) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[0]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[1]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[2])
          + VectorizedArray<number>(numbers::PI / 2.0) * (
            std::cos(VectorizedArray<number>(numbers::PI / 2.0) * p[0]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[1]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[2]) +
            std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[0]) * std::cos(VectorizedArray<number>(numbers::PI / 2.0) * p[1]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[2]) +
            std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[0]) * std::sin(VectorizedArray<number>(numbers::PI / 2.0) * p[1]) * std::cos(VectorizedArray<number>(numbers::PI / 2.0) * p[2])
          );
      }
  };


  template <int dim>
  class DiffusionCoefficient : public Function<dim> {
    public:
      DiffusionCoefficient() : Function<dim>() {}

      virtual double value(const Point<dim> &, const unsigned int = 0) const override {
        return 1.0;
      }
  
      template <typename number>
      VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &,const unsigned int = 0) const {
        return VectorizedArray<number>(number(1.0));
      }
  };


  template <int dim>
  class AdvectionCoefficient : public Function<dim> {
    public:
      AdvectionCoefficient() : Function<dim>(dim) {}

      virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {
        // const double speed_factor = 1; 
        for (size_t i = 0; i < dim; ++i)
          values[i] = 1.0;
      }

      template <typename number>
      Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &p) const {
        VectorizedArray<number> one(number(1.0));
        // VectorizedArray<number> point_five(number(0.5));

        Tensor<1, dim, VectorizedArray<number>> result;
        for (size_t i = 0; i < dim; ++i)
          result[i] = one;
        return result;
      }
  };


  template <int dim>
  class ReactionCoefficient : public Function<dim> {
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


  template <int dim>
  class DirichletBoundaryInlet : public Function<dim> {
    public:
      DirichletBoundaryInlet() : Function<dim>() {}
      
      virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
        return 0;
      }

      template <typename number>
      VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
        return VectorizedArray<number>(number(0.0));
      }
  };

  template <int dim>
  class DirichletBoundaryOutlet : public Function<dim> {
    public:
      DirichletBoundaryOutlet() : Function<dim>() {}
      
      virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
        return 0;
      }

      template <typename number>
      VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
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
    NeumannBoundaryValues() : Function<dim>(1) {}

    virtual double value(const Point<dim> &, const unsigned int = 0) const override {
      return 0.0;
    }

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
      return VectorizedArray<number>(number(0.0));
    }
  };


  // -------
  // EXACT SOLUTION
  // ------
  template <int dim>
  class ExactSolution : public Function<dim> {
    public:
      ExactSolution() : Function<dim>(1) {}

      virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
        return std::sin((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2]);
      }

      virtual Tensor<1, dim> gradient(const Point<dim> &p,const unsigned int = 0) const override {
        Tensor<1, dim> grad;
        grad[0] = (numbers::PI / 2.0) * std::cos((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2]);
        grad[1] = (numbers::PI / 2.0) * std::sin((numbers::PI / 2.0) * p[0]) * std::cos((numbers::PI / 2.0) * p[1]) * std::sin((numbers::PI / 2.0) * p[2]);
        grad[2] = (numbers::PI / 2.0) * std::sin((numbers::PI / 2.0) * p[0]) * std::sin((numbers::PI / 2.0) * p[1]) * std::cos((numbers::PI / 2.0) * p[2]);
        return grad;
      }
  };

}

#endif
