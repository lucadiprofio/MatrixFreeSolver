#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace Project
{
  using namespace dealii;

  template<int dim>
  class Mu : public Function<dim>
  {
  public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      (void)p;
      return 1.0;
    }

    template<typename number>
    number value_t(const Point<dim,number> &p) const
    {
      (void)p;
      return number(1.0);
    }
  };

  template<int dim>
  class Gamma : public Function<dim>
  {
  public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      (void)p;
      return 0.0;
    }

    template<typename number>
    number value_t(const Point<dim,number> &p) const
    {
      (void)p;
      return number(0.0);
    }
  };

  template<int dim>
  class Beta : public TensorFunction<1,dim>
  {
  public:
    Tensor<1,dim> value(const Point<dim> &p) const override
    {
      (void)p;
      Tensor<1,dim> b;
      return b;
    }

    template<typename number>
    Tensor<1,dim,number> value_t(const Point<dim,number> &p) const
    {
      (void)p;
      Tensor<1,dim,number> b;
      return b;
    }
  };

  template<int dim>
  class WellKnownTerm : public Function<dim>
  {
  public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      (void)p;
      return  value_t<double>(p);
    }

    template <typename number>
    number value_t(const dealii::Point<dim, number> &p) const
    {
      (void)p;
      return 3*M_PI*M_PI*std::sin(M_PI*p[0])*std::sin(M_PI*p[1])*std::sin(M_PI*p[2]);
    }
  };

  template<int dim>
  class DirichletBC : public dealii::Function<dim>
  {
  public:
    double value(const dealii::Point<dim> &p,
                const unsigned int = 0) const override
    {
      (void)p;
      return 0.0 ;
    }
  };

  template<int dim>
  class NeumannBC : public dealii::Function<dim>
  {
  public:
    // versione double "classica"
    double value(const dealii::Point<dim> &p,
                const unsigned int = 0) const override
    {
      return value_t<double>(p);
    }

    template <typename number>
    number value_t(const dealii::Point<dim, number> &p) const
    {
      (void)p;
      return number(0.0); 
    }
  };

}
