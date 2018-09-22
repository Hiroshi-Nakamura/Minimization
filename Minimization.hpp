#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <iostream>
#include <vector>

namespace Minimization {
    template<int DIM>
    bool minimization(
        const AutomaticDifferentiation::FuncPtr<double,DIM>& f,
        Eigen::Matrix<double,DIM,1>& x,
        unsigned int max_num_iteration=1000)
    {
        auto jac=AutomaticDifferentiation::jacobian<double,DIM>(f);
        auto hes=AutomaticDifferentiation::hessian<double,DIM>(f);
        for(unsigned int i=0; true; i++){
            auto jac_val=jac(x);
            auto hes_val=hes(x);
            auto delta_x=hes_val.fullPivLu().solve(-jac_val);
            if(delta_x.norm()<10e-10) return true;
            if(max_num_iteration<i) return false;
            x += delta_x;
//            std::cout << "x=" << std::endl << x << std::endl;
        }
        return true;
    }

    template<int DIM>
    bool minimization(
        std::function<AutomaticDifferentiation::FuncPtr<double,DIM>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM>,DIM>&)> f,
        Eigen::Matrix<double,DIM,1>& x,
        unsigned int max_num_iteration=1000)
    {
        AutomaticDifferentiation::FuncPtr<double,DIM> y=f(AutomaticDifferentiation::createVariables<double,DIM>());
        return minimization(y,x);
    }

    template<int DIM, int NUM_EQUALITY>
    bool minimization_with_equality_constraints(
        std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)> f,
        std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)>,NUM_EQUALITY> g,
        Eigen::Matrix<double,DIM+NUM_EQUALITY,1>& x_val,
        unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> y=f(x);
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,NUM_EQUALITY> z;
        for(size_t i=0; i<NUM_EQUALITY; i++){
            AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> z=g[i](x);
            y=y+x[DIM+i]*z;
        }
        return Minimization::minimization(y,x_val);
    }

}
#endif // MINIMIZATION_HPP_INCLUDED
