#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <iostream>
#include <vector>

namespace Minimization {

    template<int DIM_EXTENDED>
    std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM_EXTENDED>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM_EXTENDED>,DIM_EXTENDED>&)>,0> NO_CONSTRAINT;


    /**
        DIM is the dimention of variables.
    */
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

    /**
        DIM is the dimention of variables.
    */
    template<int DIM>
    bool minimization(
        std::function<AutomaticDifferentiation::FuncPtr<double,DIM>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM>,DIM>&)> f,
        Eigen::Matrix<double,DIM,1>& x,
        unsigned int max_num_iteration=1000)
    {
        AutomaticDifferentiation::FuncPtr<double,DIM> y=f(AutomaticDifferentiation::createVariables<double,DIM>());
        return minimization(y,x);
    }

    /**
        DIM is the dimention of variables, NUM_EQUALITY is the number of equality_constraints.
        Note: the variable x_val should have the "DIM+NUM_EQUALITY" dimension,
        such that the extended dimention will be used for the Lagrange multipliers method.
    */
    template<int DIM, int NUM_EQUALITY>
    bool minimization_with_equality_constraints(
        std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)> f,
        std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)>,NUM_EQUALITY> g,
        Eigen::Matrix<double,DIM+NUM_EQUALITY,1>& x_val,
        unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> y=f(x);
        for(size_t i=0; i<NUM_EQUALITY; i++){
            AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> z=g[i](x);
            y=y+x[DIM+i]*z;
        }
        return Minimization::minimization(y,x_val);
    }

    /**
        DIM is the dimention of variables,
        NUM_EQUALITY is the number of equality_constraints and NUM_INEQUALITY is the number of inequality_constraints.
        Note: the variable x_val should have the "DIM+NUM_EQUALITY+NUM_INEQUALITY" dimension,
        such that the extended dimention will be used for the Lagrange multipliers method.
    */
    template<int DIM, int NUM_EQUALITY, int NUM_INEQUALITY>
    bool minimization_with_constraints(
        std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)> f,
        std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)>,NUM_EQUALITY> g_eq,
        std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)>,NUM_INEQUALITY> g_in,
        Eigen::Matrix<double,DIM+NUM_EQUALITY+NUM_INEQUALITY,1>& x_val,
        unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY> y_base=f(x);
        for(size_t i=0; i<NUM_EQUALITY; i++){
            AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY> z=g_eq[i](x);
            y_base=y_base+x[DIM+i]*z;
        }
        if(!Minimization::minimization(y_base,x_val)){ return false; }

        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,NUM_INEQUALITY> z_in;
        for(size_t i=0; i<NUM_INEQUALITY; i++){
            z_in[i]=g_in[i](x);
        }
        while(true){
            auto y=y_base;
            bool flag_break=true;
            for(size_t i=0; i<NUM_INEQUALITY; i++){
                if( 0<(*z_in[i])(x_val) ){
//                    std::cout << "Inequality #" << i << " is not satisfied." << std::endl;
                    y=y+x[DIM+NUM_EQUALITY+i]*z_in[i];
                    flag_break=false;
                }
            }
            if(flag_break) break;
            if(!Minimization::minimization(y,x_val)){ return false; }
        }
        return true;
    }
}
#endif // MINIMIZATION_HPP_INCLUDED
