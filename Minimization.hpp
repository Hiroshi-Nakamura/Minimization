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
        const unsigned int max_num_iteration=1000)
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
        Variant of minimization() for usability.
        The second argument is acceptable for std::function.
    */
    template<int DIM>
    bool minimization(
        const std::function<AutomaticDifferentiation::FuncPtr<double,DIM>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM>,DIM>&)>& f,
        Eigen::Matrix<double,DIM,1>& x,
        const unsigned int max_num_iteration=1000)
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
        const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>& x,
        const AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>& y,
        const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,NUM_EQUALITY>& z,
        Eigen::Matrix<double,DIM+NUM_EQUALITY,1>& x_val,
        const unsigned int max_num_iteration=1000)
    {
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> y_extended=y;
        for(size_t i=0; i<NUM_EQUALITY; i++){
            y_extended=y_extended+x[DIM+i]*z[i];
        }
        return Minimization::minimization(y_extended,x_val,max_num_iteration);
    }

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::array<std::function>.
    */
    template<int DIM, int NUM_EQUALITY>
    bool minimization_with_equality_constraints(
        const std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)>& f,
        const std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)>,NUM_EQUALITY>& g,
        Eigen::Matrix<double,DIM+NUM_EQUALITY,1>& x_val,
        const unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> y=f(x);
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,NUM_EQUALITY> z;
        for(size_t i=0; i<NUM_EQUALITY; i++){
            z[i]=g[i](x);
        }
        return Minimization::minimization_with_equality_constraints<DIM,NUM_EQUALITY>(x,y,z,x_val,max_num_iteration);
    }

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::array.
    */
    template<int DIM, int NUM_EQUALITY>
    bool minimization_with_equality_constraints(
        const std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&)>& f,
        const std::function< std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,NUM_EQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY>&) >& g,
        Eigen::Matrix<double,DIM+NUM_EQUALITY,1>& x_val,
        const unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,DIM+NUM_EQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY> y=f(x);
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY>,NUM_EQUALITY> z=g(x);
        return Minimization::minimization_with_equality_constraints(x,y,z,x_val,max_num_iteration);
    }


    /**
        DIM is the dimention of variables,
        NUM_EQUALITY is the number of equality_constraints and NUM_INEQUALITY is the number of inequality_constraints.
        Note: the variable x_val should have the "DIM+NUM_EQUALITY+NUM_INEQUALITY" dimension,
        such that the extended dimention will be used for the Lagrange multipliers method.
    */
    template<int DIM, int NUM_EQUALITY, int NUM_INEQUALITY>
    bool minimization_with_constraints(
        const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>& x,
        const AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>& y,
        const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,NUM_EQUALITY>& z_eq,
        const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,NUM_INEQUALITY>& z_in,
        Eigen::Matrix<double,DIM+NUM_EQUALITY+NUM_INEQUALITY,1>& x_val,
        const unsigned int max_num_iteration=1000)
    {
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY> y_base=y;
        for(size_t i=0; i<NUM_EQUALITY; i++){
            y_base=y_base+x[DIM+i]*z_eq[i];
        }
        if(!Minimization::minimization(y_base,x_val,max_num_iteration)){ return false; }

        while(true){
            auto y_extended=y_base;
            bool flag_break=true;
            for(size_t i=0; i<NUM_INEQUALITY; i++){
                if( 0<(*z_in[i])(x_val) ){
//                    std::cout << "Inequality #" << i << " is not satisfied." << std::endl;
                    y_extended=y_extended+x[DIM+NUM_EQUALITY+i]*z_in[i];
                    flag_break=false;
                }
            }
            if(flag_break) break;
            if(!Minimization::minimization(y_extended,x_val,max_num_iteration)){ return false; }
        }
        return true;
    }

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::array.
    */
    template<int DIM, int NUM_EQUALITY, int NUM_INEQUALITY>
    bool minimization_with_constraints(
        const std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)>& f,
        const std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)>,NUM_EQUALITY>& g_eq,
        const std::array<std::function<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>(const std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY>&)>,NUM_INEQUALITY>& g_in,
        Eigen::Matrix<double,DIM+NUM_EQUALITY+NUM_INEQUALITY,1>& x_val,
        const unsigned int max_num_iteration=1000)
    {
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,DIM+NUM_EQUALITY+NUM_INEQUALITY> x=AutomaticDifferentiation::createVariables<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>();
        AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY> y=f(x);
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,NUM_INEQUALITY> z_eq;
        for(size_t i=0; i<NUM_INEQUALITY; i++){
            z_eq[i]=g_eq[i](x);
        }
        std::array<AutomaticDifferentiation::FuncPtr<double,DIM+NUM_EQUALITY+NUM_INEQUALITY>,NUM_INEQUALITY> z_in;
        for(size_t i=0; i<NUM_INEQUALITY; i++){
            z_in[i]=g_in[i](x);
        }
        return minimization_with_constraints<DIM,NUM_EQUALITY,NUM_INEQUALITY>(x,y,z_eq,z_in,x_val,max_num_iteration);
    }
}
#endif // MINIMIZATION_HPP_INCLUDED
