#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <iostream>
#include <vector>
#include <cassert>

namespace Minimization {

    using namespace AutomaticDifferentiation;

    const std::vector<std::function< FuncPtr<double>(const std::vector<FuncPtr<double>>&) >> NO_CONSTRAINT_VEC;
    const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) > NO_CONSTRAINT_FUN=nullptr;
    constexpr double EPSILON=1.0e-9;
    constexpr double EPSILON_INEQUALITY=1.0e-6;

    /**
        minimization without any constraints.
    */
    inline bool minimization(
        const FuncPtr<double>& f,
        Eigen::VectorXd& x,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x.rows();
        MatFuncPtr<double> jac=jacobian<double>(f,dim);
        MatFuncPtr<double> hes=hessian<double>(f,dim);
        for(unsigned int i=0; true; i++){
            Eigen::MatrixXd jac_val=jac(x);
            Eigen::MatrixXd hes_val=hes(x);
            Eigen::VectorXd delta_x=hes_val.fullPivLu().solve(-jac_val);
            if(delta_x.norm()<EPSILON) return true;
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
    inline bool minimization(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x_val.rows();
        FuncPtr<double> y=f(createVariables<double>(dim));
        return minimization(y,x_val);
    }

    /**
        minimization with equality constraints.
    */
    inline bool minimization_with_equality_constraints(
        const std::vector<FuncPtr<double>>& x,
        const FuncPtr<double>& y,
        const std::vector<FuncPtr<double>>& z,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        assert((int)x.size()==x_val.rows());
        size_t dim=x_val.rows();
        size_t num_equality=z.size();
        std::vector<FuncPtr<double>> x_entended=x;
        FuncPtr<double> y_extended=y;
        for(size_t i=0; i<num_equality; i++){
            x_entended.emplace_back(new Variable<double>(dim+i));
            y_extended=y_extended+x_entended.back()*z[i];
        }
        Eigen::VectorXd x_val_extended(dim+num_equality);
        x_val_extended.block(0,0,dim,1)=x_val;
        x_val_extended.block(dim,0,num_equality,1).setZero();
        bool rtn=minimization(y_extended,x_val_extended,max_num_iteration);
        x_val=x_val_extended.block(0,0,dim,1);
        return rtn;
    }

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::vector<std::function>.
    */
    inline bool minimization_with_equality_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x_val.rows();
        size_t num_equality=g.size();
        std::vector<FuncPtr<double>> x=createVariables<double>(dim);
        FuncPtr<double> y=f(x);
        std::vector<FuncPtr<double>> z;
        for(size_t i=0; i<num_equality; i++){
            z.push_back(std::move(g[i](x)));
        }
        return minimization_with_equality_constraints(x,y,z,x_val,max_num_iteration);
    }

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::vector.
    */
    inline bool minimization_with_equality_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x_val.rows();
        std::vector<FuncPtr<double>> x=createVariables<double>(dim);
        FuncPtr<double> y=f(x);
        std::vector<FuncPtr<double>> z=g(x);
        return minimization_with_equality_constraints(x,y,z,x_val,max_num_iteration);
    }


    /**
        minimization with equality constraints of both of equality and inequality.
    */
    inline bool minimization_with_constraints(
        const std::vector<FuncPtr<double>>& x,
        const FuncPtr<double>& y,
        const std::vector<FuncPtr<double>>& z_eq,
        const std::vector<FuncPtr<double>>& z_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        assert((int)x.size()==x_val.rows());

        /// calculate without inequality constraints
        if(!minimization_with_equality_constraints(x,y,z_eq,x_val,max_num_iteration)){ return false; }

        std::vector<bool> flag_in(z_in.size(),true);
        while(true){
            /// check inequality constraints.
            bool flag=true;
            for(size_t i=0; i<z_in.size(); i++){
                if( EPSILON_INEQUALITY<(*z_in[i])(x_val) ){
                    if(!flag_in[i]){ /// also in privious trial, cannot find a solution.
                        throw std::string("Cannot find solution satisfied #")+std::to_string(i)+std::string(" inequation.");
                    }
                    std::cout << "Inequality #" << i << " is not satisfied." << std::endl;
                    flag_in[i]=false;
                    flag=false;
                }
            }
            if(flag){
                /// if safisfied, current x_val is the solution.
                break;
            }else{
                /// if not safisfied yet, again calculate with the inequality constrains.
                std::cout << "re-try minimization." << std::endl;
                std::vector<FuncPtr<double>> z_eq_extended=z_eq;
                for(size_t i=0; i<z_in.size(); i++){
                    if( !flag_in[i] ){
                        std::cout << "Inequality #" << i << " will be used as an equality constraint." << std::endl;
                        z_eq_extended.push_back(z_in[i]);
                    }
                }
                if(!minimization_with_equality_constraints(x,y,z_eq_extended,x_val,max_num_iteration)){ return false;}
            }
        }
        return true;
    }

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::vector<std::function>.
    */
    inline bool minimization_with_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_eq,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x_val.rows();
        size_t num_equality=g_eq.size();
        size_t num_inequality=g_in.size();
        std::vector<FuncPtr<double>> x=createVariables<double>(dim);
        FuncPtr<double> y=f(x);
        std::vector<FuncPtr<double>> z_eq;
        for(size_t i=0; i<num_equality; i++){
            z_eq.push_back(std::move(g_eq[i](x)));
        }
        std::vector<FuncPtr<double>> z_in;
        for(size_t i=0; i<num_inequality; i++){
            z_in.push_back(std::move(g_in[i](x)));
        }
        return minimization_with_constraints(x,y,z_eq,z_in,x_val,max_num_iteration);
    }

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::vector.
    */
    inline bool minimization_with_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_eq,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=1000)
    {
        size_t dim=x_val.rows();
        std::vector<FuncPtr<double>> x=createVariables<double>(dim);
        FuncPtr<double> y=f(x);
        std::vector<FuncPtr<double>> z_eq;
        if(g_eq!=nullptr){ z_eq=g_eq(x); }
        std::vector<FuncPtr<double>> z_in;
        if(g_in!=nullptr){ z_in=g_in(x); }
        return minimization_with_constraints(x,y,z_eq,z_in,x_val,max_num_iteration);
    }

}

#endif // MINIMIZATION_HPP_INCLUDED
