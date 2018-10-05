#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <iostream>
#include <vector>
#include <cassert>

namespace Minimization {

    using namespace AutomaticDifferentiation;

    std::vector<std::function< FuncPtr<double>(const std::vector<FuncPtr<double>>&) >> NO_CONSTRAINT;

    /**
        minimization without any constraints.
    */
    bool minimization(
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
    bool minimization(
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
    bool minimization_with_equality_constraints(
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
        and the argument of constraints is acceptable for std::vector<std::function>.    */
    bool minimization_with_equality_constraints(
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
    bool minimization_with_equality_constraints(
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
    bool minimization_with_constraints(
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

        while(true){
            std::vector<FuncPtr<double>> z_eq_extended=z_eq;
            bool flag=true;
            /// check inequality constraints.
            for(auto z: z_in){
                if( 0<(*z)(x_val) ){
//                    std::cout << "Inequality #" << i << " is not satisfied." << std::endl;
                    z_eq_extended.push_back(z);
                    flag=false;
                }
            }
            if(flag){
                /// if safisfied, current x_val is the solution.
                break;
            }else{
                /// if not safisfied yet, again calculate with the inequality constrains.
                if(!minimization_with_equality_constraints(x,y,z_eq_extended,x_val,max_num_iteration)){ return false;}
            }
        }
        return true;
    }

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::array.
    */
    bool minimization_with_constraints(
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
}

#endif // MINIMIZATION_HPP_INCLUDED
