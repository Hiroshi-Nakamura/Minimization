#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <iostream>
#include <vector>
#include <cassert>
#include <atomic>

namespace Minimization {

    using namespace AutomaticDifferentiation;

    const std::vector<std::function< FuncPtr<double>(const std::vector<FuncPtr<double>>&) >> NO_CONSTRAINT_VEC;
    const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) > NO_CONSTRAINT_FUN=nullptr;
    constexpr unsigned int MAX_NUM_ITERATION=1000;
    constexpr double EPSILON=1.0e-9;
    constexpr double EPSILON_INEQUALITY=1.0e-6;
    static std::atomic<bool> STOP_FLAG(false);

    /**
        minimization without any constraints.
    */
    bool minimization(
        const FuncPtr<double>& f,
        Eigen::VectorXd& x,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        Variant of minimization() for usability.
        The second argument is acceptable for std::function.
    */
    bool minimization(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        minimization with equality constraints.
    */
    bool minimization_with_equality_constraints(
        const std::vector<FuncPtr<double>>& x,
        const FuncPtr<double>& y,
        const std::vector<FuncPtr<double>>& z,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::vector<std::function>.
    */
    bool minimization_with_equality_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        Variant of minimization_with_equality_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::vector.
    */
    bool minimization_with_equality_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        minimization with equality constraints of both of equality and inequality.
    */
    bool minimization_with_constraints(
        const std::vector<FuncPtr<double>>& x,
        const FuncPtr<double>& y,
        const std::vector<FuncPtr<double>>& z_eq,
        const std::vector<FuncPtr<double>>& z_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        const double epsilon_inequality=EPSILON_INEQUALITY,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::vector<std::function>.
    */
    bool minimization_with_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_eq,
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        const double epsilon_inequality=EPSILON_INEQUALITY,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    /**
        Variant of minimization_with_constraints() for usability.
        The argument of cost function is acceptable for std::function,
        and the argument of constraints is acceptable for std::function, which return std::vector.
    */
    bool minimization_with_constraints(
        const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_eq,
        const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_in,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        const double epsilon_inequality=EPSILON_INEQUALITY,
        std::atomic<bool>& stop_flag=STOP_FLAG);
}

///
/// implementation
///

inline bool Minimization::minimization(
    const FuncPtr<double>& f,
    Eigen::VectorXd& x,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x.rows();
    MatFuncPtr<double> jac=jacobian<double>(f,dim);
    MatFuncPtr<double> hes=hessian<double>(f,dim);
    bool flag_deficient=false;
#ifdef DEBUG
    std::cout << "x:" << std::endl << x << std::endl;
    std::vector<double> x_val(x.data(),x.data()+dim);
    std::cout << "f(x)=" << (*f)(x_val) << std::endl;
#endif
    for(unsigned int i=0; !(stop_flag.load())&& i<max_num_iteration; i++){
        Eigen::MatrixXd jac_val=jac(x);
        Eigen::MatrixXd hes_val=hes(x);
        /// Calculate delta_x, which satisfies the equation: H delta_x = -jac
        /// Direct method is calculating this simultaneous equations.
        Eigen::VectorXd delta_x;
        auto lu=hes_val.fullPivLu();
        if(lu.rank()!=(signed int)dim){
            std::cout << "Warning: rank deficient! rank is " << lu.rank() << " (should be " << dim << ")" << std::endl;
            flag_deficient=true;
#ifdef USE_CONJUGATE_GRADIENT
            /// Avoiding the rank deficient of H, Conjugate Gradient Method will be used for solving the equation "H delta_x = -jac".
            Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
            cg.compute(hes_val);
            delta_x=cg.solve(-jac_val);
#endif // USE_CONJUGATE_GRADIENT
        }else{
            if(flag_deficient){
                std::cout << "Info: rank becomes sufficient." << std::endl;
                flag_deficient=false;
            }
            delta_x=lu.solve(-jac_val);
        }
        if(delta_x.norm()<epsilon) return true;
        x += delta_x;
#ifdef DEBUG
        std::cout << "x:" << std::endl << x << std::endl;
        std::vector<double> x_val(x.data(),x.data()+dim);
        std::cout << "f(x)=" << (*f)(x_val) << std::endl;
#endif
    }
    return false;
}

inline bool Minimization::minimization(
    const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    FuncPtr<double> y=f(createVariables<double>(dim));
    return minimization(y,x_val,max_num_iteration,epsilon,stop_flag);
}


inline bool Minimization::minimization_with_equality_constraints(
    const std::vector<FuncPtr<double>>& x,
    const FuncPtr<double>& y,
    const std::vector<FuncPtr<double>>& z,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
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
    x_val_extended.block(dim,0,num_equality,1).setOnes();
    bool rtn=minimization(y_extended,x_val_extended,max_num_iteration,epsilon,stop_flag);
    x_val=x_val_extended.block(0,0,dim,1);
    return rtn;
}

inline bool Minimization::minimization_with_equality_constraints(
    const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
    const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    size_t num_equality=g.size();
    std::vector<FuncPtr<double>> x=createVariables<double>(dim);
    FuncPtr<double> y=f(x);
    std::vector<FuncPtr<double>> z;
    for(size_t i=0; i<num_equality; i++){
        z.push_back(std::move(g[i](x)));
    }
    return minimization_with_equality_constraints(x,y,z,x_val,max_num_iteration,epsilon,stop_flag);
}

inline bool Minimization::minimization_with_equality_constraints(
    const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
    const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    std::vector<FuncPtr<double>> x=createVariables<double>(dim);
    FuncPtr<double> y=f(x);
    std::vector<FuncPtr<double>> z=g(x);
    return minimization_with_equality_constraints(x,y,z,x_val,max_num_iteration,epsilon,stop_flag);
}

inline bool Minimization::minimization_with_constraints(
    const std::vector<FuncPtr<double>>& x,
    const FuncPtr<double>& y,
    const std::vector<FuncPtr<double>>& z_eq,
    const std::vector<FuncPtr<double>>& z_in,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    const double epsilon_inequality,
    std::atomic<bool>& stop_flag)
{
    assert((int)x.size()==x_val.rows());

    /// calculate without inequality constraints
    if(!minimization_with_equality_constraints(x,y,z_eq,x_val,max_num_iteration,epsilon,stop_flag)){ return false; }

    std::vector<bool> flag_in(z_in.size(),true);
    while(true){
        /// check inequality constraints.
        bool flag=true;
        for(size_t i=0; i<z_in.size(); i++){
            if( epsilon_inequality<(*z_in[i])(x_val) ){
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
            if(!minimization_with_equality_constraints(x,y,z_eq_extended,x_val,max_num_iteration,epsilon,stop_flag)){ return false;}
        }
    }
    return true;
}

inline bool Minimization::minimization_with_constraints(
    const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
    const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_eq,
    const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& g_in,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    const double epsilon_inequality,
    std::atomic<bool>& stop_flag)
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
    return minimization_with_constraints(x,y,z_eq,z_in,x_val,max_num_iteration,epsilon,epsilon_inequality,stop_flag);
}

inline bool Minimization::minimization_with_constraints(
    const std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>& f,
    const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_eq,
    const std::function< std::vector<FuncPtr<double>>(const std::vector<FuncPtr<double>>&) >& g_in,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    const double epsilon_inequality,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    std::vector<FuncPtr<double>> x=createVariables<double>(dim);
    FuncPtr<double> y=f(x);
    std::vector<FuncPtr<double>> z_eq;
    if(g_eq!=nullptr){ z_eq=g_eq(x); }
    std::vector<FuncPtr<double>> z_in;
    if(g_in!=nullptr){ z_in=g_in(x); }
    return minimization_with_constraints(x,y,z_eq,z_in,x_val,max_num_iteration,epsilon,epsilon_inequality,stop_flag);
}
#endif // MINIMIZATION_HPP_INCLUDED
