#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#if 1
#include <eigen3/Eigen/Eigenvalues>
#endif
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
    constexpr double EPSILON_SOLVE_HESSIAN_JACOBIAN_EQUATION=2.0e-2;
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
#ifdef DEBUG
    std::cout << std::endl << "==Initial==" << std::endl;
    std::cout << "x:" << std::endl << x << std::endl;
    std::cout << "f(x)=" << (*f)(x) << std::endl;
#endif
    //
    Eigen::VectorXd delta_x;
    for(unsigned int i=0; !(stop_flag.load())&& i<max_num_iteration; i++){
#ifdef DEBUG
        std::cout << std::endl << "==Iteration #" << i << "==" << std::endl;
#endif // DEBUG
        const Eigen::MatrixXd jac_val=jac(x);
        const Eigen::MatrixXd hes_val=hes(x);

#if 0
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_hes(hes_val,false);
        std::cout << "eigenvalues of Hessian: " << std::endl << eigen_hes.eigenvalues() << std::endl;
#endif

        /// Calculate delta_x, which satisfies the equation: H delta_x = -jac
        /// First, calculating this simultaneous equations directly.
        std::cout << "Direct Method (LU)" << std::endl;
        Eigen::FullPivLU lu=hes_val.fullPivLu();

#if 0
        std::cout << "hes:" << std::endl << hes_val << std::endl;
        std::cout << "jac:" << std::endl << jac_val << std::endl;
#endif

        lu.setThreshold(1.0e-5);
        delta_x=lu.solve(-jac_val);
        //
        double relative_error=(hes_val*delta_x+jac_val).norm() / jac_val.norm();
        std::cout << "relative_error=" << relative_error << std::endl;
        if(!(relative_error<EPSILON_SOLVE_HESSIAN_JACOBIAN_EQUATION)){
            std::cout << "Conjugate Gradient Method" << std::endl;
            Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
            cg.compute(hes_val);
            delta_x=cg.solve(-jac_val);
            std::cout << "CG:       #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
            //
            relative_error=(hes_val*delta_x+jac_val).norm() / jac_val.norm();
            std::cout << "relative_error=" << relative_error << std::endl;
            if(!(cg.error()<EPSILON_SOLVE_HESSIAN_JACOBIAN_EQUATION)){
                std::cout << "Bi Conjugate Gradient STABilized Method" << std::endl;
                Eigen::BiCGSTAB<Eigen::MatrixXd> bicg;
                bicg.compute(hes_val);
                delta_x=bicg.solve(-jac_val);
                std::cout << "BiCGSTAB:       #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
                relative_error=(hes_val*delta_x+jac_val).norm() / jac_val.norm();
                std::cout << "relative_error=" << relative_error << std::endl;
                if(!(relative_error<EPSILON_SOLVE_HESSIAN_JACOBIAN_EQUATION)){
                    return false;
                }
            }
        }

        /// update x
        double norm=delta_x.norm();
#ifdef DEBUG
        std::cout << "delta_x:" << std::endl << delta_x << std::endl;
        std::cout << "delta_x.norm=" << norm << std::endl;
#endif
        if(norm<epsilon){
            return true;
        }
        x += delta_x;
#ifdef DEBUG
        std::cout << "x:" << std::endl << x << std::endl;
        std::cout << "f(x)=" << (*f)(x) << std::endl;
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
