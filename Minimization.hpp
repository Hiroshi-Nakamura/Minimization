#ifndef MINIMIZATION_HPP_INCLUDED
#define MINIMIZATION_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <iostream>
#include <vector>

namespace Minimization {
    template<typename T, int DIM>
    bool minimization(
        const AutomaticDifferentiation::FuncPtr<T,DIM>& f,
        Eigen::Matrix<T,DIM,1>& x,
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
    }

}
#endif // MINIMIZATION_HPP_INCLUDED
