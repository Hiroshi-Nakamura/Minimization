#include "Minimization.hpp"
#include <iostream>

template <typename T,int DIM>
T f(const std::array<T,DIM>& x)
{
    return x[0]*x[0]+x[1]*x[1]+2.0*x[0]-4.0*x[1]+1.0;
} /// int DIM used for constrained minimization

template<typename T, int DIM> T g(const std::array<T,DIM>& x)
{
    return x[0]-x[1]; /// means x[0]=x[1]
}

Eigen::Matrix<double,2,1> without_constraint()
{
    Eigen::Matrix<double,2,1> x_val{0.0,0.0};
    Minimization::minimization<2>(f<AutomaticDifferentiation::FuncPtr<double,2>,2>,x_val);
    return x_val;
}

Eigen::Matrix<double,2,1> with_equality_constraint()
{
    Eigen::Matrix<double,3,1> x_val{0.0,0.0,0.0};
    Minimization::minimization_with_equality_constraints<2,1>(f<AutomaticDifferentiation::FuncPtr<double,3>,3>,{g<AutomaticDifferentiation::FuncPtr<double,3>,3>},x_val);
    Eigen::Matrix<double,2,1> rtn;
    rtn[0]=x_val[0];
    rtn[1]=x_val[1];
    return rtn;
}

Eigen::Matrix<double,2,1> with_inequality_constraint()
{
    AutomaticDifferentiation::FuncPtr<double,2> z=g<AutomaticDifferentiation::FuncPtr<double,2>,2>(AutomaticDifferentiation::createVariables<double,2>());
    //
    Eigen::Matrix<double,2,1> x_val{0.0,0.0};
    Minimization::minimization<2>(f<AutomaticDifferentiation::FuncPtr<double,2>,2>,x_val);
    if(0.0<(*z)(x_val)) return x_val;
    //
    Eigen::Matrix<double,3,1> x_val_extended{x_val[0],x_val[1],0.0};
    Minimization::minimization_with_equality_constraints<2,1>(f<AutomaticDifferentiation::FuncPtr<double,3>,3>,{g<AutomaticDifferentiation::FuncPtr<double,3>,3>},x_val_extended);
    Eigen::Matrix<double,2,1> rtn;
    rtn[0]=x_val_extended[0];
    rtn[1]=x_val_extended[1];
    return rtn;
}

int main(int argc, char** argv)
{
    std::cout << "without_constraint()=" << std::endl << without_constraint() << std::endl;
    std::cout << "with_equality_constraint()=" << std::endl << with_equality_constraint() << std::endl;
    std::cout << "with_inequality_constraint()=" << std::endl << with_inequality_constraint() << std::endl;
}
