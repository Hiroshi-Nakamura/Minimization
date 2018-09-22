#include "Minimization.hpp"
#include <iostream>

template <typename T,int DIM> T f(const std::array<T,DIM>& x)
{
    return x[0]*x[0]+x[1]*x[1]+2.0*x[0]-4.0*x[1]+1.0;
} /// int DIM used for constrained minimization

template<typename T, int DIM> T g(const std::array<T,DIM>& x)
{
    return x[1]-x[0]; /// means x[0]=x[1]
}

Eigen::Matrix<double,2,1> without_constraint()
{
    std::array<AutomaticDifferentiation::FuncPtr<double,2>,2> x=AutomaticDifferentiation::createVariables<double,2>();
    AutomaticDifferentiation::FuncPtr<double,2> y=f<AutomaticDifferentiation::FuncPtr<double,2>,2>(x);
    Eigen::Matrix<double,2,1> x_val{0.0,0.0};
    Minimization::minimization(y,x_val);
    return x_val;
}

Eigen::Matrix<double,2,1> with_equality_constraint()
{
        std::array<AutomaticDifferentiation::FuncPtr<double,3>,3> x=AutomaticDifferentiation::createVariables<double,3>();
        AutomaticDifferentiation::FuncPtr<double,3> y=f<AutomaticDifferentiation::FuncPtr<double,3>,3>(x);
        AutomaticDifferentiation::FuncPtr<double,3> z=g<AutomaticDifferentiation::FuncPtr<double,3>,3>(x);
        AutomaticDifferentiation::FuncPtr<double,3> y_constraint=y+x[2]*z;
        Eigen::Matrix<double,3,1> x_val{0.0,0.0,0.0};
        Minimization::minimization(y_constraint,x_val);
        Eigen::Matrix<double,2,1> rtn;
        rtn[0]=x_val[0];
        rtn[1]=x_val[1];
        return rtn;
}

Eigen::Matrix<double,2,1> with_inequality_constraint()
{

    Eigen::Matrix<double,2,1> without=without_constraint();
    AutomaticDifferentiation::FuncPtr<double,2> z=g<AutomaticDifferentiation::FuncPtr<double,2>,2>(AutomaticDifferentiation::createVariables<double,2>());
    if(0.0<(*z)(without)) return without;
    return with_equality_constraint();
}

int main(int argc, char** argv)
{
    std::cout << "without_constraint()=" << std::endl << without_constraint() << std::endl;
    std::cout << "with_equality_constraint()=" << std::endl << with_equality_constraint() << std::endl;
    std::cout << "with_inequality_constraint()=" << std::endl << with_inequality_constraint() << std::endl;
}
