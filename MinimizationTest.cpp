#include "Minimization.hpp"
#include <iostream>

template <typename T,int DIM>
T f(const std::array<T,DIM>& x)
{
    return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
} /// int DIM used for constrained minimization

template<typename T, int DIM> T g_eq(const std::array<T,DIM>& x)
{
    return 4.0*x[0]+x[1]+2.0*x[2]-2.0;
}

template<typename T, int DIM> T g_in(const std::array<T,DIM>& x)
{
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]-1.0;
//    return 1.0-x[0]*x[0]-x[1]*x[1]-x[2]*x[2];
}


Eigen::Matrix<double,3,1> without_constraint()
{
    Eigen::Matrix<double,3,1> x_val{0.0, 0.0, 0.0};
    Minimization::minimization<3>(f<AutomaticDifferentiation::FuncPtr<double,3>,3>,x_val);
    return x_val;
}

Eigen::Matrix<double,4,1> with_equality_constraint()
{
    Eigen::Matrix<double,4,1> x_val_extended{0.0, 0.0, 0.0, 0.0};
    Minimization::minimization_with_equality_constraints<3,1>(f<AutomaticDifferentiation::FuncPtr<double,4>,4>,{g_eq<AutomaticDifferentiation::FuncPtr<double,4>,4>},x_val_extended);

/*
    Minimization::minimization_with_constraints<2,1,1>(
        f<AutomaticDifferentiation::FuncPtr<double,4>,4>,
        {g_eq<AutomaticDifferentiation::FuncPtr<double,4>,4>},
//        {g_in<AutomaticDifferentiation::FuncPtr<double,4>,4>},
        Minimization::NO_CONSTRAINT<4>,
        x_val_extended);
*/


    return x_val_extended;
}

#if 0
Eigen::Matrix<double,2,1> with_inequality_constraint()
{
    Eigen::Matrix<double,2,1> x_val{0.0,0.0};
    Minimization::minimization<2>(f<AutomaticDifferentiation::FuncPtr<double,2>,2>,x_val);
    //
    AutomaticDifferentiation::FuncPtr<double,2> z=g_in<AutomaticDifferentiation::FuncPtr<double,2>,2>(AutomaticDifferentiation::createVariables<double,2>());
    if(0.0<(*z)(x_val)) return x_val;
    //
    Eigen::Matrix<double,3,1> x_val_extended{x_val[0],x_val[1],0.0};
    Minimization::minimization_with_equality_constraints<2,1>(f<AutomaticDifferentiation::FuncPtr<double,3>,3>,{g<AutomaticDifferentiation::FuncPtr<double,3>,3>},x_val_extended);

    Eigen::Matrix<double,2,1> rtn;
    rtn[0]=x_val_extended[0];
    rtn[1]=x_val_extended[1];
    return rtn;
}

Eigen::Matrix<double,4,1> with_inequality_constraint()
{
    Eigen::Matrix<double,4,1> x_val_extended({0.0, 0.0, 0.0, 0.0});
    Minimization::minimization_with_constraints<3,0,1>(
        f<AutomaticDifferentiation::FuncPtr<double,4>,4>,
        Minimization::NO_CONSTRAINT<4>,
//        {g_eq<AutomaticDifferentiation::FuncPtr<double,4>,4>},
        {g_in<AutomaticDifferentiation::FuncPtr<double,4>,4>},
        x_val_extended);
    return x_val_extended;
}
#endif

int main(int argc, char** argv)
{
    try{
        std::cout << "without_constraint()=" << std::endl << without_constraint() << std::endl;
        std::cout << "with_equality_constraint()=" << std::endl << with_equality_constraint() << std::endl;
//        std::cout << "with_inequality_constraint()=" << std::endl << with_inequality_constraint() << std::endl;
    }catch(std::string message){
        std::cout << message << std::endl;
    }
}
