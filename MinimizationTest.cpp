#include "Minimization.hpp"
#include <iostream>

template <typename T,int DIM>
T f(const std::array<T,DIM>& x)
{
    return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
} /// int DIM used for constrained minimization

template<typename T, int DIM> T g_eq(const std::array<T,DIM>& x)
{
    return x[0]+x[1]+x[2]-2.0;
}

template<typename T, int DIM> T g_in(const std::array<T,DIM>& x)
{
    return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]-1.0;
//    return 1.0-x[0]*x[0]-x[1]*x[1]-x[2]*x[2];
}

int main(int argc, char** argv)
{
    try{
        {
            std::cout << "without_constraint:" << std::endl;
            Eigen::Matrix<double,3,1> x_val{0.0, 0.0, 0.0};
            Minimization::minimization<3>(f<AutomaticDifferentiation::FuncPtr<double,3>,3>,x_val);
            std::cout << x_val << std::endl;
        }
        {
            std::cout << "with_equality_constraint:" << std::endl;
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
            std::cout << x_val_extended << std::endl;
        }
        {
            std::cout << "with_inequality_constraint:" << std::endl;
            Eigen::Matrix<double,5,1> x_val_extended;
            Minimization::minimization_with_constraints<3,1,1>(
                f<AutomaticDifferentiation::FuncPtr<double,5>,5>,
//                Minimization::NO_CONSTRAINT<4>,
                {g_eq<AutomaticDifferentiation::FuncPtr<double,5>,5>},
                {g_in<AutomaticDifferentiation::FuncPtr<double,5>,5>},
                x_val_extended);
            std::cout << x_val_extended << std::endl;
        }
    }catch(std::string message){
        std::cout << message << std::endl;
    }
}
