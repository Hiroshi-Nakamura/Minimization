#include "Minimization.hpp"
#include <iostream>

template <typename T>
T f(const std::vector<T>& x)
{
    return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
} /// int DIM used for constrained minimization

template<typename T> T g_eq(const std::vector<T>& x)
{
    return x[0]+x[1]+x[2]-2.0;
}

template<typename T> T g_in(const std::vector<T>& x)
{
    return x[0];
//    return 1.0-x[0]*x[0]-x[1]*x[1]-x[2]*x[2];
}

int main(int argc, char** argv)
{
    using namespace AutomaticDifferentiation;
    using namespace Minimization;
    try{
        {
            std::cout << "without_constraint:" << std::endl;
            Eigen::VectorXd x_val(3);
            x_val << 0.0, 0.0, 0.0;
            minimization(f<FuncPtr<double>>,x_val);
            std::cout << x_val << std::endl;
        }
        {
            std::cout << "with_equality_constraint:" << std::endl;
            Eigen::VectorXd x_val(3);
            x_val << 0.0, 0.0, 0.0;
            minimization_with_equality_constraints(f<FuncPtr<double>>,{g_eq<FuncPtr<double>>},x_val);
            std::cout << x_val << std::endl;
        }
        {
            std::cout << "with_inequality_constraint:" << std::endl;
            Eigen::VectorXd x_val(3);
            x_val.setOnes();
            minimization_with_constraints(
                f<FuncPtr<double>>,
                {g_eq<FuncPtr<double>>},
                {g_in<FuncPtr<double>>},
                x_val);
            std::cout << x_val << std::endl;
        }
    }catch(std::string message){
        std::cout << message << std::endl;
    }
}
