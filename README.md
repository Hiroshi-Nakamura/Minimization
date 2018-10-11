# Minimization
This provides you Non-Linear Minimization with Constraints by using of Newton method.
For Hessian calculation in Newton method, Automatic Differentiation technique will be used,
which is inplemented in "AutomaticDifferentiation.hpp" is needed, which you can get from my other repository.

Because of using automatic differentiation, the cost function and constraint functions should be implemented with templete.

    template <typename T> T f(const std::vector<T>& x)
    {
        return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
    }
    
The function argument is `std::vector`. Also, in "MinimizationTest.cpp", you can see how to use this library.

If you want to no-constraint minimization, call Minimization::minimization() function.
The first argument ('std::function') is cost function, which you want to minimization.
The second argument ('std::vector<double>') is an initial value as an input, 
and the value will be updated by the solustion as an output.
Note that the dimension of the second argument should be the same which you want to solve for x..
