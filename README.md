# Minimization
Implementation of Non-Linear Minimization with constraints by using of Newton method.
For Hessian calculation in Newton method, Automatic Differentiation techniqu will be used,
which is inplemented in "AutomaticDifferentiation.hpp" is needed, which you can get from my other repository.

Because of using automatic differentiation, the cost function and constraint functions should be implemented with templete.

    template <typename T> T f(const std::vector<T>& x)
    {
        return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
    }
    
The function argument is `std::vector`.
