# Minimization
Implementation of Non-Linear Minimization with constraints by using of Newton method and Automatic Differentiation.
For Automatic Differentiation, this needs "AutomaticDifferentiation.hpp", which you can get from my other repository.

Because of using automatic differentiation, the cost function and constraint functions should be implemented with templete.

    template <typename T,int DIM> T f(const std::array<T,DIM>& x)
    {
        return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
    }
    
The function argument is `std::array`, whose size will be determined by DIM number of the template.
