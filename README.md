# Minimization
This provides you Non-Linear Minimization with Constraints by using of Newton method.
For Hessian calculation in Newton method, Automatic Differentiation technique will be used,
which is implemented in "AutomaticDifferentiation.hpp", and which you can get from my other repository.
In "MinimizationTest.cpp", you can see how to use this library.

Because of using automatic differentiation, the cost function and constraint functions should be implemented with templete.
The example are shown the below:

    template <typename T> T f(const std::vector<T>& x)
    {
        return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)+(x[2]-1.0)*(x[2]-1.0);
    }

If you want no-constraint minimization, call `Minimization::minimization()` function.
The first argument ('std::function') is cost function, which you want to minimization.
The second argument ('std::vector<double>') is an initial value as an input, 
and the value will be updated by the solustion as an output.
Note that the dimension of the second argument should be the same which you want to solve for x.

If you want with-constraint minimimzation, call `Minimization::minimization_with_constraint()` function.
The first and the 4th argument is cost function and x (initial / solution), similarly as `minimization()`.
If the constraints are only equalities, `Minimization::minimization_with_equality_constraints()` function is possible.
The 2nd and 3rd arguments are the constrains. Because they are multiple, the arguments are `std::vector<std::function>` 
or `std::function<std::vector<double>(std::vector<double>)>`.
