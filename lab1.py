# def lagrange_interpolation(x_current, nodes, values)->float:
#     res = 0;
#     for i in range(len(values)):
#         cur = 1;
#         for j in range(len(nodes)):
#             if i != j:
#                 cur *= (x_current - nodes[j]) / (nodes[i] - nodes[j]);
#         res += values[i] * cur;
#     return res;

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.interpolate import lagrange


def lagrange_interpolation(x_current, nodes, function)->float:
    res = 0;
    for i in range(len(nodes)):
        cur = 1;
        for j in range(len(nodes)):
            if i != j:
                cur *= (x_current - nodes[j]) / (nodes[i] - nodes[j]);
        res += function(nodes[i]) * cur;
    return res;

def newton_interpolation(x_current, nodes, function)->float:
    
    def divided_difference(nodes, function):
        summ = 0;
        for i in range(len(nodes)):
            mult = function(nodes[i]);
            for j in range (len(nodes)):
                if i != j:
                    mult /= (nodes[i] - nodes[j]);
            summ += mult;
        return summ;
    
    res = function(nodes[0]);
    for i in range(1, len(nodes)):
        mult = divided_difference(nodes[0:i+1], function);
        for j in range(i):
            mult *= (x_current - nodes[j]);
        res += mult;
    return res;

def barycentric_interpolation(x_current, nodes, function)->float:
    def find_beta(current_index, nodes):
        divider = 1;
        for i in range(len(nodes)):
            if current_index != i:
                divider *= (nodes[current_index] - nodes[i]);
        return (1/divider);

    numinator = 0;
    denominator = 0;
    
    for i in range(len(nodes)):
        numinator += (find_beta(i, nodes) * function(nodes[i])) / (x_current - nodes[i]);
        denominator += find_beta(i, nodes) / (x_current - nodes[i]);

    return numinator/denominator;
                    
    
if __name__ == "__main__":
    def f(x): return np.sqrt(np.tan(x)**2 + 7*x**2);
    a = -0.3;
    b = 0.3;
    num_nodes_list = [5, 10, 15, 20];
    
    x_start = 0.1;
    errors = [];
    
    for num_nodes in num_nodes_list:
        nodes = np.linspace(a, b, num_nodes)
        values = f(nodes)
        y_start = barycentric_interpolation(x_start, nodes, f)
        errors.append(abs(f(x_start) - y_start))
        nodes = np.linspace(a, b, num_nodes)
        #values = f(nodes)
        x_values = np.linspace(a, b, 1000)
        interpolated_values = barycentric_interpolation(x_values, nodes, f)
        y_start = barycentric_interpolation(x_start, nodes, f)
        plt.plot(x_values, f(x_values), label='f(x) = tan^2(x) + 7x^2')
        plt.plot(x_values, interpolated_values, label='Интерполяция')
        plt.scatter(nodes, values, color='black', label='Узлы интерполяции')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Интерполяция')
        plt.legend()
        #plt.grid(True)
        plt.show()
        
    plt.plot(num_nodes_list, errors, marker='o')
    plt.xlabel('Число узлов интерполяции')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Анализ погрешности')
    #plt.grid(True)
    plt.show()
 
    n = 1;
    x_nodes = np.linspace(a, b, n+1)
    y_nodes = f(x_nodes)
    #lag_poly_arbitrary = lagrange(x_nodes, y_nodes)
    lag_poly_arbitrary = lambda x: lagrange_interpolation(x, x_nodes, f);
    newton_poly = lambda x: newton_interpolation(x, x_nodes, f)
    x_values = np.linspace(a, b, 1000)
    f_values = f(x_values)
    lagrange_interpolated_values = lag_poly_arbitrary(x_values)
    lagrange_practical_error = np.max(np.abs(f_values - lagrange_interpolated_values))
    x_uniform = np.linspace(a, b, n+1)
    y_uniform = f(x_uniform)
    #lag_poly_uniform = lagrange(x_uniform, y_uniform)
    lag_poly_uniform = lambda x: lagrange_interpolation(x, x_uniform, f);
    x_chebyshev = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*np.arange(n+1) + 1) / (2*(n+1)) * np.pi)
    y_chebyshev = f(x_chebyshev)
    #lag_poly_barycentric = lagrange(x_chebyshev, y_chebyshev)
    lag_poly_barycentric = lambda x: barycentric_interpolation(x, x_chebyshev, f);
    newton_interpolated_values = newton_poly(x_values)
    newton_practical_error = np.max(np.abs(f_values - newton_interpolated_values))
    x_values = np.linspace(a, b, 1000)
    f_values = f(x_values)
    lagrange_uniform_interpolated_values = lag_poly_uniform(x_values)
    lagrange_uniform_practical_error = np.max(np.abs(f_values - lagrange_uniform_interpolated_values))
    lagrange_barycentric_interpolated_values = lag_poly_barycentric(x_values)
    lagrange_barycentric_practical_error = np.max(np.abs(f_values - lagrange_barycentric_interpolated_values))
    print("Погрешность интерполяции формулой Лагранжа с произвольными узлами:", "{:.5f}".format(lagrange_practical_error))
    print("Погрешность интерполяции формулой Лагранжа с равноотстоящими узлами:", "{:.5f}".format(lagrange_uniform_practical_error))
    print("Погрешность интерполяции в барицентрическом виде:", "{:.5f}".format(lagrange_barycentric_practical_error))
    print("Погрешность интерполяции формулой Нютона через разделенные разности:", "{:.5f}".format(newton_practical_error))
