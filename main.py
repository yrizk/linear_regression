import numpy as np

NUM_ITERATIONS = 1000
GRADIENT_DESC_PORTION = 0.0001

def avg_mse(golden_data, m, b):
    """Calculates the MSE from the list of 2-tuples."""
    error = 0
    for i in range(len(golden_data)):
        actual_x = golden_data[i][0]
        actual_y = golden_data[i][1]
        error += pow(actual_y - (actual_x * m + b), 2)
    return error / float(len(golden_data))

def grad_step(golden_data, m, b):
    """
        uses the slope of the MSE function to figure out where to go
        let f be the error function that accepts m,b
        d/dm = -2/N sum -xi (yi - mxi + b)
        d/db = -2/N sum -xi (yi - mxi + b)
    """
    N = float(len(golden_data))
    b_gradient = 0
    m_gradient = 0
    for i in range(int(N)):
        x = golden_data[i][0]
        y = golden_data[i][1]
        m_gradient += -(2/N) * x * (y - ((m * x) + b))
        b_gradient += -(2/N) * (y - ((m * x) + b))
    updated_m = m  - (GRADIENT_DESC_PORTION * m_gradient)
    updated_b = b  - (GRADIENT_DESC_PORTION * b_gradient)
    # the partial derivative of a parabola (y = x^2) is 2x
    return updated_m, updated_b

def start_grad_desc(golden_data, initial_m, initial_b):
    """ Returns the m,b with the minimal error after running iteratively."""
    b = initial_b
    m = initial_m
    for i in range(NUM_ITERATIONS):
        m, b = grad_step(golden_data, m, b)
    return m, b, mse(golden_data, m, b)

def main():
    x_y_points = np.genfromtxt("data.csv", delimiter = ",")
    final_m, final_b, final_error = start_grad_desc(x_y_points, float(0), float(0))
    print("Linear Regression Complete. The line has been fitted with m = {} , b = {} with MSE = {}".format(final_m, final_b, final_error))


if __name__ == "__main__":
    main()
