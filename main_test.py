import pytest
import main

golden_data = [
    (1, 2),
    (2, 4),
    (3, 6)
]

def test_avg_mse_zero():
    calculated = (2, 0)
    assert main.avg_mse(golden_data, calculated[0], calculated[1]) == 0

def test_avg_mse_nonzero():
    calculated = (1, 2)
    assert main.avg_mse(golden_data, calculated[0], calculated[1]) == 2 / 3.

def test_grad_step():
    assert main.grad_step(golden_data,1, 2) == (1.0001333333333333, 2.)

