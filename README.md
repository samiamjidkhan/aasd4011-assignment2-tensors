# Computational Derivation and Matrix Multiplication

This repository contains solutions to exercises on computational derivation, root mean square error (RMSE) implementation, and matrix multiplication using tensors.

## Files

1. **`computational_derivation.py`**: Implements the function `compute_derivative` to calculate the derivative of a function at a given point using the definition of the derivative.
   
2. **`rmse.py`**: Implements the function `rmse` to calculate the root mean square error between predicted and true values.

3. **`tensor_multiplication.py`**: Implements the function `matrix_multiplication` to perform matrix multiplication using tensors, ensuring correct dimension matching by transposing the weight matrix.

## Exercise Details

### Computational Derivation

Given a function \( f(x) \), the derivative \( f'(x) \) at point \( x \) is calculated using:
- The definition of the derivative \( f'(x) = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon} \).

### RMSE Implementation

Root Mean Square Error (RMSE) is calculated as:
\[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{true}^{(i)} - y_{pred}^{(i)})^2} \]
where \( y_{true} \) and \( y_{pred} \) are tensors of true and predicted values, respectively.

### Matrix Multiplication

Matrix multiplication is performed using tensors:
\[ Z = X \cdot W^T \]
where \( X \) is a tensor of shape \( (m, n) \) and \( W \) is a tensor of shape \( (p, n) \). \( W^T \) is used to ensure correct dimensions for multiplication.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/samiamjidkhan/computational-derivation-matrix-multiplication.git
   cd computational-derivation-matrix-multiplication
   ```

2. Run individual scripts or integrate functions into your projects as needed.

### Running Tests

To validate locally:

1. Install pytest if not already installed:
   ```sh
   pip install pytest
   ```

2. Run tests:
   ```sh
   pytest
   ```

