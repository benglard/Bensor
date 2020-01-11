#include "tensor.h"

int main(void) {
  std::cout << Tensor<double>::empty({2, 2}) << std::endl;

  Tensor<int> a({1, 2, 3, 4, 5, 6}, {1, 2, 1, 3});
  std::cout << a;

  a.at({0, 1, 0, 1}) = 10;

  for (int i{0}; i < a.size(0); ++i)
    for (int j{0}; j < a.size(1); ++j)
      for (int k{0}; k < a.size(2); ++k)
        for (int l{0}; l < a.size(3); ++l)
          std::cout << "a(" << i << ", " << j << ", " << k << ", " << l
                    << ") = " << a.at({i, j, k, l}) << std::endl;
  std::cout << std::endl;

  a.select(1, 1).at({0, 0, 1}) = 5;
  std::cout << a << std::endl;

  std::cout << a.select(1, 0) << std::endl;
  std::cout << a.select(1, 1) << std::endl;
  std::cout << a.select(3, 0) << std::endl;
  std::cout << a.select(3, 1) << std::endl;
  std::cout << a.select(3, 2) << std::endl;

  std::cout << a.permute({0, 1, 3, 2}) << std::endl;

  Tensor<int> b({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  std::cout << b.select(1, 1).select(1, 1) << std::endl;

  Tensor<int> c({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3});
  std::cout << c << std::endl << c.narrow(1, 1, 2).narrow(2, 0, 2) << std::endl;

  Tensor<int> d({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  std::cout << d << std::endl << d.narrow(1, 0, 1).narrow(2, 0, 1) << std::endl;

  d.narrow(1, 0, 1).narrow(2, 0, 1).at({0, 0, 0}) = 0;
  std::cout << d << std::endl;

  return 0;
}
