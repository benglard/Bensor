# Bensor

Ben's tiny Tensor Library. PyTorch inspired syntax but fully templated, e.g. not limited to just numeric types.

Supports initialization from vectors, creating views (select + narrow + indexing), iterators, reshaping, flattening, permuting, and getting access the underlying data.

## Sample

```c++
#include <array>

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
  std::cout << c.narrow(1, 0, 3, 2) << std::endl;

  Tensor<int> d({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  std::cout << d << std::endl << d.narrow(1, 0, 1).narrow(2, 0, 1) << std::endl;

  d.narrow(1, 0, 1).narrow(2, 0, 1).at({0, 0, 0}) = 0;
  std::cout << d << std::endl;

  Tensor<int> e({1});
  std::cout << e << "item = " << e.item() << std::endl << std::endl;

  Tensor<int> f({1, 2, 3, 4}, {2, 2});
  std::cout << f << std::endl << f.t() << std::endl;
  std::cout << f[0] << std::endl << f[1] << std::endl;
  std::cout << b[0][1][1] << b[0][1][1].item() << std::endl << std::endl;

  std::array<int, 10> garr{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  auto g = Tensor<int>::fromBlob(garr.data(), {2, 5});
  std::cout << g << std::endl;

  Tensor<int> h({1, 2, 3, 4, 5, 6}, {1, 1, 2, 1, 3, 1, 1, 1});
  std::cout << h << std::endl << h.squeeze() << std::endl;

  auto h2 = h.squeeze().unsqueeze(0).unsqueeze(3);
  auto h3 = h2.squeeze(3).squeeze(0);
  std::cout << h2 << std::endl << h3 << std::endl;

  for (auto elem : d)
    for (auto row : elem.second)
      for (auto col : row.second)
        std::cout << "(" << elem.first << ", " << row.first << ", " << col.first
                  << ") " << col.second << std::endl;

  return 0;
}
```
