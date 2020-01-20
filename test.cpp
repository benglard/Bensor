#include "tensor.h"

using namespace bensor;

struct Point {
  Point() {
    x = (float)rand() / RAND_MAX;
    y = (float)rand() / RAND_MAX;
    z = (float)rand() / RAND_MAX;
  }
  Point(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
  float x, y, z;
};
std::ostream &operator<<(std::ostream &os, const Point &p) {
  os << "Point(" << p.x << ", " << p.y << ", " << p.z << ")";
  return os;
}

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
  auto &x = e.item();
  x = 3;
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
                  << ") " << col.second << col.second.item() << std::endl
                  << std::endl;

  std::cout << c[0][{0, -1, 2}] << std::endl
            << c[0][{0, 3, 2}][{1, 2, 1}] << std::endl;

  h.fill_(1);
  std::cout << h << std::endl;
  g[0][{0, -1, 2}].fill_(1);
  std::cout << g << std::endl;

  h.apply_([](const int v) { return v + 1; });
  std::cout << h << std::endl;

  auto ha = h.apply([](const int v) { return v + 1; });
  std::cout << h << std::endl << ha << std::endl;

  auto had = h.apply<double>([](const int v) { return std::sqrt(v); });
  std::cout << had << std::endl;

  auto cloud = Tensor<Point>::empty({5});
  auto cloud_x = cloud.apply<float>([](const Point &p) { return p.x; });
  std::cout << cloud << std::endl << cloud_x << std::endl;

  auto cloud_x_gt = cloud.where([](const Point &p) { return p.x > 0.5; });
  std::cout << cloud_x_gt << std::endl;
  cloud_x.masked_fill_(cloud_x_gt, 0);
  cloud.masked_fill_(cloud_x_gt, {1, 2, 3});
  std::cout << cloud_x << std::endl;
  std::cout << cloud << std::endl
            << cloud.masked_select(cloud_x_gt) << std::endl;

  auto index = Tensor<IndexType>(std::vector<IndexType>{0});
  auto values = Tensor<Point>::empty({1});
  cloud.index_put_(index, values);
  std::cout << cloud << std::endl;

  cloud.masked_scatter_(cloud_x_gt, Tensor<Point>::empty({1, 3}));
  std::cout << cloud << std::endl;

  std::cout << had.cast<int>() << std::endl;

  return 0;
}
