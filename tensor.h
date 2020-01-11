#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

template <typename TensorType> class Tensor final {
public:
  using IndexType = std::int64_t;
  using TensorShape = std::vector<IndexType>;
  using AdvancedTensorShape = std::vector<std::vector<std::vector<IndexType>>>;

  Tensor(const std::vector<TensorType> &initial_data, const TensorShape &shape)
      : data_(initial_data), sizes_(shape) {
    fillSize();
    fillStrides();
    beginning_ = data_.data();
    ending_ = beginning_ + data_.size() * element_size();
  }
  Tensor(const std::vector<TensorType> &initial_data)
      : data_(initial_data),
        sizes_({static_cast<IndexType>(initial_data.size())}) {
    fillSize();
    fillStrides();
    beginning_ = data_.data();
    ending_ = beginning_ + data_.size() * element_size();
  }
  Tensor(TensorType *ptr, TensorType *end_ptr, const TensorShape &sizes,
         const TensorShape &strides, bool copy = false)
      : data_{}, beginning_(ptr), ending_(end_ptr), sizes_(sizes),
        strides_(strides) {
    fillSize();
    if (copy) {
      data_ = {beginning_, ending_};
    }
  }
  Tensor(TensorType *ptr, TensorType *end_ptr, const TensorShape &sizes,
         bool copy = false)
      : data_{}, beginning_(ptr), ending_(end_ptr), sizes_(sizes) {
    fillSize();
    fillStrides();
    if (copy) {
      data_ = {beginning_, ending_};
    }
  }

  static Tensor empty(const TensorShape &shape) {
    std::size_t num_elements{shape.size() ? 1UL : 0UL};
    for (const auto s : shape) {
      num_elements *= s;
    }

    return {std::vector<TensorType>(num_elements), shape};
  }

  static Tensor fromBlob(TensorType *ptr, const TensorShape &shape) {
    std::size_t num_elements{shape.size() ? 1UL : 0UL};
    for (const auto s : shape) {
      num_elements *= s;
    }
    auto ending = ptr + num_elements * element_size();

    return Tensor(ptr, ending, shape);
  }

  Tensor copy() const { return Tensor(data_, sizes_); }

  TensorType &at(const TensorShape &indices) {
    return beginning_[indexListToSingleIndex(indices)];
  }
  const TensorType at(const TensorShape &indices) const {
    return beginning_[indexListToSingleIndex(indices)];
  }
  TensorType &operator[](const TensorShape &indices) {
    return beginning_[indexListToSingleIndex(indices)];
  }

  Tensor operator[](IndexType index) { return narrow(0, index, 1, 1, false); }

  Tensor operator[](const AdvancedTensorShape &indices) {
    auto t = this;

    for (std::size_t d{0}, nd{indices.size()}; d < nd; ++nd) {
      const auto &dim_index = indices[d];
      const auto index_size = dim_index.size();
      assert(index_size < 4);

      const auto current_dim_size = sizes_[d];

      switch (index_size) {
      case 0: {
        t = t.narrow(d, 0, current_dim_size);
        break;
      }
      case 1: {
        t = t.narrow(d, dim_index[0][0], 1, 1);
        break;
      }
      case 2: {
        const auto start = dim_index[0];

        break;
      }
      case 3: {
        break;
      }
      }
    }
    return t;
  }

  Tensor select(IndexType dim, IndexType val) const {
    auto new_sizes = sizes_;
    new_sizes.erase(new_sizes.begin() + dim);
    auto new_strides = strides_;
    auto offset = new_strides[dim] * val;
    new_strides.erase(new_strides.begin() + dim);

    return Tensor(beginning_ + offset, ending_, new_sizes, new_strides);
  }

  Tensor narrow(IndexType dim, IndexType start, IndexType length,
                IndexType stride = 1, bool keepdim = true) const {
    auto new_sizes = sizes_;
    auto dim_size = new_sizes[dim] =
        std::ceil(static_cast<double>(length) / stride);

    auto new_begin = beginning_;
    new_begin += start * strides_[dim];

    auto new_end = new_begin + length * strides_[dim];

    auto new_strides = strides_;
    new_strides[dim] *= stride;

    if (!keepdim && dim_size == 1 && new_sizes.size() > 1) {
      new_sizes.erase(new_sizes.begin() + dim);
      new_strides.erase(new_strides.begin() + dim);
    }

    return Tensor(new_begin, new_end, new_sizes, new_strides);
  }

  Tensor flatten() const {
    std::vector<std::vector<IndexType>> ranges;
    for (const auto d : sizes_) {
      ranges.emplace_back(d);
      std::iota(ranges.back().begin(), ranges.back().end(), 0);
    }
    ranges = cartesianProduct(ranges);

    auto new_size = std::max(1UL, num_elements_);
    std::vector<TensorType> new_data(new_size);
    TensorShape new_sizes{static_cast<IndexType>(new_size)};
    std::size_t new_data_index{0};
    for (const auto row : ranges) {
      new_data[new_data_index++] = at(row);
    }

    return Tensor(new_data, new_sizes);
  }

  Tensor permute(const TensorShape &new_indices) {
    assert(new_indices.size() == sizes_.size());

    auto new_sizes = sizes_;
    auto new_strides = strides_;

    for (std::size_t i{0}, l{new_indices.size()}; i < l; ++i) {
      const auto new_index = new_indices[i];
      new_sizes[i] = sizes_[new_index];
      new_strides[i] = strides_[new_index];
    }

    return Tensor(beginning_, ending_, new_sizes, new_strides);
  }

  Tensor t_() {
    auto new_sizes = sizes_;
    std::reverse(new_sizes.begin(), new_sizes.end());
    auto new_strides = strides_;
    std::reverse(new_strides.begin(), new_strides.end());
    return Tensor(beginning_, ending_, new_sizes, new_strides);
  }
  Tensor t() const {
    auto new_sizes = sizes_;
    std::reverse(new_sizes.begin(), new_sizes.end());
    auto new_strides = strides_;
    std::reverse(new_strides.begin(), new_strides.end());
    return Tensor(beginning_, ending_, new_sizes, new_strides, true);
  }

  Tensor reshape(const TensorShape &shape) {
    return Tensor(beginning_, ending_, shape);
  }

  Tensor squeeze() {
    auto new_sizes = sizes_;
    new_sizes.erase(std::remove(new_sizes.begin(), new_sizes.end(), 1),
                    new_sizes.end());
    return reshape(new_sizes);
  }
  Tensor squeeze(std::size_t dim) {
    assert(dim < sizes_.size() && sizes_[dim] == 1);
    auto new_sizes = sizes_;
    new_sizes.erase(new_sizes.begin() + dim);
    return reshape(new_sizes);
  }
  Tensor unsqueeze(std::size_t dim) {
    assert(dim <= sizes_.size());
    auto new_sizes = sizes_;
    new_sizes.insert(new_sizes.begin() + dim, 1);
    return reshape(new_sizes);
  }

  TensorType *data() {
    checkStrides();
    return beginning_;
  }
  const TensorType *data() const {
    checkStrides();
    return beginning_;
  }
  static constexpr std::size_t element_size() { return sizeof(TensorType); }
  std::size_t numel() const { return num_elements_; }
  std::size_t dim() const { return sizes_.size(); }
  const TensorShape sizes() const { return sizes_; }
  const TensorShape strides() const { return strides_; }

  IndexType size(const std::size_t dim) const {
    assert(dim < sizes_.size());
    return sizes_[dim];
  }

  TensorType item() const {
    assert(numel() == 1);
    return *beginning_;
  }

  template <typename FriendTensorType>
  friend std::ostream &operator<<(std::ostream &stream,
                                  const Tensor<FriendTensorType> &tensor);

private:
  std::vector<TensorType> data_{};
  TensorType *beginning_{nullptr};
  TensorType *ending_{nullptr};

  TensorShape sizes_{};
  TensorShape strides_{};
  std::size_t num_elements_{0};

  void fillStrides() {
    strides_ = {1};
    IndexType stride{1};
    for (auto it{sizes_.rbegin()}, it_end{sizes_.rend() - 1}; it != it_end;
         ++it) {
      stride *= *it;
      strides_.emplace(strides_.begin(), stride);
    }
  }

  void fillSize() {
    num_elements_ = sizes_.size() ? 1UL : 0UL;
    for (const auto s : sizes_) {
      num_elements_ *= s;
    }
  }

  std::vector<std::vector<IndexType>>
  cartesianProduct(const std::vector<std::vector<IndexType>> &v) const {
    std::vector<std::vector<IndexType>> s{{}};
    for (const auto &u : v) {
      std::vector<std::vector<IndexType>> r;
      for (const auto &x : s) {
        for (const auto y : u) {
          r.push_back(x);
          r.back().push_back(y);
        }
      }
      s = std::move(r);
    }
    return s;
  }

  void checkStrides() {
    if (strides_.size() > 0 && strides_[strides_.size() - 1] != 1) {
      std::cout << "Final stride != 1. Data is not contiguous. Please flatten"
                << std::endl;
      assert(false);
    }
  }

  std::size_t indexListToSingleIndex(const TensorShape &indices) const {
    assert(indices.size() == sizes_.size());

    IndexType tensor_index{0};
    for (std::size_t shape_index{0}, shape_length{sizes_.size()};
         shape_index < shape_length; ++shape_index) {
      assert(indices[shape_index] < sizes_[shape_index]);
      tensor_index += indices[shape_index] * strides_[shape_index];
    }

    return tensor_index;
  }
};

template <typename TensorType>
std::ostream &operator<<(std::ostream &os, const Tensor<TensorType> &tensor) {
  os << "Data: [ ";
  auto tc = tensor.flatten();
  auto data = tc.data();
  for (std::size_t e{0}; e < tensor.numel(); ++e) {
    os << data[e] << " ";
  }
  os << "]" << std::endl;
  os << "Sizes: [ ";
  for (const auto val : tensor.sizes()) {
    os << val << " ";
  }
  os << "]" << std::endl;
  os << "Strides: [ ";
  for (const auto val : tensor.strides()) {
    os << val << " ";
  }
  os << "]" << std::endl;
  return os;
}
