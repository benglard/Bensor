#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

template <typename TensorType> class Tensor final {
  using IndexType = std::int64_t;
  using TensorShape = std::vector<IndexType>;

public:
  Tensor(const std::vector<TensorType> &initial_data, const TensorShape &shape)
      : data_(initial_data), sizes_(shape) {
    fillSize();
    fillStrides();
    beginning_ = data_.data();
    ending_ = beginning_ + data_.size() * sizeof(TensorType);
  }
  Tensor(const std::vector<TensorType> &initial_data)
      : data_(initial_data),
        sizes_({static_cast<IndexType>(initial_data.size())}) {
    fillSize();
    fillStrides();
    beginning_ = data_.data();
    ending_ = beginning_ + data_.size() * sizeof(TensorType);
  }
  Tensor(TensorType *ptr, TensorType *end_ptr, const TensorShape &sizes,
         const TensorShape &strides)
      : data_{}, beginning_(ptr), ending_(end_ptr), sizes_(sizes),
        strides_(strides) {
    fillSize();
  }

  static Tensor empty(const TensorShape &shape) {
    std::size_t num_elements{shape.size() ? 1UL : 0UL};
    for (const auto s : shape) {
      num_elements *= s;
    }

    return {std::vector<TensorType>(num_elements), shape};
  }

  Tensor copy() const { return Tensor(data_, sizes_); }

  TensorType &at(const TensorShape &indices) {
    assert(indices.size() == sizes_.size());

    IndexType tensor_index{0};
    for (std::size_t shape_index{0}, shape_length{sizes_.size()};
         shape_index < shape_length; ++shape_index) {
      assert(indices[shape_index] < sizes_[shape_index]);
      tensor_index += indices[shape_index] * strides_[shape_index];
    }

    return beginning_[tensor_index];
  }
  const TensorType at(const TensorShape &indices) const {
    assert(indices.size() == sizes_.size());

    IndexType tensor_index{0};
    for (std::size_t shape_index{0}, shape_length{sizes_.size()};
         shape_index < shape_length; ++shape_index) {
      assert(indices[shape_index] < sizes_[shape_index]);
      tensor_index += indices[shape_index] * strides_[shape_index];
    }

    return beginning_[tensor_index];
  }

  Tensor reshape(const TensorShape &shape) {
    assert(shape.size() == sizes_.size());
    sizes_ = shape;
  }

  Tensor select(IndexType dim, IndexType val) const {
    auto new_sizes = sizes_;
    new_sizes.erase(new_sizes.begin() + dim);
    auto new_strides = strides_;
    auto offset = new_strides[dim] * val;
    new_strides.erase(new_strides.begin() + dim);

    return Tensor(beginning_ + offset, ending_, new_sizes, new_strides);
  }

  Tensor narrow(IndexType dim, IndexType start, IndexType length) const {
    auto new_sizes = sizes_;
    new_sizes[dim] = length;

    auto new_begin = beginning_;
    new_begin += start * strides_[dim];

    auto new_end = new_begin + length * strides_[dim];

    return Tensor(new_begin, new_end, new_sizes, strides_);
  }

  Tensor flatten() const {
    std::vector<std::vector<IndexType>> ranges;
    for (const auto d : sizes_) {
      ranges.emplace_back(d);
      std::iota(ranges.back().begin(), ranges.back().end(), 0);
    }
    ranges = cartesianProduct(ranges);

    std::vector<TensorType> new_data(num_elements_);
    TensorShape new_sizes{static_cast<IndexType>(num_elements_)};
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

  TensorType *data() {
    checkStrides();
    return beginning_;
  }
  const TensorType *data() const {
    checkStrides();
    return beginning_;
  }
  std::size_t num_elements() const { return num_elements_; }
  const TensorShape sizes() const { return sizes_; }
  const TensorShape strides() const { return strides_; }

  IndexType size(const std::size_t dim) const {
    assert(dim < sizes_.size());
    return sizes_[dim];
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
};

template <typename TensorType>
std::ostream &operator<<(std::ostream &os, const Tensor<TensorType> &tensor) {
  os << "Data: [ ";
  auto tc = tensor.flatten();
  auto data = tc.data();
  for (std::size_t e{0}; e < tensor.num_elements(); ++e) {
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
