#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <typeinfo>
#include <vector>

namespace bensor {

using IndexType = std::int64_t;
using TensorShape = std::vector<IndexType>;

enum Bool : std::uint8_t { False = 0, True = 1 };

template <typename TensorType> class Tensor final {
public:
  using TensorApplyFunction = std::function<TensorType(const TensorType &)>;
  using TensorConditionalFunction = std::function<bool(const TensorType &)>;

  template <typename IteratorType> class Iterator {
    using reference =
        std::pair<IndexType, IteratorType>; // Tensor<IteratorType>;
    using pointer = IteratorType *;         // Tensor<IteratorType> *;

  public:
    Iterator(pointer ptr, IndexType index) : ptr_(ptr), index_(index) {}
    Iterator operator++() {
      auto i = *this;
      ++index_;
      return i;
    }
    reference operator*() {
      return std::make_pair(index_, ptr_->operator[](index_));
    }
    pointer operator->() { return ptr_; }
    bool operator==(const Iterator &rhs) {
      return ptr_ == rhs.ptr_ && index_ == rhs.index_;
    }
    bool operator!=(const Iterator &rhs) {
      return ptr_ != rhs.ptr_ || index_ != rhs.index_;
    }
    reference operator*() const { return ptr_->operator[](index_); }
    pointer operator->() const { return ptr_; }
    bool operator==(const Iterator &rhs) const {
      return ptr_ == rhs.ptr_ && index_ == rhs.index_;
    }
    bool operator!=(const Iterator &rhs) const {
      return ptr_ != rhs.ptr_ || index_ != rhs.index_;
    }

    IndexType index() const { return index_; }

  private:
    pointer ptr_{nullptr};
    IndexType index_{0};
  };

  using iterator = Iterator<Tensor<TensorType>>;
  using const_iterator = Iterator<const Tensor<TensorType>>;

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, size(0)); }
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, size(0)); }

  Tensor(const std::vector<TensorType> &initial_data, const TensorShape &shape)
      : data_(initial_data), sizes_(shape) {
    fillSize();
    fillStrides();
    beginning_ = data_.data();
    ending_ = beginning_ + data_.size() * element_size();
  }
  Tensor(const std::vector<TensorType> &initial_data, const TensorShape &shape,
         const TensorShape &strides)
      : data_(initial_data), sizes_(shape), strides_(strides) {
    fillSize();
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
  static Tensor empty(const TensorShape &shape, const TensorShape &strides) {
    std::size_t num_elements{shape.size() ? 1UL : 0UL};
    for (const auto s : shape) {
      num_elements *= s;
    }

    return {std::vector<TensorType>(num_elements), shape, strides};
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
  TensorType &at(const Tensor<IndexType> &indices) {
    assert(indices.sizes().size() == 1 && indices.strides().size() == 1 &&
           indices.strides()[0] == 1);
    return beginning_[indexListToSingleIndex(indices.toNaiveVector())];
  }
  const TensorType at(const Tensor<IndexType> &indices) const {
    assert(indices.sizes().size() == 1 && indices.strides().size() == 1 &&
           indices.strides()[0] == 1);
    return beginning_[indexListToSingleIndex(indices.toNaiveVector())];
  }

  Tensor operator[](IndexType index) { return narrow(0, index, 1, 1, false); }
  Tensor operator[](IndexType index) const {
    return narrow(0, index, 1, 1, false);
  }
  Tensor operator[](const std::array<IndexType, 3> &index) {
    const IndexType start = index[0];
    IndexType end = index[1];
    if (end == -1) {
      end = size(0);
    }
    const IndexType length{end - start};
    assert(length > 0);
    const IndexType stride = index[2];

    return narrow(0, start, length, stride, false);
  }

  Tensor select(IndexType dim, IndexType val, bool keepdim = false) const {
    auto new_sizes = sizes_;
    auto new_strides = strides_;
    auto offset = new_strides[dim] * val;

    if (!keepdim && new_sizes.size() > 1) {
      new_sizes.erase(new_sizes.begin() + dim);
      new_strides.erase(new_strides.begin() + dim);
    }

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
    const auto new_size = std::max(1UL, numel());
    std::vector<TensorType> new_data(new_size);
    const TensorShape new_sizes{static_cast<IndexType>(new_size)};
    std::size_t new_data_index{0};
    for (const auto &row : generateIndexList()) {
      new_data[new_data_index++] = at(row);
    }

    return Tensor(new_data, new_sizes);
  }

  Tensor contiguous() const {
    std::vector<TensorType> new_data(numel());
    std::size_t new_data_index{0};
    for (const auto &row : generateIndexList()) {
      new_data[new_data_index++] = at(row);
    }
    return Tensor(new_data, sizes_);
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

  TensorType &item() const {
    assert(numel() == 1);
    return *beginning_;
  }

  void fill_(const TensorType &item) {
    for (const auto &row : generateIndexList()) {
      at(row) = item;
    }
  }

  void apply_(const TensorApplyFunction &fn) {
    for (const auto &row : generateIndexList()) {
      at(row) = fn(at(row));
    }
  }
  Tensor apply(const TensorApplyFunction &fn) const {
    std::vector<TensorType> new_data(numel());
    std::size_t i{0};
    for (const auto &row : generateIndexList()) {
      new_data[i++] = fn(at(row));
    }
    return Tensor(new_data, sizes_, strides_);
  }
  template <typename ReturnTemplateType>
  Tensor<ReturnTemplateType>
  apply(const std::function<ReturnTemplateType(const TensorType &)> &fn) const {
    std::vector<ReturnTemplateType> new_data(numel());
    std::size_t i{0};
    for (const auto &row : generateIndexList()) {
      new_data[i++] = fn(at(row));
    }
    return Tensor<ReturnTemplateType>(new_data, sizes_, strides_);
  }
  template <typename ReturnTemplateType>
  Tensor<ReturnTemplateType> cast() const {
    return apply<ReturnTemplateType>(
        [](const TensorType &v) -> ReturnTemplateType {
          return static_cast<ReturnTemplateType>(v);
        });
  }

  Tensor<Bool> where(const TensorConditionalFunction &fn) const {
    auto output = Tensor<Bool>::empty(sizes_, strides_);
    for (const auto &row : generateIndexList()) {
      output.at(row) = static_cast<Bool>(fn(at(row)));
    }
    return output;
  }
  void masked_scatter_(const Tensor<Bool> &index,
                       const Tensor<TensorType> &values) {
    const auto vf = values.flatten();
    IndexType i{0};
    for (const auto &row : index.generateIndexList()) {
      if (index.at(row)) {
        at(row) = vf.at({i++});
      }
    }
  }
  void masked_fill_(const Tensor<Bool> &index, const TensorType &item) {
    for (const auto &row : index.generateIndexList()) {
      if (index.at(row)) {
        at(row) = item;
      }
    }
  }
  void index_put_(const Tensor<IndexType> &index,
                  const Tensor<TensorType> &values) {
    assert(index.numel() == values.numel());
    for (const auto &row : index) {
      at(row.second) = values.at(row.second);
    }
  }
  Tensor masked_select(const Tensor<Bool> &index) const {
    std::vector<TensorType> new_data;
    std::size_t i{0};
    for (const auto &row : index.generateIndexList()) {
      if (index.at(row)) {
        new_data.emplace_back(at(row));
        ++i;
      }
    }

    const TensorShape new_sizes{static_cast<IndexType>(i)};

    return Tensor(new_data, new_sizes);
  }

  std::string type() const { return typeid(TensorType).name(); }

  template <typename FriendTensorType>
  friend std::ostream &operator<<(std::ostream &stream,
                                  const Tensor<FriendTensorType> &tensor);

  std::vector<std::vector<IndexType>> generateIndexList() const {
    std::vector<std::vector<IndexType>> ranges;
    for (const auto d : sizes_) {
      ranges.emplace_back(d);
      std::iota(ranges.back().begin(), ranges.back().end(), 0);
    }
    return cartesianProduct(ranges);
  }

  std::vector<TensorType> toNaiveVector() const {
    return {beginning_, ending_};
  }

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
};

template <typename TensorType>
Tensor<TensorType> cat(const std::vector<Tensor<TensorType>> tensors,
                       std::size_t dim = 0) {
  const auto num_tensors = tensors.size();
  assert(num_tensors > 0);

  const auto &s1 = tensors[0].sizes();
  const auto num_sizes = s1.size();
  assert(dim < num_sizes);
  std::vector<IndexType> new_sizes{s1};

  for (std::size_t ti{1}; ti < num_tensors; ++ti) {
    const auto &t = tensors[ti];
    const auto &st = t.sizes();
    const auto num_other_sizes = st.size();
    assert(num_sizes == num_other_sizes);
    for (std::size_t i{0}; i < num_sizes; ++i) {
      if (i != dim) {
        assert(s1[i] == st[i]);
      } else {
        new_sizes[i] += st[i];
      }
    }
  }

  std::size_t numel = new_sizes.size() ? 1UL : 0UL;
  for (const auto s : new_sizes) {
    numel *= s;
  }
  assert(numel > 0);

  std::vector<TensorType> new_data(numel);
  std::size_t i{0};
  for (const auto &t : tensors) {
    for (const auto &row : t.generateIndexList()) {
      new_data[i++] = t.at(row);
    }
  }

  return Tensor<TensorType>(new_data, new_sizes);
}

template <typename TensorType>
std::ostream &operator<<(std::ostream &os, const Tensor<TensorType> &tensor) {
  os << "Data: [ ";
  auto tc = tensor.contiguous();
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
  os << "Type: " << tensor.type() << std::endl;
  return os;
}

} // namespace bensor
