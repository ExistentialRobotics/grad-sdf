#include "erl_geometry/abstract_quadtree.hpp"

#include "erl_common/logging.hpp"
#include "erl_common/serialization.hpp"

namespace erl::geometry {

    template<typename Dtype>
    AbstractQuadtree<Dtype>::AbstractQuadtree(std::shared_ptr<NdTreeSetting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_setting_->tree_depth <= 16, "tree_depth > 16 is not supported.");
    }

    template<typename Dtype>
    std::string
    AbstractQuadtree<Dtype>::GetTreeType() const {
        return demangle(typeid(*this).name());
    }

    template<typename Dtype>
    std::shared_ptr<AbstractQuadtree<Dtype>>
    AbstractQuadtree<Dtype>::CreateTree(
        const std::string &tree_id,
        const std::shared_ptr<NdTreeSetting> &setting) {
        return Factory::GetInstance().Create(tree_id, setting);
    }

    template<typename Dtype>
    bool
    AbstractQuadtree<Dtype>::ReadSetting(std::istream &s) const {
        std::streamsize len;
        s.read(reinterpret_cast<char *>(&len), sizeof(std::size_t));
        std::string yaml_str(len, '\0');
        s.read(yaml_str.data(), len);
        return m_setting_->FromYamlString(yaml_str);
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::WriteSetting(std::ostream &s) const {
        const std::string yaml_str = m_setting_->AsYamlString();
        const auto len = static_cast<std::streamsize>(yaml_str.size());
        s.write(reinterpret_cast<const char *>(&len), sizeof(std::size_t));
        s.write(yaml_str.data(), len);
        s << '\n';  // add newline to separate from data
    }

    template<typename Dtype>
    bool
    AbstractQuadtree<Dtype>::operator!=(const AbstractQuadtree &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    uint32_t
    AbstractQuadtree<Dtype>::GetTreeDepth() const {
        return m_setting_->tree_depth;
    }

    template<typename Dtype>
    Dtype
    AbstractQuadtree<Dtype>::GetResolution() const {
        return static_cast<Dtype>(m_setting_->resolution);
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricMin() {
        Vector2 min;
        GetMetricMin(min.x(), min.y());
        return min;
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricMin() const {
        Vector2 min;
        GetMetricMin(min.x(), min.y());
        return min;
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMin(Vector2 &min) {
        GetMetricMin(min.x(), min.y());
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMin(Vector2 &min) const {
        GetMetricMin(min.x(), min.y());
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricMax() {
        Vector2 max;
        GetMetricMax(max.x(), max.y());
        return max;
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricMax() const {
        Vector2 max;
        GetMetricMax(max.x(), max.y());
        return max;
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMax(Vector2 &max) {
        GetMetricMax(max.x(), max.y());
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMax(Vector2 &max) const {
        GetMetricMax(max.x(), max.y());
    }

    template<typename Dtype>
    Aabb<Dtype, 2>
    AbstractQuadtree<Dtype>::GetMetricAabb() {
        Vector2 min, max;
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    Aabb<Dtype, 2>
    AbstractQuadtree<Dtype>::GetMetricAabb() const {
        Vector2 min, max;
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    std::pair<typename AbstractQuadtree<Dtype>::Vector2, typename AbstractQuadtree<Dtype>::Vector2>
    AbstractQuadtree<Dtype>::GetMetricMinMax() {
        Vector2 min, max;
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    std::pair<typename AbstractQuadtree<Dtype>::Vector2, typename AbstractQuadtree<Dtype>::Vector2>
    AbstractQuadtree<Dtype>::GetMetricMinMax() const {
        Vector2 min, max;
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMinMax(Vector2 &min, Vector2 &max) {
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricMinMax(Vector2 &min, Vector2 &max) const {
        GetMetricMinMax(min.x(), min.y(), max.x(), max.y());
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricSize() {
        Vector2 size;
        GetMetricSize(size.x(), size.y());
        return size;
    }

    template<typename Dtype>
    typename AbstractQuadtree<Dtype>::Vector2
    AbstractQuadtree<Dtype>::GetMetricSize() const {
        Vector2 size;
        GetMetricSize(size.x(), size.y());
        return size;
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricSize(Vector2 &size) {
        GetMetricSize(size.x(), size.y());
    }

    template<typename Dtype>
    void
    AbstractQuadtree<Dtype>::GetMetricSize(Vector2 &size) const {
        GetMetricSize(size.x(), size.y());
    }

    template<typename Dtype>
    bool
    AbstractQuadtree<Dtype>::Write(std::ostream &s) const {
        static const common::TokenWriteFunctionPairs<AbstractQuadtree> token_function_pairs = {
            {
                "setting",
                [](const AbstractQuadtree *self, std::ostream &stream) {
                    self->WriteSetting(stream);
                    return stream.good();
                },
            },
            {
                "data",
                [](const AbstractQuadtree *self, std::ostream &stream) {
                    const std::size_t size = self->GetSize();
                    stream << size << '\n';
                    if (size > 0) { return self->WriteData(stream).good(); }
                    return stream.good();
                },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    AbstractQuadtree<Dtype>::Read(std::istream &s) {
        static const common::TokenReadFunctionPairs<AbstractQuadtree> token_function_pairs = {
            {
                "setting",
                [](AbstractQuadtree *self, std::istream &stream) {
                    self->Clear();  // clear the tree before reading the setting
                    if (!self->ReadSetting(stream)) { return false; }
                    self->ApplySetting();
                    return stream.good();
                },
            },
            {
                "data",
                [](AbstractQuadtree *self, std::istream &stream) {
                    std::size_t size;
                    stream >> size;
                    common::SkipLine(stream);
                    if (size > 0) { return self->ReadData(stream).good(); }
                    ERL_DEBUG("Load {} nodes", size);
                    return stream.good();
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template class AbstractQuadtree<double>;
    template class AbstractQuadtree<float>;
}  // namespace erl::geometry
