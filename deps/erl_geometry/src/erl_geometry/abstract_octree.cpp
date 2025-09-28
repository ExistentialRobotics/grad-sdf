#include "erl_geometry/abstract_octree.hpp"

#include "erl_common/logging.hpp"
#include "erl_common/serialization.hpp"

namespace erl::geometry {

    template<typename Dtype>
    AbstractOctree<Dtype>::AbstractOctree(std::shared_ptr<NdTreeSetting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_setting_->tree_depth <= 16, "tree_depth > 16 is not supported.");
    }

    template<typename Dtype>
    std::string
    AbstractOctree<Dtype>::GetTreeType() const {
        return demangle(typeid(*this).name());
    }

    template<typename Dtype>
    std::shared_ptr<AbstractOctree<Dtype>>
    AbstractOctree<Dtype>::CreateTree(
        const std::string &tree_id,
        const std::shared_ptr<NdTreeSetting> &setting) {
        return Factory::GetInstance().Create(tree_id, setting);
    }

    template<typename Dtype>
    bool
    AbstractOctree<Dtype>::ReadSetting(std::istream &s) const {
        std::streamsize len;
        s.read(reinterpret_cast<char *>(&len), sizeof(std::size_t));
        std::string yaml_str(len, '\0');
        s.read(yaml_str.data(), len);
        return m_setting_->FromYamlString(yaml_str);
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::WriteSetting(std::ostream &s) const {
        const std::string yaml_str = m_setting_->AsYamlString();
        const auto len = static_cast<std::streamsize>(yaml_str.size());
        s.write(reinterpret_cast<const char *>(&len), sizeof(std::size_t));
        s.write(yaml_str.data(), len);
        s << '\n';  // add newline to separate from data
    }

    template<typename Dtype>
    bool
    AbstractOctree<Dtype>::operator!=(const AbstractOctree &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    uint32_t
    AbstractOctree<Dtype>::GetTreeDepth() const {
        return m_setting_->tree_depth;
    }

    template<typename Dtype>
    Dtype
    AbstractOctree<Dtype>::GetResolution() const {
        return static_cast<Dtype>(m_setting_->resolution);
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricMin() {
        Vector3 min;
        GetMetricMin(min.x(), min.y(), min.z());
        return min;
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricMin() const {
        Vector3 min;
        GetMetricMin(min.x(), min.y(), min.z());
        return min;
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMin(Vector3 &min) {
        GetMetricMin(min.x(), min.y(), min.z());
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMin(Vector3 &min) const {
        GetMetricMin(min.x(), min.y(), min.z());
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricMax() {
        Vector3 max;
        GetMetricMax(max.x(), max.y(), max.z());
        return max;
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricMax() const {
        Vector3 max;
        GetMetricMax(max.x(), max.y(), max.z());
        return max;
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMax(Vector3 &max) {
        GetMetricMax(max.x(), max.y(), max.z());
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMax(Vector3 &max) const {
        GetMetricMax(max.x(), max.y(), max.z());
    }

    template<typename Dtype>
    Aabb<Dtype, 3>
    AbstractOctree<Dtype>::GetMetricAabb() {
        Vector3 min, max;
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    Aabb<Dtype, 3>
    AbstractOctree<Dtype>::GetMetricAabb() const {
        Vector3 min, max;
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    std::pair<typename AbstractOctree<Dtype>::Vector3, typename AbstractOctree<Dtype>::Vector3>
    AbstractOctree<Dtype>::GetMetricMinMax() {
        Vector3 min, max;
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    std::pair<typename AbstractOctree<Dtype>::Vector3, typename AbstractOctree<Dtype>::Vector3>
    AbstractOctree<Dtype>::GetMetricMinMax() const {
        Vector3 min, max;
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
        return {std::move(min), std::move(max)};
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMinMax(Vector3 &min, Vector3 &max) {
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricMinMax(Vector3 &min, Vector3 &max) const {
        GetMetricMinMax(min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricSize() {
        Vector3 size;
        GetMetricSize(size.x(), size.y(), size.z());
        return size;
    }

    template<typename Dtype>
    typename AbstractOctree<Dtype>::Vector3
    AbstractOctree<Dtype>::GetMetricSize() const {
        Vector3 size;
        GetMetricSize(size.x(), size.y(), size.z());
        return size;
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricSize(Vector3 &size) {
        GetMetricSize(size.x(), size.y(), size.z());
    }

    template<typename Dtype>
    void
    AbstractOctree<Dtype>::GetMetricSize(Vector3 &size) const {
        GetMetricSize(size.x(), size.y(), size.z());
    }

    template<typename Dtype>
    bool
    AbstractOctree<Dtype>::Write(std::ostream &s) const {
        static const common::TokenWriteFunctionPairs<AbstractOctree> token_function_pairs = {
            {
                "setting",
                [](const AbstractOctree *self, std::ostream &stream) {
                    self->WriteSetting(stream);
                    return stream.good();
                },
            },
            {
                "data",
                [](const AbstractOctree *self, std::ostream &stream) {
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
    AbstractOctree<Dtype>::Read(std::istream &s) {
        static const common::TokenReadFunctionPairs<AbstractOctree> token_function_pairs = {
            {
                "setting",
                [](AbstractOctree *self, std::istream &stream) {
                    self->Clear();  // clear the tree before reading the setting
                    if (!self->ReadSetting(stream)) { return false; }
                    self->ApplySetting();
                    return stream.good();
                },
            },
            {
                "data",
                [](AbstractOctree *self, std::istream &stream) {
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

    template class AbstractOctree<double>;
    template class AbstractOctree<float>;
}  // namespace erl::geometry
