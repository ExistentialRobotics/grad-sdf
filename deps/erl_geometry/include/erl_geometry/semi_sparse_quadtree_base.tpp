#pragma once

#include "find_voxel_indices.hpp"
#include "semi_sparse_quadtree_base.hpp"

namespace erl::geometry {

    template<typename Dtype, class Node, class Setting>
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::SemiSparseQuadtreeBase(
        const std::shared_ptr<Setting> &setting)
        : Super(setting),
          m_setting_(setting) {

        m_parents_.resize(m_setting_->init_voxel_num);
        m_children_.resize(Eigen::NoChange, m_setting_->init_voxel_num);
        m_voxels_.resize(Eigen::NoChange, m_setting_->init_voxel_num);
        m_vertices_.resize(Eigen::NoChange, m_setting_->init_voxel_num);

        if (m_setting_->cache_voxel_centers) {
            m_voxel_centers_.resize(Eigen::NoChange, m_setting_->init_voxel_num);
        }

        m_key_to_vertex_map_.reserve(m_setting_->init_voxel_num << 2);
        m_vertex_keys_.reserve(m_setting_->init_voxel_num << 2);

        // create the root node
        const uint32_t &x = this->m_tree_key_offset_;
        const QuadtreeKey root_key(x, x);
        this->m_root_.reset(CreateNode(root_key, m_setting_->tree_depth).first);
    }

    template<typename Dtype, class Node, class Setting>
    std::shared_ptr<AbstractQuadtree<Dtype>>
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::Clone() const {
        std::shared_ptr<AbstractQuadtree<Dtype>> tree = Super::Clone();
        std::shared_ptr<SemiSparseQuadtreeBase> semi_sparse_tree =
            std::dynamic_pointer_cast<SemiSparseQuadtreeBase>(tree);
        semi_sparse_tree->m_children_ = m_children_;
        semi_sparse_tree->m_voxels_ = m_voxels_;
        semi_sparse_tree->m_vertices_ = m_vertices_;
        semi_sparse_tree->m_key_to_vertex_map_ = m_key_to_vertex_map_;
        semi_sparse_tree->m_vertex_keys_ = m_vertex_keys_;
        return tree;
    }

    template<typename Dtype, class Node, class Setting>
    const typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::BufferParents &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetParents() const {
        return m_parents_;
    }

    template<typename Dtype, class Node, class Setting>
    const typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::BufferChildren &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetChildren() const {
        return m_children_;
    }

    template<typename Dtype, class Node, class Setting>
    const typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::BufferVoxels &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetVoxels() const {
        return m_voxels_;
    }

    template<typename Dtype, class Node, class Setting>
    const typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::Matrix2X &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetVoxelCenters() const {
        return m_voxel_centers_;
    }

    template<typename Dtype, class Node, class Setting>
    const typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::BufferVertices &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetVertices() const {
        return m_vertices_;
    }

    template<typename Dtype, class Node, class Setting>
    std::size_t
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetVertexCount() const {
        return m_vertex_keys_.size();
    }

    template<typename Dtype, class Node, class Setting>
    const QuadtreeKeyVector &
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::GetVertexKeys() const {
        return m_vertex_keys_;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::InsertPoints(const Matrix2X &points) {
        return InsertPoints(points.data(), points.cols());
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::InsertPoints(
        const Dtype *points,
        const long num_points) {

        NodeIndices voxel_indices(num_points);

        auto p = points;
        for (long i = 0; i < num_points; ++i, p += 2) {
            if (QuadtreeKey key; !this->CoordToKeyChecked(p[0], p[1], key)) {
                ERL_WARN("Point ({}, {}) is out of range.", p[0], p[1]);
                voxel_indices[i] = -1;
            } else {
                voxel_indices[i] = InsertKey(key, 0);
            }
        }

        return voxel_indices;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::InsertKeys(
        const QuadtreeKey *keys,
        const long num_points) {

        NodeIndices voxel_indices(num_points);

        auto k = keys;
        for (long i = 0; i < num_points; ++i) { voxel_indices[i] = InsertKey(*k++, 0); }

        return voxel_indices;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndex
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::InsertKey(
        const QuadtreeKey &key,
        uint32_t max_depth) {
        Node *node = this->m_root_.get();
        NodeIndex node_index = 0;

        const uint32_t tree_depth = m_setting_->tree_depth;
        if (max_depth == 0) { max_depth = tree_depth; }
        const int min_level = tree_depth - max_depth;

        int child_level = tree_depth - 1;
        const uint64_t code = key.ToMortonCode();
        uint64_t shift = child_level << 1;
        uint64_t mask = 0b11 << shift;
        while (child_level >= min_level) {
            if (const auto child_index = static_cast<int>((code & mask) >> shift);
                node->HasChild(child_index)) {
                node = this->GetNodeChild(node, child_index);
                node_index = m_children_(child_index, node_index);
            } else {
                std::tie(node, node_index) = CreateNode(key, child_level, node, child_index);
            }
            --child_level;
            shift -= 2;
            mask >>= 2;
        }

        return node_index;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::FindVoxelIndices(
        const Matrix2X &points,
        bool parallel) const {
        return FindVoxelIndices(points.data(), points.cols(), parallel);
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::FindVoxelIndices(
        const Dtype *points,
        const long num_points,
        bool parallel) const {

        NodeIndices voxel_indices(num_points);

#pragma omp parallel if (parallel) default(none) shared(num_points, points, voxel_indices)
        for (long i = 0; i < num_points; ++i) {
            long j = i << 1;
            if (QuadtreeKey key; !this->CoordToKeyChecked(points[j], points[j + 1], key)) {
                voxel_indices[i] = -1;
            } else {
                voxel_indices[i] = FindVoxelIndex(key);
            }
        }

        return voxel_indices;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndices
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::FindVoxelIndices(
        const QuadtreeKey *keys,
        const long num_points,
        bool parallel) const {

        NodeIndices voxel_indices(num_points);

#pragma omp parallel if (parallel) default(none) shared(num_points, keys, voxel_indices)
        for (long i = 0; i < num_points; ++i) { voxel_indices[i] = FindVoxelIndex(keys[i]); }
        return voxel_indices;
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndex
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::FindVoxelIndex(const QuadtreeKey &key) const {
        return geometry::FindVoxelIndex<NodeIndex, uint64_t, 2>(
            key.ToMortonCode(),
            m_setting_->tree_depth - 1,
            m_children_.data());
    }

    template<typename Dtype, class Node, class Setting>
    typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndex
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::AllocateVoxelEntry(
        const QuadtreeKey &key,
        const QuadtreeKey::KeyType level,
        const NodeIndex parent_node_index,
        const NodeIndex child_index) {

        if (m_recycled_node_indices_.empty()) {
            const NodeIndex node_index = m_buf_head_;
            if (m_buf_head_ >= m_parents_.size()) {  // need to expand the buffers
                const long new_size = 2 * m_buf_head_ + 1;
                m_parents_.conservativeResize(new_size);
                m_children_.conservativeResize(Eigen::NoChange, new_size);
                m_voxels_.conservativeResize(Eigen::NoChange, new_size);
                m_vertices_.conservativeResize(Eigen::NoChange, new_size);
                if (m_setting_->cache_voxel_centers) {
                    m_voxel_centers_.conservativeResize(Eigen::NoChange, new_size);
                }
            }
            m_parents_[m_buf_head_] = parent_node_index;
            m_children_.col(m_buf_head_).setConstant(-1);
            m_voxels_.col(m_buf_head_) << key[0], key[1], (1 << level);
            m_vertices_.col(m_buf_head_).setConstant(-1);
            if (m_setting_->cache_voxel_centers) {
                const auto r = static_cast<Dtype>(m_setting_->resolution);
                const auto key_offset = this->m_tree_key_offset_;
                if (level == 0) {
                    m_voxel_centers_.col(m_buf_head_)
                        << (static_cast<Dtype>(key[0]) - static_cast<Dtype>(key_offset) + 0.5f) * r,
                        (static_cast<Dtype>(key[1]) - static_cast<Dtype>(key_offset) + 0.5f) * r;
                } else {
                    m_voxel_centers_.col(m_buf_head_)
                        << (static_cast<Dtype>(key[0]) - static_cast<Dtype>(key_offset)) * r,
                        (static_cast<Dtype>(key[1]) - static_cast<Dtype>(key_offset)) * r;
                }
            }
            ++m_buf_head_;

            if (parent_node_index >= 0) {
                m_children_(child_index, parent_node_index) = node_index;
            }
            return node_index;
        }
        const auto it = m_recycled_node_indices_.begin();
        const NodeIndex node_index = *it;
        m_recycled_node_indices_.erase(it);
        m_parents_[node_index] = parent_node_index;
        m_voxels_.col(node_index) << key[0], key[1], (1 << level);
        if (m_setting_->cache_voxel_centers) {
            const auto r = static_cast<Dtype>(m_setting_->resolution);
            const auto key_offset = this->m_tree_key_offset_;
            if (level == 0) {
                m_voxel_centers_.col(node_index)
                    << (static_cast<Dtype>(key[0]) - static_cast<Dtype>(key_offset) + 0.5f) * r,
                    (static_cast<Dtype>(key[1]) - static_cast<Dtype>(key_offset) + 0.5f) * r;
            } else {
                m_voxel_centers_.col(node_index)
                    << (static_cast<Dtype>(key[0]) - static_cast<Dtype>(key_offset)) * r,
                    (static_cast<Dtype>(key[1]) - static_cast<Dtype>(key_offset)) * r;
            }
        }

        // already reset in OnDeleteNodeChild
        // m_children_[node_index] = {-1, -1, -1, -1};

        // will be set in RecordVertices
        // m_vertices_[node_index] = {-1, -1, -1, -1};

        if (parent_node_index >= 0) { m_children_(child_index, parent_node_index) = node_index; }

        return node_index;
    }

    template<typename Dtype, class Node, class Setting>
    std::pair<Node *, typename SemiSparseQuadtreeBase<Dtype, Node, Setting>::NodeIndex>
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::CreateNode(
        QuadtreeKey node_key,
        uint32_t level,
        Node *parent,
        int child_index) {

        NodeIndex node_index;
        Node *node = nullptr;

        const uint32_t depth = m_setting_->tree_depth - level;
        node_key = this->AdjustKeyToDepth(node_key, depth);

        if (parent != nullptr) {
            // We are going to create the child node. But we need to make sure its siblings are
            // also created if the depth is still smaller than full_depth.
            const NodeIndex parent_node_idx = parent->GetNodeIndex();
            if (BuildFullTree(this->AdjustKeyToDepth(node_key, depth - 1), parent)) {
                node = this->GetNodeChild(parent, child_index);
                node_index = m_children_(child_index, parent_node_idx);
            } else {
                // the child is not created in BuildFullTree, create it now.
                node = this->CreateNodeChild(parent, child_index);
                node_index = AllocateVoxelEntry(node_key, level, parent_node_idx, child_index);
                RecordVertices(node_key, node_index, level);
                node->SetNodeIndex(node_index);
            }
        } else {  // root
            node_index = AllocateVoxelEntry(node_key, level, -1, -1);
            node = new Node;
            node->SetNodeIndex(node_index);
            ++this->m_tree_size_;
            RecordVertices(node_key, node_index, level);
        }

        return {node, node_index};
    }

    template<typename Dtype, class Node, class Setting>
    void
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::RecordVertices(
        const QuadtreeKey &node_key,
        const NodeIndex node_idx,
        uint32_t level) {
        QuadtreeKey vertex_key;
        for (int i = 0; i < 4; ++i) {
            QuadtreeKey::ComputeVertexKey(i, level, node_key, vertex_key);

            auto [it, inserted] =
                m_key_to_vertex_map_.try_emplace(vertex_key, m_vertex_keys_.size());
            if (inserted) { m_vertex_keys_.push_back(vertex_key); }
            m_vertices_(i, node_idx) = it->second;
        }
    }

    template<typename Dtype, class Node, class Setting>
    bool
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::BuildFullTree(
        const QuadtreeKey &node_key,
        Node *node) {

        const uint32_t depth = node->GetDepth();
        if (depth >= m_setting_->semi_sparse_depth) { return false; }
        if (node->GetNumChildren() > 0) { return false; }

        this->ExpandNode(node);
        const QuadtreeKey::KeyType child_level = m_setting_->tree_depth - depth - 1;
        const QuadtreeKey::KeyType offset = (1 << child_level) >> 1;
        const NodeIndex parent_node_idx = node->GetNodeIndex();
        for (int i = 0; i < 4; ++i) {
            QuadtreeKey child_key;
            QuadtreeKey::ComputeChildKey(i, offset, node_key, child_key);
            auto child_node_idx = AllocateVoxelEntry(child_key, child_level, parent_node_idx, i);
            this->GetNodeChild(node, i)->SetNodeIndex(child_node_idx);
            RecordVertices(child_key, child_node_idx, child_level);
        }
        return true;
    }

    template<typename Dtype, class Node, class Setting>
    void
    SemiSparseQuadtreeBase<Dtype, Node, Setting>::OnDeleteNodeChild(
        Node *node,
        Node *child,
        const QuadtreeKey &key) {
        Super::OnDeleteNodeChild(node, child, key);

        const NodeIndex node_index = node->GetNodeIndex();
        const NodeIndex child_node_index = child->GetNodeIndex();
        const int child_pos = child->GetChildIndex();
        m_children_(child_pos, node_index) = -1;
        m_recycled_node_indices_.emplace(child_node_index);

        m_parents_[child_node_index] = -1;
        m_children_.col(child_node_index).setConstant(-1);
        m_voxels_.col(child_node_index).setZero();
        m_vertices_.col(child_node_index).setConstant(-1);
    }
}  // namespace erl::geometry
