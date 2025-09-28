#pragma once

#include <cstdint>

namespace erl::geometry {

    /**
     * Find the index of a voxel in a tree given its morton code.
     * @tparam IndexType index type of the voxel, usually int32_t or int64_t.
     * @tparam MortonType type of the morton code, usually uint32_t or uint64_t.
     * @tparam Dim space dimension, 2 for quadtree, 3 for octree.
     * @param code the morton code of the voxel to find.
     * @param level the level to start the search, usually tree_depth - 1, where level=tree_depth is
     * the root.
     * @param children the buffer that stores the child voxel indices of each voxel.
     * @return the index of the found voxel, or -1 if children is nullptr or the voxel is not found.
     */
    template<typename IndexType, typename MortonType, int Dim>
    IndexType
    FindVoxelIndex(const MortonType code, int level, const IndexType *children) {
        if (children == nullptr) { return -1; }

        uint64_t shift = level * Dim;
        uint64_t mask = ((1 << Dim) - 1) << shift;
        IndexType index = 0;

        while (level >= 0) {
            if (const auto child_index = static_cast<int>((code & mask) >> shift) + (index << Dim);
                children[child_index] >= 0) {
                index = children[child_index];
            } else {
                return index;
            }
            --level;
            shift -= Dim;
            mask >>= Dim;
        }
        return index;
    }

    template<typename IndexType, typename MortonType, int Dim>
    void
    FindVoxelIndices(
        const MortonType *codes,
        std::size_t num_codes,
        int level,
        const IndexType *children,
        IndexType *voxel_indices,
        bool parallel) {

#pragma omp parallel if (parallel) default(none) \
    shared(num_codes, codes, level, children, voxel_indices)
        for (std::size_t i = 0; i < num_codes; ++i) {
            voxel_indices[i] =
                FindVoxelIndex<IndexType, MortonType, Dim>(codes[i], level, children);
        }
    }

}  // namespace erl::geometry
