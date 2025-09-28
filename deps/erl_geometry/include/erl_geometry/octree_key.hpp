#pragma once

#include "libmorton/morton.h"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <vector>

namespace erl::geometry {

    /**
     * OctreeKey is a simple class that represents a key for an octree node. It is a 3D vector of
     * uint32_t. Each element counts the number of cells from the origin as a discrete address of a
     * voxel.
     */
    class OctreeKey {
    public:
        using KeyType = uint32_t;

    private:
        KeyType m_k_[3] = {0, 0, 0};

    public:
        // Hash function for OctreeKey when used with absl containers.
        template<typename H>
        [[maybe_unused]] friend H
        AbslHashValue(H h, const OctreeKey& key) {
            return H::combine(std::move(h), key.m_k_[0], key.m_k_[1], key.m_k_[2]);
        }

        // Hash function for OctreeKey when used with std hash containers.
        struct [[maybe_unused]] KeyHash {
            [[nodiscard]] std::size_t
            operator()(const OctreeKey& key) const {
                return (static_cast<std::size_t>(key.m_k_[0]) << 32) |
                       (static_cast<std::size_t>(key.m_k_[1]) << 16) |
                       static_cast<std::size_t>(key.m_k_[2]);
            }
        };

        OctreeKey() = default;

        OctreeKey(const KeyType a, const KeyType b, const KeyType c)
            : m_k_{a, b, c} {}

        explicit OctreeKey(const uint64_t& morton_code) {
            uint_fast32_t x, y, z;
            libmorton::morton3D_64_decode(morton_code, x, y, z);
            m_k_[0] = static_cast<KeyType>(x);
            m_k_[1] = static_cast<KeyType>(y);
            m_k_[2] = static_cast<KeyType>(z);
        }

        OctreeKey(const OctreeKey& other)
            : m_k_{other.m_k_[0], other.m_k_[1], other.m_k_[2]} {}

        OctreeKey&
        operator=(const OctreeKey& other) {
            if (this == &other) { return *this; }
            m_k_[0] = other.m_k_[0];
            m_k_[1] = other.m_k_[1];
            m_k_[2] = other.m_k_[2];
            return *this;
        }

        OctreeKey(OctreeKey&& other) noexcept
            : m_k_{
                  std::exchange(other.m_k_[0], 0),
                  std::exchange(other.m_k_[1], 0),
                  std::exchange(other.m_k_[2], 0)} {}

        OctreeKey&
        operator=(OctreeKey&& other) noexcept {
            if (this == &other) { return *this; }
            m_k_[0] = std::exchange(other.m_k_[0], 0);
            m_k_[1] = std::exchange(other.m_k_[1], 0);
            m_k_[2] = std::exchange(other.m_k_[2], 0);
            return *this;
        }

        [[nodiscard]] bool
        operator==(const OctreeKey& other) const {
            return !std::memcmp(m_k_, other.m_k_, sizeof(m_k_));
        }

        [[nodiscard]] bool
        operator!=(const OctreeKey& other) const {
            return std::memcmp(m_k_, other.m_k_, sizeof(m_k_));
        }

        KeyType&
        operator[](const uint32_t i) {
            return m_k_[i];
        }

        [[nodiscard]] const KeyType&
        operator[](const uint32_t i) const {
            return m_k_[i];
        }

        [[nodiscard]] bool
        operator<(const OctreeKey& other) const {
            return std::memcmp(m_k_, other.m_k_, sizeof(m_k_)) < 0;
        }

        [[nodiscard]] bool
        operator<=(const OctreeKey& other) const {
            return std::memcmp(m_k_, other.m_k_, sizeof(m_k_)) <= 0;
        }

        [[nodiscard]] bool
        operator>(const OctreeKey& other) const {
            return std::memcmp(m_k_, other.m_k_, sizeof(m_k_)) > 0;
        }

        [[nodiscard]] bool
        operator>=(const OctreeKey& other) const {
            return std::memcmp(m_k_, other.m_k_, sizeof(m_k_)) >= 0;
        }

        [[nodiscard]] explicit
        operator std::string() const {
            return "[" +                             //
                   std::to_string(m_k_[0]) + ", " +  //
                   std::to_string(m_k_[1]) + ", " +  //
                   std::to_string(m_k_[2])           //
                   + "]";
        }

        [[nodiscard]] uint64_t
        ToMortonCode() const {
            return libmorton::morton3D_64_encode(m_k_[0], m_k_[1], m_k_[2]);
        }

        /**
         * Compute the key of a child node from the key of its parent node and the index of the
         * child node.
         * @param pos index of child node (0..7)
         * @param center_offset_key
         * @param parent_key
         * @param child_key
         */
        static void
        ComputeChildKey(
            const uint32_t pos,
            const KeyType center_offset_key,
            const OctreeKey& parent_key,
            OctreeKey& child_key) {
            if (center_offset_key == 0) {
                child_key.m_k_[0] = parent_key.m_k_[0] + ((pos & 1) ? 0 : -1);
                child_key.m_k_[1] = parent_key.m_k_[1] + ((pos & 2) ? 0 : -1);
                child_key.m_k_[2] = parent_key.m_k_[2] + ((pos & 4) ? 0 : -1);
            } else {
                child_key.m_k_[0] =
                    parent_key.m_k_[0] + ((pos & 1) ? center_offset_key : -center_offset_key);
                child_key.m_k_[1] =
                    parent_key.m_k_[1] + ((pos & 2) ? center_offset_key : -center_offset_key);
                child_key.m_k_[2] =
                    parent_key.m_k_[2] + ((pos & 4) ? center_offset_key : -center_offset_key);
            }
        }

        static KeyType
        AdjustKeyToLevel(const KeyType key, const uint32_t level) {
            if (level == 0) { return key; }
            return ((key >> level) << level) + (1 << (level - 1));
        }

        [[nodiscard]] OctreeKey
        AdjustToLevel(const uint32_t level) const {
            if (level == 0) { return *this; }
            return {
                AdjustKeyToLevel(m_k_[0], level),
                AdjustKeyToLevel(m_k_[1], level),
                AdjustKeyToLevel(m_k_[2], level)};
        }

        static void
        ComputeVertexKey(
            const uint32_t vertex_index,
            const uint32_t level,
            const OctreeKey& voxel_key,
            OctreeKey& vertex_key) {

            const KeyType voxel_size = 1 << level;

            vertex_key.m_k_[0] = (voxel_key.m_k_[0] >> level) << level;
            vertex_key.m_k_[0] += ((vertex_index & 0b001) ? voxel_size : 0);

            vertex_key.m_k_[1] = (voxel_key.m_k_[1] >> level) << level;
            vertex_key.m_k_[1] += ((vertex_index & 0b010) ? voxel_size : 0);

            vertex_key.m_k_[2] = (voxel_key.m_k_[2] >> level) << level;
            vertex_key.m_k_[2] += ((vertex_index & 0b100) ? voxel_size : 0);
        }

        /**
         * Compute child index (0..7) from a key at a given level.
         * @param key
         * @param level level=0 means the leaf level
         * @return
         */
        static int
        ComputeChildIndex(const OctreeKey& key, const uint32_t level) {
            int pos = 0;
            const KeyType mask = 1 << level;
            if (key.m_k_[0] & mask) { pos += 1; }
            if (key.m_k_[1] & mask) { pos += 2; }
            if (key.m_k_[2] & mask) { pos += 4; }
            return pos;
        }

        static bool
        KeyInAabb(
            const OctreeKey& key,
            const KeyType center_offset_key,
            const OctreeKey& aabb_min_key,
            const OctreeKey& aabb_max_key) {
            return (aabb_min_key.m_k_[0] <= (key.m_k_[0] + center_offset_key)) &&  //
                   (aabb_min_key.m_k_[1] <= (key.m_k_[1] + center_offset_key)) &&  //
                   (aabb_min_key.m_k_[2] <= (key.m_k_[2] + center_offset_key)) &&  //
                   (aabb_max_key.m_k_[0] >= (key.m_k_[0] - center_offset_key)) &&  //
                   (aabb_max_key.m_k_[1] >= (key.m_k_[1] - center_offset_key)) &&  //
                   (aabb_max_key.m_k_[2] >= (key.m_k_[2] - center_offset_key));
        }
    };

    /**
     * Data structure to efficiently compute the nodes to update from a scan insertion using a hash
     * set.
     */
    using OctreeKeySet = absl::flat_hash_set<OctreeKey>;
    using OctreeKeyVectorMap = absl::flat_hash_map<OctreeKey, std::vector<long>>;
    using OctreeKeyVector = std::vector<OctreeKey>;

    /**
     * Data structure to efficiently track changed nodes.
     */
    using OctreeKeyBoolMap = absl::flat_hash_map<OctreeKey, bool>;
    using OctreeKeyLongMap = absl::flat_hash_map<OctreeKey, long>;
    using OctreeKeyRay = std::vector<OctreeKey>;
}  // namespace erl::geometry
