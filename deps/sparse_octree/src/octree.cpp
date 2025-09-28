#include "octree.h"
#include "utils.h"
#include <queue>
#include <iostream>

// #define MAX_HIT_VOXELS 10
// #define MAX_NUM_VOXELS 10000

int Octant::next_node_index_ = 0;
int Octant::feature_next_index_ = 0;

int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

int incr_x_init[8] = {0, 1, 0, 1, 0, 1, 0, 1}; // x_bit
int incr_y_init[8] = {0, 0, 1, 1, 0, 0, 1, 1}; // y_bit
int incr_z_init[8] = {0, 0, 0, 0, 1, 1, 1, 1}; // z_bit

Octree::Octree()
{
}

Octree::~Octree()
{
}

void Octree::init(int64_t grid_dim, int64_t grid_num, double voxel_size, int64_t full_depth)
{
    size_ = grid_dim;
    voxel_size_ = voxel_size;
    full_depth_ = full_depth;
    max_level_ = log2(size_);
    // root_ = std::make_shared<Octant>();
    root_ = new Octant();
    root_->set_node_index_();
    root_->side_ = size_;
    // root_->depth_ = 0;
    root_->is_leaf_ = false;

    grid_num_ = grid_num;

    children_.resize(grid_num, 8);
    features_.resize(grid_num, 8);
    voxel_.resize(grid_num, 4);
    vox_has_value.resize(grid_num, 1);

    for (int i = 0; i <grid_num_; i ++ ){
        for (int j = 0; j < 8; j ++){
            if (j < 4)
                voxel_.set(i, j, 0.0);
            if (j < 1)
                vox_has_value.set(i, j, 0);
            children_.set(i, j, -1.0);
            features_.set(i, j, -1);
        }
    }
    vox_has_value.set(root_->node_index_, 0, 1);
    auto xyz = decode(root_->code_);
    voxel_.set(root_->node_index_, 0, float(xyz[0]));
    voxel_.set(root_->node_index_, 1, float(xyz[1]));
    voxel_.set(root_->node_index_, 2, float(xyz[2]));
    voxel_.set(root_->node_index_, 3, float(root_->side_));
    for (int i = 0; i < 8; i++){
        int x = 0 + incr_x[i] * size_;
        int y = 0 + incr_y[i] * size_;
        int z = 0 + incr_z[i] * size_;
        uint64_t feature_key = encode(x, y, z);
        Octant* root_feature = new Octant();
        root_feature->type_ = FEATURE;
        root_feature->set_feature_index_();
        feature_keys.insert(std::pair < uint64_t, int > (feature_key, root_feature->feature_index_));
        features_.set(root_->node_index_, i, root_feature->feature_index_);
    }

    buildFullTreeRec(root_, 1, full_depth_);
}

void Octree::buildFullTreeRec(Octant* node, int64_t depth, int64_t full_depth){
    if (depth > full_depth){
        return;
    }
    const int edge = node->side_ >> 1;
    auto xyz = decode(node->code_);
    for (int i = 0; i < 8; i++){
        auto child = new Octant();
        node->child(i) = child;
        child->set_node_index_();
        child->side_ = edge;
        child->is_leaf_ = false;
        child->type_ = NONLEAF;
        int x_child = xyz[0] + incr_x_init[i] * edge;
        int y_child = xyz[1] + incr_y_init[i] * edge;
        int z_child = xyz[2] + incr_z_init[i] * edge;
        child->code_ = encode(x_child, y_child, z_child);
        voxel_.set(child->node_index_, 0, x_child);
        voxel_.set(child->node_index_, 1, y_child);
        voxel_.set(child->node_index_, 2, z_child);
        voxel_.set(child->node_index_, 3, float(child->side_));
        vox_has_value.set(child->node_index_, 0, 1);
        children_.set(node->node_index_, i, child->node_index_);
        for (int k = 0; k < 8; k++){
            int x_feature = x_child + incr_x[k] * edge;
            int y_feature = y_child + incr_y[k] * edge;
            int z_feature = z_child + incr_z[k] * edge;
            uint64_t feature_key = encode(x_feature, y_feature, z_feature);
            auto feature = feature_keys.find(feature_key);
            if (feature != feature_keys.end()){
                features_.set(child->node_index_, k, feature->second);
                continue;
            }
            Octant* tmp_feature = new Octant();
            tmp_feature->type_ = FEATURE;
            tmp_feature->set_feature_index_();
            feature_keys.insert(std::pair < uint64_t, int > (feature_key, tmp_feature->feature_index_));
            features_.set(child->node_index_, k, tmp_feature->feature_index_);
        }
        buildFullTreeRec(child, depth + 1, full_depth);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor ,torch::Tensor, torch::Tensor>  Octree::insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        //        return;
    }

    int frame_voxel_idx[points.size(0)][1];
    for (int i = 0; i < points.size(0); ++i)
    {
//        std::vector<int> features;
//        std::vector<int> features_id;

        int x = points[i][0];
        int y = points[i][1];
        int z = points[i][2];

        const unsigned int shift = MAX_BITS - max_level_ - 1;

        auto n = root_;

        unsigned edge = size_ / 2;

        for (int d = 1; d <= max_level_; edge /= 2, ++d)
        {
            const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0); // current point belongs to which child 0-8
            const int parentid = n->node_index_;
            auto tmp = n->child(childid);
            if (!tmp){
                uint64_t key = encode(x, y, z);
                const uint64_t code = key & MASK[d + shift];
                const bool is_leaf = (d == max_level_); // the leaf node only exist in the deepest level
                int tmp_type;
                tmp_type = is_leaf ? SURFACE : NONLEAF;
                tmp = new Octant();
                tmp->code_ = code; //position
                tmp->side_ = edge; //side length
                tmp->is_leaf_ = is_leaf;
                tmp->type_ = tmp_type;
                tmp->set_node_index_();
                vox_has_value.set(tmp->node_index_, 0, 1);
                // n->children_mask_ = n->children_mask_ | (1 << childid);
                n->child(childid) = tmp;
                children_.set(parentid, childid, tmp->node_index_);

                auto xyz = decode(tmp->code_);
                voxel_.set(tmp->node_index_, 0, xyz[0]);
                voxel_.set(tmp->node_index_, 1, xyz[1]);
                voxel_.set(tmp->node_index_, 2, xyz[2]);
                voxel_.set(tmp->node_index_, 3, float(tmp->side_));
                for (int k = 0; k < 8; k++){
                    int x = xyz[0] + incr_x[k] * edge;
                    int y = xyz[1] + incr_y[k] * edge;
                    int z = xyz[2] + incr_z[k] * edge;
                    uint64_t feature_key = encode(x, y, z);
                    auto feature = feature_keys.find(feature_key);
                    if (feature != feature_keys.end()){
                        features_.set(tmp->node_index_, k, feature->second);
                        continue;
                    }

                    // const uint64_t code = feature_key & MASK[d + shift];
                    Octant* tmp_feature = new Octant();
                    // tmp_feature->code_ = code;
                    tmp_feature->type_ = FEATURE;
                    tmp_feature->set_feature_index_();
                    feature_keys.insert(std::pair < uint64_t, int > (feature_key, tmp_feature->feature_index_));
                    features_.set(tmp->node_index_, k, tmp_feature->feature_index_);
                }
            }
            if (tmp->type_ == SURFACE){
                frame_voxel_idx[i][0] = tmp->node_index_;
            }
            n = tmp;
        }
    }
    return std::make_tuple(torch::from_blob(voxel_.ptr(), {grid_num_,4}, dtype(torch::kFloat32)).clone(),
                           torch::from_blob(children_.ptr(), {grid_num_,8}, dtype(torch::kFloat32)).clone(),
                           torch::from_blob(features_.ptr(), {grid_num_,8}, dtype(torch::kInt32)).clone(),
                           torch::from_blob(vox_has_value.ptr(), {grid_num_,1}, dtype(torch::kInt32)).clone(),
                           torch::from_blob(frame_voxel_idx, {points.size(0),1}, dtype(torch::kInt32)).clone());
}

Octant *Octree::find_octant(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return nullptr;

        n = tmp;
    }
    return n;
}

bool Octree::has_voxel(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 1>();
    if (points.size(0) != 3)
    {
        return false;
    }

    int x = int(points[0]);
    int y = int(points[1]);
    int z = int(points[2]);

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return false;

        n = tmp;
    }

    if (!n)
        return false;
    else
        return true;
}

torch::Tensor Octree::get_features(torch::Tensor pts)
{
}

torch::Tensor Octree::get_leaf_voxels()
{
    std::vector<float> voxel_coords = get_leaf_voxel_recursive(root_);

    int N = voxel_coords.size() / 3;
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 3});
    return voxels.clone();
}

std::vector<float> Octree::get_leaf_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    if (n->is_leaf_ && n->type_ == SURFACE)
    {
        auto xyz = decode(n->code_);
        return {xyz[0], xyz[1], xyz[2]};
    }

    std::vector<float> coords;
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_leaf_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

torch::Tensor Octree::get_voxels()
{
    std::vector<float> voxel_coords = get_voxel_recursive(root_);
    int N = voxel_coords.size() / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 4}, options);
    return voxels.clone();
}

std::vector<float> Octree::get_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    auto xyz = decode(n->code_);
    std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(n->side_)};
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

std::pair<int64_t, int64_t> Octree::count_recursive_internal(Octant *n)
{
    if (!n)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

int64_t Octree::count_nodes()
{
    return count_recursive(root_);
}

int64_t Octree::count_recursive(Octant *n)
{
    if (!n)
        return 0;

    int64_t sum = 1;

    for (int i = 0; i < 8; i++)
    {
        sum += count_recursive(n->child(i));
    }

    return sum;
}

int64_t Octree::count_leaf_nodes()
{
    return leaves_count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::leaves_count_recursive(Octant *n)
{
    if (!n)
        return 0;

    if (n->type_ == SURFACE)
    {
        return 1;
    }

    int64_t sum = 0;

    for (int i = 0; i < 8; i++)
    {
        sum += leaves_count_recursive(n->child(i));
    }

    return sum;
}
