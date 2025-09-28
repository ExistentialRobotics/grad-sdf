#include "erl_geometry/intersection.hpp"

#include <cmath>

namespace erl::geometry {

    template<typename Dtype>
    Dtype
    ComputeNearestDistanceFromPointToLineSegment2D(
        const Dtype x0,
        const Dtype y0,
        const Dtype x1,
        const Dtype y1,
        const Dtype x2,
        const Dtype y2) {

        const Dtype dx_20 = x2 - x0;
        const Dtype dx_21 = x2 - x1;
        const Dtype dy_20 = y2 - y0;
        const Dtype dy_21 = y2 - y1;
        const Dtype d = dx_21 * dx_21 + dy_21 * dy_21;
        const Dtype lam = (dx_20 * dx_21 + dy_20 * dy_21) / d;

        Dtype dist = (dy_21 * dx_20 - dy_20 * dx_21) / std::sqrt(d);
        dist = std::abs(dist);

        if (lam > 1.) {
            const Dtype dx_10 = x1 - x0;
            const Dtype dy_10 = y1 - y0;
            dist = std::sqrt(dx_10 * dx_10 + dy_10 * dy_10);
        } else if (lam < 0.) {
            dist = std::sqrt(dx_20 * dx_20 + dy_20 * dy_20);
        }

        return dist;
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenTwoLines2D(
        const Eigen::Vector2<Dtype> &p00,
        const Eigen::Vector2<Dtype> &p01,
        const Eigen::Vector2<Dtype> &p10,
        const Eigen::Vector2<Dtype> &p11,
        Dtype &lam1,
        Dtype &lam2,
        bool &intersected) {

        Eigen::Vector2<Dtype> v_0 = p01 - p00;
        Eigen::Vector2<Dtype> v_1 = p11 - p10;

        const Dtype tmp = v_1.x() * v_0.y() - v_1.y() * v_0.x();  // are the two lines parallel?
        intersected = std::abs(tmp) > std::numeric_limits<Dtype>::min();
        if (!intersected) {
            lam1 = std::numeric_limits<Dtype>::infinity();
            lam2 = std::numeric_limits<Dtype>::infinity();
            return;
        }
        lam1 = (v_1.x() * (p00.y() - p10.y()) - v_1.y() * (p00.x() - p10.x())) / tmp;
        lam2 = (v_0.x() * (p01.y() - p10.y()) - v_0.y() * (p01.x() - p10.x())) / tmp;
    }

    template<typename Dtype>
    void
    ComputeClosestPointsBetweenTwoLines3D(
        const Eigen::Vector3<Dtype> &p11,
        const Eigen::Vector3<Dtype> &p12,
        const Eigen::Vector3<Dtype> &p21,
        const Eigen::Vector3<Dtype> &p22,
        Dtype &lam1,
        Dtype &lam2,
        bool &valid) {

        Eigen::Vector3<Dtype> d1 = p12 - p11;
        Dtype d1_norm = d1.norm();
        if (d1_norm < std::numeric_limits<Dtype>::min()) {  // line 1 is a point
            lam1 = 0;
            lam2 = 0;
            valid = false;
            return;
        }
        Eigen::Vector3<Dtype> d2 = p22 - p21;
        Dtype d2_norm = d2.norm();
        if (d2_norm < std::numeric_limits<Dtype>::min()) {  // line 2 is a point
            lam1 = 0;
            lam2 = 0;
            valid = false;
            return;
        }

        d1 /= d1_norm;
        d2 /= d2_norm;
        Eigen::Vector3<Dtype> v = p21 - p11;
        Dtype a = v.dot(d1);
        Dtype b = d2.dot(d1);
        // Dtype c = d1.dot(d1); // c = 1
        Dtype d = v.dot(d2);
        // Dtype e = d2.dot(d2); // e = 1

        Dtype tmp = 1 - b * b;
        if (std::abs(tmp) <
            std::numeric_limits<Dtype>::min()) {  // two lines are parallel, no unique solution
            lam1 = 0;
            lam2 = -a;
            valid = false;
            return;
        }

        lam1 = (a - b * d) / tmp;
        lam2 = (a * b - d) / tmp;
        valid = true;
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenTwoLines3D(
        const Eigen::Vector3<Dtype> &p11,
        const Eigen::Vector3<Dtype> &p12,
        const Eigen::Vector3<Dtype> &p21,
        const Eigen::Vector3<Dtype> &p22,
        Dtype &lam1,
        Dtype &lam2,
        bool &intersected) {
        ComputeClosestPointsBetweenTwoLines3D(p11, p12, p21, p22, lam1, lam2, intersected);
        if (!intersected) { return; }
        Eigen::Vector3<Dtype> v1 = (1 - lam1) * p11 + lam1 * p12;
        Eigen::Vector3<Dtype> v2 = (1 - lam2) * p21 + lam2 * p22;
        intersected = (v1 - v2).squaredNorm() < std::numeric_limits<Dtype>::min();
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenRayAndLine2D(
        const Eigen::Vector2<Dtype> &p0,
        const Eigen::Vector2<Dtype> &v,
        const Eigen::Vector2<Dtype> &p1,
        const Eigen::Vector2<Dtype> &p2,
        Dtype &lam,
        Dtype &dist,
        bool &intersected) {

        Eigen::Vector2<Dtype> v_21 = p2 - p1;
        Eigen::Vector2<Dtype> v_20 = p2 - p0;

        const Dtype tmp = v_21.x() * v.y() - v_21.y() * v.x();  // tmp = (p2 - p1).cross(v)
        intersected = std::abs(tmp) > std::numeric_limits<Dtype>::min();
        if (!intersected) {
            lam = std::numeric_limits<Dtype>::infinity();
            dist = std::numeric_limits<Dtype>::infinity();
            return;
        }
        lam = (v_20.x() * v.y() - v_20.y() * v.x()) / tmp;  // (p2 - p0).cross(v) / tmp
        dist = (v_21.x() * v_20.y() - v_21.y() * v_20.x()) /
               tmp;  // dist = (p2 - p1).cross(p2 - p0) / tmp
    }

    template<typename Dtype, int Dim>
    std::enable_if_t<Dim == 2>
    ComputeIntersectionBetweenRayAndAabb(
        const Eigen::Vector2<Dtype> &p,
        const Eigen::Vector2<Dtype> &v_inv,
        const Eigen::Vector2<Dtype> &box_min,
        const Eigen::Vector2<Dtype> &box_max,
        Dtype &d1,
        Dtype &d2,
        bool &intersected,
        bool &is_inside) {
        ComputeIntersectionBetweenRayAndAabb2D<Dtype>(
            p,
            v_inv,
            box_min,
            box_max,
            d1,
            d2,
            intersected,
            is_inside);
    }

    template<typename Dtype, int Dim>
    std::enable_if_t<Dim == 3>
    ComputeIntersectionBetweenRayAndAabb(
        const Eigen::Vector3<Dtype> &p,
        const Eigen::Vector3<Dtype> &v_inv,
        const Eigen::Vector3<Dtype> &box_min,
        const Eigen::Vector3<Dtype> &box_max,
        Dtype &d1,
        Dtype &d2,
        bool &intersected,
        bool &is_inside) {
        ComputeIntersectionBetweenRayAndAabb3D<Dtype>(
            p,
            v_inv,
            box_min,
            box_max,
            d1,
            d2,
            intersected,
            is_inside);
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenRayAndAabb2D(
        const Eigen::Vector2<Dtype> &p,
        const Eigen::Vector2<Dtype> &v_inv,
        const Eigen::Vector2<Dtype> &box_min,
        const Eigen::Vector2<Dtype> &box_max,
        Dtype &d1,
        Dtype &d2,
        bool &intersected,
        bool &is_inside) {

        Dtype tx_1, tx_2, ty_1, ty_2;
        if (p[0] == box_min[0]) {
            tx_1 = 0;
        } else {
            tx_1 = (box_min[0] - p[0]) * v_inv[0];
        }
        if (p[0] == box_max[0]) {
            tx_2 = 0;
        } else {
            tx_2 = (box_max[0] - p[0]) * v_inv[0];
        }
        Dtype t_min = std::min(tx_1, tx_2);
        Dtype t_max = std::max(tx_1, tx_2);

        if (p[1] == box_min[1]) {
            ty_1 = 0;
        } else {
            ty_1 = (box_min[1] - p[1]) * v_inv[1];
        }
        if (p[1] == box_max[1]) {
            ty_2 = 0;
        } else {
            ty_2 = (box_max[1] - p[1]) * v_inv[1];
        }
        t_min = std::max(t_min, std::min(ty_1, ty_2));
        t_max = std::min(t_max, std::max(ty_1, ty_2));

        intersected = t_max >= t_min;
        d1 = std::numeric_limits<Dtype>::infinity();
        d2 = std::numeric_limits<Dtype>::infinity();
        is_inside = p[0] >= box_min[0] && p[0] <= box_max[0] &&  // check x
                    p[1] >= box_min[1] && p[1] <= box_max[1];    // check y
        if (intersected) {
            if (is_inside) {       // ray start point is inside the box
                d1 = t_max;        // forward intersection
                d2 = t_min;        // backward intersection
            } else {               // ray start point is outside the box
                if (t_min >= 0) {  // forward intersection
                    d1 = t_min;    // first intersection point
                    d2 = t_max;    // second intersection point
                } else {           // backward intersection
                    d1 = t_max;    // first intersection point
                    d2 = t_min;    // second intersection point
                }
            }
        }
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenRayAndAabb3D(
        const Eigen::Vector3<Dtype> &p,
        const Eigen::Vector3<Dtype> &r_inv,
        const Eigen::Vector3<Dtype> &box_min,
        const Eigen::Vector3<Dtype> &box_max,
        Dtype &d1,
        Dtype &d2,
        bool &intersected,
        bool &is_inside) {

        Dtype tx_1, tx_2, ty_1, ty_2, tz_1, tz_2;
        if (p[0] == box_min[0]) {
            tx_1 = 0;
        } else {
            tx_1 = (box_min[0] - p[0]) * r_inv[0];
        }
        if (p[0] == box_max[0]) {
            tx_2 = 0;
        } else {
            tx_2 = (box_max[0] - p[0]) * r_inv[0];
        }
        Dtype t_min = std::min(tx_1, tx_2);
        Dtype t_max = std::max(tx_1, tx_2);

        if (p[1] == box_min[1]) {
            ty_1 = 0;
        } else {
            ty_1 = (box_min[1] - p[1]) * r_inv[1];
        }
        if (p[1] == box_max[1]) {
            ty_2 = 0;
        } else {
            ty_2 = (box_max[1] - p[1]) * r_inv[1];
        }
        t_min = std::max(t_min, std::min(ty_1, ty_2));
        t_max = std::min(t_max, std::max(ty_1, ty_2));

        if (p[2] == box_min[2]) {
            tz_1 = 0;
        } else {
            tz_1 = (box_min[2] - p[2]) * r_inv[2];
        }
        if (p[2] == box_max[2]) {
            tz_2 = 0;
        } else {
            tz_2 = (box_max[2] - p[2]) * r_inv[2];
        }
        t_min = std::max(t_min, std::min(tz_1, tz_2));
        t_max = std::min(t_max, std::max(tz_1, tz_2));

        intersected = t_max >= t_min;
        d1 = std::numeric_limits<Dtype>::infinity();
        d2 = std::numeric_limits<Dtype>::infinity();
        is_inside = p[0] >= box_min[0] && p[0] <= box_max[0] &&  // check x
                    p[1] >= box_min[1] && p[1] <= box_max[1] &&  // check y
                    p[2] >= box_min[2] && p[2] <= box_max[2];
        if (intersected) {
            if (is_inside) {       // ray start point is inside the box
                d1 = t_max;        // forward intersection
                d2 = t_min;        // backward intersection
            } else {               // ray start point is outside the box
                if (t_min >= 0) {  // forward intersection
                    d1 = t_min;    // first intersection point
                    d2 = t_max;    // second intersection point
                } else {           // backward intersection
                    d1 = t_max;    // first intersection point
                    d2 = t_min;    // second intersection point
                }
            }
        }
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenLineAndEllipse2D(
        const Dtype x0,
        const Dtype y0,
        const Dtype x1,
        const Dtype y1,
        const Dtype a,
        const Dtype b,
        Dtype &lam1,
        Dtype &lam2,
        bool &intersected) {

        // ellipse equation: (x - cx)^2 / a^2 + (y - cy)^2 / b^2 = 1
        // line equation: x = x0 + lam * (x1 - x0), y = y0 + lam * (y1 - y0)
        // substitute line equation into ellipse equation and solve for lam

        const Dtype a_sq = a * a;
        const Dtype b_sq = b * b;
        const Dtype x_diff = x0 - x1;
        const Dtype y_diff = y0 - y1;
        const Dtype x_diff_sq = x_diff * x_diff;
        const Dtype y_diff_sq = y_diff * y_diff;
        const Dtype cross_term = x0 * y1 - x1 * y0;
        const Dtype cross_term_sq = cross_term * cross_term;
        Dtype tmp0 = a_sq * y_diff_sq + b_sq * x_diff_sq;

        if (const Dtype tmp1 = tmp0 - cross_term_sq; tmp1 < 0) {  // no intersection
            intersected = false;
        } else {
            const Dtype tmp2 = a_sq * y0 * y_diff + b_sq * x0 * x_diff;
            const Dtype tmp3 = std::sqrt(tmp1) * a * b;
            tmp0 = 1.0 / tmp0;
            lam1 = (-tmp3 + tmp2) * tmp0;
            lam2 = (tmp3 + tmp2) * tmp0;
            intersected = true;
        }

        /*
         * The following code is equivalent to the above code. Although its mathematical form is
        more compact, but uses 13 multiplications and 3 divisions.
         * The above code uses 17 multiplications and 1 division. 1 division cost is about 6
        multiplications.

        a = 1.0 / a;
        b = 1.0 / b;

        x0 *= a;
        y0 *= b;
        x1 *= a;
        y1 *= b;
        const Dtype x_diff = x0 - x1;
        const Dtype y_diff = y0 - y1;
        const Dtype x_diff_sq = x_diff * x_diff;
        const Dtype y_diff_sq = y_diff * y_diff;
        const Dtype cross_term = x0 * y1 - x1 * y0;
        const Dtype cross_term_sq = cross_term * cross_term;
        Dtype tmp0 = x_diff_sq + y_diff_sq;
        if (const Dtype tmp1 = tmp0 - cross_term_sq; tmp1 < 0) {
            intersected = false;
        } else {
            const Dtype tmp2 = x0 * x_diff + y0 * y_diff;
            const Dtype tmp3 = std::sqrt(tmp1);
            tmp0 = 1.0 / tmp0;
            lam1 = (-tmp3 + tmp2) * tmp0;
            lam2 = (tmp3 + tmp2) * tmp0;
            intersected = true;
        }
        */
    }

    template<typename Dtype>
    void
    ComputeIntersectionBetweenLineAndEllipsoid3D(
        const Dtype x0,
        const Dtype y0,
        const Dtype z0,
        const Dtype x1,
        const Dtype y1,
        const Dtype z1,
        const Dtype a,
        const Dtype b,
        const Dtype c,
        Dtype &lam1,
        Dtype &lam2,
        bool &intersected) {

        // ellipse equation: (x - cx)^2 / a^2 + (y - cy)^2 / b^2 + (z - cz)^2 / c^2 = 1
        // line equation: x = x0 + lam * (x1 - x0), y = y0 + lam * (y1 - y0), z = z0 + lam * (z1 -
        // z0) substitute line equation into ellipse equation and solve for lam

        const Dtype a_sq = a * a;
        const Dtype b_sq = b * b;
        const Dtype c_sq = c * c;

        const Dtype a_sq_b_sq = a_sq * b_sq;
        const Dtype a_sq_c_sq = a_sq * c_sq;
        const Dtype b_sq_c_sq = b_sq * c_sq;

        const Dtype x_diff = x0 - x1;
        const Dtype y_diff = y0 - y1;
        const Dtype z_diff = z0 - z1;
        const Dtype x_diff_sq = x_diff * x_diff;
        const Dtype y_diff_sq = y_diff * y_diff;
        const Dtype z_diff_sq = z_diff * z_diff;

        const Dtype cross_x = y0 * z1 - y1 * z0;
        const Dtype cross_y = z0 * x1 - z1 * x0;
        const Dtype cross_z = x0 * y1 - x1 * y0;
        const Dtype cross_term_sq =
            a_sq * cross_x * cross_x + b_sq * cross_y * cross_y + c_sq * cross_z * cross_z;

        Dtype tmp0 = b_sq_c_sq * x_diff_sq + a_sq_c_sq * y_diff_sq + a_sq_b_sq * z_diff_sq;

        if (const Dtype tmp1 = tmp0 - cross_term_sq; tmp1 < 0) {  // no intersection
            intersected = false;
        } else {
            const Dtype tmp2 =
                b_sq_c_sq * x0 * x_diff + a_sq_c_sq * y0 * y_diff + a_sq_b_sq * z0 * z_diff;
            const Dtype tmp3 = std::sqrt(tmp1) * a * b * c;
            tmp0 = 1.0 / tmp0;
            lam1 = (-tmp3 + tmp2) * tmp0;
            lam2 = (tmp3 + tmp2) * tmp0;
            intersected = true;
        }
    }

}  // namespace erl::geometry
