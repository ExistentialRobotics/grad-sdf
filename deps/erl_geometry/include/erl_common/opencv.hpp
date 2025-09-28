#pragma once

#ifdef ERL_USE_OPENCV
    #include "grid_map_info.hpp"
    #include "logging.hpp"

    #include <Eigen/Dense>
    #include <opencv2/core.hpp>
    #include <opencv2/core/eigen.hpp>
    #include <opencv2/highgui.hpp>
    #include <opencv2/imgproc.hpp>

namespace erl::common {
    inline const std::vector<cv::Vec3b> kCustomColorMap = {
        // BGR            // RGB
        {154, 154, 154},  // #9a9a9a
        {0, 0, 215},      // #d700
        {255, 60, 140},   // #8c3cff
        {0, 136, 2},      // #2880
        {199, 172, 0},    // #0acc7
        {0, 255, 152},    // #98ff0
        {209, 127, 255},  // #ff7fd1
        {79, 0, 108},     // #6c04f
        {48, 165, 255},   // #ffa530
        {157, 0, 0},      // #009d
        {104, 112, 134},  // #867068
        {66, 73, 0},      // #04942
        {0, 42, 79},      // #4f2a0
        {207, 253, 0},    // #0fdcf
        {255, 183, 188},  // #bcb7ff
        {122, 180, 149},  // #95b47a
        {185, 4, 192},    // #c04b9
        {162, 102, 37},   // #2566a2
        {65, 0, 40},      // #28041
        {175, 179, 220},  // #dcb3af
        {144, 245, 254},  // #fef590
        {91, 69, 80},     // #50455b
        {0, 124, 164},    // #a47c0
        {102, 113, 255},  // #ff7166
        {110, 129, 63},   // #3f816e
        {13, 0, 130},     // #820d
        {179, 123, 163},  // #a37bb3
        {0, 78, 52},      // #344e0
        {255, 228, 155},  // #9be4ff
        {119, 0, 235},    // #eb077
        {10, 0, 45},      // #2d0a
        {255, 144, 94},   // #5e90ff
        {32, 199, 0},     // #0c720
        {170, 1, 88},     // #581aa
        {0, 30, 0},       // #01e0
        {0, 71, 154},     // #9a470
        {166, 159, 150},  // #969fa6
        {92, 66, 155},    // #9b425c
        {50, 31, 0},      // #01f32
        {0, 196, 200},    // #c8c40
        {255, 208, 255},  // #ffd0ff
        {154, 190, 0},    // #0be9a
        {255, 21, 55},    // #3715ff
        {37, 37, 45},     // #2d2525
        {255, 88, 223},   // #df58ff
        {192, 231, 190},  // #bee7c0
        {152, 69, 127},   // #7f4598
        {60, 79, 82},     // #524f3c
        {0, 102, 216},    // #d8660
        {56, 116, 100},   // #647438
        {136, 115, 193},  // #c17388
        {138, 116, 110},  // #6e748a
        {3, 157, 128},    // #809d3
        {101, 139, 190},  // #be8b65
        {57, 51, 99},     // #633339
        {218, 205, 202},  // #cacdda
        {131, 235, 108},  // #6ceb83
        {105, 64, 34},    // #224069
        {255, 127, 162},  // #a27fff
        {203, 3, 254},    // #fe3cb
        {253, 188, 118},  // #76bcfd
        {130, 195, 217},  // #d9c382
        {206, 163, 206},  // #cea3ce
        {0, 80, 109},     // #6d500
        {116, 105, 0},    // #06974
        {94, 159, 71},    // #479f5e
        {191, 198, 148},  // #94c6bf
        {0, 255, 249},    // #f9ff0
        {69, 84, 192},    // #c05445
        {60, 101, 0},     // #0653c
        {168, 80, 91},    // #5b50a8
        {100, 32, 83},    // #532064
        {255, 95, 79},    // #4f5fff
        {119, 143, 126},  // #7e8f77
        {250, 8, 185},    // #b98fa
        {195, 146, 139},  // #8b92c3
        {53, 0, 179},     // #b3035
        {126, 96, 136},   // #88607e
        {117, 0, 159},    // #9f075
        {196, 222, 255},  // #ffdec4
        {0, 8, 81},       // #5180
        {0, 8, 26},       // #1a80
        {182, 137, 76},   // #4c89b6
        {223, 223, 0},    // #0dfdf
        {250, 255, 200},  // #c8fffa
        {21, 53, 48},     // #303515
        {71, 39, 255},    // #ff2747
        {170, 151, 255},  // #ff97aa
        {26, 0, 4},       // #401a
        {177, 96, 201},   // #c960b1
        {55, 162, 195},   // #c3a237
        {58, 79, 124},    // #7c4f3a
        {119, 158, 249},  // #f99e77
        {100, 101, 86},   // #566564
        {255, 147, 209},  // #d193ff
        {105, 31, 45},    // #2d1f69
        {52, 27, 65},     // #411b34
        {152, 147, 175},  // #af9398
        {153, 158, 98},   // #629e99
        {123, 222, 189},  // #bdde7b
        {148, 94, 255},   // #ff5e94
        {35, 41, 15},     // #f2923
        {172, 190, 184},  // #b8beac
        {101, 59, 116},   // #743b65
        {13, 0, 16},      // #100d
        {189, 110, 127},  // #7f6ebd
        {59, 107, 158},   // #9e6b3b
        {0, 70, 255},     // #ff460
        {135, 0, 127},    // #7f087
        {62, 206, 255},   // #ffce3e
        {67, 59, 48},     // #303b43
        {255, 165, 254},  // #fea5ff
        {62, 2, 138},     // #8a23e
        {1, 44, 118},     // #762c1
        {150, 138, 10},   // #a8a96
        {82, 0, 5},       // #5052
        {50, 214, 142},   // #8ed632
        {115, 196, 83},   // #53c473
        {113, 89, 71},    // #475971
        {34, 2, 88},      // #58222
        {1, 34, 166},     // #a6221
        {76, 147, 144},   // #90934c
        {30, 67, 0},      // #0431e
        {209, 0, 129},    // #810d1
        {63, 38, 47},     // #2f263f
        {132, 57, 191},   // #bf3984
        {213, 255, 245},  // #f5ffd5
        {255, 211, 0},    // #0d3ff
        {248, 0, 106},    // #6a0f8
        {210, 187, 156},  // #9cbbd2
        {171, 217, 122},  // #7ad9ab
        {93, 87, 105},    // #69575d
        {5, 105, 0},      // #0695
        {156, 54, 54},    // #36369c
        {71, 131, 1},     // #18347
        {24, 30, 68},     // #441e18
        {239, 165, 7},    // #7a5ef
        {48, 129, 255},   // #ff8130
        {184, 85, 167},   // #a755b8
        {131, 90, 104},   // #685a83
        {255, 255, 115},  // #73ffff
        {2, 135, 217},    // #d9872
        {255, 211, 187},  // #bbd3ff
        {47, 55, 142},    // #8e372f
        {128, 160, 167},  // #a7a080
        {227, 125, 0},    // #07de3
        {143, 126, 142},  // #8e7e8f
        {136, 68, 153},   // #994488
        {53, 241, 0},     // #0f135
        {201, 170, 174},  // #aeaac9
        {98, 97, 160},    // #a06162
        {119, 58, 76},    // #4c3a77
        {131, 130, 108},  // #6c8283
        {231, 221, 241},  // #f1dde7
        {211, 187, 255},  // #ffbbd3
        {35, 165, 56},    // #38a523
        {168, 255, 180},  // #b4ffa8
        {7, 18, 12},      // #c127
        {110, 82, 215},   // #d7526e
        {254, 159, 149},  // #959ffe
        {0, 127, 125},    // #7d7f0
        {185, 159, 118},  // #769fb9
        {127, 135, 219},  // #db877f
        {25, 19, 17},     // #111319
        {212, 130, 212},  // #d482d4
        {191, 0, 159},    // #9f0bf
        {255, 239, 220},  // #dcefff
        {154, 171, 142},  // #8eab9a
        {66, 100, 113},   // #716442
        {62, 60, 74},     // #4a3c3e
        {95, 78, 8},      // #84e5f
        {68, 184, 156},   // #9cb844
        {213, 222, 216},  // #d8ded5
        {108, 255, 203},  // #cbff6c
        {235, 100, 179},  // #b364eb
        {51, 93, 70},     // #465d33
        {125, 158, 0},    // #09e7d
        {0, 65, 194},     // #c2410
        {187, 188, 79},   // #4fbcbb
        {177, 139, 217},  // #d98bb1
        {182, 115, 91},   // #5b73b6
        {215, 0, 0},      // #00d7
        {140, 60, 255},   // #ff3c8c
        {2, 136, 0},      // #0882
        {0, 172, 199},    // #c7ac0
        {152, 255, 0},    // #0ff98
        {255, 127, 209},  // #d17fff
        {108, 0, 79},     // #4f06c
        {255, 165, 48},   // #30a5ff
        {0, 0, 157},      // #9d00
        {134, 112, 104},  // #687086
        {0, 73, 66},      // #42490
        {79, 42, 0},      // #02a4f
        {0, 253, 207},    // #cffd0
        {188, 183, 255},  // #ffb7bc
        {149, 180, 122},  // #7ab495
        {192, 4, 185},    // #b94c0
        {37, 102, 162},   // #a26625
        {40, 0, 65},      // #41028
        {220, 179, 175},  // #afb3dc
        {254, 245, 144},  // #90f5fe
        {80, 69, 91},     // #5b4550
        {164, 124, 0},    // #07ca4
        {255, 113, 102},  // #6671ff
        {63, 129, 110},   // #6e813f
        {130, 0, 13},     // #d082
        {163, 123, 179},  // #b37ba3
        {52, 78, 0},      // #04e34
        {155, 228, 255},  // #ffe49b
        {235, 0, 119},    // #770eb
        {45, 0, 10},      // #a02d
        {94, 144, 255},   // #ff905e
        {0, 199, 32},     // #20c70
        {88, 1, 170},     // #aa158
        {154, 71, 0},     // #0479a
        {150, 159, 166},  // #a69f96
        {155, 66, 92},    // #5c429b
        {0, 31, 50},      // #321f0
        {200, 196, 0},    // #0c4c8
        {0, 190, 154},    // #9abe0
        {55, 21, 255},    // #ff1537
        {45, 37, 37},     // #25252d
        {223, 88, 255},   // #ff58df
        {190, 231, 192},  // #c0e7be
        {127, 69, 152},   // #98457f
        {82, 79, 60},     // #3c4f52
        {216, 102, 0},    // #066d8
        {100, 116, 56},   // #387464
        {193, 115, 136},  // #8873c1
        {110, 116, 138},  // #8a746e
        {128, 157, 3},    // #39d80
        {190, 139, 101},  // #658bbe
        {99, 51, 57},     // #393363
        {202, 205, 218},  // #dacdca
        {108, 235, 131},  // #83eb6c
        {34, 64, 105},    // #694022
        {162, 127, 255},  // #ff7fa2
        {254, 3, 203},    // #cb3fe
        {118, 188, 253},  // #fdbc76
        {217, 195, 130},  // #82c3d9
        {109, 80, 0},     // #0506d
        {0, 105, 116},    // #74690
        {71, 159, 94},    // #5e9f47
        {148, 198, 191},  // #bfc694
        {249, 255, 0},    // #0fff9
        {192, 84, 69},    // #4554c0
        {0, 101, 60},     // #3c650
        {91, 80, 168},    // #a8505b
        {83, 32, 100},    // #642053
        {79, 95, 255},    // #ff5f4f
        {126, 143, 119},  // #778f7e
        {185, 8, 250},    // #fa8b9
        {139, 146, 195},  // #c3928b
        {179, 0, 53},     // #350b3
        {136, 96, 126},   // #7e6088
        {159, 0, 117},    // #7509f
        {255, 222, 196},  // #c4deff
        {81, 8, 0},       // #0851
        {26, 8, 0},       // #081a
        {76, 137, 182},   // #b6894c
        {0, 223, 223},    // #dfdf0
        {200, 255, 250},  // #faffc8
        {48, 53, 21},     // #153530
        {255, 39, 71},    // #4727ff
        {255, 151, 170},  // #aa97ff
        {4, 0, 26},       // #1a04
        {201, 96, 177},   // #b160c9
        {195, 162, 55},   // #37a2c3
        {124, 79, 58},    // #3a4f7c
        {249, 158, 119},  // #779ef9
        {86, 101, 100},   // #646556
        {209, 147, 255},  // #ff93d1
        {45, 31, 105},    // #691f2d
        {65, 27, 52},     // #341b41
        {175, 147, 152},  // #9893af
        {98, 158, 153},   // #999e62
        {189, 222, 123},  // #7bdebd
        {255, 94, 148},   // #945eff
        {15, 41, 35},     // #2329f
        {184, 190, 172},  // #acbeb8
        {116, 59, 101},   // #653b74
        {16, 0, 13},      // #d010
        {127, 110, 189},  // #bd6e7f
        {158, 107, 59},   // #3b6b9e
        {255, 70, 0},     // #046ff
        {127, 0, 135},    // #8707f
        {255, 206, 62},   // #3eceff
        {48, 59, 67},     // #433b30
        {254, 165, 255},  // #ffa5fe
        {138, 2, 62},     // #3e28a
        {118, 44, 1},     // #12c76
        {10, 138, 150},   // #968aa
        {5, 0, 82},       // #5205
        {142, 214, 50},   // #32d68e
        {83, 196, 115},   // #73c453
        {71, 89, 113},    // #715947
        {88, 2, 34},      // #22258
        {166, 34, 1},     // #122a6
        {144, 147, 76},   // #4c9390
        {0, 67, 30},      // #1e430
        {129, 0, 209},    // #d1081
        {47, 38, 63},     // #3f262f
        {191, 57, 132},   // #8439bf
        {245, 255, 213},  // #d5fff5
        {0, 211, 255},    // #ffd30
        {106, 0, 248},    // #f806a
        {156, 187, 210},  // #d2bb9c
        {122, 217, 171},  // #abd97a
        {105, 87, 93},    // #5d5769
        {0, 105, 5},      // #5690
        {54, 54, 156},    // #9c3636
        {1, 131, 71},     // #47831
        {68, 30, 24},     // #181e44
        {7, 165, 239},    // #efa57
        {255, 129, 48},   // #3081ff
        {167, 85, 184},   // #b855a7
        {104, 90, 131},   // #835a68
        {115, 255, 255},  // #ffff73
        {217, 135, 2},    // #287d9
        {187, 211, 255},  // #ffd3bb
        {142, 55, 47},    // #2f378e
        {167, 160, 128},  // #80a0a7
        {0, 125, 227},    // #e37d0
        {142, 126, 143},  // #8f7e8e
        {153, 68, 136},   // #884499
        {0, 241, 53},     // #35f10
        {174, 170, 201},  // #c9aaae
        {160, 97, 98},    // #6261a0
        {76, 58, 119},    // #773a4c
        {108, 130, 131},  // #83826c
        {241, 221, 231},  // #e7ddf1
        {255, 187, 211},  // #d3bbff
        {56, 165, 35},    // #23a538
        {180, 255, 168},  // #a8ffb4
        {12, 18, 7},      // #712c
        {215, 82, 110},   // #6e52d7
        {149, 159, 254},  // #fe9f95
        {125, 127, 0},    // #07f7d
        {118, 159, 185},  // #b99f76
        {219, 135, 127},  // #7f87db
        {17, 19, 25},     // #191311
        {159, 0, 191},    // #bf09f
        {220, 239, 255},  // #ffefdc
        {142, 171, 154},  // #9aab8e
        {113, 100, 66},   // #426471
        {74, 60, 62},     // #3e3c4a
        {8, 78, 95},      // #5f4e8
        {156, 184, 68},   // #44b89c
        {216, 222, 213},  // #d5ded8
        {203, 255, 108},  // #6cffcb
        {179, 100, 235},  // #eb64b3
        {70, 93, 51},     // #335d46
        {0, 158, 125},    // #7d9e0
        {194, 65, 0},     // #041c2
        {79, 188, 187},   // #bbbc4f
        {217, 139, 177},  // #b18bd9
        {91, 115, 182},   // #b6735b
        // 357 colors
    };

    /**
     *
     * @param mat
     * @param window_name
     * @param delay_ms in ms
     * @param mouse_callback
     * @param userdata
     */
    inline void
    ShowCvMat(
        const cv::Mat &mat,
        const std::string &window_name = "cv_mat",
        const int delay_ms = 0,
        void (*mouse_callback)(int event, int x, int y, int flags, void *userdata) = nullptr,
        void *userdata = nullptr) {
        bool window_exists;
        try {
            window_exists = cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) > 0;
        } catch (const std::exception &) { window_exists = false; }
        if (!window_exists) { cv::namedWindow(window_name, cv::WINDOW_NORMAL); }
        cv::imshow(window_name, mat);
        if (!window_exists) {
            cv::resizeWindow(window_name, 1000, 800);
        }  // only resize when first show
        if (mouse_callback) { cv::setMouseCallback(window_name, mouse_callback, userdata); }
        cv::waitKey(delay_ms);
    }

    void
    ColorGrayCustom(const cv::Mat &gray, cv::Mat &custom);

    inline cv::Mat
    ColorGrayCustom(const cv::Mat &gray) {
        cv::Mat custom;
        ColorGrayCustom(gray, custom);
        return custom;
    }

    void
    ColorGrayToJet(const cv::Mat &gray, cv::Mat &jet, bool normalize = true);

    /**
     * @brief Show a Eigen matrix as a cv::Mat.
     * @param mat
     * @param nan_value value to replace nan
     * @param inf_value value to replace inf
     * @param window_name
     * @param delay_ms
     */
    template<typename T>
    cv::Mat
    ShowEigenMatrix(
        const Eigen::MatrixX<T> &mat,
        double nan_value,
        double inf_value,
        const std::string &window_name,
        int delay_ms = 0) {
        Eigen::MatrixXd normalized_mat = mat.template cast<double>();
        double min = std::numeric_limits<double>::infinity();
        double max = -std::numeric_limits<double>::infinity();
        const long rows = normalized_mat.rows();
        const long cols = normalized_mat.cols();
        for (long r = 0; r < rows; ++r) {
            for (long c = 0; c < cols; ++c) {
                const double value = mat(r, c);
                if (std::isnan(value) || std::isinf(value)) { continue; }
                if (value < min) { min = value; }
                if (value > max) { max = value; }
            }
        }
        if (nan_value < min) { min = nan_value; }
        if (nan_value > max) { max = nan_value; }
        if (inf_value < min) { min = inf_value; }
        if (inf_value > max) { max = inf_value; }

        double value_range = (max - min) / 255.;
        nan_value = (nan_value - min) / value_range;
        inf_value = (inf_value - min) / value_range;

        for (long r = 0; r < rows; ++r) {
            for (long c = 0; c < cols; ++c) {
                double &value = normalized_mat(r, c);
                if (std::isnan(value)) {
                    value = nan_value;
                } else if (std::isinf(value)) {
                    value = inf_value;
                } else {
                    value = (value - min) / value_range;
                }
            }
        }
        Eigen::MatrixXi normalized_mat_int = normalized_mat.cast<int>();
        cv::Mat cv_mat;
        cv::eigen2cv(normalized_mat_int, cv_mat);
        cv_mat = ColorGrayCustom(cv_mat);
        // connect mouse callback
        auto callback = [](const int event, int x, int y, const int &flags, void *userdata) {
            (void) flags;
            if (event == cv::EVENT_LBUTTONDOWN) {
                std::cout << "x: " << x << ", y: " << y << ", mat(x, y): "  //
                          << static_cast<double>(
                                 (*static_cast<const Eigen::MatrixX<T> *>(userdata))(y, x))
                          << std::endl;
            }
        };
        ShowCvMat(cv_mat, window_name, delay_ms, callback, const_cast<Eigen::MatrixX<T> *>(&mat));
        return cv_mat;
    }

    cv::Mat
    AlphaBlending(const cv::Mat &foreground, const cv::Mat &background);

    template<typename Dtype>
    cv::Mat &
    DrawTrajectoryInplace(
        cv::Mat &map,
        const Eigen::Ref<const Eigen::Matrix2X<Dtype>> &trajectory,
        const std::shared_ptr<const GridMapInfo2D<Dtype>> &grid_map_info,
        const cv::Scalar &color,
        const int thickness,
        const bool pixel_based) {

        const long num_points = trajectory.cols();
        if (num_points == 0) { return map; }

        std::vector<cv::Point2i> points;
        points.reserve(num_points);
        if (pixel_based) {
            for (long i = 0; i < num_points; ++i) {
                points.emplace_back(
                    grid_map_info->MeterToGridAtDim(trajectory(0, i), 0),
                    grid_map_info->Shape(1) - grid_map_info->MeterToGridAtDim(trajectory(1, i), 1));
            }
        } else {
            for (long i = 0; i < num_points; ++i) {
                points.emplace_back(
                    grid_map_info->MeterToGridAtDim(trajectory(1, i), 1),
                    grid_map_info->MeterToGridAtDim(trajectory(0, i), 0));
            }
        }
        cv::polylines(map, points, false, color, thickness);
        return map;
    }

    template<typename Dtype>
    cv::Mat &
    DrawLidarScanAreaInplace(
        cv::Mat &map,
        const Eigen::Ref<const Eigen::Vector2<Dtype>> &position,
        const Eigen::Ref<const Eigen::VectorX<Dtype>> &angles_in_world,
        const Eigen::Ref<const Eigen::VectorX<Dtype>> &ranges,
        const std::shared_ptr<GridMapInfo2D<Dtype>> &grid_map_info,
        const cv::Scalar &color) {

        const long num_points = angles_in_world.size();
        if (num_points == 0) { return map; }

        std::vector<std::vector<cv::Point2i>> contours(1);
        auto &contour = contours[0];
        contour.reserve(num_points + 1);
        contour.emplace_back(
            grid_map_info->MeterToGridAtDim(position[1], 1),
            grid_map_info->MeterToGridAtDim(position[0], 0));
        for (long i = 0; i < num_points; ++i) {
            const double &angle = angles_in_world[i];
            const double &range = ranges[i];
            contour.emplace_back(
                grid_map_info->MeterToGridAtDim(position[1] + range * std::sin(angle), 1),
                grid_map_info->MeterToGridAtDim(position[0] + range * std::cos(angle), 0));
        }
        cv::fillPoly(map, contours, color);

        return map;
    }

    template<typename Dtype>
    cv::Mat &
    DrawLidarRaysInplace(
        cv::Mat &map,
        const Eigen::Ref<const Eigen::Vector2<Dtype>> &position,
        const Eigen::Ref<const Eigen::VectorX<Dtype>> &angles_in_world,
        const Eigen::Ref<const Eigen::VectorX<Dtype>> &ranges,
        const std::shared_ptr<GridMapInfo2D<Dtype>> &grid_map_info,
        const cv::Scalar &color,
        const int &ray_thickness) {

        const long num_points = angles_in_world.size();
        if (num_points == 0) { return map; }

        std::vector<cv::Point2i> points;
        points.reserve(num_points * 2);
        const cv::Point2i start_point(
            grid_map_info->MeterToGridAtDim(position[1], 1),
            grid_map_info->MeterToGridAtDim(position[0], 0));
        for (long i = 0; i < num_points; ++i) {
            const double &angle = angles_in_world[i];
            const double &range = ranges[i];
            points.push_back(start_point);
            points.emplace_back(
                grid_map_info->MeterToGridAtDim(position[1] + range * std::sin(angle), 1),
                grid_map_info->MeterToGridAtDim(position[0] + range * std::cos(angle), 0));
        }
        cv::polylines(map, points, false, color, ray_thickness);

        return map;
    }

    inline cv::Mat
    MergeColoredMask(const std::vector<cv::Mat> &masks, const std::vector<cv::Scalar> &colors) {
        ERL_ASSERTM(
            masks.size() == colors.size(),
            "The number of masks and colors must be the same.");
        ERL_ASSERTM(!masks.empty(), "The number of masks and colors must be greater than 0.");

        cv::Mat merged_mask = cv::Mat::zeros(masks[0].size(), CV_8UC4);
        for (size_t i = 0; i < masks.size(); ++i) {
            cv::Mat colored_mask;
            cv::cvtColor(masks[i], colored_mask, cv::COLOR_GRAY2RGBA);
            cv::multiply(colored_mask, colors[i], colored_mask);
            cv::add(merged_mask, colored_mask, merged_mask);
        }
        return merged_mask;
    }
}  // namespace erl::common

#endif
