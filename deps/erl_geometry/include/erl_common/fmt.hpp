#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#ifndef FMT_HEADER_ONLY
    #define FMT_HEADER_ONLY
#endif
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#if FMT_VERSION >= 60200  // 6.2.0
    #include <fmt/os.h>
#endif
#if FMT_VERSION >= 90000
    #include <fmt/std.h>
#endif
#pragma GCC diagnostic pop
