#pragma once

#include "progress_bar.hpp"

#include <ctime>
#include <mutex>
#include <sstream>
#include <string>

namespace erl::common {
    class LoggingNoFmt {

    public:
        enum Level {
            kInfo,
            kDebug,
            kWarn,
            kError,
            kSilent,
        };

    private:
        static Level s_level_;
        static std::mutex g_print_mutex;

        // ANSI color codes for terminal output
        static constexpr const char* COLOR_RESET = "\033[0m";
        static constexpr const char* COLOR_BLUE = "\033[1;36m";      // Deep sky blue + bold
        static constexpr const char* COLOR_ORANGE = "\033[1;33m";    // Orange + bold
        static constexpr const char* COLOR_RED = "\033[1;31m";       // Red + bold
        static constexpr const char* COLOR_DARK_RED = "\033[1;91m";  // Dark red + bold
        static constexpr const char* COLOR_GREEN = "\033[1;92m";     // Spring green + bold

        template<typename T>
        static void
        AppendToStream(std::ostringstream& oss, T&& value) {
            oss << std::forward<T>(value);
        }

        template<typename T, typename... Args>
        static void
        AppendToStream(std::ostringstream& oss, T&& first, Args&&... args) {
            oss << std::forward<T>(first);
            AppendToStream(oss, std::forward<Args>(args)...);
        }

    public:
        static void
        SetLevel(Level level);

        static Level
        GetLevel();

        static std::string
        GetDateStr();

        static std::string
        GetTimeStr();

        static std::string
        GetDateTimeStr();

        static std::string
        GetTimeStamp();

        template<typename... Args>
        static void
        Info(Args&&... args) {
            if (s_level_ > kInfo) { return; }
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_BLUE << "[";
            oss << std::put_time(time, "%X");
            oss << "][INFO]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        template<typename... Args>
        static void
        Debug(Args&&... args) {
            if (s_level_ > kDebug) { return; }
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_ORANGE << "[";
            oss << std::put_time(time, "%X");
            oss << "][DEBUG]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        template<typename... Args>
        static void
        Warn(Args&&... args) {
            if (s_level_ > kWarn) { return; }
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_ORANGE << "[";
            oss << std::put_time(time, "%X");
            oss << "][WARN]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        /**
         * Report the error but not fatal message when an exception is handled properly and the
         * program can continue
         * @tparam Args
         * @param args
         */
        template<typename... Args>
        static void
        Error(Args&&... args) {
            if (s_level_ > kError) { return; }
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_RED << "[";
            oss << std::put_time(time, "%X");
            oss << "][ERROR]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        /**
         * Report a fatal message ignoring the logging level.
         * @tparam Args
         * @param args
         */
        template<typename... Args>
        static void
        Fatal(Args&&... args) {
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_DARK_RED << "[";
            oss << std::put_time(time, "%X");
            oss << "][FATAL]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        /**
         * Report a success message ignoring the logging level.
         * @tparam Args
         * @param args
         */
        template<typename... Args>
        static void
        Success(Args&&... args) {
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream oss;
            oss << COLOR_GREEN << "[";
            oss << std::put_time(time, "%X");
            oss << "][SUCCESS]: " << COLOR_RESET;
            AppendToStream(oss, std::forward<Args>(args)...);

            std::string msg = oss.str();
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        /**
         * Report a failure message ignoring the logging level.
         * @tparam Args
         * @param args
         * @return
         */
        template<typename... Args>
        static std::string
        Failure(Args&&... args) {
            std::lock_guard lock(g_print_mutex);
            time_t now = std::time(nullptr);
            auto time = std::localtime(&now);

            std::ostringstream header_oss;
            header_oss << COLOR_RED << "[";
            header_oss << std::put_time(time, "%X");
            header_oss << "][FAILURE]: " << COLOR_RESET;

            std::ostringstream content_oss;
            AppendToStream(content_oss, std::forward<Args>(args)...);

            std::string header_msg = header_oss.str();
            std::string failure_msg = content_oss.str();
            if (ProgressBar::GetNumBars() == 0) { failure_msg += "\n"; }
            ProgressBar::Write(header_msg + failure_msg);
            return failure_msg;
        }

        static void
        Write(const std::string& msg) {
            std::lock_guard lock(g_print_mutex);
            ProgressBar::Write(msg);
        }
    };
}  // namespace erl::common

#define LOGGING_NO_FMT_LABELS (std::string(__FILE__) + ":" + std::to_string(__LINE__))
#define LOGGING_NO_FMT_LABELED_MSG(msg) \
    (std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + std::string(msg))

#if defined(ERL_ROS_VERSION_1)
    #include <ros/assert.h>
    #include <ros/console.h>

// Helper function to convert variadic args to string
namespace erl::common::detail {
    template<typename... Args>
    inline std::string
    FormatArgs(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }
}  // namespace erl::common::detail

    #define ERL_NO_FMT_FATAL(...) \
        ROS_FATAL("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_ERROR(...) \
        ROS_ERROR("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN(...) \
        ROS_WARN("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN_ONCE(...) \
        ROS_WARN_ONCE("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN_COND(condition, ...) \
        ROS_WARN_COND(condition, "%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_INFO(...) \
        ROS_INFO("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_DEBUG(...) \
        ROS_DEBUG("%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #ifdef ROS_ASSERT_ENABLED
        #define ERL_NO_FMT_ASSERT(expr) ROS_ASSERT(expr)
        #define ERL_NO_FMT_ASSERTM(expr, ...)                                                     \
            do {                                                                                  \
                ROS_ASSERT_MSG(expr, "%s", erl::common::detail::FormatArgs(__VA_ARGS__).c_str()); \
            } while (false)
    #endif
#elif defined(ERL_ROS_VERSION_2)
    #include <rclcpp/rclcpp.hpp>

// Helper function to convert variadic args to string
namespace erl::common::detail {
    template<typename... Args>
    inline std::string
    FormatArgs(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }
}  // namespace erl::common::detail

    #define ERL_NO_FMT_FATAL(...)         \
        RCLCPP_FATAL(                     \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_ERROR(...)         \
        RCLCPP_ERROR(                     \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN(...)          \
        RCLCPP_WARN(                      \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN_ONCE(...)     \
        RCLCPP_WARN_ONCE(                 \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_WARN_COND(condition, ...)                           \
        do {                                                               \
            if (condition)                                                 \
                RCLCPP_WARN(                                               \
                    rclcpp::get_logger("rclcpp"),                          \
                    "%s",                                                  \
                    erl::common::detail::FormatArgs(__VA_ARGS__).c_str()); \
        } while (false)
    #define ERL_NO_FMT_INFO(...)          \
        RCLCPP_INFO(                      \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
    #define ERL_NO_FMT_DEBUG(...)         \
        RCLCPP_DEBUG(                     \
            rclcpp::get_logger("rclcpp"), \
            "%s",                         \
            erl::common::detail::FormatArgs(__VA_ARGS__).c_str())
#else

// Helper function to convert variadic args to string
namespace erl::common::detail {
    template<typename... Args>
    std::string
    FormatArgs(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }
}  // namespace erl::common::detail

    #define ERL_NO_FMT_FATAL(...)                              \
        do {                                                   \
            erl::common::LoggingNoFmt::Fatal(                  \
                __FILE__,                                      \
                ":",                                           \
                __LINE__,                                      \
                ": ",                                          \
                erl::common::detail::FormatArgs(__VA_ARGS__)); \
            exit(1);                                           \
        } while (false)

    #define ERL_NO_FMT_ERROR(...)                              \
        do {                                                   \
            erl::common::LoggingNoFmt::Error(                  \
                __FILE__,                                      \
                ":",                                           \
                __LINE__,                                      \
                ": ",                                          \
                erl::common::detail::FormatArgs(__VA_ARGS__)); \
        } while (false)

    #define ERL_NO_FMT_WARN(...)                               \
        do {                                                   \
            erl::common::LoggingNoFmt::Warn(                   \
                __FILE__,                                      \
                ":",                                           \
                __LINE__,                                      \
                ": ",                                          \
                erl::common::detail::FormatArgs(__VA_ARGS__)); \
        } while (false)

    #define ERL_NO_FMT_WARN_ONCE(...)         \
        do {                                  \
            static bool warned = false;       \
            if (!warned) {                    \
                warned = true;                \
                ERL_NO_FMT_WARN(__VA_ARGS__); \
            }                                 \
        } while (false)

    #define ERL_NO_FMT_WARN_COND(condition, ...)             \
        do {                                                 \
            if (condition) { ERL_NO_FMT_WARN(__VA_ARGS__); } \
        } while (false)

    #define ERL_NO_FMT_INFO(...)                               \
        do {                                                   \
            erl::common::LoggingNoFmt::Info(                   \
                __FILE__,                                      \
                ":",                                           \
                __LINE__,                                      \
                ": ",                                          \
                erl::common::detail::FormatArgs(__VA_ARGS__)); \
        } while (false)

    #ifndef NDEBUG
        #define ERL_NO_FMT_DEBUG(...)                              \
            do {                                                   \
                erl::common::LoggingNoFmt::Debug(                  \
                    __FILE__,                                      \
                    ":",                                           \
                    __LINE__,                                      \
                    ": ",                                          \
                    erl::common::detail::FormatArgs(__VA_ARGS__)); \
            } while (false)
        #define ERL_NO_FMT_DEBUG_ASSERT(expr, ...) ERL_NO_FMT_ASSERTM(expr, __VA_ARGS__)
    #else
        #define ERL_NO_FMT_DEBUG(...)              ((void) 0)
        #define ERL_NO_FMT_DEBUG_ASSERT(expr, ...) (void) 0
    #endif
#endif

#define ERL_NO_FMT_INFO_ONCE(...)         \
    do {                                  \
        static bool infoed = false;       \
        if (!infoed) {                    \
            infoed = true;                \
            ERL_NO_FMT_INFO(__VA_ARGS__); \
        }                                 \
    } while (false)

#define ERL_NO_FMT_WARN_ONCE_COND(condition, ...) \
    do {                                          \
        static bool warned = false;               \
        if (!warned && (condition)) {             \
            warned = true;                        \
            ERL_NO_FMT_WARN(__VA_ARGS__);         \
        }                                         \
    } while (false)

#ifndef ERL_NO_FMT_ASSERTM
    #define ERL_NO_FMT_ASSERTM(expr, ...)                                     \
        do {                                                                  \
            if (!(expr)) {                                                    \
                std::string failure_msg = erl::common::LoggingNoFmt::Failure( \
                    "assertion (",                                            \
                    #expr,                                                    \
                    ") at ",                                                  \
                    __FILE__,                                                 \
                    ":",                                                      \
                    __LINE__,                                                 \
                    ": ",                                                     \
                    erl::common::detail::FormatArgs(__VA_ARGS__));            \
                throw std::runtime_error(failure_msg);                        \
            }                                                                 \
        } while (false)
#endif

#ifndef ERL_NO_FMT_ASSERT
    #define ERL_NO_FMT_ASSERT(expr) ERL_NO_FMT_ASSERTM(expr, "Assertion ", #expr, " failed.")
#endif

#ifndef NDEBUG
    #define ERL_NO_FMT_DEBUG_ASSERT(expr, ...)         ERL_NO_FMT_ASSERTM(expr, __VA_ARGS__)
    #define ERL_NO_FMT_DEBUG_WARN_COND(condition, ...) ERL_NO_FMT_WARN_COND(condition, __VA_ARGS__)
    #define ERL_NO_FMT_DEBUG_WARN_ONCE_COND(condition, ...) \
        ERL_NO_FMT_WARN_ONCE_COND(condition, __VA_ARGS__)
#else
    #define ERL_NO_FMT_DEBUG_ASSERT(expr, ...)              (void) 0
    #define ERL_NO_FMT_DEBUG_WARN_COND(condition, ...)      (void) 0
    #define ERL_NO_FMT_DEBUG_WARN_ONCE_COND(condition, ...) (void) 0
#endif
