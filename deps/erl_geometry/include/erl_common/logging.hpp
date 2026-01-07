#pragma once

#include "logging_level.hpp"

#ifdef ERL_USE_FMT
    #include "fmt.hpp"
    #include "progress_bar.hpp"

    #include <mutex>

namespace erl::common {

    class Logging {
        static LoggingLevel s_level_;
        static std::mutex g_print_mutex;

    public:
        static void
        SetLevel(LoggingLevel level);

        static LoggingLevel
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
        Info(Args... args) {
            if (s_level_ > kInfo) { return; }
            // https://fmt.dev/latest/syntax.html
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(
                fmt::fg(fmt::color::deep_sky_blue) | fmt::emphasis::bold,
                "[{:%X}][INFO]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        template<typename... Args>
        static void
        Debug(Args... args) {
            if (s_level_ > kDebug) { return; }
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(  //
                fmt::fg(fmt::color::orange) | fmt::emphasis::bold,
                "[{:%X}][DEBUG]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
            if (ProgressBar::GetNumBars() == 0) { msg += "\n"; }
            ProgressBar::Write(msg);
        }

        template<typename... Args>
        static void
        Warn(Args... args) {
            if (s_level_ > kWarn) { return; }
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(
                fmt::fg(fmt::color::orange_red) | fmt::emphasis::bold,
                "[{:%X}][WARN]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
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
        Error(Args... args) {
            if (s_level_ > kError) { return; }
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(  //
                fmt::fg(fmt::color::red) | fmt::emphasis::bold,
                "[{:%X}][ERROR]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
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
        Fatal(Args... args) {
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(
                fmt::fg(fmt::color::dark_red) | fmt::emphasis::bold,
                "[{:%X}][FATAL]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
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
        Success(Args... args) {
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            std::string msg = fmt::format(
                fmt::fg(fmt::color::spring_green) | fmt::emphasis::bold,
                "[{:%X}][SUCCESS]: ",
                time);
            fmt::format_to(std::back_inserter(msg), std::forward<Args>(args)...);
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
        Failure(Args... args) {
            const std::scoped_lock lock(g_print_mutex);
            const time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
            auto time = *std::localtime(&now);
    #else
            auto time = fmt::localtime(now);
    #endif
            const std::string msg = fmt::format(  //
                fmt::fg(fmt::color::red) | fmt::emphasis::bold,
                "[{:%X}][FAILURE]: ",
                time);
            std::string failure_msg = fmt::format(std::forward<Args>(args)...);
            if (ProgressBar::GetNumBars() == 0) { failure_msg += "\n"; }
            ProgressBar::Write(msg + failure_msg);
            return failure_msg;
        }

        static void
        Write(const std::string &msg) {
            const std::scoped_lock lock(g_print_mutex);
            ProgressBar::Write(msg);
        }
    };
}  // namespace erl::common

    #define LOGGING_LABELS           fmt::format("{}:{}", __FILE__, __LINE__)
    #define LOGGING_LABELED_MSG(msg) fmt::format("{}:{}: {}", __FILE__, __LINE__, msg)

    #ifdef ERL_ROS_VERSION_1
        #include <ros/assert.h>
        #include <ros/console.h>
        #define ERL_FATAL(...)     ROS_FATAL(fmt::format(__VA_ARGS__).c_str())
        #define ERL_ERROR(...)     ROS_ERROR(fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN(...)      ROS_WARN(fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN_ONCE(...) ROS_WARN_ONCE(fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN_COND(condition, ...) \
            ROS_WARN_COND(condition, fmt::format(__VA_ARGS__).c_str())
        #define ERL_INFO(...)  ROS_INFO(fmt::format(__VA_ARGS__).c_str())
        #define ERL_DEBUG(...) ROS_DEBUG(fmt::format(__VA_ARGS__).c_str())
        #ifdef ROS_ASSERT_ENABLED
            #define ERL_ASSERT(expr) ROS_ASSERT(expr)
            #define ERL_ASSERTM(expr, ...) \
                do { ROS_ASSERT_MSG(expr, fmt::format(__VA_ARGS__).c_str()); } while (false)
        #endif
    #elif defined(ERL_ROS_VERSION_2)
        #include <rclcpp/rclcpp.hpp>
        #define ERL_FATAL(...) \
            RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
        #define ERL_ERROR(...) \
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN(...) \
            RCLCPP_WARN(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN_ONCE(...) \
            RCLCPP_WARN_ONCE(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
        #define ERL_WARN_COND(condition, ...)                                                    \
            do {                                                                                 \
                if (condition)                                                                   \
                    RCLCPP_WARN(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str()); \
            } while (false)
        #define ERL_INFO(...) \
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
        #define ERL_DEBUG(...) \
            RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), fmt::format(__VA_ARGS__).c_str())
    #else

        #define ERL_FATAL(...)                 \
            do {                               \
                erl::common::Logging::Fatal(   \
                    "{}:{}: {}",               \
                    __FILE__,                  \
                    __LINE__,                  \
                    fmt::format(__VA_ARGS__)); \
                exit(1);                       \
            } while (false)

        #define ERL_ERROR(...)                 \
            do {                               \
                erl::common::Logging::Error(   \
                    "{}:{}: {}",               \
                    __FILE__,                  \
                    __LINE__,                  \
                    fmt::format(__VA_ARGS__)); \
            } while (false)

        #define ERL_WARN(...)                  \
            do {                               \
                erl::common::Logging::Warn(    \
                    "{}:{}: {}",               \
                    __FILE__,                  \
                    __LINE__,                  \
                    fmt::format(__VA_ARGS__)); \
            } while (false)

        #define ERL_WARN_ONCE(...)          \
            do {                            \
                static bool warned = false; \
                if (!warned) {              \
                    warned = true;          \
                    ERL_WARN(__VA_ARGS__);  \
                }                           \
            } while (false)

        #define ERL_WARN_COND(condition, ...)             \
            do {                                          \
                if (condition) { ERL_WARN(__VA_ARGS__); } \
            } while (false)

        #define ERL_INFO(...)                  \
            do {                               \
                erl::common::Logging::Info(    \
                    "{}:{}: {}",               \
                    __FILE__,                  \
                    __LINE__,                  \
                    fmt::format(__VA_ARGS__)); \
            } while (false)

        #define ERL_DEBUG(...)                 \
            do {                               \
                erl::common::Logging::Debug(   \
                    "{}:{}: {}",               \
                    __FILE__,                  \
                    __LINE__,                  \
                    fmt::format(__VA_ARGS__)); \
            } while (false)

        #ifndef NDEBUG
            #define ERL_DEBUG_ASSERT(expr, ...) ERL_ASSERTM(expr, __VA_ARGS__)
        #else
            #define ERL_DEBUG_ASSERT(expr, ...) (void) 0
        #endif
    #endif

    #define ERL_INFO_ONCE(...)          \
        do {                            \
            static bool infoed = false; \
            if (!infoed) {              \
                infoed = true;          \
                ERL_INFO(__VA_ARGS__);  \
            }                           \
        } while (false)

    #define ERL_WARN_ONCE_COND(condition, ...) \
        do {                                   \
            static bool warned = false;        \
            if (!warned && (condition)) {      \
                warned = true;                 \
                ERL_WARN(__VA_ARGS__);         \
            }                                  \
        } while (false)

    #ifndef ERL_ASSERTM
        #define ERL_ASSERTM(expr, ...)                                             \
            do {                                                                   \
                if (!(expr)) {                                                     \
                    const std::string failure_msg = erl::common::Logging::Failure( \
                        "assertion ({}) at {}:{}: {}",                             \
                        #expr,                                                     \
                        __FILE__,                                                  \
                        __LINE__,                                                  \
                        fmt::format(__VA_ARGS__));                                 \
                    throw std::runtime_error(failure_msg);                         \
                }                                                                  \
            } while (false)
    #endif

    #ifndef ERL_ASSERT
        #define ERL_ASSERT(expr) ERL_ASSERTM(expr, "Assertion {} failed.", #expr)
    #endif

    #ifndef NDEBUG
        #define ERL_DEBUG_ASSERT(expr, ...)              ERL_ASSERTM(expr, __VA_ARGS__)
        #define ERL_DEBUG_WARN_COND(condition, ...)      ERL_WARN_COND(condition, __VA_ARGS__)
        #define ERL_DEBUG_WARN_ONCE_COND(condition, ...) ERL_WARN_ONCE_COND(condition, __VA_ARGS__)
    #else
        #define ERL_DEBUG_ASSERT(expr, ...)              (void) 0
        #define ERL_DEBUG_WARN_COND(condition, ...)      (void) 0
        #define ERL_DEBUG_WARN_ONCE_COND(condition, ...) (void) 0
    #endif

#else

    #include "logging_no_fmt.hpp"

namespace erl::common {
    using Logging = LoggingNoFmt;
}

    #define ERL_FATAL(...)                      ERL_NO_FMT_FATAL(__VA_ARGS__)
    #define ERL_ERROR(...)                      ERL_NO_FMT_ERROR(__VA_ARGS__)
    #define ERL_WARN(...)                       ERL_NO_FMT_WARN(__VA_ARGS__)
    #define ERL_WARN_ONCE(...)                  ERL_NO_FMT_WARN_ONCE(__VA_ARGS__)
    #define ERL_WARN_COND(condition, ...)       ERL_NO_FMT_WARN_COND(condition, __VA_ARGS__)
    #define ERL_WARN_ONCE_COND(condition, ...)  ERL_NO_FMT_WARN_ONCE_COND(condition, __VA_ARGS__)
    #define ERL_INFO(...)                       ERL_NO_FMT_INFO(__VA_ARGS__)
    #define ERL_INFO_ONCE(...)                  ERL_NO_FMT_INFO_ONCE(__VA_ARGS__)
    #define ERL_DEBUG(...)                      ERL_NO_FMT_DEBUG(__VA_ARGS__)
    #define ERL_ASSERT(expr)                    ERL_NO_FMT_ASSERT(expr)
    #define ERL_ASSERTM(expr, ...)              ERL_NO_FMT_ASSERTM(expr, __VA_ARGS__)
    #define ERL_DEBUG_ASSERT(condition, ...)    ERL_NO_FMT_DEBUG_ASSERT(condition, __VA_ARGS__)
    #define ERL_DEBUG_WARN_COND(condition, ...) ERL_NO_FMT_DEBUG_WARN_COND(condition, __VA_ARGS__)
    #define ERL_DEBUG_WARN_ONCE_COND(condition, ...) \
        ERL_NO_FMT_DEBUG_WARN_ONCE_COND(condition, __VA_ARGS__)

#endif

#define ERL_ASSERT_EQ(a, b)     ERL_ASSERTM((a) == (b), "{} == {} failed", (a), (b))
#define ERL_ASSERT_NE(a, b)     ERL_ASSERTM((a) != (b), "{} != {} failed", (a), (b))
#define ERL_ASSERT_LT(a, b)     ERL_ASSERTM((a) < (b), "{} < {} failed", (a), (b))
#define ERL_ASSERT_LE(a, b)     ERL_ASSERTM((a) <= (b), "{} <= {} failed", (a), (b))
#define ERL_ASSERT_GT(a, b)     ERL_ASSERTM((a) > (b), "{} > {} failed", (a), (b))
#define ERL_ASSERT_GE(a, b)     ERL_ASSERTM((a) >= (b), "{} >= {} failed", (a), (b))
#define ERL_ASSERT_PTR(ptr)     ERL_ASSERTM((ptr) != nullptr, "{} != nullptr failed", #ptr)
#define ERL_ASSERT_NULL(ptr)    ERL_ASSERTM((ptr) == nullptr, "{} == nullptr failed", #ptr)
#define ERL_ASSERT_POS_EQ(a, b) ERL_ASSERTM((a) <= 0 || (a) == (b), "{} == {} failed", (a), (b))
#define ERL_ASSERT_POS_LT(a, b) ERL_ASSERTM((a) <= 0 || (a) < (b), "{} < {} failed", (a), (b))
#define ERL_ASSERT_POS_LE(a, b) ERL_ASSERTM((a) <= 0 || (a) <= (b), "{} <= {} failed", (a), (b))
#define ERL_ASSERT_POS_GT(a, b) ERL_ASSERTM((a) <= 0 || (a) > (b), "{} > {} failed", (a), (b))
#define ERL_ASSERT_POS_GE(a, b) ERL_ASSERTM((a) <= 0 || (a) >= (b), "{} >= {} failed", (a), (b))

#define ERL_DEBUG_ASSERT_EQ(a, b)  ERL_DEBUG_ASSERT((a) == (b), "{} == {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_NE(a, b)  ERL_DEBUG_ASSERT((a) != (b), "{} != {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_LT(a, b)  ERL_DEBUG_ASSERT((a) < (b), "{} < {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_LE(a, b)  ERL_DEBUG_ASSERT((a) <= (b), "{} <= {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_GT(a, b)  ERL_DEBUG_ASSERT((a) > (b), "{} > {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_GE(a, b)  ERL_DEBUG_ASSERT((a) >= (b), "{} >= {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_PTR(ptr)  ERL_DEBUG_ASSERT((ptr) != nullptr, "{} != nullptr failed", #ptr)
#define ERL_DEBUG_ASSERT_NULL(ptr) ERL_DEBUG_ASSERT((ptr) == nullptr, "{} == nullptr failed", #ptr)
#define ERL_DEBUG_ASSERT_POS_EQ(a, b) \
    ERL_DEBUG_ASSERT((a) <= 0 || (a) == (b), "{} == {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_POS_LT(a, b) \
    ERL_DEBUG_ASSERT((a) <= 0 || (a) < (b), "{} < {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_POS_LE(a, b) \
    ERL_DEBUG_ASSERT((a) <= 0 || (a) <= (b), "{} <= {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_POS_GT(a, b) \
    ERL_DEBUG_ASSERT((a) <= 0 || (a) > (b), "{} > {} failed", (a), (b))
#define ERL_DEBUG_ASSERT_POS_GE(a, b) \
    ERL_DEBUG_ASSERT((a) <= 0 || (a) >= (b), "{} >= {} failed", (a), (b))

/**
 * Assert expression and return a value if the assertion fails.
 * @param expr Expression to assert.
 * @param retval Return value if the assertion fails.
 * @param ... Message format and arguments.
 */
#define ERL_ASSERTM_RETURN(expr, retval, ...) \
    do {                                      \
        if (!(expr)) {                        \
            ERL_ERROR(__VA_ARGS__);           \
            return retval;                    \
        }                                     \
    } while (false)
