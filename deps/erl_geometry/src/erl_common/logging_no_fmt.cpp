#include "erl_common/logging_no_fmt.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace erl::common {

    LoggingNoFmt::Level LoggingNoFmt::s_level_ = LoggingNoFmt::Level::kInfo;
    std::mutex LoggingNoFmt::g_print_mutex;

    void
    LoggingNoFmt::SetLevel(Level level) {
        s_level_ = level;
    }

    LoggingNoFmt::Level
    LoggingNoFmt::GetLevel() {
        return s_level_;
    }

    std::string
    LoggingNoFmt::GetDateStr() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d");
        return oss.str();
    }

    std::string
    LoggingNoFmt::GetTimeStr() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%H:%M:%S");
        return oss.str();
    }

    std::string
    LoggingNoFmt::GetDateTimeStr() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    std::string
    LoggingNoFmt::GetTimeStamp() {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        return std::to_string(millis);
    }

}  // namespace erl::common
