#ifdef ERL_USE_FMT
    #include "erl_common/logging.hpp"

namespace erl::common {
    #ifndef NDEBUG
    LoggingLevel Logging::s_level_ = kDebug;
    #else
    LoggingLevel Logging::s_level_ = kInfo;
    #endif
    std::mutex Logging::g_print_mutex;

    void
    Logging::SetLevel(const LoggingLevel level) {
        s_level_ = level;
    }

    LoggingLevel
    Logging::GetLevel() {
        return s_level_;
    }

    std::string
    Logging::GetDateStr() {
        time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
        return fmt::format("{:%Y-%m-%d}", *std::localtime(&now));
    #else
        return fmt::format("{:%Y-%m-%d}", fmt::localtime(now));
    #endif
    }

    std::string
    Logging::GetTimeStr() {
        time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
        return fmt::format("{:%X}", *std::localtime(&now));
    #else
        return fmt::format("{:%X}", fmt::localtime(now));
    #endif
    }

    std::string
    Logging::GetDateTimeStr() {
        time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
        return fmt::format("{:%Y-%m-%d %X}", *std::localtime(&now));
    #else
        return fmt::format("{:%Y-%m-%d %X}", fmt::localtime(now));
    #endif
    }

    std::string
    Logging::GetTimeStamp() {
        time_t now = std::time(nullptr);
    #if FMT_VERSION >= 110200
        return fmt::format("{:%Y%m%d-%H%M%S}", *std::localtime(&now));
    #else
        return fmt::format("{:%Y%m%d-%H%M%S}", fmt::localtime(now));
    #endif
    }

}  // namespace erl::common
#endif
