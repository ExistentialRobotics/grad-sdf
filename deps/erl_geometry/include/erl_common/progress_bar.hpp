#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

namespace erl::common {

    class ProgressBar : public std::enable_shared_from_this<ProgressBar> {
    public:
        struct Setting {
            std::size_t line_width = 120;
            std::string description{};
            char symbol_done = '#';
            char symbol_todo = ' ';
            std::size_t position = 0;
            std::size_t total = 100;

            [[nodiscard]] std::string
            GetSymbolBuffer() const {
                constexpr std::size_t tail_width = sizeof " [100.00 %]";
                const std::size_t n = line_width - tail_width - description.size();
                return std::string(n, symbol_done) + std::string(n, symbol_todo);
            }
        };

    private:
        static std::vector<std::weak_ptr<const ProgressBar>> s_progress_bars_;

        std::shared_ptr<Setting> m_setting_ = nullptr;
        double m_fraction_ = 0.0;
        std::size_t m_prev_count_ = -1;
        std::size_t m_count_ = 0;
        std::chrono::high_resolution_clock::time_point m_t0_ =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> m_prev_duration_{};
        std::chrono::duration<double> m_duration_{};
        std::ostream &m_out_;
        bool m_displayed_ = false;

    public:
        static std::shared_ptr<ProgressBar>
        Open(std::shared_ptr<Setting> setting = nullptr, std::ostream &out = std::cout);

        ProgressBar(const ProgressBar &) = delete;
        ProgressBar &
        operator=(const ProgressBar &) = delete;
        ProgressBar(ProgressBar &&) = delete;
        ProgressBar &
        operator=(ProgressBar &&) = delete;

        ~ProgressBar() { Close(); }

        void
        Close() {
            for (auto itr = s_progress_bars_.begin(); itr != s_progress_bars_.end(); ++itr) {
                if (itr->expired()) {
                    s_progress_bars_.erase(itr--);  // erase and decrement iterator
                    continue;
                }
                if (itr->lock().get() == this) {
                    s_progress_bars_.erase(itr);
                    break;
                }
            }
            m_displayed_ = false;
        }

        static std::size_t
        GetNumBars() {
            return s_progress_bars_.size();
        }

        void
        SetDescription(const std::string &description) const {
            m_setting_->description = description;
        }

        void
        SetTotal(const std::size_t total) const {
            m_setting_->total = total;
        }

        void
        Print(const std::string &msg) const {
            Write(msg, m_out_);
        }

        template<typename... Args>
        static void
        Write(std::ostream &out, Args &&...args) {
            std::stringstream ss;
            (ss << ... << args);  // https://en.cppreference.com/w/cpp/language/fold
            Write(ss.str(), out);
        }

        static void
        Write(const std::string &str = "", std::ostream &out = std::cout) {
            if (s_progress_bars_.empty()) {
                out << str << std::flush;
                return;
            }
            std::vector<std::size_t> idx = ArgSortProgressBars();
            if (const std::size_t num_displayed = std::count_if(  //
                    s_progress_bars_.begin(),
                    s_progress_bars_.end(),
                    [](const auto& pbar) -> bool {
                        return !pbar.expired() && pbar.lock()->m_displayed_;
                    });
                num_displayed >= 1) {
                GoBackLines(out, num_displayed);
            }
            auto idx_itr = idx.begin();
            if (str.empty()) {
                out << s_progress_bars_[*idx_itr++].lock()->AsString();
            } else {
                out << str << std::endl << s_progress_bars_[*idx_itr++].lock()->AsString();
            }

            for (; idx_itr != idx.end(); ++idx_itr) {
                out << std::endl << s_progress_bars_[*idx_itr].lock()->AsString();
            }
            out << std::flush;
        }

        void
        Update(const std::size_t n = 1, const std::string &msg = "") {
            UpdateCount(n);
            UpdateFraction();
            Write(msg, m_out_);
        }

        void
        UpdateProgress(const std::size_t n, const std::string &msg = "") {
            m_prev_count_ = m_count_;
            m_count_ = n;
            UpdateDuration();
            UpdateFraction();
            Write(msg, m_out_);
        }

        void
        Reset() {
            Register();
            m_prev_duration_ = std::chrono::duration<double>{};
            m_duration_ = std::chrono::duration<double>{};
            m_prev_count_ = 0;
            m_count_ = 0;
            m_fraction_ = 0.0;
            m_t0_ = std::chrono::high_resolution_clock::now();
        }

    private:
        explicit ProgressBar(
            std::shared_ptr<Setting> setting = nullptr,
            std::ostream &out = std::cout)
            : m_setting_(std::move(setting)), m_out_(out) {
            if (m_setting_ == nullptr) { m_setting_ = std::make_shared<Setting>(); }
        }

        static std::vector<std::size_t>
        ArgSortProgressBars() {
            std::vector<std::size_t> idx(s_progress_bars_.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [](const std::size_t i, const std::size_t j) {
                return s_progress_bars_[i].lock()->m_setting_->position <
                       s_progress_bars_[j].lock()->m_setting_->position;
            });
            return idx;
        }

        static void
        GoBackLines(std::ostream &out, const std::size_t num_lines) {
            // https://web.archive.org/web/20121225024852/http://www.climagic.org/mirrors/VT100_Escape_Codes.html
            out << "\33[2K\r";  // clear line
            // move the cursor up
            for (std::size_t i = 1; i < num_lines; i++) { out << "\033[F\033[0K"; }
        }

        static std::string
        DurationToString(double seconds) {
            std::stringstream ss;
            if (constexpr double day = 24 * 3600; seconds > day) {
                const int days = static_cast<int>(seconds / day);
                ss << days << "d,";
                seconds -= static_cast<double>(days) * day;
            }
            if (constexpr double hour = 3600; seconds > hour) {
                const int hours = static_cast<int>(seconds / hour);
                ss << std::setfill('0') << std::setw(2) << hours << ":";
                seconds -= static_cast<double>(hours) * hour;
            }
            if (seconds > 60) {
                const int minutes = static_cast<int>(seconds / 60);
                ss << std::setfill('0') << std::setw(2) << minutes << ":";
                seconds -= static_cast<double>(minutes) * 60;
            } else {
                ss << "00:";
            }
            if (seconds >= 1) {
                const int sec = static_cast<int>(seconds);
                ss << std::setfill('0') << std::setw(2) << sec;
            } else {
                ss << "00";
            }
            return ss.str();
        }

        void
        UpdateFraction() {
            if (m_setting_->total == 0) { return; }
            m_fraction_ = static_cast<double>(m_count_) / static_cast<double>(m_setting_->total);
            if (m_fraction_ < 0) { m_fraction_ = 0; }
            if (m_fraction_ > 1) { m_fraction_ = 1; }
        }

        void
        UpdateCount(const std::size_t n) {
            if (n == 0) { return; }
            m_prev_count_ = m_count_;
            m_count_ += n;
            UpdateDuration();
        }

        void
        UpdateDuration() {
            m_prev_duration_ = m_duration_;
            m_duration_ = std::chrono::high_resolution_clock::now() - m_t0_;
        }

        [[nodiscard]] double
        ComputeFps() const {
            if (m_count_ == m_prev_count_) {
                return static_cast<double>(m_count_) / (m_duration_.count() + 0.001);
            }
            return static_cast<double>(m_count_ - m_prev_count_) /
                   (m_duration_.count() - m_prev_duration_.count());
        }

        [[nodiscard]] std::string
        AsString() const {
            std::stringstream ss_time;
            double passed_seconds = m_duration_.count();
            double remaining_seconds = passed_seconds / m_fraction_ - passed_seconds;
            ss_time << "| [" << DurationToString(remaining_seconds) << "/"
                    << DurationToString(passed_seconds) << "]";
            std::string time_str = ss_time.str();

            std::stringstream ss_count;
            if (m_setting_->total > 0) {
                ss_count << "[" << std::setw(3) << m_count_ << "/" << m_setting_->total << ": "
                         << std::setprecision(2) << ComputeFps() << "it/s]";
            }
            std::string count_str = ss_count.str();

            std::stringstream ss_desc;
            ss_desc << m_setting_->description << ": ";
            std::string desc_str = ss_desc.str();

            std::string symbol_buffer = m_setting_->GetSymbolBuffer();
            std::size_t n = symbol_buffer.size() / 2;
            std::size_t bar_width = n - count_str.size() - time_str.size();
            std::size_t offset =  // static_cast from positive real to unsigned integer is fine.
                n - static_cast<std::size_t>(static_cast<double>(bar_width) * m_fraction_);

            std::stringstream ss_bar;
            ss_bar << desc_str;  // write description
            ss_bar.write(
                symbol_buffer.data() + offset,
                static_cast<std::streamsize>(bar_width));  // write bar
            ss_bar << time_str << count_str;               // write time and count
            ss_bar << "[" << std::fixed << std::setprecision(2) << std::setw(6) << 100 * m_fraction_
                   << "%]";  // write time and count
            return ss_bar.str();
        }

        void
        Register() const {
            if (std::none_of(
                    s_progress_bars_.begin(),
                    s_progress_bars_.end(),
                    [this](const auto &bar) { return bar.lock().get() == this; })) {
                s_progress_bars_.push_back(shared_from_this());
            }
        }
    };
}  // namespace erl::common
