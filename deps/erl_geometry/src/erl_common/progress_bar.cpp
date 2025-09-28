#include "erl_common/progress_bar.hpp"

namespace erl::common {

    std::vector<std::weak_ptr<const ProgressBar>> ProgressBar::s_progress_bars_ = {};

    std::shared_ptr<ProgressBar>
    ProgressBar::Open(std::shared_ptr<Setting> setting, std::ostream &out) {
        auto bar = std::shared_ptr<ProgressBar>(new ProgressBar(std::move(setting), out));
        s_progress_bars_.push_back(bar);
        bar->Update(0);
        bar->m_displayed_ = true;
        return bar;
    }

}  // namespace erl::common
