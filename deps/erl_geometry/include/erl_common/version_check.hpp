#pragma once

#define ERL_CHECK_VERSION_GE(major, minor, patch, major_min, minor_min, patch_min) \
    ((major) > (major_min) ||                                                      \
     ((major) == (major_min) &&                                                    \
      ((minor) > (minor_min) || ((minor) == (minor_min) && (patch) >= (patch_min)))))

#define ERL_CHECK_VERSION_LE(major, minor, patch, major_max, minor_max, patch_max) \
    ((major) < (major_max) ||                                                      \
     ((major) == (major_max) &&                                                    \
      ((minor) < (minor_max) || ((minor) == (minor_max) && (patch) <= (patch_max)))))

#define ERL_CHECK_VERSION_EQ(major, minor, patch, major_eq, minor_eq, patch_eq) \
    ((major) == (major_eq) && (minor) == (minor_eq) && (patch) == (patch_eq))
