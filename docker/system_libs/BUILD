package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "k8",
    values = {"cpu": "k8"},
)

UNSUPPORTED_CPU_ERROR = "Unsupported target CPU"

cc_library(
    name = "gstreamer",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstreamer-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstreamer-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["usr/include/gstreamer-1.0"] + select(
        {
            ":aarch64": ["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include"],
            ":k8": ["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
    deps = [
        ":glib",
    ],
)

cc_library(
    name = "gstallocators",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstallocators-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstallocators-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/allocators/*.h",
        ],
    ),
    includes = ["usr/include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "gstvideo",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstvideo-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstvideo-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/video/*.h",
            "usr/include/gstreamer-1.0/gst/base/*.h",
        ],
    ),
    includes = ["usr/include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "gstaudio",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstaudio-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstaudio-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/audio/*.h",
            "usr/include/gstreamer-1.0/gst/base/*.h",
        ],
    ),
    includes = ["usr/include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "gstgl",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstgl-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstgl-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/gl/*.h",
	    "usr/include/gstreamer-1.0/gst/gl/glprototypes/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/gl/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/gl/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/gl/glprototypes/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/gl/glprototypes/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["usr/include/gstreamer-1.0"] + select(
        {
            ":aarch64": ["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include"],
            ":k8": ["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "gstpbutils",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstpbutils-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstpbutils-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/pbutils/*.h",
        ],
    ),
    includes = ["usr/include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)


cc_library(
    name = "x11",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libX11.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libX11.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/X11*.h",
        ],
    ),
    includes = ["usr/include/X11"],
    linkstatic = 0,
    deps = [
    ],
)

cc_library(
    name = "glib",
    srcs = select(
        {
            ":aarch64": glob([
                "usr/lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "usr/lib/aarch64-linux-gnu/libglib-2.0.so.0*",
                "lib/aarch64-linux-gnu/libglib-2.0.so.0*",
            ]),
            ":k8": glob([
                "usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "usr/lib/x86_64-linux-gnu/libglib-2.0.so.0*",
                "lib/x86_64-linux-gnu/libglib-2.0.so.0*",
            ]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/glib-2.0/*.h",
            "usr/include/glib-2.0/*/*.h",
            "usr/include/glib-2.0/glib/deprecated/*.h",
            "usr/include/glib-2.0/gobject/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/glib-2.0/include/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/glib-2.0/include/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["usr/include/glib-2.0"] + select(
        {
            ":aarch64": ["usr/lib/aarch64-linux-gnu/glib-2.0/include"],
            ":k8": ["usr/lib/x86_64-linux-gnu/glib-2.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
)
