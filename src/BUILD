cc_library(
    name = "Bin",
    srcs = ["Bin.cpp"],
    hdrs = ["Bin.h"],
    deps = [
            "@system_libs//:gstreamer",
    ],
)

cc_library(
    name = "MixerBin",
    srcs = ["MixerBin.cpp"],
    hdrs = ["MixerBin.h"],
    deps = [
            ":Bin",
	    ":InferencerBin",
	    ":InferencerBase",
	    ":TwoModelInferencerBin",
            "@system_libs//:gstreamer",
    ],
)

cc_library(
    name = "InferencerBin",
    srcs = ["InferencerBin.cpp"],
    hdrs = ["InferencerBin.h", "MixerBin.h"],
    deps = [
            ":Bin",
	    ":InferencerBase",
	    ":DetectionInferencer",
	    ":ManufacturingInferencer",
	    ":PipelinedInferencer",
	    ":Utility",
            "@com_google_absl//absl/strings:strings",
            "@system_libs//:gstreamer",
            "@system_libs//:gstallocators",
            "@system_libs//:gstgl",
            "@system_libs//:gstaudio",
            "@system_libs//:gstvideo",
            "@system_libs//:gstpbutils",
    ],
)

cc_library(
    name = "TwoModelInferencerBin",
    srcs = ["TwoModelInferencerBin.cpp"],
    hdrs = ["TwoModelInferencerBin.h"],
    deps = [
	    ":ClassificationInferencer",
            ":InferencerBin",
	    ":InferencerBase",
            "@com_google_absl//absl/strings:strings",
            "@system_libs//:gstreamer",
            "@system_libs//:gstvideo",
    ],
)

cc_library(
    name = "Utility",
    srcs = ["Utility.cpp"],
    hdrs = ["Utility.h"],
    deps = [
    ],
)


cc_library(
    name = "Pipeline",
    srcs = ["Pipeline.cpp"],
    hdrs = ["Pipeline.h"],
    deps = [
            ":ClassificationInferencer",
            ":DetectionInferencer",
            ":InferencerBase",
	    ":InferencerBin",
	    ":MixerBin",
            ":ManufacturingInferencer",
            ":PipelinedInferencer",
	    ":SegmentationInferencer",
	    ":TwoModelInferencerBin",
            "@system_libs//:gstreamer",
	    "@system_libs//:x11",
    ],
)

cc_library(
    name = "ManufacturingInferencer",
    srcs = ["ManufacturingInferencer.cpp"],
    hdrs = ["ManufacturingInferencer.h"],
    deps = [
    	    ":DetectionInferencer",
	    ":Utility",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
	    "@org_tensorflow//tensorflow/lite:builtin_op_data",
	    "@org_tensorflow//tensorflow/lite:framework",
	    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "SegmentationInferencer",
    srcs = ["SegmentationInferencer.cpp"],
    hdrs = ["SegmentationInferencer.h"],
    deps = [
    	    ":InferencerBase",
	    "@libcoral//coral:tflite_utils",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
	    "@org_tensorflow//tensorflow/lite:builtin_op_data",
	    "@org_tensorflow//tensorflow/lite:framework",
	    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "DetectionInferencer",
    srcs = ["DetectionInferencer.cpp"],
    hdrs = ["DetectionInferencer.h"],
    deps = [
    	    ":InferencerBase",
	    "@libcoral//coral:error_reporter",
	    "@libcoral//coral:tflite_utils",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
	    "@org_tensorflow//tensorflow/lite:builtin_op_data",
	    "@org_tensorflow//tensorflow/lite:framework",
	    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "ClassificationInferencer",
    srcs = ["ClassificationInferencer.cpp"],
    hdrs = ["ClassificationInferencer.h"],
    deps = [
    	    ":InferencerBase",
	    "@libcoral//coral/classification:adapter",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
	    "@org_tensorflow//tensorflow/lite:builtin_op_data",
	    "@org_tensorflow//tensorflow/lite:framework",
	    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "PipelinedInferencer",
    srcs = ["PipelinedInferencer.cpp"],
    hdrs = ["PipelinedInferencer.h"],
    deps = [
    	    ":DetectionInferencer",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
	    "@libcoral//coral:error_reporter",
            "@libcoral//coral/pipeline:pipelined_model_runner",
            "@libcoral//coral:tflite_utils",
	    "@org_tensorflow//tensorflow/lite:builtin_op_data",
	    "@org_tensorflow//tensorflow/lite:framework",
	    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "InferencerBase",
    srcs = ["InferencerBase.cpp"],
    hdrs = ["InferencerBase.h"],
    deps = [
        ":Utility",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
        "@libcoral//coral:error_reporter",
        "@libcoral//coral/pipeline:pipelined_model_runner",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "MultiVideoStreamsDemo",
    srcs = ["MultiVideoStreamsDemo.cpp"],
    deps = [
        ":Pipeline",
    ],
)
