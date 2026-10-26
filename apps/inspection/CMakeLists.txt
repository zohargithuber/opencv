ocv_add_application(opencv_inspection
    MODULES opencv_core opencv_imgproc opencv_geometry opencv_imgcodecs opencv_dnn
    SRCS main.cpp inspection.cpp missing_part.cpp scratch_diff.cpp scratch_dnn.cpp dual_inspector.cpp stripe_pool.cpp)
