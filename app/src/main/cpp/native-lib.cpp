#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>



extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_tflite_MainActivity_loadModelJNItest(JNIEnv *env, jobject thiz,
                                                      jobject asset_manager, jstring file_name,
                                                      jlong native_obj_addr) {
    char* buffer = nullptr;
    long size = 0;
    const char* modelpath = env->GetStringUTFChars(file_name, 0);

    if (!(env->IsSameObject(asset_manager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
        AAsset *asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromBuffer(buffer, size);
    assert(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    assert(interpreter != nullptr);

    // Allocate tensor buffers.
    assert(interpreter->AllocateTensors() == kTfLiteOk);

    //IMPUT IMAGE
    cv::Mat& mRgb = *(cv::Mat*)native_obj_addr;

    cv::cvtColor(mRgb, mRgb, cv::COLOR_RGBA2RGB);
    cv::resize(mRgb, mRgb, cv::Size(256,256), cv::INTER_LINEAR);

    mRgb.convertTo(mRgb, CV_32FC3);


    float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

    memcpy(input, mRgb.data,
           sizeof(float) * 256 * 256 * 3);


    //OUTPUTT

    float* output = interpreter->typed_output_tensor<float>(0);


    memcpy(mRgb.data, output,
           sizeof(float) * 256 * 256 * 1);

    mRgb = mRgb / 16;
    mRgb.convertTo(mRgb, CV_8UC4);

    cv::resize(mRgb, mRgb, cv::Size(640,480), cv::INTER_LINEAR);


    std::string status = "Load TF Lite model successfully!";
    free(buffer);

    return env->NewStringUTF(status.c_str());

}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_tflite_MainActivity_MidasInference(JNIEnv *env, jobject thiz, jlong mat_addr_gr) {
    cv::Mat& mGr  = *(cv::Mat*)mat_addr_gr;
    cv::cvtColor(mGr, mGr, cv::COLOR_RGBA2GRAY);
}