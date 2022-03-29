#include <iostream>
#include <CL/cl.hpp>
#include "time.h"

void default_example();

cl::Platform getPlatform();

cl::Device getDevice(const cl::Platform &default_platform);

const std::string matmul_kernel = "__kernel void matmul(__global float* A,\n"
                                  "                     __global float* B,\n"
                                  "                     __global float* C,\n"
                                  "                     int N) {\n"
                                  "    int i = get_global_id(0);\n"
                                  "    for (size_t j = 0; j < N; ++j) {\n"
                                  "        float acc = 0.0f;\n"
                                  "        for (size_t k = 0; k < N; ++k) {\n"
                                  "            acc += A[i*N + k] * B[k*N + j];\n"
                                  "        }\n"
                                  "        C[i*N + j] = acc;\n"
                                  "    }\n"
                                  "}";

int main() {
    cl::Platform default_platform = getPlatform();
    cl::Device default_device = getDevice(default_platform);
    cl::Context context({default_device});
    cl::Program::Sources sources;
    sources.push_back({matmul_kernel.c_str(), matmul_kernel.length()});
    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    const int n = 256;
    constexpr int n2 = n * n;

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * n2);
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float) * n2);
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * n2);

    float *A = (float *) malloc(n2 * sizeof(float));
    float *B = (float *) malloc(n2 * sizeof(float));
    srand(42);
    for (size_t i = 0; i != n2; ++i) {
        A[i] = (rand() % 10) - 10;
        B[i] = (rand() % 10) - 10;
    }

    cl::CommandQueue queue(context, default_device, CL_QUEUE_PROFILING_ENABLE);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * n2, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * n2, B);
    cl::Kernel kernel = cl::Kernel(program, "matmul");
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, n);
//    const int TS = ;
//    auto local = cl::NDRange(TS, TS);
    cl::Event event;
    auto global = cl::NDRange(n, n);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    clock_t start = clock();
    event.wait();
    queue.finish();
    clock_t end = clock();
    std::cout << "Overall time:" << (end - start) * 1.0 / CLOCKS_PER_SEC << std::endl;

    auto elapsed = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
               event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Kernel time:" << elapsed / pow(10,9) << std::endl;


    float *C = (float *) malloc(n2 * sizeof(float));

    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * n2, C);

    std::cout << " result: \n";
    std::cout << C[0] << " ";
    return 0;
}

void default_example() {
    cl::Platform default_platform = getPlatform();
    cl::Device default_device = getDevice(default_platform);
    cl::Context context({default_device});
    cl::Program::Sources sources;
    // kernel calculates for each element C=A+B
    std::string kernel_code =
            "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
            "   }                                                                               ";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }


    // create buffers on the device
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);


    //run the kernel
//    cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10),
//                                 cl::NullRange);
//    simple_add(buffer_A, buffer_B, buffer_C);

    //alternative way to run the kernel
    cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
    kernel_add.setArg(0, buffer_A);
    kernel_add.setArg(1, buffer_B);
    kernel_add.setArg(2, buffer_C);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
    queue.finish();

    int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

    std::cout << " result: \n";
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
}

cl::Device getDevice(const cl::Platform &default_platform) {
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    return default_device;
}

cl::Platform getPlatform() {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    return default_platform;
}
