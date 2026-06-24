// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Minimal dynamic OpenCL loader (no OpenCL.lib, no SDK needed to build).    ║
// ║                                                                            ║
// ║  Loads OpenCL.dll (Windows) / libOpenCL.so.1 (Linux) at runtime and binds  ║
// ║  the ~22 core functions we use. Benefits:                                  ║
// ║    * builds with ONLY the OpenCL HEADERS (CUDA ships CL/cl.h; Linux:       ║
// ║      apt install opencl-headers) — no import library required;             ║
// ║    * the resulting exe has NO link-time OpenCL dependency, so it runs on   ║
// ║      any machine that has a GPU driver (the ICD loader ships with it).     ║
// ║                                                                            ║
// ║  Function pointer TYPES are taken from cl.h via decltype(&clXxx), so the   ║
// ║  signatures can never drift from the header. After loading, the trailing   ║
// ║  #defines remap clXxx -> the loaded pointer, so calling code reads like    ║
// ║  ordinary OpenCL.                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <cstdio>

#ifdef _WIN32
  #include <windows.h>
  static inline void* bogo_cl_dlopen(){ return (void*)LoadLibraryA("OpenCL.dll"); }
  static inline void* bogo_cl_sym(void* h, const char* n){ return (void*)GetProcAddress((HMODULE)h, n); }
#else
  #include <dlfcn.h>
  static inline void* bogo_cl_dlopen(){
      void* h = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_GLOBAL);
      if (!h) h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
      return h;
  }
  static inline void* bogo_cl_sym(void* h, const char* n){ return dlsym(h, n); }
#endif

// The core functions we bind.
#define BOGO_CL_FUNCS \
  X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) \
  X(clCreateContext) X(clCreateCommandQueue) X(clCreateProgramWithSource) \
  X(clBuildProgram) X(clGetProgramBuildInfo) X(clCreateKernel) X(clCreateBuffer) \
  X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueReadBuffer) \
  X(clEnqueueWriteBuffer) X(clFinish) X(clReleaseMemObject) X(clReleaseKernel) \
  X(clReleaseProgram) X(clReleaseCommandQueue) X(clReleaseContext) \
  X(clGetKernelWorkGroupInfo)

// Declare a pointer per function, typed straight from the header.
#define X(name) static decltype(&name) p_##name = nullptr;
BOGO_CL_FUNCS
#undef X

// Load OpenCL once. Returns false (with a printed reason) if the runtime is
// missing or too old to expose one of the core entry points.
static inline bool bogo_cl_load(){
    static int state = 0;            // 0 = untried, 1 = ok, -1 = failed
    if (state == 1) return true;
    if (state == -1) return false;
    void* h = bogo_cl_dlopen();
    if (!h){
        fprintf(stderr, "[ERROR] OpenCL runtime not found (OpenCL.dll / libOpenCL.so). "
                        "Install your GPU driver (it ships the OpenCL runtime).\n");
        state = -1; return false;
    }
    #define X(name) \
        p_##name = (decltype(&name))bogo_cl_sym(h, #name); \
        if (!p_##name){ fprintf(stderr, "[ERROR] OpenCL is missing %s (driver too old?)\n", #name); state=-1; return false; }
    BOGO_CL_FUNCS
    #undef X
    state = 1; return true;
}

// Make calling code read like ordinary OpenCL.
#define clGetPlatformIDs           p_clGetPlatformIDs
#define clGetPlatformInfo          p_clGetPlatformInfo
#define clGetDeviceIDs             p_clGetDeviceIDs
#define clGetDeviceInfo            p_clGetDeviceInfo
#define clCreateContext            p_clCreateContext
#define clCreateCommandQueue       p_clCreateCommandQueue
#define clCreateProgramWithSource  p_clCreateProgramWithSource
#define clBuildProgram             p_clBuildProgram
#define clGetProgramBuildInfo      p_clGetProgramBuildInfo
#define clCreateKernel             p_clCreateKernel
#define clCreateBuffer             p_clCreateBuffer
#define clSetKernelArg             p_clSetKernelArg
#define clEnqueueNDRangeKernel     p_clEnqueueNDRangeKernel
#define clEnqueueReadBuffer        p_clEnqueueReadBuffer
#define clEnqueueWriteBuffer       p_clEnqueueWriteBuffer
#define clFinish                   p_clFinish
#define clReleaseMemObject         p_clReleaseMemObject
#define clReleaseKernel            p_clReleaseKernel
#define clReleaseProgram           p_clReleaseProgram
#define clReleaseCommandQueue      p_clReleaseCommandQueue
#define clReleaseContext           p_clReleaseContext
#define clGetKernelWorkGroupInfo   p_clGetKernelWorkGroupInfo
