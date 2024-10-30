#ifndef LOGGINGH
#define LOGGINGH

#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvrtc.h>
#include <regex>  // NOLINT(*)
#include <string>
#include <type_traits>


/*! \brief Substitute regex occurances in string
 *
 * This is a convenience wrapper around std::regex_replace.
 */
template <typename T>
inline std::string regex_replace(const std::string &str,
                                 const std::string &pattern,
                                 const T &replacement) {
  return std::regex_replace(str,
                            std::regex(pattern),
                            to_string_like(replacement));
}

/*! \brief Convert to C-style or C++-style string */
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string to_string_like(const T &val) {
  return std::to_string(val);
}

inline const std::string& to_string_like(const std::string& val) noexcept {
  return val;
}

constexpr const char *to_string_like(const char *val) noexcept {
  return val;
}

/*! \brief Convert arguments to strings and concatenate */
template <typename... Ts>
inline std::string concat_strings(const Ts &... args) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  (..., (str += to_string_like(args)));
  return str;
}


#define DHELLAM_ERROR(...)                                         \
  do {                                                          \
    throw ::std::runtime_error(                                 \
      ::concat_strings(                     \
        __FILE__ ":", __LINE__,                                 \
        " in function ", __func__, ": ",                        \
        ::concat_strings(__VA_ARGS__)));    \
  } while (false)

#define DHELLAM_CHECK(expr, ...)                                           \
  do {                                                                  \
    if (!(expr)) {                                                      \
      DHELLAM_ERROR("Assertion failed: " #expr ". ",                       \
                 ::concat_strings(__VA_ARGS__));    \
    }                                                                   \
  } while (false)

#define DHELLAM_CHECK_CUDA(expr)                                           \
  do {                                                                  \
    const cudaError_t status_DHELLAM_CHECK_CUDA = (expr);                  \
    if (status_DHELLAM_CHECK_CUDA != cudaSuccess) {                        \
      DHELLAM_ERROR("CUDA Error: ",                                        \
                 cudaGetErrorString(status_DHELLAM_CHECK_CUDA));           \
    }                                                                   \
  } while (false)

#define DHELLAM_CHECK_CUBLAS(expr)                                         \
  do {                                                                  \
    const cublasStatus_t status_DHELLAM_CHECK_CUBLAS = (expr);             \
    if (status_DHELLAM_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS) {            \
      DHELLAM_ERROR("cuBLAS Error: ",                                      \
                 cublasGetStatusString(status_DHELLAM_CHECK_CUBLAS));      \
    }                                                                   \
  } while (false)

#define DHELLAM_CHECK_CUDNN(expr)                                          \
  do {                                                                  \
    const cudnnStatus_t status_DHELLAM_CHECK_CUDNN = (expr);               \
    if (status_DHELLAM_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {              \
      DHELLAM_ERROR("cuDNN Error: ",                                       \
                 cudnnGetErrorString(status_DHELLAM_CHECK_CUDNN),          \
                 ". "                                                   \
                 "For more information, enable cuDNN error logging "    \
                 "by setting CUDNN_LOGERR_DBG=1 and "                   \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");       \
    }                                                                   \
  } while (false)

#define DHELLAM_CHECK_CUDNN_FE(expr)                                       \
  do {                                                                  \
    const auto error = (expr);                                          \
    if (error.is_bad()) {                                               \
      DHELLAM_ERROR("cuDNN Error: ",                                       \
                 error.err_msg,                                         \
                 ". "                                                   \
                 "For more information, enable cuDNN error logging "    \
                 "by setting CUDNN_LOGERR_DBG=1 and "                   \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");       \
    }                                                                   \
  } while (false)

#define DHELLAM_CHECK_NVRTC(expr)                                  \
  do {                                                          \
    const nvrtcResult status_DHELLAM_CHECK_NVRTC = (expr);         \
    if (status_DHELLAM_CHECK_NVRTC != NVRTC_SUCCESS) {             \
      DHELLAM_ERROR("NVRTC Error: ",                               \
                 nvrtcGetErrorString(status_DHELLAM_CHECK_NVRTC)); \
    }                                                           \
  } while (false)

#endif