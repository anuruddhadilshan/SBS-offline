#ifndef SBSGEMMLHITFINDER_H
#define SBSGEMMLHITFINDER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>


class SBSGEMMLHitFinder {

public:

  SBSGEMMLHitFinder();
  ~SBSGEMMLHitFinder() = default;

  // Load model once
  bool Initialize(const std::string& model_path);

  bool IsInitialized() const {
    return fInitialized;
  }

  // ----------------------------------------------------------
  // Run inference
  //
  // x3d:
  //   shape [1, 2, 6, H, W]
  //   vector size = 2 * 6 * H * W
  //
  // extra:
  //   shape [1, 14, H, W]
  //   vector size = 14 * H * W
  //
  // logits:
  //   output shape [1, 1, H, W]
  //   vector size = H * W
  // ----------------------------------------------------------

  bool Run(
      const std::vector<float>& x3d,
      const std::vector<float>& extra,
      std::size_t H,
      std::size_t W,
      std::vector<float>& logits);

  const std::string& GetInputNameX3D() const {
    return fInputNameX3D;
  }

  const std::string& GetInputNameExtra() const {
    return fInputNameExtra;
  }

  const std::string& GetOutputName() const {
    return fOutputName;
  }

private:

  bool fInitialized = false;

  Ort::Env fEnv;
  Ort::SessionOptions fSessionOptions;

  std::unique_ptr<Ort::Session> fSession;

  Ort::MemoryInfo fMemoryInfo;

  std::string fInputNameX3D;
  std::string fInputNameExtra;
  std::string fOutputName;
};

#endif