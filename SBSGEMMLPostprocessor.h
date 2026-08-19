#ifndef SBSGEMMLPOSTPROCESSOR_H
#define SBSGEMMLPOSTPROCESSOR_H

#include "SBSGEMMLPreprocessor.h"

#include <vector>


// ============================================================
// One predicted ML blob
//
// Naming matches the Python output:
//   x -> U
//   y -> V
// ============================================================

struct SBSGEMMLBlob {

  int blob_id = -1;

  // Centroid in image coordinates
  double cy = 0.0;
  double cx = 0.0;

  // Final snapped image indices
  int iy = -1;
  int ix = -1;

  // Physical GEM strip IDs
  int y_strip = -1;  // V
  int x_strip = -1;  // U

  // Number of predicted pixels in connected component
  int area = 0;

  // Number of real physical pixels used for centroid
  int real_area = 0;
};


// ============================================================
// Full postprocessing result
//
// Keeping these arrays temporarily will make Python/C++
// postprocessing parity testing straightforward.
// ============================================================

struct SBSGEMMLPostprocessResult {

  // sigmoid(logits)
  // Shape [H,W]
  std::vector<float> prob;

  // prob * valid_mask
  // Shape [H,W]
  std::vector<float> prob_valid;

  // prob_valid > threshold
  // Shape [H,W]
  std::vector<unsigned char> pred_mask;

  // Connected predicted blobs
  std::vector<SBSGEMMLBlob> blobs;
};


class SBSGEMMLPostprocessor {

public:

  SBSGEMMLPostprocessor() = default;
  ~SBSGEMMLPostprocessor() = default;


  bool Process(
      const std::vector<float>& logits,
      const SBSGEMMLInput& input,
      float threshold,
      SBSGEMMLPostprocessResult& output
  ) const;
};


#endif