#ifndef SBSGEMMLPREPROCESSOR_H
#define SBSGEMMLPREPROCESSOR_H

#include <array>
#include <cstddef>
#include <vector>

// One decoded physical GEM strip before ML preprocessing.
// ADC samples are the regular SBS pedestal/common-mode processed
// fADCsamples values, but NOT ML-normalized yet.
struct SBSGEMMLStrip {
  int strip_id = -1;
  std::array<float,6> adc{{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}};
};


// Complete input needed by the ONNX model.
struct SBSGEMMLInput {

  // Shape [2,6,H,W]
  std::vector<float> x3d;

  // Shape [14,H,W]
  std::vector<float> extra;

  // Image-column -> physical U strip ID
  // Negative IDs are fake spacer columns.
  std::vector<int> x_ids;

  // Image-row -> physical V strip ID
  // Negative IDs are fake spacer rows.
  std::vector<int> y_ids;

  // Shape [H,W].
  // 1 only where both x_ids[x] and y_ids[y] are real.
  std::vector<unsigned char> valid_mask;

  std::size_t H = 0;
  std::size_t W = 0;
};


class SBSGEMMLPreprocessor {

public:

  SBSGEMMLPreprocessor() = default;
  ~SBSGEMMLPreprocessor() = default;

  bool Build(
      const std::vector<SBSGEMMLStrip>& u_strips,
      const std::vector<SBSGEMMLStrip>& v_strips,
      SBSGEMMLInput& output
  ) const;
};

#endif