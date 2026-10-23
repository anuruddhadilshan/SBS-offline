#include "SBSGEMMLHitFinder.h"

#include <iostream>
#include <vector>
#include <chrono>

// ============================================================
// Constructor
// ============================================================

SBSGEMMLHitFinder::SBSGEMMLHitFinder()
  : fEnv(
      ORT_LOGGING_LEVEL_WARNING,
      "SBSGEMMLHitFinder"
    ),
    fMemoryInfo(
      Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
      )
    )
{
  std::cout
    << "[SBSGEMMLHitFinder] "
    << "ONNX Runtime version: "
    << OrtGetApiBase()->GetVersionString()
    << " !!!!!!!    !!!!!!!!   !!!!!!!!!!!!!!!!!!    !!!!!!!!   !!!!!!!    !!!!!!!!  !!!!! !!!!!!!!!!!!!    !!!!!!!!!!!!!!!!       !!!!!!!!!!!!!!!!! "
    << std::endl;
}



// ============================================================
// Initialize
// ============================================================

bool SBSGEMMLHitFinder::Initialize(
    const std::string& model_path)
{
  fInitialized = false;

  try {

    // Start conservatively.
    // SBS replay may already use external parallelism.
    fSessionOptions.SetIntraOpNumThreads(1);
    fSessionOptions.SetInterOpNumThreads(1);

    fSessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL
    );


    // --------------------------------------------------------
    // Load ONNX model
    //
    // gem_hitfinder.onnx.data is automatically resolved
    // by ONNX Runtime if it is beside gem_hitfinder.onnx.
    // --------------------------------------------------------

    fSession =
      std::make_unique<Ort::Session>(
        fEnv,
        model_path.c_str(),
        fSessionOptions
      );


    // --------------------------------------------------------
    // Check number of inputs / outputs
    // --------------------------------------------------------

    const std::size_t n_inputs =
      fSession->GetInputCount();

    const std::size_t n_outputs =
      fSession->GetOutputCount();


    std::cout
      << "[SBSGEMMLHitFinder] Model loaded:"
      << std::endl
      << "  " << model_path
      << std::endl;

    std::cout
      << "[SBSGEMMLHitFinder] Number of inputs  = "
      << n_inputs
      << std::endl;

    std::cout
      << "[SBSGEMMLHitFinder] Number of outputs = "
      << n_outputs
      << std::endl;


    if (n_inputs != 2) {

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "expected 2 model inputs, got "
        << n_inputs
        << std::endl;

      fSession.reset();
      return false;
    }


    if (n_outputs != 1) {

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "expected 1 model output, got "
        << n_outputs
        << std::endl;

      fSession.reset();
      return false;
    }


    // --------------------------------------------------------
    // Obtain input/output names
    // --------------------------------------------------------

    Ort::AllocatorWithDefaultOptions allocator;


    auto input_name_0 =
      fSession->GetInputNameAllocated(
        0,
        allocator
      );

    auto input_name_1 =
      fSession->GetInputNameAllocated(
        1,
        allocator
      );

    auto output_name_0 =
      fSession->GetOutputNameAllocated(
        0,
        allocator
      );


    fInputNameX3D =
      input_name_0.get();

    fInputNameExtra =
      input_name_1.get();

    fOutputName =
      output_name_0.get();


    std::cout
      << "[SBSGEMMLHitFinder] Input 0 = "
      << fInputNameX3D
      << std::endl;

    std::cout
      << "[SBSGEMMLHitFinder] Input 1 = "
      << fInputNameExtra
      << std::endl;

    std::cout
      << "[SBSGEMMLHitFinder] Output  = "
      << fOutputName
      << std::endl;


    // --------------------------------------------------------
    // Verify names match exported model
    // --------------------------------------------------------

    if (
      fInputNameX3D != "x3d" ||
      fInputNameExtra != "extra"
    ) {

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "unexpected model input names."
        << std::endl;

      std::cerr
        << "  expected: x3d, extra"
        << std::endl;

      std::cerr
        << "  found: "
        << fInputNameX3D
        << ", "
        << fInputNameExtra
        << std::endl;

      fSession.reset();
      return false;
    }


    if (fOutputName != "logits") {

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "unexpected output name: "
        << fOutputName
        << std::endl;

      std::cerr
        << "  expected: logits"
        << std::endl;

      fSession.reset();
      return false;
    }


    // --------------------------------------------------------
    // Print model shapes
    // --------------------------------------------------------

    for (std::size_t i = 0; i < n_inputs; ++i) {

      auto type_info =
        fSession->GetInputTypeInfo(i);

      auto tensor_info =
        type_info.GetTensorTypeAndShapeInfo();

      auto shape =
        tensor_info.GetShape();


      std::cout
        << "[SBSGEMMLHitFinder] Input "
        << i
        << " shape = [";

      for (std::size_t j = 0; j < shape.size(); ++j) {

        if (j > 0)
          std::cout << ", ";

        std::cout << shape[j];
      }

      std::cout << "]"
                << std::endl;
    }


    {
      auto type_info =
        fSession->GetOutputTypeInfo(0);

      auto tensor_info =
        type_info.GetTensorTypeAndShapeInfo();

      auto shape =
        tensor_info.GetShape();


      std::cout
        << "[SBSGEMMLHitFinder] Output shape = [";

      for (std::size_t j = 0; j < shape.size(); ++j) {

        if (j > 0)
          std::cout << ", ";

        std::cout << shape[j];
      }

      std::cout << "]"
                << std::endl;
    }

  }
  catch (const Ort::Exception& e) {

    std::cerr
      << "[SBSGEMMLHitFinder] "
      << "ONNX Runtime initialization failed:"
      << std::endl
      << "  "
      << e.what()
      << std::endl;

    fSession.reset();

    return false;
  }


  fInitialized = true;

  std::cout
    << "[SBSGEMMLHitFinder] "
    << "Initialization successful."
    << std::endl;

  return true;
}


// ============================================================
// Run inference
// ============================================================
bool SBSGEMMLHitFinder::Run(
    const std::vector<float>& x3d,
    const std::vector<float>& extra,
    std::size_t H,
    std::size_t W,
    std::vector<float>& logits)
{
  logits.clear();

  if( !fInitialized || !fSession ){

    std::cerr
      << "[SBSGEMMLHitFinder] ERROR: "
      << "Run called before initialization."
      << std::endl;

    return false;
  }


  if( H == 0 || W == 0 ){

    std::cerr
      << "[SBSGEMMLHitFinder] ERROR: "
      << "Invalid input dimensions H="
      << H
      << ", W="
      << W
      << std::endl;

    return false;
  }


  // ==========================================================
  // Check ORIGINAL input sizes
  // ==========================================================

  const std::size_t expected_x3d =
    2 * 6 * H * W;

  const std::size_t expected_extra =
    14 * H * W;


  if( x3d.size() != expected_x3d ){

    std::cerr
      << "[SBSGEMMLHitFinder] ERROR: "
      << "x3d size mismatch. Got "
      << x3d.size()
      << ", expected "
      << expected_x3d
      << std::endl;

    return false;
  }


  if( extra.size() != expected_extra ){

    std::cerr
      << "[SBSGEMMLHitFinder] ERROR: "
      << "extra size mismatch. Got "
      << extra.size()
      << ", expected "
      << expected_extra
      << std::endl;

    return false;
  }


  // ==========================================================
  // IMPORTANT:
  //
  // Original PyTorch UNet pads H and W to multiples of 8
  // before the three downsampling stages.
  //
  // The exported ONNX graph is not correctly preserving this
  // dynamic padding operation, so reproduce it here.
  //
  // Padding is on the BOTTOM and RIGHT, matching:
  //
  // F.pad(x, (0, padW, 0, padH))
  // ==========================================================

  using MLClock = std::chrono::steady_clock;

  const auto run_t0 = MLClock::now();

  double padding_ms = 0.0;
  double copies_ms = 0.0;
  double tensor_creation_ms = 0.0;
  double ort_ms = 0.0;
  double output_crop_ms = 0.0;


  // Start timing padding/allocation
  const auto padding_t0 = MLClock::now();

  // constexpr std::size_t FACTOR = 8;

  // const std::size_t Hpad =
  //   ((H + FACTOR - 1) / FACTOR) * FACTOR;

  // const std::size_t Wpad =
  //   ((W + FACTOR - 1) / FACTOR) * FACTOR;
  // adding flag to add padding to 8 for speedup tests - Bhasitha speedup tests
  constexpr std::size_t FACTOR = 8;

  std::size_t Hrun = H;
  std::size_t Wrun = W;

  if( fRequiresPaddingTo8 ){

    Hrun =
    ((H + FACTOR - 1) / FACTOR) * FACTOR;

    Wrun =
    ((W + FACTOR - 1) / FACTOR) * FACTOR;
  }


  const std::size_t padH =
      Hrun - H;

  const std::size_t padW =
      Wrun - W;

  // ==========================================================
  // Create zero-padded x3d:
  //
  // original shape:
  //   [2,6,H,W]
  //
  // padded shape:
  //   [2,6,Hpad,Wpad]
  // ==========================================================

  // std::vector<float> x3d_padded(
  //   2 * 6 * Hpad * Wpad,
  //   0.0f
  // );
  // std::vector<float> extra_padded(
  //   14 * Hpad * Wpad,
  //   0.0f
  // );
  // adding flag to add padding to 8 for speedup tests - Bhasitha speedup tests
const std::vector<float>* x3d_run  = &x3d;
const std::vector<float>* extra_run = &extra;

std::vector<float> x3d_padded;
std::vector<float> extra_padded;


// ==========================================================
// Only create/copy padded tensors if this model requires them
// ==========================================================

const auto copies_t0 = MLClock::now();

if( Hrun != H || Wrun != W ){

  x3d_padded.assign(
      2ULL * 6ULL * Hrun * Wrun,
    0.0f
  );

  extra_padded.assign(
      14ULL * Hrun * Wrun,
      0.0f
  );


  // ========================================================
  // Copy x3d:
  // [2,6,H,W] -> [2,6,Hrun,Wrun]
  // ========================================================

  for( std::size_t c = 0; c < 2; ++c ){

    for( std::size_t t = 0; t < 6; ++t ){

      for( std::size_t y = 0; y < H; ++y ){

        for( std::size_t x = 0; x < W; ++x ){

          const std::size_t src =
            (((c * 6 + t) * H + y) * W + x);

          const std::size_t dst =
              (((c * 6 + t) * Hrun + y) * Wrun + x);

          x3d_padded[dst] =
            x3d[src];
        }
      }
    }
  }


  // ========================================================
  // Copy extra:
  // [14,H,W] -> [14,Hrun,Wrun]
  // ========================================================

  for( std::size_t c = 0; c < 14; ++c ){

    for( std::size_t y = 0; y < H; ++y ){

      for( std::size_t x = 0; x < W; ++x ){

        const std::size_t src =
          ((c * H + y) * W + x);

        const std::size_t dst =
            ((c * Hrun + y) * Wrun + x);

        extra_padded[dst] =
          extra[src];
      }
    }
  }


  x3d_run = &x3d_padded;
  extra_run = &extra_padded;
}

const auto copies_t1 = MLClock::now();

copies_ms =
    std::chrono::duration<double, std::milli>(
        copies_t1 - copies_t0
    ).count();

  // ==========================================================
  // ONNX tensor shapes
  // ==========================================================

  // flag to add padding to 8 for speedup tests - Bhasitha speedup tests
  // const std::vector<int64_t> x3d_shape = {
  //   1,
  //   2,
  //   6,
  //   static_cast<int64_t>(Hpad),
  //   static_cast<int64_t>(Wpad)
  // };
  std::array<int64_t,5> x3d_shape = {
    1,
    2,
    6,
      static_cast<int64_t>(Hrun),
      static_cast<int64_t>(Wrun)
  };

  // flag to add padding to 8 for speedup tests - Bhasitha speedup tests
  // const std::vector<int64_t> extra_shape = {
  //   1,
  //   14,
  //   static_cast<int64_t>(Hpad),
  //   static_cast<int64_t>(Wpad)
  // };
  std::array<int64_t,4> extra_shape = {
    1,
    14,
      static_cast<int64_t>(Hrun),
      static_cast<int64_t>(Wrun)
  };

  try {

    // ========================================================
    // Create ONNX input tensors
    // ========================================================

    const auto tensor_t0 = MLClock::now();

    // Ort::Value x3d_tensor =
    //   Ort::Value::CreateTensor<float>(
    //     fMemoryInfo,
    //     x3d_padded.data(),
    //     x3d_padded.size(),
    //     x3d_shape.data(),
    //     x3d_shape.size()
    //   );


    // Ort::Value extra_tensor =
    //   Ort::Value::CreateTensor<float>(
    //     fMemoryInfo,
    //     extra_padded.data(),
    //     extra_padded.size(),
    //     extra_shape.data(),
    //     extra_shape.size()
    //   );
    // flag to add padding to 8 for speedup tests - Bhasitha speedup tests
    Ort::Value x3d_tensor =
      Ort::Value::CreateTensor<float>(
        fMemoryInfo,
            const_cast<float*>(x3d_run->data()),
            x3d_run->size(),
        x3d_shape.data(),
        x3d_shape.size()
      );

    Ort::Value extra_tensor =
      Ort::Value::CreateTensor<float>(
        fMemoryInfo,
            const_cast<float*>(extra_run->data()),
            extra_run->size(),
        extra_shape.data(),
        extra_shape.size()
      );


    std::vector<Ort::Value> input_tensors;

    input_tensors.emplace_back(
      std::move(x3d_tensor)
    );

    input_tensors.emplace_back(
      std::move(extra_tensor)
    );


    const char* input_names[] = {
      fInputNameX3D.c_str(),
      fInputNameExtra.c_str()
    };


    const char* output_names[] = {
      fOutputName.c_str()
    };

    const auto tensor_t1 = MLClock::now();

    tensor_creation_ms =
      std::chrono::duration<double, std::milli>(
        tensor_t1 - tensor_t0
      ).count();

    // ========================================================
    // Run ONNX
    // ========================================================

    const auto ort_t0 = MLClock::now();

    auto output_tensors =
      fSession->Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        1
      );


      const auto ort_t1 = MLClock::now();

      ort_ms =
        std::chrono::duration<double, std::milli>(
          ort_t1 - ort_t0
        ).count();



    if( output_tensors.empty() ){

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "ONNX Runtime returned no output."
        << std::endl;

      return false;
    }


    if( !output_tensors[0].IsTensor() ){

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "ONNX output is not a tensor."
        << std::endl;

      return false;
    }


    // ========================================================
    // Inspect padded output
    // ========================================================

    const auto crop_t0 = MLClock::now();

    auto info =
      output_tensors[0]
        .GetTensorTypeAndShapeInfo();


    const std::size_t output_count =
      info.GetElementCount();


    // flag to add padding to 8 for speedup tests - Bhasitha speedup tests
    // const std::size_t expected_output_count =
    //   Hpad * Wpad;
    const std::size_t expected_output_count =
        Hrun * Wrun;

    if( output_count != expected_output_count ){

      std::cerr
        << "[SBSGEMMLHitFinder] ERROR: "
        << "output size mismatch. Got "
        << output_count
        << ", expected inference size "
        << expected_output_count
        << " ("
        << Hrun
        << " x "
        << Wrun
        << ")"
        << std::endl;

      return false;
    }


    const float* output_data =
      output_tensors[0]
        .GetTensorData<float>();


    // ========================================================
    // Crop output back to ORIGINAL H x W.
    //
    // Equivalent to Python:
    //
    // logits = logits[..., :H0, :W0]
    // ========================================================

    logits.resize(
      H * W
    );


    // for( std::size_t y = 0; y < H; ++y ){

    //   for( std::size_t x = 0; x < W; ++x ){

    //     const std::size_t src =
    //       y * Wpad + x;


    //     const std::size_t dst =
    //       y * W + x;


    //     logits[dst] =
    //       output_data[src];
    //   }
    // }
    // flag to add padding to 8 for speedup tests - Bhasitha speedup tests
    // ============================================================
    // Copy ONNX output back to original H x W
    //
    // Fast model:
    //   Hrun == H and Wrun == W
    //   -> output is already exactly the required size.
    //
    // Old U-Net:
    //   Hrun/Wrun may be padded
    //   -> crop padded output back to H x W.
    // ============================================================

    if( Hrun == H && Wrun == W ){

      // Fast model: no padding was used.
      // ONNX output is already [1,1,H,W].
      logits.assign(
          output_data,
          output_data + H * W
      );

    }
    else{

      // Old U-Net: ONNX output is [1,1,Hrun,Wrun].
      // Crop away the bottom/right padding.

      logits.resize(H * W);

      for( std::size_t y = 0; y < H; y++ ){

        for( std::size_t x = 0; x < W; x++ ){

        const std::size_t src =
              y * Wrun + x;

        const std::size_t dst =
          y * W + x;

          logits[dst] = output_data[src];
      }
    }
    }

    const auto crop_t1 = MLClock::now();

    output_crop_ms =
      std::chrono::duration<double, std::milli>(
        crop_t1 - crop_t0
      ).count();

    const auto run_t1 = MLClock::now();

    // ========================================================
    // Temporary diagnostic
    // ========================================================

    static bool printed_once = false;

    if( !printed_once ){

      std::cout
        << "[SBSGEMMLHitFinder] ONNX padding:"
        << std::endl;

      std::cout
        << "  original H,W = "
        << H
        << ", "
        << W
        << std::endl;

      std::cout
        << "  inference H,W = "
        << Hrun
        << ", "
        << Wrun
        << std::endl;

      std::cout
        << "  padH,padW = "
        << padH
        << ", "
        << padW
        << std::endl;

      std::cout
        << "  cropped output elements = "
        << logits.size()
        << std::endl;


      printed_once = true;
    }


    

    const double run_total_ms =
      std::chrono::duration<double, std::milli>(
        run_t1 - run_t0
      ).count();

    const double accounted_ms =
        padding_ms
      + copies_ms
      + tensor_creation_ms
      + ort_ms
      + output_crop_ms;

    const double other_ms =
      run_total_ms - accounted_ms;


    static unsigned long timing_count = 0;
    timing_count++;

    // if(
    //   timing_count <= 40 ||
    //   timing_count % 100 == 0
    // ){

    //   std::cout
    //     << "\n[ML HITFINDER TIMING]"
    //     << " count=" << timing_count
    //     << "\n"
    //     << "  input:"
    //     << " H=" << H
    //     << " W=" << W
    //     << " Hpad=" << Hpad
    //     << " Wpad=" << Wpad
    //     << "\n"
    //     << "  timing_ms:"
    //     << " padding=" << padding_ms
    //     << " copies=" << copies_ms
    //     << " tensor=" << tensor_creation_ms
    //     << " ORT=" << ort_ms
    //     << " crop=" << output_crop_ms
    //     << "\n"
    //     << "  accounted=" << accounted_ms
    //     << " other=" << other_ms
    //     << " Run_total=" << run_total_ms
    //     << "\n"
    //     << "  ORT/Run="
    //     << (
    //         run_total_ms > 0.0
    //         ? 100.0 * ort_ms / run_total_ms
    //         : 0.0
    //       )
    //     << "%"
    //     << std::endl;
    // }


    return true;
  }


  catch( const Ort::Exception& e ){

    std::cerr
      << "[SBSGEMMLHitFinder] ONNX inference failed:"
      << std::endl
      << "  "
      << e.what()
      << std::endl;

    return false;
  }
}