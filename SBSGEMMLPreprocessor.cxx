#include "SBSGEMMLPreprocessor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace {

constexpr int T = 6;

constexpr float ADC_SCALE     = 3539.0f;
constexpr float ADC_CLIP_MAX  = 1.5f;

constexpr float COMPACT_MIN_Q = 0.00001f;

constexpr int GAP_MIN   = 3;
constexpr int N_SPACER  = 3;

constexpr float THR_WIDTH    = 0.11f;
constexpr float QTHR_CLUSTER = 0.15f;

constexpr int   CLIP_CLUST  = 8;
constexpr float QRATIO_CLIP = 10.0f;

constexpr float EPS = 1.0e-6f;

using Waveform = std::array<float,T>;


struct NormalizedStrip {
  Waveform adc{{0,0,0,0,0,0}};
  float qsum = 0.0f;
};


// ============================================================
// Normalize exactly like Python:
//
// adc / 3539
// clip [0,1.5]
//
// Then apply min_q and duplicate-strip max-q behavior.
// ============================================================

std::map<int,NormalizedStrip>
NormalizeAndDeduplicate(
    const std::vector<SBSGEMMLStrip>& input
){
  std::map<int,NormalizedStrip> result;

  for( const auto& strip : input ){

    if( strip.strip_id < 0 )
      continue;

    NormalizedStrip candidate;

    candidate.qsum = 0.0f;

    for( int t = 0; t < T; t++ ){

      float value = strip.adc[t];

      // Python np.nan_to_num(..., nan=0, posinf=0, neginf=0)
      if( !std::isfinite(value) )
        value = 0.0f;

      value /= ADC_SCALE;

      if( value < 0.0f )
        value = 0.0f;

      if( value > ADC_CLIP_MAX )
        value = ADC_CLIP_MAX;

      candidate.adc[t] = value;
      candidate.qsum += value;
    }


    // Python:
    // qsum >= COMPACT_MIN_Q
    if( candidate.qsum < COMPACT_MIN_Q )
      continue;


    // Python:
    //
    // groupby(strip_id)["qsum"].idxmax()
    //
    // Keep highest-q duplicate.
    auto found = result.find(strip.strip_id);

    if(
        found == result.end() ||
        candidate.qsum > found->second.qsum
      ){
      result[strip.strip_id] = candidate;
    }
  }

  return result;
}


// ============================================================
// Exact Python ids_with_spacers()
//
// gap >= 3 -> insert EXACTLY THREE fake positions.
// ============================================================

std::vector<int>
IDsWithSpacers(
    const std::map<int,NormalizedStrip>& strips
){
  std::vector<int> real_ids;

  real_ids.reserve(strips.size());

  for( const auto& item : strips )
    real_ids.push_back(item.first);


  if( real_ids.empty() )
    return {};


  std::vector<int> output;

  output.push_back(real_ids.front());

  int fake = -10000000;


  for( std::size_t i = 0; i + 1 < real_ids.size(); i++ ){

    const int a = real_ids[i];
    const int b = real_ids[i+1];


    if( (b-a) >= GAP_MIN ){

      for( int n = 0; n < N_SPACER; n++ ){
        output.push_back(fake);
        fake--;
      }
    }

    output.push_back(b);
  }

  return output;
}


// ============================================================
// Python strip_scalar_features()
// ============================================================

void ScalarFeatures(
    const std::vector<Waveform>& adc,
    std::vector<float>& qsum,
    std::vector<float>& peak,
    std::vector<int>& tpeak,
    std::vector<int>& width
){
  const std::size_t N = adc.size();

  qsum.assign(N,0.0f);
  peak.assign(N,0.0f);
  tpeak.assign(N,0);
  width.assign(N,0);


  for( std::size_t i = 0; i < N; i++ ){

    float q = 0.0f;

    float p = adc[i][0];
    int tp = 0;

    int w = 0;


    for( int t = 0; t < T; t++ ){

      const float value = adc[i][t];

      q += value;

      // numpy argmax returns first maximum
      if( value > p ){
        p = value;
        tp = t;
      }

      if( value > THR_WIDTH )
        w++;
    }


    qsum[i] = q;
    peak[i] = p;
    tpeak[i] = tp;
    width[i] = w;
  }
}


// ============================================================
// Exact Python neighbor_cluster_size()
// ============================================================

std::vector<float>
NeighborClusterSize(
    const std::vector<int>& ids,
    const std::vector<float>& qsum
){
  const std::size_t N = ids.size();

  std::vector<float> clust(N,0.0f);

  std::size_t i = 0;


  while( i < N ){

    const bool active =
      ids[i] >= 0 &&
      qsum[i] >= QTHR_CLUSTER;


    if( !active ){
      i++;
      continue;
    }


    std::size_t j = i;


    while( j + 1 < N ){

      if( ids[j] < 0 || ids[j+1] < 0 )
        break;

      if( qsum[j+1] < QTHR_CLUSTER )
        break;

      if( ids[j+1] != ids[j] + 1 )
        break;

      j++;
    }


    const float cluster_size =
      static_cast<float>(j-i+1);


    for( std::size_t k = i; k <= j; k++ )
      clust[k] = cluster_size;


    i = j + 1;
  }

  return clust;
}


// ============================================================
// Python xy_sim
// ============================================================

float XYSimilarity(
    const Waveform& x,
    const Waveform& y
){
  float dot = 0.0f;
  float nx2 = 0.0f;
  float ny2 = 0.0f;


  for( int t = 0; t < T; t++ ){

    dot += x[t] * y[t];

    nx2 += x[t] * x[t];
    ny2 += y[t] * y[t];
  }


  // Python:
  //
  // xn = norm(x) + 1e-6
  // yn = norm(y) + 1e-6

  const float xn = std::sqrt(nx2) + EPS;
  const float yn = std::sqrt(ny2) + EPS;

  return dot / (xn * yn);
}


// ============================================================
// Exact shifted normalized xcorr
//
// shifts = (-1,0,+1)
// ============================================================

float MaxNormXCorr(
    const Waveform& x,
    const Waveform& y
){
  const int shifts[3] = {-1,0,1};

  float best = -1.0e9f;


  for( int ishift = 0; ishift < 3; ishift++ ){

    const int s = shifts[ishift];

    float dot = 0.0f;
    float nx2 = 0.0f;
    float ny2 = 0.0f;


    if( s == 0 ){

      for( int t = 0; t < T; t++ ){

        dot += x[t] * y[t];

        nx2 += x[t] * x[t];
        ny2 += y[t] * y[t];
      }

    } else if( s > 0 ){

      for( int t = 0; t < T-s; t++ ){

        dot += x[t] * y[t+s];

        nx2 += x[t] * x[t];
        ny2 += y[t+s] * y[t+s];
      }

    } else {

      const int ss = -s;

      for( int t = 0; t < T-ss; t++ ){

        dot += x[t+ss] * y[t];

        nx2 += x[t+ss] * x[t+ss];
        ny2 += y[t] * y[t];
      }
    }


    // Note difference from xy_sim:
    //
    // Python xcorr:
    // norm(x)*norm(y) + eps

    const float den =
      std::sqrt(nx2) * std::sqrt(ny2) + EPS;


    const float score = dot / den;

    if( score > best )
      best = score;
  }


  if( best < -1.0f )
    best = -1.0f;

  if( best > 1.0f )
    best = 1.0f;

  return best;
}

} // namespace



// ============================================================
// MAIN PREPROCESSOR
// ============================================================

bool SBSGEMMLPreprocessor::Build(
    const std::vector<SBSGEMMLStrip>& u_strips,
    const std::vector<SBSGEMMLStrip>& v_strips,
    SBSGEMMLInput& output
) const
{
  output = SBSGEMMLInput{};


  // ============================================================
  // Normalize, q-cut and duplicate handling
  // ============================================================

  const auto Ax =
    NormalizeAndDeduplicate(u_strips);

  const auto Ay =
    NormalizeAndDeduplicate(v_strips);


  if( Ax.empty() || Ay.empty() )
    return false;


  // ============================================================
  // Insert strip-gap spacers
  //
  // Python:
  // x = U
  // y = V
  //
  // W = number of x/U image columns
  // H = number of y/V image rows
  // ============================================================

  output.x_ids =
    IDsWithSpacers(Ax);

  output.y_ids =
    IDsWithSpacers(Ay);


  output.W = output.x_ids.size();
  output.H = output.y_ids.size();


  const std::size_t W = output.W;
  const std::size_t H = output.H;


  if( W == 0 || H == 0 )
    return false;


  // ============================================================
  // Build per-axis ADC arrays
  // ============================================================

  std::vector<Waveform> x_adc(W);
  std::vector<Waveform> y_adc(H);


  for( auto& wf : x_adc )
    wf.fill(0.0f);

  for( auto& wf : y_adc )
    wf.fill(0.0f);


  for( std::size_t x = 0; x < W; x++ ){

    const int id = output.x_ids[x];

    if( id < 0 )
      continue;

    const auto found = Ax.find(id);

    if( found != Ax.end() )
      x_adc[x] = found->second.adc;
  }


  for( std::size_t y = 0; y < H; y++ ){

    const int id = output.y_ids[y];

    if( id < 0 )
      continue;

    const auto found = Ay.find(id);

    if( found != Ay.end() )
      y_adc[y] = found->second.adc;
  }


  // ============================================================
  // valid_mask
  // ============================================================

  output.valid_mask.assign(H*W,0);


  for( std::size_t y = 0; y < H; y++ ){

    for( std::size_t x = 0; x < W; x++ ){

      if(
          output.y_ids[y] >= 0 &&
          output.x_ids[x] >= 0
        ){
        output.valid_mask[y*W+x] = 1;
      }
    }
  }


  // ============================================================
  // Strip scalar features
  // ============================================================

  std::vector<float> qx, qy;
  std::vector<float> peakx, peaky;

  std::vector<int> tpx, tpy;
  std::vector<int> widthx, widthy;


  ScalarFeatures(
      x_adc,
      qx,
      peakx,
      tpx,
      widthx
  );


  ScalarFeatures(
      y_adc,
      qy,
      peaky,
      tpy,
      widthy
  );


  const auto clx =
    NeighborClusterSize(output.x_ids,qx);

  const auto cly =
    NeighborClusterSize(output.y_ids,qy);


  // ============================================================
  // Allocate model tensors
  //
  // x3d  [2,6,H,W]
  // extra[14,H,W]
  // ============================================================

  output.x3d.assign(
      2*T*H*W,
      0.0f
  );

  output.extra.assign(
      14*H*W,
      0.0f
  );


  auto x3d_index =
    [H,W](
      std::size_t channel,
      std::size_t time,
      std::size_t y,
      std::size_t x
    )
    {
      return
        ((channel*T + time)*H + y)*W + x;
    };


  auto extra_index =
    [H,W](
      std::size_t channel,
      std::size_t y,
      std::size_t x
    )
    {
      return
        (channel*H + y)*W + x;
    };


  // ============================================================
  // x3d
  //
  // x3d[0,t,y,x] = U waveform
  // x3d[1,t,y,x] = V waveform
  // ============================================================

  for( std::size_t y = 0; y < H; y++ ){

    for( std::size_t x = 0; x < W; x++ ){

      for( int t = 0; t < T; t++ ){

        output.x3d[
          x3d_index(0,t,y,x)
        ] = x_adc[x][t];


        output.x3d[
          x3d_index(1,t,y,x)
        ] = y_adc[y][t];
      }
    }
  }


  // ============================================================
  // Exact 14 feature channels
  // ============================================================

  const float log_qratio_den =
    std::log1p(QRATIO_CLIP);


  for( std::size_t y = 0; y < H; y++ ){

    // np.linspace(-1,1,H)
    const float ypos =
      H == 1
      ? -1.0f
      : -1.0f
        + 2.0f * static_cast<float>(y)
        / static_cast<float>(H-1);


    for( std::size_t x = 0; x < W; x++ ){

      const float xpos =
        W == 1
        ? -1.0f
        : -1.0f
          + 2.0f * static_cast<float>(x)
          / static_cast<float>(W-1);


      const float xy_sim =
        XYSimilarity(x_adc[x],y_adc[y]);


      const float xcorr =
        MaxNormXCorr(x_adc[x],y_adc[y]);


      const float qsum_x_n =
        qx[x] / static_cast<float>(T);

      const float qsum_y_n =
        qy[y] / static_cast<float>(T);


      const float peak_x_n = peakx[x];
      const float peak_y_n = peaky[y];


      const float width_x_n =
        static_cast<float>(widthx[x])
        / static_cast<float>(T);

      const float width_y_n =
        static_cast<float>(widthy[y])
        / static_cast<float>(T);


      const float clust_x_n =
        std::min(
          clx[x],
          static_cast<float>(CLIP_CLUST)
        )
        / static_cast<float>(CLIP_CLUST);


      const float clust_y_n =
        std::min(
          cly[y],
          static_cast<float>(CLIP_CLUST)
        )
        / static_cast<float>(CLIP_CLUST);


      const float dtpeak_abs =
        std::fabs(
          static_cast<float>(tpx[x]-tpy[y])
        )
        / static_cast<float>(T-1);


      float qratio =
        qx[x] / (qy[y] + EPS);


      if( qratio < 0.0f )
        qratio = 0.0f;

      if( qratio > QRATIO_CLIP )
        qratio = QRATIO_CLIP;


      qratio =
        std::log1p(qratio)
        / log_qratio_den;


      output.extra[extra_index( 0,y,x)] = xy_sim;
      output.extra[extra_index( 1,y,x)] = xcorr;

      output.extra[extra_index( 2,y,x)] = qsum_x_n;
      output.extra[extra_index( 3,y,x)] = qsum_y_n;

      output.extra[extra_index( 4,y,x)] = peak_x_n;
      output.extra[extra_index( 5,y,x)] = peak_y_n;

      output.extra[extra_index( 6,y,x)] = width_x_n;
      output.extra[extra_index( 7,y,x)] = width_y_n;

      output.extra[extra_index( 8,y,x)] = clust_x_n;
      output.extra[extra_index( 9,y,x)] = clust_y_n;

      output.extra[extra_index(10,y,x)] = dtpeak_abs;
      output.extra[extra_index(11,y,x)] = qratio;

      output.extra[extra_index(12,y,x)] = xpos;
      output.extra[extra_index(13,y,x)] = ypos;
    }
  }


  return true;
}