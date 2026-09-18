#include "SBSGEMMLPostprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>


namespace {


// ============================================================
// Numerically stable sigmoid
//
// Same mathematical operation as:
//     torch.sigmoid(logit)
// ============================================================

float Sigmoid(float z)
{
  if( z >= 0.0f ){

    const float e =
      std::exp(-z);

    return 1.0f / (1.0f + e);

  } else {

    const float e =
      std::exp(z);

    return e / (1.0f + e);
  }
}


// ============================================================
// Match Python round() behavior.
//
// Python uses round-to-nearest with ties-to-even:
//
//   round(2.5) -> 2
//   round(3.5) -> 4
//
// std::round() does NOT use the same tie rule,
// so don't use std::round here.
//
// Our image coordinates are non-negative.
// ============================================================

int PythonRoundToInt(double value)
{
  const double lo_d =
    std::floor(value);

  const double frac =
    value - lo_d;

  const long long lo =
    static_cast<long long>(lo_d);


  if( frac < 0.5 )
    return static_cast<int>(lo);


  if( frac > 0.5 )
    return static_cast<int>(lo + 1);


  // Exact half:
  // choose the even integer.
  if( (lo % 2) == 0 )
    return static_cast<int>(lo);

  return static_cast<int>(lo + 1);
}


// ============================================================
// Snap an image coordinate to nearest REAL strip position.
//
// Python equivalent:
//
// real_idx[np.argmin(abs(real_idx - initial))]
//
// np.argmin chooses the FIRST minimum.
// Since we scan from low index to high index and only update
// for strictly smaller distance, ties select the lower index.
// ============================================================

int NearestRealIndex(
    int initial,
    const std::vector<int>& ids
)
{
  if(
    initial >= 0 &&
    initial < static_cast<int>(ids.size()) &&
    ids[initial] >= 0
  ){
    return initial;
  }


  int best_index = -1;
  int best_distance =
    std::numeric_limits<int>::max();


  for(
    int i = 0;
    i < static_cast<int>(ids.size());
    i++
  ){

    if( ids[i] < 0 )
      continue;


    const int distance =
      std::abs(i - initial);


    if( distance < best_distance ){

      best_distance = distance;
      best_index = i;
    }
  }


  return best_index;
}


} // namespace



// ============================================================
// MAIN POSTPROCESSOR
// ============================================================

bool SBSGEMMLPostprocessor::Process(
    const std::vector<float>& logits,
    const SBSGEMMLInput& input,
    float threshold,
    SBSGEMMLPostprocessResult& output
) const
// ADC averaged hit position implementataion - Bhasitha
// bool SBSGEMMLPostprocessor::Process(
//     const std::vector<float>& logits,
//     const SBSGEMMLInput& input,
//     float threshold,
//     SBSGEMMLPostprocessResult& output,
//     bool use_adc_weighted_center
// ) const
{
  output = SBSGEMMLPostprocessResult{};


  const std::size_t H = input.H;
  const std::size_t W = input.W;


  // ============================================================
  // Basic input validation
  // ============================================================

  if( H == 0 || W == 0 )
    return false;


  if( logits.size() != H * W )
    return false;


  if( input.valid_mask.size() != H * W )
    return false;


  if( input.x_ids.size() != W )
    return false;


  if( input.y_ids.size() != H )
    return false;
  // ADC-weighted centroid needs the model ADC tensor:
  //
  // x3d shape = [2,6,H,W]
  // if(
  //     use_adc_weighted_center &&
  //     input.x3d.size() != 2 * 6 * H * W
  // ){
  //   return false;
  // }

  if(
    threshold < 0.0f ||
    threshold > 1.0f
  ){
    return false;
  }


  // ============================================================
  // Python:
  //
  // prob = sigmoid(logits)
  //
  // prob_valid =
  //     prob * valid_mask.astype(np.float32)
  //
  // pred =
  //     (prob_valid > pred_thr).astype(np.uint8)
  //
  // IMPORTANT:
  // threshold comparison is STRICTLY >
  // ============================================================

  output.prob.resize(H * W);

  output.prob_valid.resize(
    H * W
  );

  output.pred_mask.resize(
    H * W,
    0
  );


  for(
    std::size_t i = 0;
    i < H * W;
    i++
  ){

    const float p =
      Sigmoid(logits[i]);


    output.prob[i] = p;


    const float pv =
      input.valid_mask[i]
      ? p
      : 0.0f;


    output.prob_valid[i] =
      pv;


    output.pred_mask[i] =
      pv > threshold
      ? 1
      : 0;
  }


  // ============================================================
  // Connected components
  //
  // Python:
  //
  // structure =
  //   ndimage.generate_binary_structure(2, 2)
  //
  // connectivity=2 in 2D means 8-connected:
  //
  //   x x x
  //   x P x
  //   x x x
  //
  // Scan in row-major order so blob numbering follows the same
  // natural ordering as scipy.ndimage.label.
  // ============================================================

  std::vector<unsigned char> visited(
    H * W,
    0
  );


  int component_label = 0;


  for(
    std::size_t y0 = 0;
    y0 < H;
    y0++
  ){

    for(
      std::size_t x0 = 0;
      x0 < W;
      x0++
    ){

      const std::size_t start =
        y0 * W + x0;


      if(
        output.pred_mask[start] == 0 ||
        visited[start] != 0
      ){
        continue;
      }


      // --------------------------------------------------------
      // Start one connected component
      // --------------------------------------------------------

      const int this_blob_id =
        component_label;

      component_label++;


      std::queue<std::size_t> q;

      q.push(start);
      visited[start] = 1;


      int area = 0;
      int real_area = 0;


      double sum_y_real = 0.0;
      double sum_x_real = 0.0;

      // Preserve the full real-cell membership so downstream
      // reconstruction can project this 2D blob into U and V
      // strip clusters.
      std::vector<SBSGEMMLBlobCell> cells;

      std::vector<int> u_strips;
      std::vector<int> v_strips;
      // ADC-weighted centroid needs the model ADC tensor: Bhasitha
      // Which real U columns / V rows actually participate
      // in this predicted connected component.
      // std::vector<unsigned char> blob_x_used(W, 0);
      // std::vector<unsigned char> blob_y_used(H, 0);


      while( !q.empty() ){

        const std::size_t idx =
          q.front();

        q.pop();


        const std::size_t y =
          idx / W;

        const std::size_t x =
          idx % W;


        area++;


        // Python computes the centroid using only pixels
        // where both x and y correspond to real strip IDs.

        // ADC-weighted centroid needs the model ADC tensor: Bhasitha
        // if( input.valid_mask[idx] ){

        //   real_area++;

        //   sum_y_real +=
        //     static_cast<double>(y);

        //   sum_x_real +=
        //     static_cast<double>(x);


        //   SBSGEMMLBlobCell cell;

        //   cell.iy =
        //     static_cast<int>(y);

        //   cell.ix =
        //     static_cast<int>(x);

        //   cell.u_strip =
        //     input.x_ids[x];

        //   cell.v_strip =
        //     input.y_ids[y];

        //   cell.probability =
        //     output.prob_valid[idx];

        //   cells.push_back(cell);

        //   u_strips.push_back(cell.u_strip);
        //   v_strips.push_back(cell.v_strip);
        // }
        if( input.valid_mask[idx] ){

          real_area++;

          sum_y_real +=
            static_cast<double>(y);

          sum_x_real +=
            static_cast<double>(x);


          // blob_x_used[x] = 1;
          // blob_y_used[y] = 1;


          SBSGEMMLBlobCell cell;

          cell.iy =
            static_cast<int>(y);

          cell.ix =
            static_cast<int>(x);

          cell.u_strip =
            input.x_ids[x];

          cell.v_strip =
            input.y_ids[y];

          cell.probability =
            output.prob_valid[idx];

          cells.push_back(cell);

          u_strips.push_back(cell.u_strip);
          v_strips.push_back(cell.v_strip);
        }



        // ------------------------------------------------------
        // 8-connected neighbors
        // ------------------------------------------------------

        for( int dy = -1; dy <= 1; dy++ ){

          for( int dx = -1; dx <= 1; dx++ ){

            if( dx == 0 && dy == 0 )
              continue;


            const int ny =
              static_cast<int>(y) + dy;

            const int nx =
              static_cast<int>(x) + dx;


            if(
              ny < 0 ||
              nx < 0 ||
              ny >= static_cast<int>(H) ||
              nx >= static_cast<int>(W)
            ){
              continue;
            }


            const std::size_t nidx =
              static_cast<std::size_t>(ny) * W
              + static_cast<std::size_t>(nx);


            if(
              output.pred_mask[nidx] == 0 ||
              visited[nidx] != 0
            ){
              continue;
            }


            visited[nidx] = 1;

            q.push(nidx);
          }
        }
      }


      // ========================================================
      // Python skips components that contain no real pixels.
      //
      // With our current valid_mask-before-threshold sequence,
      // this should normally never occur.
      // ========================================================

      if( real_area == 0 )
        continue;


      // Store each physical strip once and in ascending order.
      std::sort(
        u_strips.begin(),
        u_strips.end()
      );

      u_strips.erase(
        std::unique(u_strips.begin(), u_strips.end()),
        u_strips.end()
      );

      std::sort(
        v_strips.begin(),
        v_strips.end()
      );

      v_strips.erase(
        std::unique(v_strips.begin(), v_strips.end()),
        v_strips.end()
      );


      // ========================================================
      // Centroid in image coordinates
      //
      // Python:
      //
      // cy = float(ys_use.mean())
      // cx = float(xs_use.mean())
      // ========================================================

      // ADC-weighted centroid needs the model ADC tensor: Bhasitha
      // const double cy =
      //   sum_y_real /
      //   static_cast<double>(real_area);

      // const double cx =
      //   sum_x_real /
      //   static_cast<double>(real_area);
      // ========================================================
      // Representative blob position
      //
      // mode 0:
      //   geometric centroid of real predicted cells
      //
      // mode 1:
      //   ADC-sum-weighted centroid:
      //     U coordinate weighted by U-strip ADC sum
      //     V coordinate weighted by V-strip ADC sum
      //
      // IMPORTANT:
      // Weight each unique U column / V row ONCE.
      // Do not weight every 2D blob cell, because that would
      // multiply the weight by the number of opposite-axis cells.
      // ========================================================


      // --------------------------------------------------------
      // Default/current geometric centroid
      // --------------------------------------------------------

      double cy =
        sum_y_real /
        static_cast<double>(real_area);

      double cx =
        sum_x_real /
        static_cast<double>(real_area);


      // --------------------------------------------------------
      // Optional ADC-weighted centroid
      // --------------------------------------------------------

      // if( use_adc_weighted_center ){

      //   constexpr std::size_t NT = 6;


      //   // x3d flat layout:
      //   //
      //   // [channel,time,y,x]
      //   //
      //   // channel 0 = U waveform
      //   // channel 1 = V waveform
      //   auto x3d_index =
      //     [H,W](
      //       std::size_t channel,
      //       std::size_t time,
      //       std::size_t y,
      //       std::size_t x
      //     )
      //     {
      //       return
      //         ((channel * 6 + time) * H + y) * W + x;
      //     };


      //   // ======================================================
      //   // U / x weighted centroid
      //   // ======================================================

      //   double sum_q_u = 0.0;
      //   double sum_x_q_u = 0.0;


      //   for( std::size_t x = 0; x < W; ++x ){

      //     if( !blob_x_used[x] )
      //       continue;

      //     if( input.x_ids[x] < 0 )
      //       continue;


      //     double q_u = 0.0;


      //     // U waveform is repeated along y,
      //     // so y=0 is sufficient.
      //     for( std::size_t t = 0; t < NT; ++t ){

      //       q_u += static_cast<double>(
      //         input.x3d[
      //           x3d_index(
      //             0,
      //             t,
      //             0,
      //             x
      //           )
      //         ]
      //       );
      //     }


      //     if( q_u <= 0.0 )
      //       continue;


      //     sum_q_u += q_u;

      //     sum_x_q_u +=
      //       static_cast<double>(x) * q_u;
      //   }


      //   // ======================================================
      //   // V / y weighted centroid
      //   // ======================================================

      //   double sum_q_v = 0.0;
      //   double sum_y_q_v = 0.0;


      //   for( std::size_t y = 0; y < H; ++y ){

      //     if( !blob_y_used[y] )
      //       continue;

      //     if( input.y_ids[y] < 0 )
      //       continue;


      //     double q_v = 0.0;


      //     // V waveform is repeated along x,
      //     // so x=0 is sufficient.
      //     for( std::size_t t = 0; t < NT; ++t ){

      //       q_v += static_cast<double>(
      //         input.x3d[
      //           x3d_index(
      //             1,
      //             t,
      //             y,
      //             0
      //           )
      //         ]
      //       );
      //     }


      //     if( q_v <= 0.0 )
      //       continue;


      //     sum_q_v += q_v;

      //     sum_y_q_v +=
      //       static_cast<double>(y) * q_v;
      //   }


      //   // ------------------------------------------------------
      //   // Replace geometric coordinate only when ADC weighting
      //   // has a valid non-zero denominator.
      //   //
      //   // Otherwise retain the geometric centroid as fallback.
      //   // ------------------------------------------------------

      //   if( sum_q_u > 0.0 ){

      //     cx =
      //       sum_x_q_u /
      //       sum_q_u;
      //   }


      //   if( sum_q_v > 0.0 ){

      //     cy =
      //       sum_y_q_v /
      //       sum_q_v;
      //   }
      // }









      // ========================================================
      // Python:
      //
      // iy0 = clip(round(cy),0,H-1)
      // ix0 = clip(round(cx),0,W-1)
      // ========================================================

      int iy0 =
        PythonRoundToInt(cy);

      int ix0 =
        PythonRoundToInt(cx);


      iy0 =
        std::max(
          0,
          std::min(
            iy0,
            static_cast<int>(H) - 1
          )
        );


      ix0 =
        std::max(
          0,
          std::min(
            ix0,
            static_cast<int>(W) - 1
          )
        );


      // ========================================================
      // Snap centroid to nearest real row/column when the
      // rounded centroid lands on a spacer.
      // ========================================================

      const int iy =
        NearestRealIndex(
          iy0,
          input.y_ids
        );


      const int ix =
        NearestRealIndex(
          ix0,
          input.x_ids
        );


      if( iy < 0 || ix < 0 )
        continue;


      // ========================================================
      // Save blob
      // ========================================================

      SBSGEMMLBlob blob;


      blob.blob_id =
        this_blob_id;


      blob.cy = cy;
      blob.cx = cx;


      blob.iy = iy;
      blob.ix = ix;


      // Python:
      //
      // y_strip = y_ids[iy]
      // x_strip = x_ids[ix]
      //
      // x = U
      // y = V

      blob.y_strip =
        input.y_ids[iy];

      blob.x_strip =
        input.x_ids[ix];


      blob.area =
        area;

      blob.real_area =
        real_area;


      blob.cells =
        cells;

      blob.u_strips =
        u_strips;

      blob.v_strips =
        v_strips;


      output.blobs.push_back(
        blob
      );
    }
  }


  return true;
}