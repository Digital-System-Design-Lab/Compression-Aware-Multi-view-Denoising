#ifndef BM3D_H_INCLUDED
#define BM3D_H_INCLUDED

#include <fftw3.h>
#include <vector>

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_bm3d(
    const float sigma
,   std::vector<float> &img_o1
,   std::vector<float> &img_o2
,   std::vector<float> &img_o3
,   std::vector<float> &img_o4
,   std::vector<float> &img_noisy1
,   std::vector<float> &img_noisy2
,   std::vector<float> &img_noisy3
,   std::vector<float> &img_noisy4
,   std::vector<float> &img_basic
,   std::vector<float> &img_basic2
,   std::vector<float> &img_basic3
,   std::vector<float> &img_basic4
,   std::vector<float> &img_original_basic
,   std::vector<float> &img_original_basic2
,   std::vector<float> &img_original_basic3
,   std::vector<float> &img_original_basic4
,   std::vector<float> &img_basic_bm3d1
,   std::vector<float> &img_basic_bm3d2
,   std::vector<float> &img_basic_bm3d3
,   std::vector<float> &img_basic_bm3d4
        ,   std::vector<float> &img_denoised
        ,   std::vector<float> &img_denoised2
        ,   std::vector<float> &img_denoised3
        ,   std::vector<float> &img_denoised4
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const int search_range
,   const int search_range2
,   const int window_size
,   const bool useSD_h
,   const unsigned tau_2D_hard
,   const unsigned color_space
);

//! 1st step of BM3D
void bm3d_1st_step(
    const float sigma
        ,   std::vector<float> const& img_noisy
        ,   std::vector<float> const& img_noisy2
        ,   std::vector<float> const& img_noisy3
        ,   std::vector<float> const& img_noisy4
        ,   std::vector<float> const& img_sy_noisy
        ,   std::vector<float> const& img_sy_noisy2
        ,   std::vector<float> const& img_sy_noisy3
        ,   std::vector<float> const& img_sy_noisy4
        ,   std::vector<float> &img_basic
        ,   std::vector<float> &img_basic2
        ,   std::vector<float> &img_basic3
        ,   std::vector<float> &img_basic4
        ,   const unsigned width
        ,   const unsigned height
        ,   const int iFrmWidth
        ,   const int iFrmHeight
        ,   const unsigned chnls
        ,   const int search_range
        ,   const int search_range2
        ,   const int window_size
        ,   const unsigned nHard
        ,   const unsigned kHard
        ,   const unsigned NHard
        ,   const unsigned pHard
        ,   const bool     useSD
        ,   const unsigned color_space
        ,   const unsigned tau_2D
        ,   fftwf_plan *  plan_2d_for_1
        ,   fftwf_plan *  plan_2d_for_2
        ,   fftwf_plan *  plan_2d_inv
//,std::vector<std::vector<int> > &patch_table
);

//! 1st step of BM3D
void bm3d_1st_step_original_EPI(
        const float sigma
//,   std::vector<float> const& img_orig1
//,   std::vector<float> const& img_orig2
        ,   std::vector<float> const& img_noisy
        ,   std::vector<float> const& img_noisy2
        ,   std::vector<float> const& img_noisy3
        ,   std::vector<float> const& img_noisy4
        ,   std::vector<float> const& img_sy_noisy
        ,   std::vector<float> const& img_sy_noisy2
        ,   std::vector<float> const& img_sy_noisy3
        ,   std::vector<float> const& img_sy_noisy4
        ,   std::vector<float> &img_basic
        ,   std::vector<float> &img_basic2
        ,   std::vector<float> &img_basic3
        ,   std::vector<float> &img_basic4
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned nHard
        ,   const unsigned kHard
        ,   const unsigned NHard
        ,   const unsigned pHard
        ,   const bool     useSD
        ,   const unsigned color_space
        ,   const unsigned tau_2D
        ,   fftwf_plan *  plan_2d_for_1
        ,   fftwf_plan *  plan_2d_for_2
        ,   fftwf_plan *  plan_2d_inv
);
void bm3d_1st_step3(
        const float sigma
        ,   std::vector<float> const& img_noisy
        ,   std::vector<float> const& img_noisy2
        ,   std::vector<float> const& img_noisy3
        ,   std::vector<float> const& img_noisy4
        ,   std::vector<float> &img_basic
        ,   std::vector<float> &img_basic2
        ,   std::vector<float> &img_basic3
        ,   std::vector<float> &img_basic4
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned nHard
        ,   const unsigned kHard
        ,   const unsigned NHard
        ,   const unsigned pHard
        ,   const bool     useSD
        ,   const unsigned color_space
        ,   const unsigned tau_2D
        ,   fftwf_plan *  plan_2d_for_1
        ,   fftwf_plan *  plan_2d_for_2
        ,   fftwf_plan *  plan_2d_inv
);
//! 2nd step of BM3D
void bm3d_2nd_step(
    const float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> const& img_basic
,   std::vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
);

//! Process 2D dct of a group of patches
void dct_2d_process(
    std::vector<float> &DCT_table_2D
,   std::vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   std::vector<float> const& coef_norm
,   const unsigned i_min
,   const unsigned i_max
);

//! Process 2D bior1.5 transform of a group of patches
void bior_2d_process(
    std::vector<float> &bior_table_2D
,   std::vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   const unsigned i_min
,   const unsigned i_max
,   std::vector<float> &lpd
,   std::vector<float> &hpd
);

void dct_2d_inverse(
    std::vector<float> &group_3D_table
,   const unsigned kHW
,   const unsigned N
,   std::vector<float> const& coef_norm_inv
,   fftwf_plan * plan
);

void bior_2d_inverse(
    std::vector<float> &group_3D_table
,   const unsigned kHW
,   std::vector<float> const& lpr
,   std::vector<float> const& hpr
);

//! HT filtering using Welsh-Hadamard transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_hadamard(
    std::vector<float> &group_3D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kHard
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   const float lambdaThr3D
,   std::vector<float> &weight_table
,   const bool doWeight
);

//! Wiener filtering using Welsh-Hadamard transform
void wiener_filtering_hadamard(
    std::vector<float> &group_3D_img
,   std::vector<float> &group_3D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kWien
,   const unsigned chnls
,   std::vector<float> const& sigma_table
,   std::vector<float> &weight_table
,   const bool doWeight
);

//! Compute weighting using Standard Deviation
void sd_weighting(
    std::vector<float> const& group_3D
,   const unsigned nSx_r
,   const unsigned kHW
,   const unsigned chnls
,   std::vector<float> &weight_table
);

//! Apply a bior1.5 spline wavelet on a vector of size N x N.
void bior1_5_transform(
    std::vector<float> const& input
,   std::vector<float> &output
,   const unsigned N
,   std::vector<float> const& bior_table
,   const unsigned d_i
,   const unsigned d_o
,   const unsigned N_i
,   const unsigned N_o
);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
//! Preprocess coefficients of the Kaiser window and normalization coef for the DCT
void bm3d_1st_step_bm3d(
        const float sigma
        ,   std::vector<float> const& img_noisy
        ,   std::vector<float> &img_basic
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned nHard
        ,   const unsigned kHard
        ,   const unsigned NHard
        ,   const unsigned pHard
        ,   const bool     useSD
        ,   const unsigned color_space
        ,   const unsigned tau_2D
        ,   fftwf_plan *  plan_2d_for_1
        ,   fftwf_plan *  plan_2d_for_2
        ,   fftwf_plan *  plan_2d_inv
);
void preProcess(
    std::vector<float> &kaiserWindow
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned kHW
);

void precompute(
    std::vector<std::vector<int> > &patch_table1
,   std::vector<std::vector<int> > &patch_table2
,   const std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned n
,   const unsigned pHW
,   const float    tauMatch
);
void precompute_BM(
        std::vector<std::vector<int> > &patch_table1
        ,   const std::vector<float> &img
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned kHW
        ,   const unsigned NHW
        ,   const unsigned n
        ,   const unsigned pHW
        ,   const float    tauMatch
);
void bm3d_1st_step2(
        const float sigma
        ,   std::vector<float> const& img_noisy
        ,   std::vector<float> const&img_basic
        ,   std::vector<float> &img_basic1
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned nHard
        ,   const unsigned kHard
        ,   const unsigned NHard
        ,   const unsigned pHard
        ,   const bool     useSD
        ,   const unsigned color_space
        ,   const unsigned tau_2D
        ,   fftwf_plan *  plan_2d_for_1
        ,   fftwf_plan *  plan_2d_for_2
        ,   fftwf_plan *  plan_2d_inv
);

void precompute_EPI(
        std::vector<std::vector<int> > &patch_table1
        ,   std::vector<std::vector<int> > &patch_table2
        ,   std::vector<std::vector<int> > &patch_table3
        ,   std::vector<std::vector<int> > &patch_table4
        ,   std::vector<std::vector<int> > &confidence_level_flag
        ,   const std::vector<float> &img1
        ,   const std::vector<float> &img2
        ,   const std::vector<float> &img3
        ,   const std::vector<float> &img4
        ,   const unsigned width
        ,   const unsigned height
        ,   const int iFrmWidth
        ,   const int iFrmHeight
        ,   const int search_range
        ,   const int search_range2
        ,   const int window_size
        ,   const float    tauMatch
        ,   const unsigned pHard
        ,   const unsigned nHard
        ,   const unsigned kHard
        //,std::vector<std::vector<int> > &patch_table
);
double inter8(unsigned char* LF, double P_r, int P_1, int P_2, double U_r, int U_1, int U_2, double H_r, int H_1, int H_2, int c, int HEIGHT, int WIDTH);
void precompute_original_EPI(
        std::vector<std::vector<int> > &patch_table1
        ,   std::vector<std::vector<int> > &patch_table2
        ,   std::vector<std::vector<int> > &patch_table3
        ,   std::vector<std::vector<int> > &patch_table4
        ,   const std::vector<float> &img1
        ,   const std::vector<float> &img2
        ,   const std::vector<float> &img3
        ,   const std::vector<float> &img4
        //,   const std::vector<float> &img5
        //,   const std::vector<float> &img6
        //,   const std::vector<float> &img7
        //,   const std::vector<float> &img8
        //,   const std::vector<float> &img9
        //,   const std::vector<float> &img10
        //,   const std::vector<float> &img11
        //,   const std::vector<float> &img12
        //,   const std::vector<float> &img13
        //,   const std::vector<float> &img14
        //,   const std::vector<float> &img15
        //,   const std::vector<float> &img16
        ,   const unsigned width
        ,   const unsigned height
        ,   const float    tauMatch
        ,   const unsigned pHard
);
void Non_local_mean(
        const std::vector<float> &img1
        ,   const std::vector<float> &img2
        ,   const std::vector<float> &img3
        ,   const std::vector<float> &NLM_img1
        ,   const std::vector<float> &NLM_img2
        ,   const std::vector<float> &NLM_img3
        ,   const unsigned width
        ,   const unsigned height
);
void quick_sort(
        int *data
        , int start
        , int end
);
#endif // BM3D_H_INCLUDED
