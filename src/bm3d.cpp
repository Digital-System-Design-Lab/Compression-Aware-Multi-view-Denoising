/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file bm3d.cpp
 * @brief BM3D denoising functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include "bm3d.h"
#include "utilities.h"
#include "lib_transforms.h"
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <stdint.h>
#include <time.h>
#include <vector>
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475
#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6

#ifdef _OPENMP
#include <omp.h>
#endif
 int th = 0;
using namespace std;

bool ComparaisonFirst(pair<float,unsigned> pair1, pair<float,unsigned> pair2)
{
    return pair1.first < pair2.first;
}

/** ----------------- **/
/** - Main function - **/
/** ----------------- **/
/**
 * @brief run BM3D process. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        image in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param img_noisy: noisy image;
 * @param img_basic: will be the basic estimation after the 1st step
 * @param img_denoised: will be the denoised final image;
 * @param width, height, chnls: size of the image;
 * @param useSD_h (resp. useSD_w): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
 *        on every 3D group for the first (resp. second) part.
 *        Allowed values are DCT and BIOR;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/

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
){
    //! Parameters
    const unsigned nHard = 8; //! Half size of the search window
    const unsigned nHard1 = 16; //! Half size of the search window
    const unsigned nWien = 4; //! Half size of the search window
    const unsigned kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
    const unsigned NHard = 4; //! Must be a power of 2
    const unsigned NHard1 = 4; //! Must be a power of 2
    const unsigned NWien = 32; //! Must be a power of 2
    const unsigned pHard = 1;
    const unsigned pHard1 =1;
    const unsigned pWien = 1;
    clock_t before;
    clock_t original_before;
    clock_t bm;
    double clock_result;
    double original_clock_result;
    double clock_result1;
    //! Check memory allocation
    if (img_basic.size() != img_noisy1.size())
        img_basic.resize(img_noisy1.size());
    if (img_basic2.size() != img_noisy2.size())
        img_basic2.resize(img_noisy2.size());
    if (img_basic3.size() != img_noisy3.size())
        img_basic3.resize(img_noisy3.size());
    if (img_basic4.size() != img_noisy4.size())
        img_basic4.resize(img_noisy4.size());

    if (img_original_basic.size() != img_noisy1.size())
        img_original_basic.resize(img_noisy1.size());
    if (img_original_basic2.size() != img_noisy2.size())
        img_original_basic2.resize(img_noisy2.size());
    if (img_original_basic3.size() != img_noisy3.size())
        img_original_basic3.resize(img_noisy3.size());
    if (img_original_basic4.size() != img_noisy4.size())
        img_original_basic4.resize(img_noisy4.size());

    if (img_basic_bm3d1.size() != img_noisy1.size())
        img_basic_bm3d1.resize(img_noisy1.size());
    if (img_basic_bm3d2.size() != img_noisy1.size())
        img_basic_bm3d2.resize(img_noisy1.size());
    if (img_basic_bm3d3.size() != img_noisy1.size())
        img_basic_bm3d3.resize(img_noisy1.size());
    if (img_basic_bm3d4.size() != img_noisy1.size())
        img_basic_bm3d4.resize(img_noisy1.size());

    if (img_denoised.size() != img_noisy1.size())
        img_denoised.resize(img_noisy1.size());
    if (img_denoised2.size() != img_noisy2.size())
        img_denoised2.resize(img_noisy2.size());
    if (img_denoised3.size() != img_noisy3.size())
        img_denoised3.resize(img_noisy3.size());
    if (img_denoised4.size() != img_noisy4.size())
        img_denoised4.resize(img_noisy4.size());
//cout << "color space : " << color_space << endl;
    //! Transformation to YUV color space
    if (color_space_transform(img_noisy1, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy2, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy3, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy4, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Check if OpenMP is used or if number of cores of the computer is > 1
    unsigned nb_threads = 1;

#ifdef _OPENMP
    cout << "Open MP used" << endl;
    nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#endif

    cout << endl << "Number of threads which will be used: " << nb_threads;
#ifdef _OPENMP
    cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
#endif

    //! Allocate plan for FFTW library
    fftwf_plan plan_2d_for_1[nb_threads];
    fftwf_plan plan_2d_for_2[nb_threads];
    fftwf_plan plan_2d_inv[nb_threads];

    //! In the simple case
    if (nb_threads == 1)
    {
        //! Add boundaries and symetrize them
        const unsigned h_b = height + 2 * nHard;
        const unsigned w_b = width  + 2 * nHard;
        const unsigned h_b1 = height + 2 * nHard1;
        const unsigned w_b1 = width  + 2 * nHard1;
        vector<float> img_sym_noisy, img_sym_basic, img_sym_basic1, img_sym_basic6,img_sym_basic7, img_sym_basic8, img_sym_denoised;
        symetrize(img_noisy1, img_sym_noisy, width, height, chnls, nHard);
        vector<float> img_sym_noisy2, img_sym_basic2, img_sym_denoised2;
        symetrize(img_noisy2, img_sym_noisy2, width, height, chnls, nHard);
        vector<float> img_sym_noisy3, img_sym_basic3, img_sym_denoised3;
        symetrize(img_noisy3, img_sym_noisy3, width, height, chnls, nHard);
        vector<float> img_sym_noisy4, img_sym_basic4, img_sym_denoised4;
        symetrize(img_noisy4, img_sym_noisy4, width, height, chnls, nHard);
        vector<float> img_original_sym_basic1, img_original_sym_basic2, img_original_sym_basic3, img_original_sym_basic4;

        vector<float> img_sym_original_noisy;
        symetrize(img_noisy1, img_sym_original_noisy, width, height, chnls, nHard);
        vector<float> img_sym_original_noisy2;
        symetrize(img_noisy2, img_sym_original_noisy2, width, height, chnls, nHard);
        vector<float> img_sym_original_noisy3;
        symetrize(img_noisy3, img_sym_original_noisy3, width, height, chnls, nHard);
        vector<float> img_sym_original_noisy4;
        symetrize(img_noisy4, img_sym_original_noisy4, width, height, chnls, nHard);

        vector<float> img_sym_noisy_bm3d1, img_sym_basic_bm3d1, img_sym_denoised_bm3d1;
        symetrize(img_noisy1, img_sym_noisy_bm3d1, width, height, chnls, nHard1);
        vector<float> img_sym_noisy_bm3d2, img_sym_basic_bm3d2, img_sym_denoised_bm3d2;
        symetrize(img_noisy2, img_sym_noisy_bm3d2, width, height, chnls, nHard1);
        vector<float> img_sym_noisy_bm3d3, img_sym_basic_bm3d3, img_sym_denoised_bm3d3;
        symetrize(img_noisy3, img_sym_noisy_bm3d3, width, height, chnls, nHard1);
        vector<float> img_sym_noisy_bm3d4, img_sym_basic_bm3d4, img_sym_denoised_bm3d4;
        symetrize(img_noisy4, img_sym_noisy_bm3d4, width, height, chnls, nHard1);

        //! Allocating Plan for FFTW process
        if (tau_2D_hard == DCT)
        {
            const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
            allocate_plan_2d(&plan_2d_for_1[0], kHard, FFTW_REDFT10,
                             w_b * (2 * nHard + 1) * chnls);
            allocate_plan_2d(&plan_2d_for_2[0], kHard, FFTW_REDFT10,
                             w_b * pHard * chnls);
            allocate_plan_2d(&plan_2d_inv  [0], kHard, FFTW_REDFT01,
                             NHard * nb_cols * chnls);
        }

        //! Denoising, 1st Step
        cout << endl;
        cout << "step 1...";
        before = clock();
        bm3d_1st_step(sigma, img_noisy1, img_noisy2, img_noisy3, img_noisy4, img_sym_noisy, img_sym_noisy2, img_sym_noisy3, img_sym_noisy4,
                      img_sym_basic1, img_sym_basic2, img_sym_basic3, img_sym_basic4, w_b, h_b, width, height, chnls, search_range, search_range2, window_size, nHard, kHard, NHard, pHard, useSD_h, color_space, tau_2D_hard,
                      &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        clock_result = (double)(clock() - before) / CLOCKS_PER_SEC;
        printf("youna processing time : %5.2f\n", clock_result);
        cout << "youna done." << endl;


        cout << "youna done." << endl;
        bm = clock();
        bm3d_1st_step_bm3d(sigma, img_sym_noisy_bm3d1, img_sym_basic_bm3d1, w_b1, h_b1, chnls, nHard1, kHard, NHard1, pHard1, useSD_h, color_space, tau_2D_hard,
                           &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        bm3d_1st_step_bm3d(sigma, img_sym_noisy_bm3d2, img_sym_basic_bm3d2, w_b1, h_b1, chnls, nHard1, kHard, NHard1, pHard1, useSD_h, color_space, tau_2D_hard,
                           &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        bm3d_1st_step_bm3d(sigma, img_sym_noisy_bm3d3, img_sym_basic_bm3d3, w_b1, h_b1, chnls, nHard1, kHard, NHard1, pHard1, useSD_h, color_space, tau_2D_hard,
                           &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        bm3d_1st_step_bm3d(sigma, img_sym_noisy_bm3d4, img_sym_basic_bm3d4, w_b1, h_b1, chnls, nHard1, kHard, NHard1, pHard1, useSD_h, color_space, tau_2D_hard,
                           &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        clock_result1 = (double)(clock() - bm) / CLOCKS_PER_SEC;
        printf("BM3D processing time : %5.2f\n", clock_result1);
        cout << "bm3d done." << endl;
        //! To avoid boundaries problem
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc_b = c * w_b * h_b + nHard * w_b + nHard;
            unsigned dc = c * width * height;
            for (unsigned i = 0; i < height; i++)
                for (unsigned j = 0; j < width; j++, dc++){
                    img_basic[dc] = img_sym_basic1[dc_b + i * w_b + j];
                    img_basic2[dc] = img_sym_basic2[dc_b + i * w_b + j];
                    img_basic3[dc] = img_sym_basic3[dc_b + i * w_b + j];
                    img_basic4[dc] = img_sym_basic4[dc_b + i * w_b + j];

                }
        }
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc_b = c * w_b1 * h_b1 + nHard1 * w_b1 + nHard1;
            unsigned dc = c * width * height;
            for (unsigned i = 0; i < height; i++)
                for (unsigned j = 0; j < width; j++, dc++){
                    img_basic_bm3d1[dc] = img_sym_basic_bm3d1[dc_b + i * w_b1 + j];
                    img_basic_bm3d2[dc] = img_sym_basic_bm3d2[dc_b + i * w_b1 + j];
                    img_basic_bm3d3[dc] = img_sym_basic_bm3d3[dc_b + i * w_b1 + j];
                    img_basic_bm3d4[dc] = img_sym_basic_bm3d4[dc_b + i * w_b1 + j];
                }
        }

        symetrize(img_basic, img_sym_basic1, width, height, chnls, nHard);
        symetrize(img_basic2, img_sym_basic2, width, height, chnls, nHard);
        symetrize(img_basic3, img_sym_basic3, width, height, chnls, nHard);
        symetrize(img_basic4, img_sym_basic4, width, height, chnls, nHard);

        symetrize(img_basic_bm3d1, img_sym_basic_bm3d1, width, height, chnls, nHard1);
        symetrize(img_basic_bm3d2, img_sym_basic_bm3d2, width, height, chnls, nHard1);
        symetrize(img_basic_bm3d3, img_sym_basic_bm3d3, width, height, chnls, nHard1);
        symetrize(img_basic_bm3d4, img_sym_basic_bm3d4, width, height, chnls, nHard1);

    }
    //! If more than 1 threads are used

    //! Inverse color space transform to RGB
    //if (color_space_transform(img_denoised, color_space, width, height, chnls, false)
    //    != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy1, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy2, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy3, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy4, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    if (color_space_transform(img_basic, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic2, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic3, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic4, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    if (color_space_transform(img_basic_bm3d1, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic_bm3d2, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic_bm3d3, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic_bm3d4, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

/*
    //! Free Memory
    if (tau_2D_hard == DCT || tau_2D_wien == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_2d_for_1[n]);
            fftwf_destroy_plan(plan_2d_for_2[n]);
            fftwf_destroy_plan(plan_2d_inv[n]);
        }
    fftwf_cleanup();
*/
    return EXIT_SUCCESS;
}

/**
 * @brief Run the basic process of BM3D (1st step). The result
 *        is contained in img_basic. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: will contain the denoised image after the 1st step;
 * @param width, height, chnls : size of img_noisy;
 * @param nHard: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param tau_2D: DCT or BIOR;
 * @param plan_2d_for_1, plan_2d_for_2, plan_2d_inv : for convenience. Used
 *        by fftw.
 *
 * @return none.
 **/
void bm3d_1st_step(
        const float sigma
        ,   vector<float> const& noisy
        ,   vector<float> const& noisy2
        ,   vector<float> const& noisy3
        ,   vector<float> const& noisy4
        ,   vector<float> const& img_noisy // symetrized
        ,   vector<float> const& img_noisy2
        ,   vector<float> const& img_noisy3
        ,   vector<float> const& img_noisy4
        ,   vector<float> &img_basic
        ,   vector<float> &img_basic2
        ,   vector<float> &img_basic3
        ,   vector<float> &img_basic4
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
){
    clock_t EPI_time;
    clock_t trans_time;
    clock_t inv_trans_time;
    clock_t inv_trans_time1;
    clock_t inv_trans_time2;
    clock_t hard_time;
    clock_t hard_time1;
    clock_t hard_time2;
    clock_t hard_time3;
    clock_t hard_time4;
    clock_t aggreagation_time;
    clock_t aggreagation_time1;
    clock_t aggreagation_time2;
    double clock_trans;
    double clock_inv_trans;
    double clock_inv_trans1;
    double clock_inv_trans2;
    double clock_hard;
    double clock_hard1;
    double clock_hard2;
    double clock_hard3;
    double clock_hard4;
    double clock_EPI;
    double clock_aggreagation;
    double clock_aggreagation1;
    double clock_aggreagation2;
    double clock_result_trans = 0;
    double clock_result_inv_trans = 0;
    double clock_result_hard = 0;
    double clock_result_aggregation = 0;
    double clock_result_EPI = 0;
    int bound = 0;
    int num_group=0;
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    //value sigma table
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if (img_basic2.size() != img_noisy2.size())
        img_basic2.resize(img_noisy2.size());
    if (img_basic3.size() != img_noisy3.size())
        img_basic3.resize(img_noisy3.size());
    if (img_basic4.size() != img_noisy4.size())
        img_basic4.resize(img_noisy4.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);

    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);
    vector<float> denominator2(width * height * chnls, 0.0f);
    vector<float> numerator2  (width * height * chnls, 0.0f);
    vector<float> denominator3(width * height * chnls, 0.0f);
    vector<float> numerator3  (width * height * chnls, 0.0f);
    vector<float> denominator4(width * height * chnls, 0.0f);
    vector<float> numerator4  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<int> > patch_table1;
    vector<vector<int> > patch_table2;
    vector<vector<int> > patch_table3;
    vector<vector<int> > patch_table4;// bound 영역에서 비어있는 부분
    vector<vector<int> > confidence_level_flag;
    //vector<vector<int> > patch_table6; //기울기
    cout << endl;
    cout << "start EPI estimation" << endl;

    EPI_time = clock();
    precompute_EPI(patch_table1, patch_table2, patch_table3, patch_table4, confidence_level_flag, noisy, noisy2 ,noisy3 ,noisy4 ,width ,height, iFrmWidth, iFrmHeight, search_range, search_range2, window_size, tauMatch, pHard, nHard, kHard);
    clock_EPI = (double)(clock() - EPI_time) / CLOCKS_PER_SEC;
    clock_result_EPI += clock_EPI;
    printf("EPI processing time : %5.2f\n", clock_result_EPI);
    //cout << " patch_table1 size : " << patch_table1.size() << " patch_table2 size : " << patch_table3.size() << endl;
    cout << "----------------EPI--------------"<< endl;
    //cout << "width : " << width << " height : " << height << endl;
    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D_1((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_2((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_3((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_4((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    int index = 0;
    int index2 = 0;
    //! Loop on i_r

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        const unsigned i_r = row_ind[ind_i];
        //! Update of table_2D
        trans_time = clock();
        if (tau_2D == DCT) {
            dct_2d_process(table_2D_1, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_2, img_noisy2, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_3, img_noisy3, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_4, img_noisy4, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());

        }
        else if (tau_2D == BIOR){
            bior_2d_process(table_2D_1, img_noisy, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_2, img_noisy2, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_3, img_noisy3, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_4, img_noisy4, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
        }
        clock_trans = (double)(clock() - trans_time) / CLOCKS_PER_SEC;
        clock_result_trans += clock_trans;

        //왼쪽 바운더리 채우기
        wx_r_table.clear();
        group_3D_table.clear();
        num_group += patch_table2[i_r][1];
        for(int b = 0; b < patch_table2[i_r][1]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r1 = 2;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r1 * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < 1; n++)
                {
                    const unsigned ind = patch_table3[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r1 + c * kHard_2 * nSx_r1] =
                                table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }

            //! HT filtering of the 3D group
            hard_time1 = clock();
            vector<float> weight_table1(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table1, !useSD);
            clock_hard1 = (double) (clock() - hard_time1) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard1;

            const unsigned nSx_r = 2;
            group_3D.resize(chnls * nSx_r * kHard_2, 0.0f);
            //! Build of the 3D group
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table3[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }
            //! HT filtering of the 3D group
            hard_time2 = clock();
            vector<float> weight_table(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard2 = (double) (clock() - hard_time2) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard2;
            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++) {
                    //if (n == 0) continue;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);
                    }
                }

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        }

        inv_trans_time1 = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans1 = (double)(clock() - inv_trans_time1) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans1;

        //! Registration of the weighted estimation
        unsigned dec2 = 0;
        aggreagation_time1 = clock();

        for(int b = 0; b < patch_table2[i_r][1]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r = 1;
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table3[k_r][0] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {

                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kHard + q]
                                              * wx_r_table[c + b * chnls]
                                              * group_3D_table[p * kHard + q + n * kHard_2
                                                               + c * kHard_2 * nSx_r + dec2];
                            denominator[ind] += kaiser_window[p * kHard + q]
                                                * wx_r_table[c + b * chnls];
                        }
                }
            }
            dec2 += chnls * kHard_2;
        }
        clock_aggreagation1 = (double)(clock() - aggreagation_time1) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation1;

        //cout << "left" << endl;
        //오른쪽 바운더리 채우기
        wx_r_table.clear();
        group_3D_table.clear();
        num_group += patch_table2[i_r][2];
        for(int b = 0; b < patch_table2[i_r][2]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r1 = 2;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r1 * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < 1; n++)
                {
                    const unsigned ind = patch_table4[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r1 + c * kHard_2 * nSx_r1] =
                                table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }

            //! HT filtering of the 3D group
            hard_time3 = clock();
            vector<float> weight_table1(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table1, !useSD);
            clock_hard3 = (double) (clock() - hard_time3) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard3;

            const unsigned nSx_r = 2;
            group_3D.resize(chnls * nSx_r * kHard_2, 0.0f);
            //! Build of the 3D group
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table4[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }
            //! HT filtering of the 3D group
            hard_time4 = clock();
            vector<float> weight_table(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard4 = (double) (clock() - hard_time4) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard4;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++) {
                    //if (n == 0) continue;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);
                    }
                }

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        }

        inv_trans_time2 = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans2 = (double)(clock() - inv_trans_time2) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans2;

        //! Registration of the weighted estimation
        unsigned dec3 = 0;
        aggreagation_time2 = clock();
        for(int b = 0; b < patch_table2[i_r][2]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r = 1;
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table4[k_r][0] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator4[ind] += kaiser_window[p * kHard + q]
                                               * wx_r_table[c + b * chnls]
                                               * group_3D_table[p * kHard + q + n * kHard_2
                                                                + c * kHard_2 * nSx_r + dec3];
                            denominator4[ind] += kaiser_window[p * kHard + q]
                                                 * wx_r_table[c + b * chnls];
                        }
                }
            }
            dec3 += chnls * kHard_2;
        }
        clock_aggreagation2 = (double)(clock() - aggreagation_time2) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation2;
        wx_r_table.clear();
        group_3D_table.clear();
        //cout << "right" << endl;
        //! Loop on j_r

        //patch_table2[ind_i].size() : 해당 높이에 따른 EPI 수
        num_group += patch_table2[i_r][0];
        for (unsigned ind_j = 0; ind_j < patch_table2[i_r][0]; ind_j++) {

            //cout << "width : " << ind_j << endl;

            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * (width + 3000) + ind_j;
            //cout << column_ind[ind_j] << " " << k_r << endl;
            //cout << patch_table1[k_r][0];
            //cout << " " << patch_table1[k_r][1];
            //cout << " " << patch_table1[k_r][2];
            //cout << " " << patch_table1[k_r][3] << endl;
            //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1) continue;
            //! Number of similar patches8
            unsigned nSx_r = patch_table1[k_r].size();
            for(int n = 0; n <patch_table1[k_r].size(); n++)
            {
                if(patch_table1[k_r][n] == -1)
                    nSx_r--;
            }
            if(nSx_r == 3)
                nSx_r = 4;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++) {
                if( patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1) {
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 3 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //존재
                else if( patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
                {
                    //cout << ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][0] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][0] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //#01
                else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1)
                {
                    //cout << ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if(n == 0 && patch_table1[k_r][0] == -1){
                            for (unsigned c = 0; c < chnls; c++)
                                for (unsigned n = 0; n < 1; n++)
                                {
                                    const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                    for (unsigned k = 0; k < kHard_2; k++) {
                                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                                table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                                    }
                                }

                            //! HT filtering of the 3D group
                            vector<float> weight_table1(chnls);
                            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                                  lambdaHard3D, weight_table1, !useSD);
                        }
                        else if (n == 1 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][2] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                        else if (n == 3 && patch_table1[k_r][3] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][3] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //#02
                else if(patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] == -1)
                {
                    //cout <<"d "<< ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if(n == 0 && patch_table1[k_r][3] == -1){
                            for (unsigned c = 0; c < chnls; c++)
                                for (unsigned n = 0; n < 1; n++)
                                {
                                    const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                    for (unsigned k = 0; k < kHard_2; k++) {
                                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                                table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                                    }
                                }

                            //! HT filtering of the 3D group
                            vector<float> weight_table1(chnls);
                            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                                  lambdaHard3D, weight_table1, !useSD);
                        }
                        else if (n == 1 && patch_table1[k_r][0] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][0] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                        else if (n == 3 && patch_table1[k_r][2] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //존재
                else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1)
                {
                    //cout <<"dd "<< ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][2] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][3] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][3] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        }
                    }
                }
                //else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
                //{
                    //아무동작안함
                //}

            }
            //! HT filtering of the 3D group
            //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
           // {
                //아무동작안함
            //}
            //else {
                vector<float> weight_table(chnls);
                hard_time = clock();
                ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                      lambdaHard3D, weight_table, !useSD);
                clock_hard = (double) (clock() - hard_time) / CLOCKS_PER_SEC;
                clock_result_hard += clock_hard;

                //! 3D weighting using Standard Deviation
                if (useSD)
                    sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

                //! Save the 3D group. The DCT 2D inverse will be done after.
                //if(patch_table1[k_r].size() == 5) {
                //    cout << "here : " << patch_table1[k_r][0] << " / " << patch_table1[k_r][4] << endl;
                //}
                for (unsigned c = 0; c < chnls; c++) {
                    for (unsigned n = 0; n < nSx_r; n++) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            if(confidence_level_flag[k_r].size() == 1 && confidence_level_flag[k_r][0] == 1) {
                                if (n == 0) {
                                    group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = (
                                            0.7 * group_3D[0 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[1 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[2 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[3 + k * nSx_r + c * kHard_2 * nSx_r]);
                                } else if (n == 1) {
                                    group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = (
                                            0.1 * group_3D[0 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.7 * group_3D[1 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[2 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[3 + k * nSx_r + c * kHard_2 * nSx_r]);
                                } else if (n == 2) {
                                    group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = (
                                            0.1 * group_3D[0 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[1 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.7 * group_3D[2 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[3 + k * nSx_r + c * kHard_2 * nSx_r]);
                                } else if (n == 3) {
                                    group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] = (
                                            0.1 * group_3D[0 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[1 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.1 * group_3D[2 + k * nSx_r + c * kHard_2 * nSx_r]
                                            + 0.7 * group_3D[3 + k * nSx_r + c * kHard_2 * nSx_r]);
                                }

                            }

                            group_3D_table.push_back(group_3D[n + k * nSx_r + c * kHard_2 * nSx_r]);

                        }
                    }
                }

                //! Save weighting
                for (unsigned c = 0; c < chnls; c++) {
                    //cout << "weight : " << weight_table[c] << endl;
                    wx_r_table.push_back(weight_table[c]);
                }
            //}

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        inv_trans_time = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * patch_table2.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans = (double)(clock() - inv_trans_time) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans;
        //! Registration of the weighted estimation
        unsigned dec = 0;
        aggreagation_time = clock();
        for (unsigned ind_j = 0; ind_j < patch_table2[i_r][0]; ind_j++) {
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * (width + 3000) + ind_j;
            //if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 &&
            //    patch_table1[k_r][3] == -1) {
                //아무동작안함
            //} else{
                //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1) continue;
                unsigned nSx_r = patch_table1[k_r].size();
            for (int n = 0; n < patch_table1[k_r].size(); n++) {
                if (patch_table1[k_r][n] == -1)
                    nSx_r--;
            }
            if (nSx_r == 3)
                nSx_r = 4;

            for (unsigned c = 0; c < chnls; c++) {
                for (unsigned n = 0; n < nSx_r; n++) {
                    if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                        patch_table1[k_r][3] != -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {

                                if (n == 0 && patch_table1[k_r][n] != -1) {

                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    //cout << numerator[ind] << endl;
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    } else if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] == -1 &&
                               patch_table1[k_r][3] == -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 0 && patch_table1[k_r][0] != -1) {
                                    const unsigned k = patch_table1[k_r][0] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }

                    }

                        //#01
                    else if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] != -1) {
                        if (n == 0) continue;
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 1 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][3] != -1) {
                                    const unsigned k = patch_table1[k_r][3] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    }

                        // #02
                    else if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] == -1) {
                        if (n == 0) continue;
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 1 && patch_table1[k_r][0] != -1) {
                                    const unsigned k = patch_table1[k_r][0] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    }
                        //
                    else if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] != -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 0 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][3] != -1) {
                                    const unsigned k = patch_table1[k_r][3] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }

                    }
                }
            }
            dec += nSx_r * chnls * kHard_2;
        //}
        }
        clock_aggreagation = (double)(clock() - aggreagation_time) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation;

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++) {
        img_basic[k] = numerator[k] / denominator[k];
        img_basic2[k] = numerator2[k] / denominator2[k];
        img_basic3[k] = numerator3[k] / denominator3[k];
        img_basic4[k] = numerator4[k] / denominator4[k];
    }

    printf("\n< youna >\npatch num : %d\n", num_group);
    printf("hardthresholding processing time : %5.2f\n", clock_result_hard);
    printf("EPI processing time : %5.2f\n", clock_result_EPI);
    printf("transform processing time : %5.2f\n", clock_result_trans);
    printf("inverse trans processing time : %5.2f\n", clock_result_inv_trans);
    printf("aggregation processing time : %5.2f\n", clock_result_aggregation);
    cout << "threshold : " << th << endl;
}

/**
 * @brief Run the final process of BM3D (2nd step). The result
 *        is contained in img_denoised. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: contains the denoised image after the 1st step;
 * @param img_denoised: will contain the final estimate of the denoised
 *        image after the second step;
 * @param width, height, chnls : size of img_noisy;
 * @param nWien: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the second step, otherwise use the norm
 *        of Wiener coefficients of the 3D group;
 * @param tau_2D: DCT or BIOR.
 *
 * @return none.
 **/
void bm3d_2nd_step(
        const float sigma
        ,   vector<float> const& img_noisy
        ,   vector<float> const& img_basic
        ,   vector<float> &img_denoised
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
){
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    vector<float> sigma_table_uv(chnls);
    int sig = 35;
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;
    if (estimate_sigma(sig, sigma_table_uv, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float tauMatch = (sigma_table[0] < 35.0f ? 400 : 3500); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kWien + 1, nWien, pWien);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kWien + 1, nWien, pWien);
    const unsigned kWien_2 = kWien * kWien;
    vector<float> group_3D_table(chnls * kWien_2 * NWien * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> tmp(NWien);

    //! Check allocation memory
    if (img_denoised.size() != img_noisy.size())
        img_denoised.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kWien_2);
    vector<float> coef_norm(kWien_2);
    vector<float> coef_norm_inv(kWien_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kWien);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<int> > patch_table;
    vector<vector<int> > patch_table2;
    precompute(patch_table, patch_table2, img_basic, width, height, kWien, NWien, nWien, pWien, tauMatch);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! DCT_table_2D[p * N + q + (i * width + j) * kWien_2 + c * (2 * ns + 1) * width * kWien_2]
    vector<float> table_2D_img((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);
    vector<float> table_2D_est((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        const unsigned i_r = row_ind[ind_i];

        //! Update of DCT_table_2D
        if (tau_2D == DCT)
        {
            dct_2d_process(table_2D_img, img_noisy, plan_2d_for_1, plan_2d_for_2,
                           nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_est, img_basic, plan_2d_for_1, plan_2d_for_2,
                           nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
                           row_ind[0], row_ind.back());
        }
        else if (tau_2D == BIOR)
        {
            bior_2d_process(table_2D_img, img_noisy, nWien, width, height,
                            chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_est, img_basic, nWien, width, height,
                            chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
        }

        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D_est(chnls * nSx_r * kWien_2, 0.0f);
            vector<float> group_3D_img(chnls * nSx_r * kWien_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table[k_r][n] + (nWien - i_r) * width;
                    for (unsigned k = 0; k < kWien_2; k++)
                    {
                        group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r] =
                                table_2D_est[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                        group_3D_img[n + k * nSx_r + c * kWien_2 * nSx_r] =
                                table_2D_img[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                    }
                }

            //! Wiener filtering of the 3D group
            vector<float> weight_table(chnls);
            wiener_filtering_hadamard(group_3D_img, group_3D_est, tmp, nSx_r, kWien,
                                      chnls, sigma_table, weight_table, !useSD);

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D_est, nSx_r, kWien, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kWien_2; k++)
                        group_3D_table.push_back(group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        } //! End of loop on j_r

        //!  Apply 2D dct inverse
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kWien, NWien * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kWien, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            const unsigned j_r   = column_ind[ind_j];
            const unsigned k_r   = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kWien; p++)
                        for (unsigned q = 0; q < kWien; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kWien + q]
                                              * wx_r_table[c + ind_j * chnls]
                                              * group_3D_table[p * kWien + q + n * kWien_2
                                                               + c * kWien_2 * nSx_r + dec];
                            denominator[ind] += kaiser_window[p * kWien + q]
                                                * wx_r_table[c + ind_j * chnls];
                        }
                }
            }
            dec += nSx_r * chnls * kWien_2;
        }

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++)
        img_denoised[k] = numerator[k] / denominator[k];
}

/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param DCT_table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param plan_1, plan_2 : for convenience. Used by fftw;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param coef_norm : normalization coefficients of the 2D DCT;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(
        vector<float> &DCT_table_2D
        ,   vector<float> const& img
        ,   fftwf_plan * plan_1
        ,   fftwf_plan * plan_2
        ,   const unsigned nHW
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned kHW
        ,   const unsigned i_r
        ,   const unsigned step
        ,   vector<float> const& coef_norm
        ,   const unsigned i_min
        ,   const unsigned i_max
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned size = chnls * kHW_2 * width * (2 * nHW + 1);

    //! If i_r == ns, then we have to process all DCT
    if (i_r == i_min || i_r == i_max)
    {
        //! Allocating Memory
        float* vec = (float*) fftwf_malloc(size * sizeof(float));
        float* dct = (float*) fftwf_malloc(size * sizeof(float));

        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned p = 0; p < kHW; p++)
                        for (unsigned q = 0; q < kHW; q++)
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
                                    img[dc + (i_r + i - nHW + p) * width + j + q];
        }

        //! Process of all DCTs
        fftwf_execute_r2r(*plan_1, vec, dct);
        fftwf_free(vec);

        //! Getting the result
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[dc + (i * width + j) * kHW_2 + k] =
                                dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
        }
        fftwf_free(dct);
    }
    else
    {
        const unsigned ds = step * width * kHW_2;

        //! Re-use of DCT already processed
        for (unsigned c = 0; c < chnls; c++)
        {
            unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
            for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[k + (i * width + j) * kHW_2 + dc] =
                                DCT_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
        }

        //! Compute the new DCT
        float* vec = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));
        float* dct = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));

        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * step;
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned p = 0; p < kHW; p++)
                        for (unsigned q = 0; q < kHW; q++)
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
                                    img[(p + i + 2 * nHW + 1 - step + i_r - nHW)
                                        * width + j + q + dc];
        }

        //! Process of all DCTs
        fftwf_execute_r2r(*plan_2, vec, dct);
        fftwf_free(vec);

        //! Getting the result
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
            const unsigned dc_p = c * kHW_2 * width * step;
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[dc + ((i + 2 * nHW + 1 - step) * width + j) * kHW_2 + k] =
                                dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
        }
        fftwf_free(dct);
    }
}

/**
 * @brief Precompute a 2D bior1.5 transform on all patches contained in
 *        a part of the image.
 *
 * @param bior_table_2D : will contain the 2d bior1.5 transform for all
 *        chosen patches;
 * @param img : image on which the 2d transform will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it;
 * @param lpd : low pass filter of the forward bior1.5 2d transform;
 * @param hpd : high pass filter of the forward bior1.5 2d transform.
 **/
void bior_2d_process(
        vector<float> &bior_table_2D
        ,   vector<float> const& img
        ,   const unsigned nHW
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned chnls
        ,   const unsigned kHW
        ,   const unsigned i_r
        ,   const unsigned step
        ,   const unsigned i_min
        ,   const unsigned i_max
        ,   vector<float> &lpd
        ,   vector<float> &hpd
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    //! If i_r == ns, then we have to process all Bior1.5 transforms
    if (i_r == i_min || i_r == i_max)
    {
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                {
                    bior_2d_forward(img, bior_table_2D, kHW, dc +
                                                             (i_r + i - nHW) * width + j, width,
                                    dc_p + (i * width + j) * kHW_2, lpd, hpd);
                }
        }
    }
    else
    {
        const unsigned ds = step * width * kHW_2;

        //! Re-use of Bior1.5 already processed
        for (unsigned c = 0; c < chnls; c++)
        {
            unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
            for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        bior_table_2D[k + (i * width + j) * kHW_2 + dc] =
                                bior_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
        }

        //! Compute the new Bior
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                {
                    bior_2d_forward(img, bior_table_2D, kHW, dc +
                                                             (i + 2 * nHW + 1 - step + i_r - nHW) * width + j,
                                    width, dc_p + ((i + 2 * nHW + 1 - step)
                                                   * width + j) * kHW_2, lpd, hpd);
                }
        }
    }
}

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only third
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_3D : contains the 3D block for a reference patch;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kHW : size of patches (kHW x kHW);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard3D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void ht_filtering_hadamard(
        vector<float> &group_3D
        ,   vector<float> &tmp
        ,   const unsigned nSx_r
        ,   const unsigned kHard
        ,   const unsigned chnls
        ,   vector<float> const& sigma_table
        ,   const float lambdaHard3D
        ,   vector<float> &weight_table
        ,   const bool doWeight
){

    //! Declarations
    const unsigned kHard_2 = kHard * kHard;
    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = 0.0f;
    const float coef_norm = sqrtf((float) nSx_r);
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kHard_2 * chnls; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * kHard_2;
       //float T = lambdaHard3D * sigma_table[c] * coef_norm;

            const float T = lambdaHard3D * sigma_table[c] * coef_norm;
            for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
            {
                if (fabs(group_3D[k + dc]) > T) {
                    weight_table[c]++;

                }
                else {
                    group_3D[k + dc] = 0.0f;
                    th++;
                }
            }
        //const float T = lambdaHard3D * 1 * coef_norm;
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kHard_2 * chnls; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    for (unsigned k = 0; k < group_3D.size(); k++)
        group_3D[k] *= coef;

    //! Weight for aggregation
    if (doWeight)
        for (unsigned c = 0; c < chnls; c++) {
            weight_table[c] = (weight_table[c] > 0.0f ? 1.0f /
                                                        (float) (sigma_table[c] * sigma_table[c] * weight_table[c])
                                                      : 1.0f);
        }
}

/**
 * @brief Wiener filtering using Hadamard transform.
 *
 * @param group_3D_img : contains the 3D block built on img_noisy;
 * @param group_3D_est : contains the 3D block built on img_basic;
 * @param tmp: allocated vector used in hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kWien : size of patches (kWien x kWien);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void wiener_filtering_hadamard(
        vector<float> &group_3D_img
        ,   vector<float> &group_3D_est
        ,   vector<float> &tmp
        ,   const unsigned nSx_r
        ,   const unsigned kWien
        ,   const unsigned chnls
        ,   vector<float> const& sigma_table
        ,   vector<float> &weight_table
        ,   const bool doWeight
){
    //! Declarations
    const unsigned kWien_2 = kWien * kWien;
    const float coef = 1.0f / (float) nSx_r;

    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = 0.0f;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kWien_2 * chnls; n++)
    {
        hadamard_transform(group_3D_img, tmp, nSx_r, n * nSx_r);
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);
    }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * kWien_2;
        for (unsigned k = 0; k < kWien_2 * nSx_r; k++)
        {
            float value = group_3D_est[dc + k] * group_3D_est[dc + k] * coef;
            value /= (value + sigma_table[c] * sigma_table[c]);
            group_3D_est[k + dc] = group_3D_img[k + dc] * value * coef;
            weight_table[c] += value;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kWien_2 * chnls; n++)
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);

    //! Weight for aggregation
    if (doWeight)
        for (unsigned c = 0; c < chnls; c++)
            weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
                    (sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Apply 2D dct inverse to a lot of patches.
 *
 * @param group_3D_table: contains a huge number of patches;
 * @param kHW : size of patch;
 * @param coef_norm_inv: contains normalization coefficients;
 * @param plan : for convenience. Used by fftw.
 *
 * @return none.
 **/
void dct_2d_inverse(
        vector<float> &group_3D_table
        ,   const unsigned kHW
        ,   const unsigned N
        ,   vector<float> const& coef_norm_inv
        ,   fftwf_plan * plan
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned size = kHW_2 * N;
    const unsigned Ns   = group_3D_table.size() / kHW_2;

    //! Allocate Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    //! Normalization
    for (unsigned n = 0; n < Ns; n++)
        for (unsigned k = 0; k < kHW_2; k++)
            dct[k + n * kHW_2] = group_3D_table[k + n * kHW_2] * coef_norm_inv[k];

    //! 2D dct inverse
    fftwf_execute_r2r(*plan, dct, vec);
    fftwf_free(dct);

    //! Getting the result + normalization
    const float coef = 1.0f / (float)(kHW * 2);
    for (unsigned k = 0; k < group_3D_table.size(); k++)
        group_3D_table[k] = coef * vec[k];

    //! Free Memory
    fftwf_free(vec);
}

void bior_2d_inverse(
        vector<float> &group_3D_table
        ,   const unsigned kHW
        ,   vector<float> const& lpr
        ,   vector<float> const& hpr
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned N = group_3D_table.size() / kHW_2;

    //! Bior process
    for (unsigned n = 0; n < N; n++)
        bior_2d_inverse(group_3D_table, kHW, n * kHW_2, lpr, hpr);
}

/** ----------------- **/
/** - Preprocessing - **/
/** ----------------- **/
/**
 * @brief Preprocess
 *
 * @param kaiser_window[kHW * kHW]: Will contain values of a Kaiser Window;
 * @param coef_norm: Will contain values used to normalize the 2D DCT;
 * @param coef_norm_inv: Will contain values used to normalize the 2D DCT;
 * @param bior1_5_for: will contain coefficients for the bior1.5 forward transform
 * @param bior1_5_inv: will contain coefficients for the bior1.5 inverse transform
 * @param kHW: size of patches (need to be 8 or 12).
 *
 * @return none.
 **/
void preProcess(
        vector<float> &kaiserWindow
        ,   vector<float> &coef_norm
        ,   vector<float> &coef_norm_inv
        ,   const unsigned kHW
){
    //! Kaiser Window coefficients
    if (kHW == 8)
    {
        //! First quarter of the matrix
        kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2989f; kaiserWindow[0 + kHW * 2] = 0.3846f; kaiserWindow[0 + kHW * 3] = 0.4325f;
        kaiserWindow[1 + kHW * 0] = 0.2989f; kaiserWindow[1 + kHW * 1] = 0.4642f; kaiserWindow[1 + kHW * 2] = 0.5974f; kaiserWindow[1 + kHW * 3] = 0.6717f;
        kaiserWindow[2 + kHW * 0] = 0.3846f; kaiserWindow[2 + kHW * 1] = 0.5974f; kaiserWindow[2 + kHW * 2] = 0.7688f; kaiserWindow[2 + kHW * 3] = 0.8644f;
        kaiserWindow[3 + kHW * 0] = 0.4325f; kaiserWindow[3 + kHW * 1] = 0.6717f; kaiserWindow[3 + kHW * 2] = 0.8644f; kaiserWindow[3 + kHW * 3] = 0.9718f;

        //! Completing the rest of the matrix by symmetry
        for(unsigned i = 0; i < kHW / 2; i++)
            for (unsigned j = kHW / 2; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

        for (unsigned i = kHW / 2; i < kHW; i++)
            for (unsigned j = 0; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
    }
    else if (kHW == 12)
    {
        //! First quarter of the matrix
        kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2615f; kaiserWindow[0 + kHW * 2] = 0.3251f; kaiserWindow[0 + kHW * 3] = 0.3782f;  kaiserWindow[0 + kHW * 4] = 0.4163f;  kaiserWindow[0 + kHW * 5] = 0.4362f;
        kaiserWindow[1 + kHW * 0] = 0.2615f; kaiserWindow[1 + kHW * 1] = 0.3554f; kaiserWindow[1 + kHW * 2] = 0.4419f; kaiserWindow[1 + kHW * 3] = 0.5139f;  kaiserWindow[1 + kHW * 4] = 0.5657f;  kaiserWindow[1 + kHW * 5] = 0.5927f;
        kaiserWindow[2 + kHW * 0] = 0.3251f; kaiserWindow[2 + kHW * 1] = 0.4419f; kaiserWindow[2 + kHW * 2] = 0.5494f; kaiserWindow[2 + kHW * 3] = 0.6390f;  kaiserWindow[2 + kHW * 4] = 0.7033f;  kaiserWindow[2 + kHW * 5] = 0.7369f;
        kaiserWindow[3 + kHW * 0] = 0.3782f; kaiserWindow[3 + kHW * 1] = 0.5139f; kaiserWindow[3 + kHW * 2] = 0.6390f; kaiserWindow[3 + kHW * 3] = 0.7433f;  kaiserWindow[3 + kHW * 4] = 0.8181f;  kaiserWindow[3 + kHW * 5] = 0.8572f;
        kaiserWindow[4 + kHW * 0] = 0.4163f; kaiserWindow[4 + kHW * 1] = 0.5657f; kaiserWindow[4 + kHW * 2] = 0.7033f; kaiserWindow[4 + kHW * 3] = 0.8181f;  kaiserWindow[4 + kHW * 4] = 0.9005f;  kaiserWindow[4 + kHW * 5] = 0.9435f;
        kaiserWindow[5 + kHW * 0] = 0.4362f; kaiserWindow[5 + kHW * 1] = 0.5927f; kaiserWindow[5 + kHW * 2] = 0.7369f; kaiserWindow[5 + kHW * 3] = 0.8572f;  kaiserWindow[5 + kHW * 4] = 0.9435f;  kaiserWindow[5 + kHW * 5] = 0.9885f;

        //! Completing the rest of the matrix by symmetry
        for(unsigned i = 0; i < kHW / 2; i++)
            for (unsigned j = kHW / 2; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

        for (unsigned i = kHW / 2; i < kHW; i++)
            for (unsigned j = 0; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
    }
    else
        for (unsigned k = 0; k < kHW * kHW; k++)
            kaiserWindow[k] = 1.0f;

    //! Coefficient of normalization for DCT II and DCT II inverse
    const float coef = 0.5f / ((float) (kHW));
    for (unsigned i = 0; i < kHW; i++)
        for (unsigned j = 0; j < kHW; j++)
        {
            if (i == 0 && j == 0)
            {
                coef_norm    [i * kHW + j] = 0.5f * coef;
                coef_norm_inv[i * kHW + j] = 2.0f;
            }
            else if (i * j == 0)
            {
                coef_norm    [i * kHW + j] = SQRT2_INV * coef;
                coef_norm_inv[i * kHW + j] = SQRT2;
            }
            else
            {
                coef_norm    [i * kHW + j] = 1.0f * coef;
                coef_norm_inv[i * kHW + j] = 1.0f;
            }
        }
}

/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordonnate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param kHW: size of patch
 * @param NHW: maximum similar patches wanted
 * @param nHW: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *
 * @return none.
 **/
void precompute(
        vector<vector<int> > &patch_table1
        ,   vector<vector<int> > &patch_table2
//,   int line_count[]
        ,   const vector<float> &img
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned kHW
        ,   const unsigned NHW
        ,   const unsigned nHW //16 -> 8
        ,   const unsigned pHW
        ,   const float    tauMatch
){
    vector<vector<int> > pre_patch_table;
    ////////////////////////////
    //FILE *check_patchtable1;
    //check_patchtable1 = fopen("check_patchtable1.txt", "w");
    //! Declarations
    const unsigned Ns = 2 * nHW + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table1.size() != (width+3000) * height)
        patch_table1.resize((width+3000) * height);
    if (patch_table2.size() != height)
        patch_table2.resize(height);
    //cout << "patch_table1.size() : " << patch_table1.size();
    // (4,4) -> row/column 각각 +3 증가
    unsigned ind_i = 4;
    unsigned ind_j = 0;
    //int line_count[width * height];
    //memset(line_count, 0, sizeof(int)*(width * height));
    //! Precompute Bloc Matching

    //posotion file open
    string num;
    int position_num;
    ifstream position;
    //position.open("position_balloons.txt");
    //position.open("position_GVD_cham.txt");
    position.open("position.txt");
    position >> num;
    if(position.is_open())
    {
        unsigned k_r;
        string Hei = "Height";
        while(!position.eof()) {
            // 좌표 저장
            //const unsigned k_r = (ind_i * width + ind_j);
            //    patch_table1[k_r].clear();
            if (num.compare(Hei) == 0) // height 단어를 만난 경우
            {
                getline(position, num);
                getline(position, num);
                stringstream sts(num);
                bool first = true;
                while (sts >> position_num) { //숫자 한개 단위로 읽어서 저장
                    if(first == true){
                        //k_r = (ind_i * width + position_num + 4);
                        k_r = (ind_i * (width+3000) + ind_j);
                        patch_table1[k_r].clear();
                        position_num = ind_i*width + position_num + 4;
                        patch_table1[k_r].push_back(position_num);
                        first = false;
                    }else{
                        position_num = ind_i*width + position_num + 4;
                        patch_table1[k_r].push_back(position_num);
                    }
                }
                ind_j++;
            } else {
                stringstream sts(num);
                sts >> position_num;
                //k_r = (ind_i * width + position_num + 4);
                k_r = (ind_i * (width+3000) + ind_j);
                patch_table1[k_r].clear();
                position_num = ind_i*width + position_num+ 4;
                patch_table1[k_r].push_back(position_num);
                getline(position, num);
                stringstream st(num);
                while (st >> position_num) { //숫자 한개 단위로 읽어서 저장
                    position_num = ind_i*width + position_num+ 4;
                    patch_table1[k_r].push_back(position_num);
                }
                ind_j++;
            }

            stringstream sts(num);
            position >> num;
            if (num.compare(Hei) == 0) // height 단어를 만난 경우
            {
                int s = ind_j-4;
                //해당 높이에 그어진 라인 수를 저장
                patch_table2[ind_i].clear();
                patch_table2[ind_i].push_back(ind_j);
                //cout << ind_j << " " << patch_table2[ind_i][0]<< endl;
                ind_i++;
                //cout << ind_i << endl;
                ind_j = 0;
            }
        }
    }
    position.close();
    cout << "end" << endl;
    //for (int a = 0; a < patch_table1.size(); a++) {
    //  for(int b = 0; b < patch_table1[a].size(); b++){
    //if( a == 96719)
    //cout << a << " " << patch_table1[a][b] << " ";
    // }
    //cout << patch_table1[a].size();
    //cout << endl;
    //}
}

/**
 * @brief Process of a weight dependent on the standard
 *        deviation, used during the weighted aggregation.
 *
 * @param group_3D : 3D group
 * @param nSx_r : number of similar patches in the 3D group
 * @param kHW: size of patches
 * @param chnls: number of channels in the image
 * @param weight_table: will contain the weighting for each
 *        channel.
 *
 * @return none.
 **/
void sd_weighting(
        std::vector<float> const& group_3D
        ,   const unsigned nSx_r
        ,   const unsigned kHW
        ,   const unsigned chnls
        ,   std::vector<float> &weight_table
){
    const unsigned N = nSx_r * kHW * kHW;

    for (unsigned c = 0; c < chnls; c++)
    {
        //! Initialization
        float mean = 0.0f;
        float std  = 0.0f;

        //! Compute the sum and the square sum
        for (unsigned k = 0; k < N; k++)
        {
            mean += group_3D[k];
            std  += group_3D[k] * group_3D[k];
        }

        //! Sample standard deviation (Bessel's correction)
        float res = (std - mean * mean / (float) N) / (float) (N - 1);

        //! Return the weight as used in the aggregation
        weight_table[c] = (res > 0.0f ? 1.0f / sqrtf(res) : 0.0f);
    }
}

///////////////////////////////////////////
// BM3D
//////////////////////////////////////////
void precompute_BM(
        vector<vector<unsigned> > &patch_table
        ,   const vector<float> &img
        ,   const unsigned width
        ,   const unsigned height
        ,   const unsigned kHW
        ,   const unsigned NHW
        ,   const unsigned nHW
        ,   const unsigned pHW
        ,   const float    tauMatch
){
    ////////////////////////////
    FILE *check_patchtable1;
    check_patchtable1 = fopen("check_patchtable1.txt", "w");
    ////////////////////////////////
    //! Declarations
    const unsigned Ns = 2 * nHW + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    //cout << "w : " << width << " h : " << height << endl;
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

    //! For each possible distance, precompute inter-patches distance
    for (unsigned di = 0; di <= nHW; di++)
        for (unsigned dj = 0; dj < Ns; dj++)
        {
            const int dk = (int) (di * width + dj) - (int) nHW;
            const unsigned ddk = di * Ns + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned i = nHW; i < height - nHW; i++)
            {
                unsigned k = i * width + nHW;
                for (unsigned j = nHW; j < width - nHW; j++, k++)
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
            }

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHW * width + nHW;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < kHW; p++)
            {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < kHW; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHW + 1; j < width - nHW; j++)
            {
                const unsigned ind = nHW * width + j - 1;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < kHW; p++)
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHW + 1; i < height - nHW; i++)
            {
                const unsigned ind = (i - 1) * width + nHW;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < kHW; q++)
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHW + 1;
                unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
                for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
                {
                    sum_table[ddk][k] =
                            sum_table[ddk][k - 1]
                            + sum_table[ddk][k - width]
                            - sum_table[ddk][k - 1 - width]
                            + diff_table[pq]
                            - diff_table[pq - kHW]
                            - diff_table[pq - kHW * width]
                            + diff_table[pq - kHW - kHW * width];
                }

            }
        }

    //! Precompute Bloc Matching
    vector<pair<float, unsigned> > table_distance;
    //! To avoid reallocation
    table_distance.reserve(Ns * Ns);

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        //if(ind_i == 0 || ind_i == 1)
        //fprintf(check_patchtable1, "height = %d", ind_i);
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //cout << "ind_j : " << column_ind[ind_j] << endl;
            //! Initialization
            const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
            table_distance.clear();
            patch_table[k_r].clear();
            if(ind_j == column_ind.size() - 1 && ind_i == 0) {
                //cout << "k_r : " << k_r << " " << row_ind[ind_i] << " " << column_ind[ind_j]<< endl;
            }
            //if(ind_i == 0 || ind_i == 1)
            //fprintf(check_patchtable1, "k_r : %d, row_ind : %d, column_ind : %d", k_r, row_ind[ind_i], column_ind[ind_j]);
            //! Threshold distances in order to keep similar patches
            for (int dj = -(int) nHW; dj <= (int) nHW; dj++)
            {
                for (int di = 0; di <= (int) nHW; di++)
                    if (sum_table[dj + nHW + di * Ns][k_r] < threshold)
                        table_distance.push_back(make_pair(
                                sum_table[dj + nHW + di * Ns][k_r]
                                , k_r + di * width + dj));

                for (int di = - (int) nHW; di < 0; di++)
                    if (sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold)
                        table_distance.push_back(make_pair(
                                sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj]
                                , k_r + di * width + dj));
            }

            //! We need a power of 2 for the number of similar patches,
            //! because of the Welsh-Hadamard transform on the third dimension.
            //! We assume that NHW is already a power of 2
            const unsigned nSx_r = (NHW > table_distance.size() ?
                                    closest_power_of_2(table_distance.size()) : NHW);

            //! To avoid problem
            if (nSx_r == 1 && table_distance.size() == 0)
            {
                cout << "problem size" << endl;
                table_distance.push_back(make_pair(0, k_r));
            }

            //! Sort patches according to their distance to the reference one
            partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
                         table_distance.end(), ComparaisonFirst);
            if(ind_i == 0 || ind_i == 1)
                fprintf(check_patchtable1, "patch_table =");
            //! Keep a maximum of NHW similar patches
            for (unsigned n = 0; n < nSx_r; n++) {
                patch_table[k_r].push_back(table_distance[n].second);
                //cout << "table_distance[n].second : " << table_distance[n].second << endl;
                if(ind_i == 0 || ind_i == 1)
                    fprintf(check_patchtable1, " %d ", table_distance[n].second);
            }
            fprintf(check_patchtable1, "\n");
            //! To avoid problem
            if (nSx_r == 1)
                patch_table[k_r].push_back(table_distance[0].second);
        }
    }
    //for(int a =0; a < patch_table[16000].size(); a++) {
    //    cout << "patch : " << patch_table[16000][a] << endl;
    //}
}

void bm3d_1st_step_bm3d(
        const float sigma
        ,   vector<float> const& img_noisy
        ,   vector<float> &img_basic
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
){

    clock_t trans_time;
    clock_t inv_trans_time;
    clock_t hard_time;
    clock_t hard_time1;
    clock_t hard_time2;
    clock_t aggreagation_time;
    clock_t blockmatching_time;
    double clock_trans;
    double clock_inv_trans;
    double clock_hard;
    double clock_aggreagation;
    double clock_blockmatching;
    double clock_result_trans = 0;
    double clock_result_inv_trans = 0;
    double clock_result_hard = 0;
    double clock_result_aggregation = 0;
    double clock_result_blockmatching = 0;
    int num_group=0;
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);

    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<unsigned> > patch_table;
    blockmatching_time = clock();
    precompute_BM(patch_table, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);
    clock_result_blockmatching = (double)(clock() - blockmatching_time) / CLOCKS_PER_SEC;
    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        //cout << ind_i << endl;
        trans_time = clock();
        const unsigned i_r = row_ind[ind_i];
        //! Update of table_2D
        if (tau_2D == DCT)
            dct_2d_process(table_2D, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
        else if (tau_2D == BIOR)
            bior_2d_process(table_2D, img_noisy, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
        clock_trans = (double)(clock() - trans_time) / CLOCKS_PER_SEC;
        clock_result_trans += clock_trans;

        wx_r_table.clear();
        group_3D_table.clear();
        num_group += column_ind.size();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //cout << "width spacing : " << column_ind[ind_j] << endl;
            //! Initialization

            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }

            //! HT filtering of the 3D group
            vector<float> weight_table(chnls);
            hard_time = clock();
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard = (double)(clock() - hard_time) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kHard_2; k++)
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        inv_trans_time = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans = (double)(clock() - inv_trans_time) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans;
        //! Registration of the weighted estimation
        unsigned dec = 0;
        aggreagation_time = clock();
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            const unsigned j_r   = column_ind[ind_j];
            //cout << "j_r : " << j_r << endl;
            const unsigned k_r   = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    int a=0;
                    const unsigned k = patch_table[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            //a++;
                            //cout << a << endl;
                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kHard + q]
                                              * wx_r_table[c + ind_j * chnls]
                                              * group_3D_table[p * kHard + q + n * kHard_2
                                                               + c * kHard_2 * nSx_r + dec];
                            denominator[ind] += kaiser_window[p * kHard + q]
                                                * wx_r_table[c + ind_j * chnls];
                        }
                }
            }
            dec += nSx_r * chnls * kHard_2;
        }
        clock_aggreagation = (double)(clock() - aggreagation_time) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation;

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++)
        img_basic[k] = numerator[k] / denominator[k];
    printf("\n< BM3D >\npatch num : %d\n", num_group);
    printf("hardthresholding processing time : %5.2f\n", clock_result_hard);
    printf("transform processing time : %5.2f\n", clock_result_trans);
    printf("inverse trans processing time : %5.2f\n", clock_result_inv_trans);
    printf("aggregation processing time : %5.2f\n", clock_result_aggregation);
    printf("blockmatching processing time : %5.2f\n", clock_result_blockmatching);
}


void bm3d_1st_step2(
        const float sigma
        ,   vector<float> const& img_noisy // symetrized
        ,   vector<float> const&img_basic
        ,   vector<float> &img_basic1
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
){
    clock_t trans_time;
    clock_t inv_trans_time;
    clock_t hard_time;
    clock_t aggreagation_time;
    double clock_trans;
    double clock_inv_trans;
    double clock_hard;
    double clock_aggreagation;
    double clock_result_trans = 0;
    double clock_result_inv_trans = 0;
    double clock_result_hard = 0;
    double clock_result_aggregation = 0;
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic1.size() != img_noisy.size())
        img_basic1.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);

    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<int> > patch_table1;
    vector<vector<int> > patch_table2;
    int line_count[width * height];
    //precompute(patch_table1, patch_table2, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);
    cout << " patch_table1 size : " << patch_table1.size() << " patch_table2 size : " << patch_table2.size() << endl;
    cout << " end BM " << endl;
    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D_1((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_2((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_3((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_4((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    int index = 0;
    int index2 = 0;
    //! Loop on i_r

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        //cout << "height : " << row_ind[ind_i] << endl;
        const unsigned i_r = row_ind[ind_i];
        //! Update of table_2D
        trans_time = clock();
        if (tau_2D == DCT) {
            dct_2d_process(table_2D_1, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_2, img_basic, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());

        }
        else if (tau_2D == BIOR){
            bior_2d_process(table_2D_1, img_noisy, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_2, img_basic, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);

        }
        clock_trans = (double)(clock() - trans_time) / CLOCKS_PER_SEC;
        clock_result_trans += clock_trans;
        wx_r_table.clear();
        group_3D_table.clear();
        //cout << patch_table2[i_r][0] << endl;
        //! Loop on j_r
        //patch_table2[ind_i].size() : 해당 높이에 따른 EPI 수
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            //! Number of similar patches
            unsigned nSx_r = 2;

            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++) {
                for (unsigned n = 0; n < nSx_r; n++) {
                    if (n == 0) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                        }
                    } else if (n == 1) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                        }
                    }
                }
            }
            //! HT filtering of the 3D group
            vector<float> weight_table(chnls);
            hard_time = clock();
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard = (double)(clock() - hard_time) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kHard_2; k++)
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);
        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        inv_trans_time = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * patch_table2.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans = (double)(clock() - inv_trans_time) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans;
        //! Registration of the weighted estimation
        unsigned dec = 0;
        aggreagation_time = clock();

        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            unsigned nSx_r = 2;

            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = k_r + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            if(n == 0) {
                                const unsigned k = k_r + c * width * height;
                                const unsigned ind = k + p * width + q;
                                numerator[ind] += kaiser_window[p * kHard + q]
                                                  * wx_r_table[c + ind_j * chnls]
                                                  * group_3D_table[p * kHard + q + n * kHard_2
                                                                   + c * kHard_2 * nSx_r + dec];
                                denominator[ind] += kaiser_window[p * kHard + q]
                                                    * wx_r_table[c + ind_j * chnls];
                            }

                        }
                }
            }
            dec += nSx_r * chnls * kHard_2;
        }
        clock_aggreagation = (double)(clock() - aggreagation_time) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation;

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++) {
        img_basic1[k] = numerator[k] / denominator[k];
    }
    printf("\n< youna >\nhardthresholding processing time : %5.2f\n", clock_result_hard);
    printf("\ntransform processing time : %5.2f\n", clock_result_trans);
    printf("\ninverse trans processing time : %5.2f\n", clock_result_inv_trans);
    printf("\naggregation processing time : %5.2f\n", clock_result_aggregation);

}


void bm3d_1st_step3(
        const float sigma
        ,   vector<float> const& img_noisy // symetrized
        ,   vector<float> const& img_noisy2
        ,   vector<float> const& img_noisy3
        ,   vector<float> const& img_noisy4
        ,   vector<float> &img_basic
        ,   vector<float> &img_basic2
        ,   vector<float> &img_basic3
        ,   vector<float> &img_basic4
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
){
    clock_t trans_time;
    clock_t inv_trans_time;
    clock_t hard_time;
    clock_t aggreagation_time;
    double clock_trans;
    double clock_inv_trans;
    double clock_hard;
    double clock_aggreagation;
    double clock_result_trans = 0;
    double clock_result_inv_trans = 0;
    double clock_result_hard = 0;
    double clock_result_aggregation = 0;
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if (img_basic2.size() != img_noisy2.size())
        img_basic2.resize(img_noisy2.size());
    if (img_basic3.size() != img_noisy3.size())
        img_basic3.resize(img_noisy3.size());
    if (img_basic4.size() != img_noisy4.size())
        img_basic4.resize(img_noisy4.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);

    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);
    vector<float> denominator2(width * height * chnls, 0.0f);
    vector<float> numerator2  (width * height * chnls, 0.0f);
    vector<float> denominator3(width * height * chnls, 0.0f);
    vector<float> numerator3  (width * height * chnls, 0.0f);
    vector<float> denominator4(width * height * chnls, 0.0f);
    vector<float> numerator4  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<int> > patch_table1;
    vector<vector<int> > patch_table2;
    int line_count[width * height];
    precompute(patch_table1, patch_table2, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);
    cout << " patch_table1 size : " << patch_table1.size() << " patch_table2 size : " << patch_table2.size() << endl;
    cout << " end BM " << endl;
    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D_1((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_2((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_3((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_4((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    int index = 0;
    int index2 = 0;
    //! Loop on i_r

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        //cout << "height : " << row_ind[ind_i] << endl;
        const unsigned i_r = row_ind[ind_i];
        //! Update of table_2D
        trans_time = clock();
        if (tau_2D == DCT) {
            dct_2d_process(table_2D_1, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_2, img_noisy2, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_3, img_noisy3, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_4, img_noisy4, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());

        }
        else if (tau_2D == BIOR){
            bior_2d_process(table_2D_1, img_noisy, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_2, img_noisy2, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_3, img_noisy3, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_4, img_noisy4, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
        }
        clock_trans = (double)(clock() - trans_time) / CLOCKS_PER_SEC;
        clock_result_trans += clock_trans;
        wx_r_table.clear();
        group_3D_table.clear();
        //cout << patch_table2[i_r][0] << endl;
        //! Loop on j_r
        //patch_table2[ind_i].size() : 해당 높이에 따른 EPI 수
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            //! Number of similar patches
            unsigned nSx_r = 4;

            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++) {
                for (unsigned n = 0; n < nSx_r; n++) {
                    if (n == 0) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                        }
                    } else if (n == 1) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                        }
                    } else if (n == 2) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned  ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                        }
                    } else if (n == 3) {
                        for (unsigned k = 0; k < kHard_2; k++) {
                            const unsigned ind = k_r + (nHard - i_r) * width;
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                    table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                        }
                    }
                }
            }
            //! HT filtering of the 3D group
            vector<float> weight_table(chnls);
            hard_time = clock();
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard = (double)(clock() - hard_time) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kHard_2; k++)
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);
        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        inv_trans_time = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * patch_table2.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans = (double)(clock() - inv_trans_time) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans;
        //! Registration of the weighted estimation
        unsigned dec = 0;
        aggreagation_time = clock();

        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            unsigned nSx_r = 4;

            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            if(n == 0) {
                                const unsigned k = k_r + c * width * height;
                                const unsigned ind = k + p * width + q;
                                numerator[ind] += kaiser_window[p * kHard + q]
                                                  * wx_r_table[c + ind_j * chnls]
                                                  * group_3D_table[p * kHard + q + n * kHard_2
                                                                   + c * kHard_2 * nSx_r + dec];
                                denominator[ind] += kaiser_window[p * kHard + q]
                                                    * wx_r_table[c + ind_j * chnls];
                            }

                        }
                }
            }
            dec += nSx_r * chnls * kHard_2;
        }
        clock_aggreagation = (double)(clock() - aggreagation_time) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation;

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++) {
        img_basic[k] = numerator[k] / denominator[k];
        img_basic2[k] = numerator2[k] / denominator2[k];
        img_basic3[k] = numerator3[k] / denominator3[k];
        img_basic4[k] = numerator4[k] / denominator4[k];
    }
    printf("\n< youna >\nhardthresholding processing time : %5.2f\n", clock_result_hard);
    printf("\ntransform processing time : %5.2f\n", clock_result_trans);
    printf("\ninverse trans processing time : %5.2f\n", clock_result_inv_trans);
    printf("\naggregation processing time : %5.2f\n", clock_result_aggregation);

}

void precompute_EPI(
        vector<vector<int> > &patch_table1
        ,   vector<vector<int> > &patch_table2
        ,   vector<vector<int> > &patch_table3
        ,   vector<vector<int> > &patch_table4
        ,   vector<vector<int> > &confidence_level_flag
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
) {
    int th_cl = 15000;
    long k = 0;
    int iFrmHeightC;
    int iNumFrm = 4;
    const int NumFrm = 4;
    int total_frame;
    int num_fr;
    int *index_buf;
    int *pixel_buf;
    int *line_end;
    int *x_buf;
    int *z_buf;
    int *w_buf;
    int *y_buf;
    int **y_buf_1D;
    int *best_num;
    int total = 0;
    int ***cost_infor;
    int aver_cost = 0;
    int ***bestbuf_pixel_sel;
    int ***bestbuf_pixel_sel0;
    int ***bestbuf_pixel_sel1;
    int ***bestbuf_pixel_sel2;
    int left_view = 0;
    int right_view = 0;
    int view_left = 0;
    int view_right = 0;
    const int th_cost = 500;
    int **EPI_position;
    int **line_count;
    int *cost_buf;
    unsigned char *ucInFrame;			//input buffer
    unsigned char ***RaySpace;			//ray space buffer
    unsigned char ***RaySpace_Cb;
    unsigned char ***RaySpace_Cr;
    unsigned char *usEPIOutFrame;		//EPI buffer only for one height
    unsigned char *usOutFrame;			//output buffer
    int **patch_table5;
    unsigned char **ucRayFrame;			//arbitrary ray
    unsigned char **ucRayFrame_cb;
    unsigned char **ucRayFrame_cr;

    double total_var = 0;
    int total_count = 0;

    int temp_cost = INT32_MAX;
    int best_cost = INT32_MAX;
    int cost_diff = INT32_MAX;
    int cost_outlier = INT32_MAX;
    const double w_s = 55;
    const double w_s1 = 55;
    int total_cost_aver = 0;
    int total_line = 0;
    int pixel_sel0 = 0;
    int pixel_sel1 = 0;
    int pixel_sel2 = 0;
    int best_x = 0;
    double best_z = 0;
    double best_w = 0;
    int best_line = 0;
    int linecount = 0;
    int average = 0;
    int max_cost = 0;
    int cost = 0;
    int bound = 0;
    int extra_count = 0;
    int inter_count = 0;
    int bound_count = 0;
    int single_count = 0;
    int outlier_count = 0;
    int count_outlier0 = 0;
    int count_outlier1 = 0;
    int count_cl = 0;
    //int gradient = INT32_MAX;
    if (confidence_level_flag.size() != (width+3000) * (height+8))
        confidence_level_flag.resize((width+3000) * (height+8));
    if (patch_table1.size() != (width+3000) * (height+8))
        patch_table1.resize((width+3000) * (height+8));
    if (patch_table2.size() != (height+8))
        patch_table2.resize((height+8));
    if (patch_table3.size() != (height*bound))
        patch_table3.resize((height*bound));
    if (patch_table4.size() != (height*bound))
        patch_table4.resize((height*bound));
    if (patch_table4.size() != (height*bound))
        patch_table4.resize((height*bound));
    best_num = new int[iFrmWidth + 10];
    x_buf = new int[iNumFrm];
    y_buf = new int[iNumFrm];
    y_buf_1D = new int*[iNumFrm];
    for(int i = 0; i < iNumFrm; i++)
        y_buf_1D[i] = new int[2 * window_size + 1];
    RaySpace = new unsigned char**[iNumFrm];
    for (int NumFrm = 0; NumFrm < iNumFrm; NumFrm++)
    {
        RaySpace[NumFrm] = new unsigned char*[iFrmHeight];
        for (int Height = 0; Height < iFrmHeight; Height++)
            RaySpace[NumFrm][Height] = new unsigned char[iFrmWidth];
    }
    cost_infor = new int **[iNumFrm];
    for (int i = 0; i < iNumFrm; i++) {
        cost_infor[i] = new int*[iFrmWidth + 10];
        for (int j = 0; j < (iFrmWidth + 10); j++)
            cost_infor[i][j] = new int[20];
    }
    line_count = new int*[iNumFrm];
    for (int i = 0; i < iNumFrm; i++)
    {
        line_count[i] = new int[iFrmWidth];
    }
    bestbuf_pixel_sel = new int**[iNumFrm];
    for (int i = 0; i < iNumFrm; i++) {
        bestbuf_pixel_sel[i] = new int*[iFrmHeight];
        for (int Height = 0; Height < iFrmHeight; Height++)
        {
            bestbuf_pixel_sel[i][Height] = new int[iFrmWidth + 10];
        }
    }
    EPI_position = new int*[iNumFrm];
    for (int i = 0; i < iNumFrm; i++)
    {
        EPI_position[i] = new int[iFrmWidth];
    }
    line_end = new int[iNumFrm];
    index_buf = new int[iNumFrm];
    pixel_buf = new int[iNumFrm];
    cost_buf = new int[iNumFrm];

    for (int Height = 0; Height < iFrmHeight; Height++)
    {
        for (int Width = 0; Width < iFrmWidth; Width++)
        {
            RaySpace[0][Height][Width] = img1[Height*iFrmWidth + Width];
            RaySpace[1][Height][Width] = img2[Height*iFrmWidth + Width];
            RaySpace[2][Height][Width] = img3[Height*iFrmWidth + Width];
            RaySpace[3][Height][Width] = img4[Height*iFrmWidth + Width];
        }
    }
    const int size_patch5 = (width+3000)*(height+8);
    patch_table5 = new int*[size_patch5];
    for (int i = 0; i < size_patch5; i++)
    {
        patch_table5[i] = new int[iNumFrm];
    }
    for (int i = 0; i < size_patch5; i++) {
        memset(patch_table5[i], INT32_MAX, sizeof(int) * (iNumFrm));
    }

    // 첫 번째 1view&2view full search
    int ind_h = 8;
    for (int j = 0; j < iFrmHeight; j++)
    { //height
        //cout << ind_h << endl;
        printf("height: %d\n", ind_h);
        for (int i = 0; i < iNumFrm; i++) {
            memset(line_count[i], 0, sizeof(int)*(iFrmWidth));
            for (int j = 0; j < iFrmWidth + 10; j++) {
                memset(cost_infor[i][j], 0, sizeof(int) * 20);

            }
        }
        for (int i = 0; i < iNumFrm; i++) {
            memset(y_buf_1D[i], 0, sizeof(int)*(2*window_size+1));
        }
        memset(y_buf, 0, sizeof(int)*iNumFrm);
        memset(x_buf, 0, sizeof(int)*iNumFrm);
        memset(index_buf, 0, sizeof(int)*iNumFrm);
        memset(pixel_buf, 0, sizeof(int)*iNumFrm);
        memset(line_end, 0, sizeof(int)*iNumFrm);
        memset(cost_buf, 0, sizeof(int)*iNumFrm);

        index_buf[0] = bound;
        index_buf[1] = bound;
        temp_cost = INT32_MAX;
        best_cost = INT32_MAX;
        cost_outlier = INT32_MAX;
        cost_diff = INT32_MAX;
        best_x = 0;
        //int count = 0;
        int pos_x = INT32_MAX;
        int pos_z = INT32_MAX;
        k = 0;
        linecount = 0;
        int ind_j = 0;

        for (;;)
        { // width

            temp_cost = INT32_MAX;
            best_cost = INT32_MAX;
            cost_outlier = INT32_MAX;
            cost_diff = INT32_MAX;
            best_x = 0;

            //cost 구하기

            if (!line_end[0] && index_buf[0] >= bound) // 1번째 view에 대해서
            {
                if (index_buf[0] > iFrmWidth - window_size || index_buf[0] < window_size) { // 1D windo가 범위를 벗어나는 경우
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[0] - x; // 2번째 view에서 예상되는 pixel
                        // line shouldn't be on right of index pixels
                        if (pixel_sel0 > index_buf[1] && pixel_sel0 < iFrmWidth) continue;

                        y_buf[0] = RaySpace[0][j][index_buf[0]];
                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth)
                            y_buf[1] = RaySpace[1][j][pixel_sel0];

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 1) continue;

                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth) // 1view bound 넘어간 상황
                            {
                                cost_diff = (y_buf[y] - RaySpace[0][j][index_buf[0]]) * (y_buf[y] - RaySpace[0][j][index_buf[0]]);
                            }
                            else // 1view bound 넘어가지 않은 상황
                            {
                                cost_diff = (y_buf[y] - RaySpace[0][j][index_buf[0]]) * (y_buf[y] - RaySpace[0][j][index_buf[0]]) + (y_buf[y] - RaySpace[1][j][pixel_sel0]) * (y_buf[y] - RaySpace[1][j][pixel_sel0]);
                                //cost_diff = cost_diff / 2;
                            }
                            if (cost_diff < best_cost)
                            {
                                best_cost = cost_diff;
                                best_x = x;
                            }
                        }
                    }
                }
                else {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[0] - x; // 2번째 view에서 예상되는 1D window내의 중앙pixel
                        // line shouldn't be on right of index pixels
                        if (pixel_sel0 > index_buf[1] && pixel_sel0 < iFrmWidth) continue;

                        for (int s = 0; s < 2 * window_size + 2; s++) {
                            y_buf_1D[0][s] = RaySpace[0][j][index_buf[0] - window_size + s];
                        }
                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth) {
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[1][s] = RaySpace[1][j][pixel_sel0 - window_size + s];
                            }
                        }

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 1) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth) // 1view bound 넘어간 상황
                            {

                                cost_diff = (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]) * (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]);
                                }
                                //cost_diff = cost_diff / (window_size * 2 + 1);
                            }
                            else // 1view bound 넘어가지 않은 상황
                            {
                                cost_diff = (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]) * (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]);
                                }
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[1][j][pixel_sel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][pixel_sel0 + s]);
                                }
                                //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                            }

                            // 주변 기울기를 고려한 cost
                            if(j > 4 && j < 960 && index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20))
                            {
                                // 0번째 view 분산값
                                int d = INT32_MAX;
                                int m = 0;
                                int temp = INT32_MAX;
                                int h_m = INT32_MAX;
                                int w_m = INT32_MAX;
                                int grad0[28];
                                //0번째 view 중간값 계산
                                int c = 0;
                                for (int h = -1; h < 1; h++) {
                                    for (int w = -4; w < 4; w++) {
                                        if((h == 0) && (w == 0)) break;
                                        int ind = (j + h) * (width + 3000) + (index_buf[0] + w); // current pixel (= center pixel)
                                        if (patch_table5[ind][0] == 0) continue;
                                        if(temp > patch_table5[ind][0]) temp = patch_table5[ind][0];
                                        if(m < patch_table5[ind][0]) m = patch_table5[ind][0];
                                        //cout << patch_table5[ind][0] << " ";
                                        grad0[c] = patch_table5[ind][0];
                                        c++;
                                    }
                                    //cout << endl;
                                }
                                //주변 기울기의 중간값
                                int median0 = (grad0[14] + grad0[15])/2;
                                int med = (temp + m) / 2;
                                //outlier를 반영한 cost
                                cost_outlier = abs(temp - x);
                            }
                            if(j < 15 || index_buf[0] < bound + 8) cost_outlier = 0;
                            temp_cost = cost_diff + w_s * cost_outlier;
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;

                            }
                        }
                    }
                }
            }
            x_buf[0] = best_x;
            cost_buf[0] = best_cost;
            int buf1 = best_x;

            //cout << "dd1 " << endl;
            best_x = 0;
            best_z = 0;
            best_cost = INT32_MAX;
            temp_cost = INT32_MAX;
            cost_outlier = INT32_MAX;
            cost_diff = INT32_MAX;

            if (!line_end[1] && index_buf[1] >= bound) // 2번째 view에 대해서
            {
                if (index_buf[1] > iFrmWidth - window_size || index_buf[1] < window_size) {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[1] + x; // 1번째 view에서 예상되는 pixel

                        if (pixel_sel0 > index_buf[0] && pixel_sel0 < iFrmWidth) continue;

                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth)
                            y_buf[0] = RaySpace[0][j][pixel_sel0];
                        y_buf[1] = RaySpace[1][j][index_buf[1]];

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 0) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth)
                            {
                                cost_diff = (y_buf[y] - RaySpace[1][j][index_buf[1]]) * (y_buf[y] - RaySpace[1][j][index_buf[1]]);
                            }
                            else
                            {
                                cost_diff = (y_buf[y] - RaySpace[1][j][index_buf[1]]) * (y_buf[y] - RaySpace[1][j][index_buf[1]]);
                                cost_diff += (y_buf[y] - RaySpace[0][j][pixel_sel0]) * (y_buf[y] - RaySpace[0][j][pixel_sel0]);
                                //cost_diff = cost_diff / 2;
                            }
                            if (cost_diff < best_cost)
                            {
                                best_cost = cost_diff;
                                best_x = x;
                            }
                        }
                    }
                }
                else {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[1] + x; // 1번째 view에서 예상되는 pixel

                        if (pixel_sel0 > index_buf[0] && pixel_sel0 < iFrmWidth) continue;

                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth) {
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[0][s] = RaySpace[0][j][pixel_sel0 - window_size + s];
                            }
                        }
                        for (int s = 0; s < 2 * window_size + 2; s++) {
                            y_buf_1D[1][s] = RaySpace[1][j][index_buf[1] - window_size + s];
                        }

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 0) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth)
                            {
                                cost_diff = (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]) * (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]);
                                }
                                //cost_diff = cost_diff / 1 * (window_size * 2 + 1);
                            }
                            else
                            {
                                cost_diff = (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]) * (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]);
                                }
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[0][j][pixel_sel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][pixel_sel0 + s]);
                                }
                                //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                            }

                            if(j > 4 && j < 960 && index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20))
                            {
                                // 0번째 view 분산값
                                int d = INT32_MAX;
                                int m = 0;
                                int temp = INT32_MAX;
                                int h_m = INT32_MAX;
                                int w_m = INT32_MAX;
                                int grad1[28];
                                int c = 0;
                                for (int h = -1; h < 1; h++) {
                                    for (int w = -4; w < 4; w++) {
                                        if((h == 0) && (w == 0)) break;
                                        int ind = (j + h) * (width + 3000) + (index_buf[1] + w); // current pixel (= center pixel)
                                        if (patch_table5[ind][0] == 0) continue;
                                        //cout << patch_table5[ind][0] << endl;
                                        if(temp > patch_table5[ind][0]) temp = patch_table5[ind][0];
                                        if(m < patch_table5[ind][0]) m = patch_table5[ind][0];
                                        grad1[c] = patch_table5[ind][0];
                                        //cout << grad1[c] << " ";
                                        c++;
                                    }
                                    //cout << endl;
                                }
                                //주변 기울기의 중간값
                                int median1 = (grad1[14] + grad1[15])/2;
                                int med = (temp + m) / 2;
                                //cout << "med : " << med << " temp : " << temp << " m : " << m << endl;
                                //outlier를 반영한 cost
                                cost_outlier = abs(temp - x);
                            }
                            if(j < 15 || index_buf[1] < bound + 8) cost_outlier = 0;
                            temp_cost = cost_diff + w_s * cost_outlier;
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;
                            }
                        }
                    }
                }
            }
            x_buf[1] = best_x;
            cost_buf[1] = best_cost;
            int buf2 = best_x;

            //minimum cost 저장 buf
            int save_cost[2];
            int w_mv = 0;
            save_cost[0] = cost_buf[0];
            save_cost[1] = cost_buf[1];
            // best cost
            if (save_cost[0] < save_cost[1])
            {
                best_line = 0;
                if(save_cost[0] < th_cl && save_cost[0] > 0) {
                    int k_r = (ind_h * (width + 3000) + index_buf[0]);
                    confidence_level_flag[k_r].clear();
                    confidence_level_flag[k_r].push_back(1);
                    confidence_level_flag[k_r].push_back(0);
                    confidence_level_flag[k_r].push_back(0);
                }
            }
            else
            {
                best_line = 1;
                if(save_cost[1] < th_cl && save_cost[1] > 0) {
                    int k_r = (ind_h * (width + 3000) + pixel_sel0);
                    confidence_level_flag[k_r].clear();
                    confidence_level_flag[k_r].push_back(1);
                    confidence_level_flag[k_r].push_back(0);
                    confidence_level_flag[k_r].push_back(0);
                }
            }
            // 중복되는 라인이 있는지 검사하고 직선위에있는 pixel의 position 저장하기
            if ((!line_end[0]) && (best_line == 0))
            {
                // 0 번째 view에서 해당하는 위치
                pixel_sel0 = index_buf[0] - x_buf[0];
                //printf("index_buf[0] : %d\n", index_buf[0]);
                if (line_count[0][index_buf[0]] != 0) {
                    if (save_cost[0] < cost_infor[0][index_buf[0]][0]) {
                        total_cost_aver = total_cost_aver - cost_infor[0][index_buf[0]][0];
                        cost_infor[0][index_buf[0]][0] = save_cost[0];
                        total_cost_aver += cost_infor[0][index_buf[0]][0];
                        best_num[index_buf[0]] = 0;
                        int temp = bestbuf_pixel_sel[1][j][index_buf[0]];
                        line_count[1][temp]--;
                        ///////////////////////////////////////////////////////
                        //배열에 position 저장

                        if (pixel_sel0 < 0) // 1번째 view 픽셀만 라인이 그어짐
                        {
                            bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                            bestbuf_pixel_sel[1][j][index_buf[0]] = 0;
                            EPI_position[1][0] = index_buf[0];
                            line_count[1][0]++;
                        }
                        else // 1번째 2번째 view 에만 라인이 그어짐
                        {
                            bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                            bestbuf_pixel_sel[1][j][index_buf[0]] = pixel_sel0;
                            EPI_position[1][pixel_sel0] = index_buf[0];
                            line_count[1][pixel_sel0]++;
                        }
                        int k_r = (j * (width + 3000) + index_buf[0]);
                        int gradient = bestbuf_pixel_sel[0][j][index_buf[0]] - bestbuf_pixel_sel[1][j][index_buf[0]];
                        patch_table5[k_r][0]= gradient;

                    }
                }
                else // 중복된 라인이 없는 경우
                {

                    inter_count++;
                    cost_infor[0][index_buf[0]][0] = save_cost[0];
                    total_cost_aver += cost_infor[0][index_buf[0]][0];
                    total_line++;
                    line_count[0][index_buf[0]]++;
                    line_count[1][pixel_sel0]++;

                    if (pixel_sel0 < 0) // 1번째 view 픽셀만 라인이 그어짐
                    {
                        bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                        bestbuf_pixel_sel[1][j][index_buf[0]] = 0;
                        EPI_position[1][0] = index_buf[0];
                    }
                    else // 1번째 2번째 view 에만 라인이 그어짐
                    {
                        bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                        bestbuf_pixel_sel[1][j][index_buf[0]] = pixel_sel0;
                        EPI_position[1][pixel_sel0] = index_buf[0];
                    }
                }
                int k_r = (j * (width + 3000)) + index_buf[0];
                int gradient = bestbuf_pixel_sel[0][j][index_buf[0]] - bestbuf_pixel_sel[1][j][index_buf[0]];
                patch_table5[k_r][0] = gradient;
                //cout << "gradient : " << gradient << " 1view : " << bestbuf_pixel_sel[0][j][index_buf[0]] << " 2view : " << bestbuf_pixel_sel[1][j][index_buf[0]] << endl;
                for (int v = 2; v < iNumFrm; v++)
                {
                    temp_cost = INT32_MAX;
                    cost_diff = INT32_MAX;
                    best_cost = INT32_MAX;
                    best_x = 0;
                    int pixel0 = bestbuf_pixel_sel[v-2][j][index_buf[0]];
                    int pixel1 = bestbuf_pixel_sel[v-1][j][index_buf[0]];
                    int po = 0;
                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = pixel1 + x; // 2번째 view에서 예상되는 pixel
                            if (po < 0 || po > iFrmWidth) continue;
                            y_buf[0] = RaySpace[v-1][j][pixel1];
                            if (0 <= po && po < iFrmWidth)
                                y_buf[1] = RaySpace[v][j][po];

                            for (int y = 0; y < 2; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {
                                    cost_diff = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    cost_diff = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]) + (y_buf[y] - RaySpace[v][j][po]) * (y_buf[y] - RaySpace[v][j][po]);
                                    //cost_diff = cost_diff / 2;
                                }
                                if (cost_diff < best_cost)
                                {
                                    best_cost = cost_diff;
                                    best_x = po;
                                    //EPI_position[v][po] = index_buf[0];

                                }
                            }
                        }
                    }
                    else
                    {
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = 2 * pixel1 - pixel0 +  x;
                            if (po < 0 || po > iFrmWidth) continue;
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[v - 1][s] = RaySpace[v - 1][j][pixel1 - window_size + s];
                            }
                            if (0 <= po && po < iFrmWidth) {
                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[v][s] = RaySpace[v][j][po - window_size + s];
                                }
                            }

                            for (int y = v-1; y < v+1; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {

                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    //cost_diff = cost_diff / (window_size * 2 + 1);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]);
                                    }
                                    //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                                }

                                if(j > 4 && j < 960 && index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20)){
                                // 0번째 view 분산값
                                    int d = INT32_MAX;
                                    int m = 0;
                                    int temp = INT32_MAX;
                                    int h_m = INT32_MAX;
                                    int w_m = INT32_MAX;
                                    int grad0[28];
                                    //0번째 view 중간값 계산
                                    int c = 0;
                                    for (int h = -1; h < 1; h++) {
                                        for (int w = -4; w < 4; w++) {
                                            if((h == 0) && (w == 0)) break;
                                            int ind = (j + h) * (width + 3000) + (pixel1 + w); // current pixel (= center pixel)
                                            if (patch_table5[ind][v-1] == 0 || patch_table5[ind][v-1] == -1) continue;
                                            if(temp > patch_table5[ind][v-1]) temp = patch_table5[ind][v-1];
                                            if(m < patch_table5[ind][v-1]) m = patch_table5[ind][v-1];
                                            grad0[c] = patch_table5[ind][v-1];
                                            c++;
                                        }
                                        //cout << endl;
                                    }
                                    //주변 기울기의 중간값
                                    int median1 = (grad0[(c+1)/2] + grad0[(c+2)/2])/2;
                                    int med = (temp + m) / 2;
                                    //outlier를 반영한 cost
                                    cost_outlier = abs(temp - abs(x));
                                }
                                if(j < 15 || pixel1 < bound + 8) cost_outlier = 0;
                                temp_cost = cost_diff + w_s1 * cost_outlier;
                                if (temp_cost < best_cost)
                                {
                                    int k_r = (ind_h * (width + 3000) + index_buf[0]);
                                    if(temp_cost < th_cl && confidence_level_flag[k_r].size() == 3 && confidence_level_flag[k_r][0] == 1 && temp_cost > 0) {
                                        confidence_level_flag[k_r][v-1] = 1;
                                    }
                                    best_cost = temp_cost;
                                    best_x = po;
                                }
                            }

                        }
                    }
                    bestbuf_pixel_sel[v][j][index_buf[0]] = best_x;
                    line_count[v][best_x]++;
                    if(index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20)) {
                        int k_r = (j * (width + 3000) + best_x);
                        patch_table5[k_r][v-1] = pixel1 - best_x;
                    }
                    //total_line++;
                    if(index_buf[0] < iFrmWidth)
                        EPI_position[v][best_x] = index_buf[0];
                }
            }
            if ((!line_end[1]) && (best_line == 1))
            {
                // 0 번째 view에서 해당하는 위치
                pixel_sel0 = index_buf[1] + x_buf[1];
                if (line_count[0][pixel_sel0] != 0) {
                    if (save_cost[1] < cost_infor[0][pixel_sel0][0]) {

                        //if(j > 390 && j < 550 && index_buf[1] >= 540 && index_buf[1] <= 710)
                        total_cost_aver = total_cost_aver - cost_infor[0][pixel_sel0][0];
                        cost_infor[0][pixel_sel0][0] = save_cost[1];
                        //if(j > 390 && j < 550 && index_buf[1] >= 540 && index_buf[1] <= 710)
                        total_cost_aver += cost_infor[0][pixel_sel0][0];

                        int temp = bestbuf_pixel_sel[1][j][pixel_sel0];
                        line_count[1][temp]--;

                        if(pixel_sel0 < iFrmWidth) {
                            bestbuf_pixel_sel[0][j][pixel_sel0] = pixel_sel0;
                            bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                            EPI_position[1][index_buf[1]] = pixel_sel0;
                            line_count[1][index_buf[1]]++;
                        }
                        else
                        {
                            bestbuf_pixel_sel[0][j][pixel_sel0] = iFrmWidth - 1;
                            bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                            EPI_position[1][index_buf[1]] = iFrmWidth - 1;
                            line_count[1][index_buf[1]]++;
                        }
                        int k_r = (j * (width + 3000) + pixel_sel0);
                        int gradient = bestbuf_pixel_sel[0][j][pixel_sel0] - bestbuf_pixel_sel[1][j][pixel_sel0];
                        patch_table5[k_r][0] = gradient;
                    }
                }
                else
                {
                    inter_count++;
                    best_num[pixel_sel0] = 1;
                    line_count[0][pixel_sel0]++;
                    line_count[1][index_buf[1]]++;
                    cost_infor[0][pixel_sel0][0] = save_cost[1];
                    //if(j > 390 && j < 550 && index_buf[1] >= 540 && index_buf[1] <= 710)
                    total_cost_aver += cost_infor[0][pixel_sel0][0];
                    //cout << " best_cost : " << best_cost << endl;
                    total_line++;
                    if (pixel_sel0 < iFrmWidth) {
                        bestbuf_pixel_sel[0][j][pixel_sel0] = pixel_sel0;
                        bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                        EPI_position[1][index_buf[1]] = pixel_sel0;
                    }
                    else
                    {
                        bestbuf_pixel_sel[0][j][pixel_sel0] = iFrmWidth - 1;
                        bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                        EPI_position[1][index_buf[1]] = iFrmWidth - 1;
                    }
                    int k_r = (j * (width + 3000) + pixel_sel0);
                    int gradient = bestbuf_pixel_sel[0][j][pixel_sel0] - bestbuf_pixel_sel[1][j][pixel_sel0];
                    patch_table5[k_r][0] = gradient;
                    //cout << patch_table5[k_r].size() << endl;
                }
                for (int v = 2; v < iNumFrm; v++)
                {
                    temp_cost = INT32_MAX;
                    cost_diff = INT32_MAX;
                    best_cost = INT32_MAX;
                    best_x = 0;
                    int pixel0 = bestbuf_pixel_sel[v - 2][j][pixel_sel0];
                    int pixel1 = bestbuf_pixel_sel[v - 1][j][pixel_sel0];
                    int po = 0;
                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                        for (int x = -(search_range2); x < search_range2; x++) //search range 10
                        {
                            po = pixel1 + x; // 2번째 view에서 예상되는 pixel
                            if (po < 0 || po > iFrmWidth) continue;
                            y_buf[0] = RaySpace[v - 1][j][pixel1];
                            if (0 <= po && po < iFrmWidth)
                                y_buf[1] = RaySpace[v][j][po];

                            for (int y = 0; y < 2; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {
                                    cost_diff = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    cost_diff = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]) + (y_buf[y] - RaySpace[v][j][po]) * (y_buf[y] - RaySpace[v][j][po]);
                                    //cost_diff = cost_diff / 2;
                                }
                                if (cost_diff < best_cost)
                                {
                                    cost_diff = cost_diff;
                                    best_x = po;
                                    //EPI_position[v][po] = pixel_sel0;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = 2 * pixel1 - pixel0 + x;
                            if (po < 0 || po > iFrmWidth) continue;
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[v - 1][s] = RaySpace[v - 1][j][pixel1 - window_size + s];
                            }
                            if (0 <= po && po < iFrmWidth) {
                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[v][s] = RaySpace[v][j][po - window_size + s];
                                }
                            }

                            for (int y = v - 1; y < v + 1; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {

                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    //cost_diff = cost_diff / (window_size * 2 + 1);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]);
                                    }
                                    //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                                }

                                if(j > 4 && j < 960 && index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20)){
                                // 0번째 view 분산값
                                    int d = INT32_MAX;
                                    int m = 0;
                                    int temp = INT32_MAX;
                                    int h_m = INT32_MAX;
                                    int w_m = INT32_MAX;
                                    int grad1[28];

                                    int c = 0;
                                    for (int h = -1; h < 1; h++) {
                                        for (int w = -4; w < 4; w++) {
                                            if((h == 0) && (w == 0)) break;
                                            int ind = (j + h) * (width + 3000) + (pixel1 + w); // current pixel (= center pixel)
                                            if (patch_table5[ind][v-1] == 0|| patch_table5[ind][v-1] == -1) continue;
                                            if(temp > patch_table5[ind][v-1]) temp = patch_table5[ind][v-1];
                                            if(m < patch_table5[ind][v-1]) m = patch_table5[ind][v-1];
                                            grad1[c] = patch_table5[ind][0];
                                            //cout << grad1[c] << " ";
                                            c++;
                                        }
                                        //cout << endl;
                                    }
                                    //주변 기울기의 중간값
                                    int median1 = (grad1[14] + grad1[15])/2;
                                    int med = (temp + x) / 2;
                                    //outlier를 반영한 cost
                                    cost_outlier = abs(temp - abs(x));
                                }
                                if(j < 15 || pixel1 < bound + 8) cost_outlier = 0;
                                temp_cost = cost_diff + w_s1 * cost_outlier;
                                if (temp_cost < best_cost)
                                {
                                    int k_r = (ind_h * (width + 3000) + pixel_sel0);
                                    if(temp_cost < th_cl && confidence_level_flag[k_r].size() == 3 && confidence_level_flag[k_r][0] == 1 && temp_cost > 0) {
                                        confidence_level_flag[k_r][v-1] = 1;
                                    }
                                    best_cost = temp_cost;
                                    best_x = po;
                                    //EPI_position[v][po] = pixel_sel0;

                                }
                            }

                        }
                    }
                    bestbuf_pixel_sel[v][j][pixel_sel0] = best_x;

                    if(index_buf[0] >= bound + 8 && index_buf[1] >= bound + 8 && index_buf[0] <= iFrmWidth - (bound + 20) && index_buf[1] <= iFrmWidth - (bound + 20)) {
                        int k_r = (j * (width + 3000) + best_x);
                        patch_table5[k_r][v-1] = pixel1 - best_x;
                    }
                    if (pixel_sel0 < iFrmWidth)
                        EPI_position[v][best_x] = pixel_sel0;
                }

            }
            index_buf[best_line]++;

            // end line check
            if (index_buf[0] > iFrmWidth - bound) {
                line_end[0] = 1;
            }
            if (index_buf[1] > iFrmWidth - bound) {
                line_end[1] = 1;
            }
            if ((line_end[0] == 1) && (line_end[1] == 1))
            {
                for (int b = bound; b < iFrmWidth - bound; b++)
                {
                    if (bestbuf_pixel_sel[0][j][b] < iFrmWidth - 1 && bestbuf_pixel_sel[1][j][b] < iFrmWidth - 1 && bestbuf_pixel_sel[2][j][b] < iFrmWidth - 1) {
                        if (!(bestbuf_pixel_sel[0][j][b] == 0 && bestbuf_pixel_sel[1][j][b] == 0 &&
                              bestbuf_pixel_sel[2][j][b] == 0 && bestbuf_pixel_sel[3][j][b] == 0)) {
                            if(j == 500)
                                //cout << b << " " << bestbuf_pixel_sel[1][j][b] << " " << bestbuf_pixel_sel[2][j][b]  << " " << bestbuf_pixel_sel[3][j][b] << endl;
                            line_count[1][bestbuf_pixel_sel[1][j][b]]++;
                            line_count[2][bestbuf_pixel_sel[2][j][b]]++;
                            line_count[3][bestbuf_pixel_sel[3][j][b]]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int ind1 = (j * (width + 3000) + bestbuf_pixel_sel[1][j][b]);
                            int ind2 = (j * (width + 3000) + bestbuf_pixel_sel[2][j][b]);
                            int ind3 = (j * (width + 3000) + bestbuf_pixel_sel[3][j][b]);

                            int position_num1 = ind_h * width + bestbuf_pixel_sel[0][j][b] + 8;
                            int position_num2 = ind_h * width + bestbuf_pixel_sel[1][j][b] + 8;
                            int position_num3 = ind_h * width + bestbuf_pixel_sel[2][j][b] + 8;
                            int position_num4 = ind_h * width + bestbuf_pixel_sel[3][j][b] + 8;
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            //if(b > 238 && b < 288 && ind_h > 305 && ind_h < 384)
                            //{
                            //    cout << "(x, y) : " << ind_h << " , " << b << endl;
                            //    cout << "1view : " << bestbuf_pixel_sel[0][j][b] << " 2view : " << bestbuf_pixel_sel[1][j][b] << " 3view : " << bestbuf_pixel_sel[2][j][b] << " 4view : " << bestbuf_pixel_sel[3][j][b] << endl;
                            //}
                            if(confidence_level_flag[k_r].size() == 3 && confidence_level_flag[k_r][0] == 1 && confidence_level_flag[k_r][1] == 1 && confidence_level_flag[k_r][2] == 1) {
                                count_cl++;
                                confidence_level_flag[k_r].clear();
                                confidence_level_flag[k_r].push_back(1);
                            }

                            ind_j++;
                        }
                    }
                }
                // 빈영역 ㅇ채우기
                for (int b = bound; b < iFrmWidth - bound; b+= 3)
                {
                    for (int a = 1; a < iNumFrm; a++) {

                        if (line_count[a][b] == 0 &&(line_count[a][b-2] == 0 || line_count[a][b-1] == 0))
                        {
                            extra_count++;
                            //if(a==1&& ind_h==20) cout << b << endl;
                            int pixel0 = 0;
                            if (b == 0) pixel0 = EPI_position[a][0];
                            else pixel0 = EPI_position[a][b - 3]; //left pixel의 0번째 view에서의 위치
                            int pos[NumFrm];
                            int EPI[NumFrm];
                            EPI[a] = b; //b에 대한 EPI
                            for (int x = 0; x < iNumFrm; x++)
                            {
                                pos[x] = bestbuf_pixel_sel[x][j][pixel0]; //left pixel의 EPI를 가져옴

                            }

                            if (a < iNumFrm - 1) {
                                for (int p1 = a; p1 < iNumFrm - 1; p1++) //a번째 이상 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    cost_diff = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = EPI[p1] - (pos[p1] - pos[p1 + 1]); //예상위치
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p1][j][EPI[p1]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p1 + 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p1][j][EPI[p1]]) * (y_buf[y] - RaySpace[p1][j][EPI[p1]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p1][j][EPI[p1]]) * (y_buf[y] - RaySpace[p1][j][EPI[p1]]) + (y_buf[y] - RaySpace[p1 + 1][j][po]) * (y_buf[y] - RaySpace[p1 + 1][j][po]);
                                                    //cost_diff = cost_diff / 2;
                                                }

                                                if(j < 15 || index_buf[1] < bound + 8) cost_outlier = 0;
                                                temp_cost = cost_diff + w_s * cost_outlier;

                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p1 + 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p1 + 1] = best_x;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p1][s] = RaySpace[p1][j][EPI[p1] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p1 + 1][s] = RaySpace[p1 + 1][j][po - window_size + s];
                                                }
                                            }

                                            for (int y = p1; y < p1 + 2; y++)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]) * (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]);
                                                    }
                                                    //cost_diff = cost_diff / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]) * (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p1 + 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1 + 1][j][po + s]);
                                                    }
                                                    //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                                                }
                                                if(j > 4 && j < 960 && b >= bound + 8 && b >= bound + 8)
                                                {
                                                    // 0번째 view 분산값
                                                    int d = INT32_MAX;
                                                    int m = 0;
                                                    int temp = INT32_MAX;
                                                    int h_m = INT32_MAX;
                                                    int w_m = INT32_MAX;
                                                    int grad1[28];

                                                    //cout << "dd" << endl;
                                                    //0번째 view 중간값 계산
                                                    //if(index_buf[0] >530 && index_buf[0] <700 && j > 560 && j <730)
                                                    //cout << "index_buf[0] : " << index_buf[0] << "height : " << j << endl;
                                                    int c = 0;
                                                    for (int h = -1; h < 1; h++) {
                                                        for (int w = -4; w < 4; w++) {
                                                            if((h == 0) && (w == 0)) break;
                                                            int ind = (j + h) * (width + 3000) + (EPI[a] + w); // current pixel (= center pixel)
                                                            if (patch_table5[ind][a] == 0 || patch_table5[ind][a] == -1) continue;
                                                            //cout << patch_table5[ind][0] << endl;
                                                            if(temp > patch_table5[ind][a]) temp = patch_table5[ind][a];
                                                            if(m < patch_table5[ind][a]) m = patch_table5[ind][a];
                                                            grad1[c] = patch_table5[ind][a];
                                                            //cout << grad1[c] << " ";
                                                            c++;
                                                        }
                                                        //cout << endl;
                                                    }
                                                    //주변 기울기의 중간값
                                                    int median1 = (grad1[14] + grad1[15])/2;
                                                    int med = (temp + m) / 2;
                                                    //outlier를 반영한 cost
                                                    cost_outlier = abs(temp - x);
                                                }
                                                if(j < 15 || index_buf[1] < bound + 8) cost_outlier = 0;
                                                temp_cost = cost_diff + w_s1 * cost_outlier;
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p1 + 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p1 + 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    //EPI_position[p1][po] = pixel_sel0;

                                                }
                                            }

                                        }
                                    }
                                    //EPI[p1 + 1] = best_x;
                                }
                                //printf("%d %d\n", EPI_position[a][b - 1], b);
                                for (int p2 = a; p2 > 0; p2--) //a보다 이전 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    cost_diff = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = 0;
                                    pixel1 = EPI[p2] + (pos[p2 - 1] - pos[p2]); //예상위치
                                    //printf("pixel 1 : %d %d %d %d\n", pixel1, EPI[p2], pos[p2 - 1] , pos[p2]);
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p2][j][EPI[p2]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p2 - 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]) + (y_buf[y] - RaySpace[p2 - 1][j][po]) * (y_buf[y] - RaySpace[p2 - 1][j][po]);
                                                    //cost_diff = cost_diff / 2;
                                                }

                                                if (cost_diff < best_cost)
                                                {
                                                    best_cost = cost_diff;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p2][s] = RaySpace[p2][j][EPI[p2] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p2 - 1][s] = RaySpace[p2 - 1][j][po - window_size + s];
                                                }
                                            }
                                            for (int y = p2; y > p2 - 2; y--)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    //cost_diff = cost_diff / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]);
                                                    }
                                                    //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                                                }
                                                if(j > 4 && j < 960 && b >= bound + 8 && b >= bound + 8)
                                                {
                                                    // 0번째 view 분산값
                                                    int d = INT32_MAX;
                                                    int m = 0;
                                                    int temp = INT32_MAX;
                                                    int h_m = INT32_MAX;
                                                    int w_m = INT32_MAX;
                                                    int grad1[28];

                                                    //cout << "dd" << endl;
                                                    //0번째 view 중간값 계산
                                                    //if(index_buf[0] >530 && index_buf[0] <700 && j > 560 && j <730)
                                                    //cout << "index_buf[0] : " << index_buf[0] << "height : " << j << endl;
                                                    int c = 0;
                                                    for (int h = -1; h < 1; h++) {
                                                        for (int w = -4; w < 4; w++) {
                                                            if((h == 0) && (w == 0)) break;
                                                            int ind = (j + h) * (width + 3000) + (EPI[a] + w); // current pixel (= center pixel)
                                                            if (patch_table5[ind][a-1] == 0 || patch_table5[ind][a-1] == -1) continue;
                                                            //cout << patch_table5[ind][0] << endl;
                                                            if(temp > patch_table5[ind][a-1]) temp = patch_table5[ind][a-1];
                                                            if(m < patch_table5[ind][a-1]) m = patch_table5[ind][a-1];
                                                            grad1[c] = patch_table5[ind][a-1];
                                                            //cout << grad1[c] << " ";
                                                            c++;
                                                        }
                                                        //cout << endl;
                                                    }
                                                    //주변 기울기의 중간값
                                                    int median1 = (grad1[14] + grad1[15])/2;
                                                    int med = (temp + m) / 2;
                                                    //outlier를 반영한 cost
                                                    cost_outlier = abs(temp - x);
                                                }
                                                if(j < 15 || index_buf[1] < bound + 8) cost_outlier = 0;
                                                temp_cost = cost_diff + w_s1 * cost_outlier;
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    if (p2 == 1)
                                                        EPI_position[a][b] = EPI[0];
                                                }
                                            }

                                        }
                                    }
                                }
                                if (EPI[0] < iFrmWidth && EPI[1] < iFrmWidth && EPI[2] < iFrmWidth)
                                {
                                    line_count[1][EPI[1]]++;
                                    line_count[2][EPI[2]]++;
                                    line_count[3][EPI[3]]++;
                                    int k_r = (ind_h * (width+3000) + ind_j);
                                    int ind1 = (j * (width + 3000) + EPI[1]);
                                    int ind2 = (j * (width + 3000) + EPI[2]);
                                    int ind3 = (j * (width + 3000) + EPI[3]);
                                    patch_table5[ind1][0] = EPI[0] - EPI[1];
                                    patch_table5[ind2][1] = EPI[0] - EPI[2];
                                    patch_table5[ind3][2] = EPI[0] - EPI[3];
                                    int position_num1 = ind_h*width + EPI[0] + 8;
                                    int position_num2 = ind_h*width + EPI[1] + 8;
                                    int position_num3 = ind_h*width + EPI[2] + 8;
                                    int position_num4 = ind_h*width + EPI[3] + 8;
                                    patch_table1[k_r].clear();
                                    patch_table1[k_r].push_back(position_num1);
                                    patch_table1[k_r].push_back(position_num2);
                                    patch_table1[k_r].push_back(position_num3);
                                    patch_table1[k_r].push_back(position_num3);
                                    ind_j++;
                                }
                            }
                            else if (a == iNumFrm - 1)
                            {
                                for (int p2 = a; p2 > 0; p2--) //a보다 이전 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    cost_diff = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = 0;
                                    pixel1 = EPI[p2] + (pos[p2 - 1] - pos[p2]); //예상위치
                                    //printf("pixel 1 : %d %d %d %d\n", pixel1, EPI[p2], pos[p2 - 1] , pos[p2]);
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p2][j][EPI[p2]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p2 - 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]) + (y_buf[y] - RaySpace[p2 - 1][j][po]) * (y_buf[y] - RaySpace[p2 - 1][j][po]);
                                                    //cost_diff = cost_diff / 2;
                                                }

                                                if (cost_diff < best_cost)
                                                {
                                                    best_cost = cost_diff;
                                                    best_x = po;
                                                    //EPI[p2 - 1] = best_x;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    //EPI_position[p1][EPI[p1]] = po;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            //printf("%d\n", po);
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p2][s] = RaySpace[p2][j][EPI[p2] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p2 - 1][s] = RaySpace[p2 - 1][j][po - window_size + s];
                                                }
                                            }
                                            for (int y = p2; y > p2 - 2; y--)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    //cost_diff = cost_diff / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    cost_diff = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        cost_diff += (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]);
                                                    }
                                                    //cost_diff = cost_diff / 2 * (window_size * 2 + 1);
                                                }
                                                if(j > 4 && j < 960 && b >= bound + 8 && b >= bound + 8)
                                                {
                                                    // 0번째 view 분산값
                                                    int d = INT32_MAX;
                                                    int m = 0;
                                                    int temp = INT32_MAX;
                                                    int h_m = INT32_MAX;
                                                    int w_m = INT32_MAX;
                                                    int grad1[28];

                                                    //cout << "dd" << endl;
                                                    //0번째 view 중간값 계산
                                                    //if(index_buf[0] >530 && index_buf[0] <700 && j > 560 && j <730)
                                                    //cout << "index_buf[0] : " << index_buf[0] << "height : " << j << endl;
                                                    int c = 0;
                                                    for (int h = -1; h < 1; h++) {
                                                        for (int w = -4; w < 4; w++) {
                                                            if((h == 0) && (w == 0)) break;
                                                            int ind = (j + h) * (width + 3000) + (EPI[a] + w); // current pixel (= center pixel)
                                                            if (patch_table5[ind][a-1] == 0) continue;
                                                            //cout << patch_table5[ind][0] << endl;
                                                            if(temp > patch_table5[ind][a-1]) temp = patch_table5[ind][a-1];
                                                            if(m < patch_table5[ind][a-1]) m = patch_table5[ind][a-1];
                                                            grad1[c] = patch_table5[ind][a-1];
                                                            //cout << grad1[c] << " ";
                                                            c++;
                                                        }
                                                        //cout << endl;
                                                    }
                                                    //주변 기울기의 중간값
                                                    int median1 = (grad1[14] + grad1[15])/2;
                                                    int med = (temp + m) / 2;
                                                    //outlier를 반영한 cost
                                                    cost_outlier = abs(temp - x);
                                                }
                                                if(j < 15 || index_buf[1] < bound + 8) cost_outlier = 0;
                                                temp_cost = cost_diff + w_s1 * cost_outlier;
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    if (p2 == 1)
                                                        EPI_position[a][b] = EPI[0];
                                                }
                                            }

                                        }
                                    }
                                    //EPI[p2 - 1] = best_x;
                                }
                                if (EPI[0] < iFrmWidth && EPI[1] < iFrmWidth && EPI[2] < iFrmWidth) {
                                    line_count[1][EPI[1]]++;
                                    line_count[2][EPI[2]]++;
                                    line_count[3][EPI[3]]++;
                                    int k_r = (ind_h * (width + 3000) + ind_j);
                                    patch_table1[k_r].clear();
                                    int ind1 = (j * (width + 3000) + EPI[1]);
                                    int ind2 = (j * (width + 3000) + EPI[2]);
                                    int ind3 = (j * (width + 3000) + EPI[3]);
                                    patch_table5[ind1][0] = EPI[0] - EPI[1];
                                    patch_table5[ind2][1] = EPI[0] - EPI[2];
                                    patch_table5[ind3][2] = EPI[0] - EPI[3];
                                    int position_num1 = ind_h*width + EPI[0] + 8;
                                    int position_num2 = ind_h*width + EPI[1] + 8;
                                    int position_num3 = ind_h*width + EPI[2] + 8;
                                    int position_num4 = ind_h*width + EPI[3] + 8;
                                    patch_table1[k_r].clear();
                                    patch_table1[k_r].push_back(position_num1);
                                    patch_table1[k_r].push_back(position_num2);
                                    patch_table1[k_r].push_back(position_num3);
                                    patch_table1[k_r].push_back(position_num3);
                                    ind_j++;
                                }
                            }

                        }
                    }

                }

                // 왼쪽 boundary처리
                for (int a = iNumFrm - 1; a > 0; a--)
                {
                    for (int b = 0; b < bound + 1; b+=3) {
                        int pixel1 = b; //a번째 view
                        temp_cost = INT32_MAX;
                        best_cost = INT32_MAX;
                        best_x = 0;
                        bound_count++;
                        if (pixel1 < window_size) {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 + x; //a-1번째 view

                                for (int s = 0; s < 2*window_size; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[1][s] = RaySpace[a-1][j][pixel0 + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / 2*window_size;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = 0; s < 2*window_size; s++) {
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a-1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[a-1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 + x; //a-1번째 view

                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - window_size + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[1][s] = RaySpace[a-1][j][pixel0 - window_size + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {

                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / window_size * 2;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a-1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a-1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size ;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;

                                    }
                                }
                            }
                        } // 2view까지 그었음
                        int position[4];
                        for (int v = a-1; v > 0; v--)
                        {
                            temp_cost = INT32_MAX;
                            best_cost = INT32_MAX;
                            int pre = 2*best_x - pixel1; //예상 위치
                            if (pre < window_size) {
                                for (int x = 0; x < search_range2; x++) {
                                    int pixel0 = pre + x; //a-1번째 view

                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2*window_size; s++) {
                                            y_buf_1D[1][s] = RaySpace[v-1][j][pixel0 + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / 2*window_size;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = 0; s < 2*window_size; s++) {
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v-1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[v-1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v-1] = pixel0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int x = -search_range2; x < search_range2; x++) {
                                    int pixel0 = pixel1 + x; //a-1번째 view

                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x - window_size + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2 * window_size + 2; s++) {
                                            y_buf_1D[1][s] = RaySpace[v-1][j][pixel0 - window_size + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 && y == 1) continue;
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {

                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / window_size * 2;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v-1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v-1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size ;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v-1] = pixel0;

                                        }
                                    }
                                }
                            }
                        }

                        if (a == 1){
                            line_count[0][best_x]++;
                            line_count[1][pixel1]++;
                            //cout << "pixel1 " << best_x << " " << line_count[0][best_x] <<  endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + best_x + 8;
                            int position_num2 = ind_h*width + pixel1 + 8;
                            int position_num3 = -1;
                            int position_num4 = -1;
                            //cout << pixel1 + best_x << " " << pixel1 << endl;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                        else if(a == 2){
                            line_count[0][position[0]]++;
                            line_count[1][best_x]++;
                            line_count[2][pixel1]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + position[0] + 8;
                            int position_num2 = ind_h*width + best_x + 8;
                            int position_num3 = ind_h*width + pixel1 + 8;
                            int position_num4 = -1;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                        else if(a == 3){
                            line_count[0][position[0]]++;
                            line_count[1][position[1]]++;
                            line_count[2][best_x]++;
                            line_count[3][pixel1]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + position[0] + 8;
                            int position_num2 = ind_h*width + position[1] + 8;
                            int position_num3 = ind_h*width + best_x + 8;
                            int position_num4 = ind_h*width + pixel1 + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                    }
                }
                // 오른쪽 바운더리 처리
                for (int a = 0; a < iNumFrm - 1; a++)
                {
                    for (int b = iFrmWidth - bound; b < iFrmWidth; b+=3) {
                        int pixel1 = b; //a번째 view
                        temp_cost = INT32_MAX;
                        best_cost = INT32_MAX;
                        best_x = 0;
                        bound_count++;
                        if (pixel1 > iFrmWidth - window_size) {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 - x; //a+1번째 view

                                for (int s = 0; s < 2*window_size; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[1][s] = RaySpace[a+1][j][pixel0 - s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;

                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]);
                                        }
                                        temp_cost = temp_cost / 2*window_size ;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]);
                                        }
                                        for (int s = 0; s < 2*window_size; s++) {
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a+1][j][pixel0 - s]) * (y_buf_1D[y][s] - RaySpace[a+1][j][pixel0 - s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 - x; //a-1번째 view

                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - window_size + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[1][s] = RaySpace[a+1][j][pixel0 - window_size + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {

                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / window_size * 2;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a+1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a+1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        int position[4];
                        for (int v = 1; v < iNumFrm-1; v++)
                        {
                            temp_cost = INT32_MAX;
                            best_cost = INT32_MAX;
                            int pre = 2*best_x - pixel1; //예상 위치
                            if (pre < iFrmWidth - window_size) {
                                for (int x = 0; x < search_range2; x++) {
                                    int pixel0 = pre - x; //a-1번째 view

                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2*window_size; s++) {
                                            y_buf_1D[1][s] = RaySpace[v+1][j][pixel0 + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / 2*window_size;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = 0; s < 2*window_size; s++) {
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v+1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[v+1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v+1] = pixel0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int x = -search_range2; x < search_range2; x++) {
                                    int pixel0 = pixel1 + x; //a-1번째 view

                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x - window_size + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2 * window_size + 2; s++) {
                                            y_buf_1D[1][s] = RaySpace[v+1][j][pixel0 - window_size + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 && y == 1) continue;
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {

                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / window_size * 2;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v+1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v+1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size ;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v+1] = pixel0;

                                        }
                                    }
                                }
                            }
                        }


                        if (a == 0){
                            line_count[0][pixel1]++;
                            line_count[1][best_x]++;
                            line_count[2][position[2]]++;
                            line_count[3][position[3]]++;
                            //cout << "pixel1 " << pixel1 << endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + pixel1 + 8;
                            int position_num2 = ind_h*width + best_x + 8;
                            int position_num3 = ind_h*width + position[2] + 8;
                            int position_num4 = ind_h*width + position[3] + 8;
                            //cout << pixel1 + best_x << " " << pixel1 << endl;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                        else if(a == 1){
                            line_count[1][pixel1]++;
                            line_count[2][best_x]++;
                            line_count[3][position[3]]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = -1;
                            int position_num2 = ind_h*width + pixel1 + 8;
                            int position_num3 = ind_h*width + best_x + 8;
                            int position_num4 = ind_h*width + position[3] + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                        else if(a == 2){
                            line_count[2][pixel1]++;
                            line_count[3][best_x]++;
                            //cout << "haight : " << ind_h << "line : " << best_x << endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = -1;
                            int position_num2 = -1;
                            int position_num3 = ind_h*width + pixel1 + 8;
                            int position_num4 = ind_h*width + best_x + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num3);
                            ind_j++;
                        }
                    }
                }
                patch_table2[ind_h].push_back(ind_j);
                int count = 0;
                int count1 = 0;
                for(int empty = 0; empty < bound; empty++)
                {
                    cout << empty << endl;
                    int k_r = ind_h * bound + count;
                    int k_r1 = ind_h * bound + count1;
                    if(line_count[0][empty] == 0)
                    {
                        patch_table3[k_r].clear();
                        int empty_num = ind_h*width + empty + 8;
                        patch_table3[k_r].push_back(empty_num);
                        count++;
                    }
                    if(line_count[3][iFrmWidth -1 - empty] == 0){
                        patch_table4[k_r].clear();
                        int empty_num = ind_h*width + iFrmWidth -1 - empty + 8;
                        patch_table4[k_r1].push_back(empty_num);
                        count1++;
                    }
                }

                patch_table2[ind_h].push_back(count);
                patch_table2[ind_h].push_back(count1);
                single_count += count;
                single_count += count1;
                break; // 모든 Row의 예측이 끝난 경우 for문 종료
            }
            //cout << "dd1" << endl;
        } // infinite loop end  ( width ) , width에 대해 1view & 2view 끝
        ind_h++;
    }
    /*
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    for (int j = 8; j < iFrmHeight - 8; j=j+8) {
        const unsigned i_r = 8 + j;
        for (unsigned ind_j = bound + 16; ind_j < iFrmWidth - 2 * bound - 16; ind_j = ind_j + 8) {
            if(j > 540 && j < 710 && ind_j >= 390 && ind_j <= 550) {
                    double mean = 0;
                    double sum = 0;
                    double var = 0;
                    double mean2 = 0;
                    double sum2 = 0;
                    double var2 = 0;
                    int th = 300;
                    int error = 0.2;
                    int d = INT32_MAX;
                    int m = 0;
                    int temp;
                    int h_m = INT32_MAX;
                    int w_m = INT32_MAX;
                    for (int h = -1; h < 1; h++) {
                        for (int w = -5; w < 4; w++) {
                            if((h == 0) && (w == 0)) break;
                            int ind = (j + h) * (width + 3000) + (ind_j + w);
                            cout << patch_table5[ind][2] << " ";
                            sum += patch_table5[ind][2];
                        }
                        cout << endl;
                    }
                    cout << endl;
                    mean = sum / 28;
                    sum = 0;
                    for (int h = -4; h < 4; h++) {
                        for (int w = -4; w < 4; w++) {
                            if((h == 0) && (w == 0)) break;
                            int ind = (j + h) * (width + 3000) + (ind_j + w);
                            sum += (patch_table5[ind][2] - mean) * (patch_table5[ind][2] - mean);
                            temp = patch_table5[ind][2]*patch_table5[ind][2];
                            if(d > temp)
                            {
                                h_m = h;
                                w_m = w;
                                m = ind;
                            }
                            //cout <<patch_table5[ind][0] << " " <<  patch_table5[ind][0] - mean << endl;
                        }
                    }
                    var = sum / 28;
                    total_var += var;
                    total_count++;
                    //cout << var << endl;
                }
        }
    }
*/
    //cout << ind_h << endl;
    cout << "var average : " << total_var/total_count << endl;
    cout << "cost average : " << total_cost_aver/total_line << endl;
    cout << "count_outlier0 : " << count_outlier0 << endl;
    cout << "count_outlier1 : " << count_outlier1 << endl;
    cout << "average cost : :" << aver_cost/inter_count << endl;
    cout << "extra_count : " << extra_count << endl;
    cout << "inter_count : " << inter_count << endl;
    cout << "bound_count : " << bound_count << endl;
    cout << "single_count : " << single_count << endl;
    cout << "total patch number : " << extra_count + inter_count + bound_count << endl;
    cout << "outlier_count : " << outlier_count << endl;
    cout << "count_cl : " << count_cl << endl;
}

//original EPI

void precompute_original_EPI(
        vector<vector<int> > &patch_table1
        ,   vector<vector<int> > &patch_table2
        ,   vector<vector<int> > &patch_table3
        ,   vector<vector<int> > &patch_table4
        ,   const std::vector<float> &img1
        ,   const std::vector<float> &img2
        ,   const std::vector<float> &img3
        ,   const std::vector<float> &img4
        ,   const unsigned width
        ,   const unsigned height
        ,   const float    tauMatch
        ,   const unsigned pHard
) {
    using namespace std;
    int iFrmHeight = 768;
    int iFrmWidth = 1024;
    long k = 0;
    int iFrmHeightC;
    //int iNumFrm = 16;
    int iNumFrm = 4;
    //const int NumFrm = 16;
    const int NumFrm = 4;
    int total_frame;
    int num_fr;
    int *index_buf;
    int *pixel_buf;
    int *line_end;
    int *cost_buf;
    int *x_buf;
    int *z_buf;
    int *w_buf;
    int *y_buf;
    int **y_buf_1D;
    int *best_num;
    int search_range = 30;
    int search_range2 = 10;
    int window_size = 5;
    int total = 0;
    int ***cost_infor;

    int ***bestbuf_pixel_sel;
    int ***bestbuf_pixel_sel0;
    int ***bestbuf_pixel_sel1;
    int ***bestbuf_pixel_sel2;
    int left_view = 0;
    int right_view = 0;
    int view_left = 0;
    int view_right = 0;

    int **EPI_position;
    int **line_count;

    unsigned char *ucInFrame;			//input buffer
    unsigned char ***RaySpace;			//ray space buffer
    unsigned char ***RaySpace_Cb;
    unsigned char ***RaySpace_Cr;
    unsigned char *usEPIOutFrame;		//EPI buffer only for one height
    unsigned char *usOutFrame;			//output buffer

    unsigned char **ucRayFrame;			//arbitrary ray
    unsigned char **ucRayFrame_cb;
    unsigned char **ucRayFrame_cr;

    int temp_cost = INT32_MAX;
    int best_cost = INT32_MAX;
    int pixel_sel0 = 0;
    int pixel_sel1 = 0;
    int pixel_sel2 = 0;
    int best_x = 0;
    double best_z = 0;
    double best_w = 0;
    int best_line = 0;
    int linecount = 0;
    int average = 0;
    int max_cost = 0;
    int cost = 0;
    int bound = 20;
    int extra_count = 0;
    int inter_count = 0;
    int bound_count = 0;
    int single_count = 0;

    if (patch_table1.size() != (width+3000) * (height+8))
        patch_table1.resize((width+3000) * (height+8));
    if (patch_table2.size() != (height+8))
        patch_table2.resize((height+8));
    if (patch_table3.size() != (height*bound))
        patch_table3.resize((height*bound));
    if (patch_table4.size() != (height*bound))
        patch_table4.resize((height*bound));
    best_num = new int[iFrmWidth + 10];
    x_buf = new int[iNumFrm];
    y_buf = new int[iNumFrm];
    y_buf_1D = new int*[iNumFrm];
    for(int i = 0; i < iNumFrm; i++)
        y_buf_1D[i] = new int[2 * window_size + 2];
    RaySpace = new unsigned char**[iNumFrm];
    for (int NumFrm = 0; NumFrm < iNumFrm; NumFrm++)
    {
        RaySpace[NumFrm] = new unsigned char*[iFrmHeight];
        for (int Height = 0; Height < iFrmHeight; Height++)
            RaySpace[NumFrm][Height] = new unsigned char[iFrmWidth];
    }
    cost_infor = new int **[iNumFrm];
    for (int i = 0; i < iNumFrm; i++) {
        cost_infor[i] = new int*[iFrmWidth + 10];
        for (int j = 0; j < (iFrmWidth + 10); j++)
            cost_infor[i][j] = new int[20];
    }
    line_count = new int*[iNumFrm];
    for (int i = 0; i < iNumFrm; i++)
    {
        line_count[i] = new int[iFrmWidth];
    }
    bestbuf_pixel_sel = new int**[iNumFrm];
    for (int i = 0; i < iNumFrm; i++) {
        bestbuf_pixel_sel[i] = new int*[iFrmHeight];
        for (int Height = 0; Height < iFrmHeight; Height++)
        {
            bestbuf_pixel_sel[i][Height] = new int[iFrmWidth + 10];
        }
    }
    EPI_position = new int*[iNumFrm];
    for (int i = 0; i < iNumFrm; i++)
    {
        EPI_position[i] = new int[iFrmWidth];
    }
    line_end = new int[iNumFrm];
    index_buf = new int[iNumFrm];
    pixel_buf = new int[iNumFrm];
    cost_buf = new int[iNumFrm];
    for (int Height = 0; Height < iFrmHeight; Height++)
    {
        for (int Width = 0; Width < iFrmWidth; Width++)
        {
            RaySpace[0][Height][Width] = img1[Height*iFrmWidth + Width];
            RaySpace[1][Height][Width] = img2[Height*iFrmWidth + Width];
            RaySpace[2][Height][Width] = img3[Height*iFrmWidth + Width];
            RaySpace[3][Height][Width] = img4[Height*iFrmWidth + Width];
        }
    }
    // 첫 번째 1view&2view full search
    int ind_h = 8;
    for (int j = 0; j < iFrmHeight; j++)
    { //height
        //printf("height: %d\n", ind_h);
        for (int i = 0; i < iNumFrm; i++) {
            memset(line_count[i], 0, sizeof(int)*(iFrmWidth));
            for (int j = 0; j < iFrmWidth + 10; j++) {
                memset(cost_infor[i][j], 0, sizeof(int) * 20);

            }
        }
        for (int i = 0; i < iNumFrm; i++) {
            memset(y_buf_1D[i], 0, sizeof(int)*(2*window_size+1));
        }
        memset(y_buf, 0, sizeof(int)*iNumFrm);
        memset(x_buf, 0, sizeof(int)*iNumFrm);
        memset(index_buf, 0, sizeof(int)*iNumFrm);
        memset(pixel_buf, 0, sizeof(int)*iNumFrm);
        memset(line_end, 0, sizeof(int)*iNumFrm);
        memset(cost_buf, 0, sizeof(int)*iNumFrm);
        index_buf[0] = bound;
        index_buf[1] = bound;
        temp_cost = INT32_MAX;
        best_cost = INT32_MAX;

        best_x = 0;
        //int count = 0;
        int pos_x = INT32_MAX;
        int pos_z = INT32_MAX;
        k = 0;
        linecount = 0;
        int ind_j = 0;

        for (;;)
        { // width

            temp_cost = INT32_MAX;
            best_cost = INT32_MAX;
            best_x = 0;

            //cost 구하기

            if (!line_end[0] && index_buf[0] >= bound) // 1번째 view에 대해서
            {
                if (index_buf[0] > iFrmWidth - window_size || index_buf[0] < window_size) { // 1D windo가 범위를 벗어나는 경우
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[0] - x; // 2번째 view에서 예상되는 pixel
                        // line shouldn't be on right of index pixels
                        if (pixel_sel0 > index_buf[1] && pixel_sel0 < iFrmWidth) continue;

                        y_buf[0] = RaySpace[0][j][index_buf[0]];
                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth)
                            y_buf[1] = RaySpace[1][j][pixel_sel0];

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 1) continue;

                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth) // 1view bound 넘어간 상황
                            {
                                temp_cost = (y_buf[y] - RaySpace[0][j][index_buf[0]]) * (y_buf[y] - RaySpace[0][j][index_buf[0]]);
                            }
                            else // 1view bound 넘어가지 않은 상황
                            {
                                temp_cost = (y_buf[y] - RaySpace[0][j][index_buf[0]]) * (y_buf[y] - RaySpace[0][j][index_buf[0]]) + (y_buf[y] - RaySpace[1][j][pixel_sel0]) * (y_buf[y] - RaySpace[1][j][pixel_sel0]);
                                temp_cost = temp_cost / 2;
                            }
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;
                            }
                        }
                    }
                }
                else {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[0] - x; // 2번째 view에서 예상되는 1D window내의 중앙pixel
                        // line shouldn't be on right of index pixels
                        if (pixel_sel0 > index_buf[1] && pixel_sel0 < iFrmWidth) continue;

                        for (int s = 0; s < 2 * window_size + 2; s++) {
                            y_buf_1D[0][s] = RaySpace[0][j][index_buf[0] - window_size + s];
                        }
                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth) {
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[1][s] = RaySpace[1][j][pixel_sel0 - window_size + s];
                            }
                        }

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 1) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth) // 1view bound 넘어간 상황
                            {

                                temp_cost = (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]) * (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]);
                                }
                                temp_cost = temp_cost / (window_size * 2 + 1);
                            }
                            else // 1view bound 넘어가지 않은 상황
                            {
                                temp_cost = (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]) * (y_buf_1D[y][window_size] - RaySpace[0][j][index_buf[0]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][index_buf[0] + s]);
                                }
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[1][j][pixel_sel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][pixel_sel0 + s]);
                                }
                                temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                            }
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;

                            }
                        }
                    }
                }
            }
            x_buf[0] = best_x;
            cost_buf[0] = best_cost;
            //cout << "dd1 " << endl;
            best_x = 0;
            best_z = 0;
            best_cost = INT32_MAX;
            temp_cost = INT32_MAX;

            if (!line_end[1] && index_buf[1] >= bound) // 2번째 view에 대해서
            {
                if (index_buf[1] > iFrmWidth - window_size || index_buf[1] < window_size) {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[1] + x; // 1번째 view에서 예상되는 pixel

                        if (pixel_sel0 > index_buf[0] && pixel_sel0 < iFrmWidth) continue;

                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth)
                            y_buf[0] = RaySpace[0][j][pixel_sel0];
                        y_buf[1] = RaySpace[1][j][index_buf[1]];

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 0) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth)
                            {
                                temp_cost = (y_buf[y] - RaySpace[1][j][index_buf[1]]) * (y_buf[y] - RaySpace[1][j][index_buf[1]]);
                            }
                            else
                            {
                                temp_cost = (y_buf[y] - RaySpace[1][j][index_buf[1]]) * (y_buf[y] - RaySpace[1][j][index_buf[1]]);
                                temp_cost += (y_buf[y] - RaySpace[0][j][pixel_sel0]) * (y_buf[y] - RaySpace[0][j][pixel_sel0]);
                                temp_cost = temp_cost / 2;
                            }
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;
                            }
                        }
                    }
                }
                else {
                    for (int x = 0; x < search_range; x++) //search range 10
                    {
                        pixel_sel0 = index_buf[1] + x; // 1번째 view에서 예상되는 pixel

                        if (pixel_sel0 > index_buf[0] && pixel_sel0 < iFrmWidth) continue;

                        if (0 <= pixel_sel0 && pixel_sel0 < iFrmWidth) {
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[0][s] = RaySpace[0][j][pixel_sel0 - window_size + s];
                            }
                        }
                        for (int s = 0; s < 2 * window_size + 2; s++) {
                            y_buf_1D[1][s] = RaySpace[1][j][index_buf[1] - window_size + s];
                        }

                        for (int y = 0; y < 2; y++)
                        {
                            if (pixel_sel0 < 0 && y == 0) continue;
                            if (pixel_sel0 < 0 || pixel_sel0 >= iFrmWidth)
                            {
                                temp_cost = (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]) * (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]);
                                }
                                temp_cost = temp_cost / 1 * (window_size * 2 + 1);
                            }
                            else
                            {
                                temp_cost = (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]) * (y_buf_1D[y][window_size] - RaySpace[1][j][index_buf[1]]);
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    if (s == 0) continue;
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[1][j][index_buf[1] + s]);
                                }
                                for (int s = -window_size; s < window_size + 1; s++) {
                                    temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[0][j][pixel_sel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[0][j][pixel_sel0 + s]);
                                }
                                temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                            }
                            if (temp_cost < best_cost)
                            {
                                best_cost = temp_cost;
                                best_x = x;
                            }
                        }
                    }
                }
            }
            x_buf[1] = best_x;
            cost_buf[1] = best_cost;
            //cout << "dd2 " << endl;
            //minimum cost 저장 buf
            int save_cost[2];
            save_cost[0] = cost_buf[0];
            save_cost[1] = cost_buf[1];
            // best cost
            if (save_cost[0] < save_cost[1])
            {
                best_line = 0;
            }
            else
            {
                best_line = 1;
            }

            // 중복되는 라인이 있는지 검사하고 직선위에있는 pixel의 position 저장하기
            if ((!line_end[0]) && (best_line == 0))
            {

                // 0 번째 view에서 해당하는 위치
                pixel_sel0 = index_buf[0] - x_buf[0];

                //printf("index_buf[0] : %d\n", index_buf[0]);
                if (line_count[0][index_buf[0]] != 0) {
                    if (save_cost[0] < cost_infor[0][index_buf[0]][0]) {
                        cost_infor[0][index_buf[0]][0] = save_cost[0];
                        best_num[index_buf[0]] = 0;
                        int temp = bestbuf_pixel_sel[1][j][index_buf[0]];
                        line_count[1][temp]--;
                        ///////////////////////////////////////////////////////
                        //배열에 position 저장

                        if (pixel_sel0 < 0) // 1번째 view 픽셀만 라인이 그어짐
                        {
                            bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                            bestbuf_pixel_sel[1][j][index_buf[0]] = 0;
                            EPI_position[1][0] = index_buf[0];
                            line_count[1][0]++;
                        }
                        else // 1번째 2번째 view 에만 라인이 그어짐
                        {
                            bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                            bestbuf_pixel_sel[1][j][index_buf[0]] = pixel_sel0;
                            EPI_position[1][pixel_sel0] = index_buf[0];
                            line_count[1][pixel_sel0]++;
                        }

                    }
                }
                else // 중복된 라인이 없는 경우
                {
                    inter_count++;
                    cost_infor[0][index_buf[0]][0] = save_cost[0];
                    line_count[0][index_buf[0]]++;
                    line_count[1][pixel_sel0]++;

                    if (pixel_sel0 < 0) // 1번째 view 픽셀만 라인이 그어짐
                    {
                        bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                        bestbuf_pixel_sel[1][j][index_buf[0]] = 0;
                        EPI_position[1][0] = index_buf[0];
                    }
                    else // 1번째 2번째 view 에만 라인이 그어짐
                    {
                        bestbuf_pixel_sel[0][j][index_buf[0]] = index_buf[0];
                        bestbuf_pixel_sel[1][j][index_buf[0]] = pixel_sel0;
                        EPI_position[1][pixel_sel0] = index_buf[0];
                    }

                }
                for (int v = 2; v < iNumFrm; v++)
                {
                    temp_cost = INT32_MAX;
                    best_cost = INT32_MAX;
                    best_x = 0;
                    int pixel0 = bestbuf_pixel_sel[v-2][j][index_buf[0]];
                    int pixel1 = bestbuf_pixel_sel[v-1][j][index_buf[0]];
                    int po = 0;
                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = pixel1 + x; // 2번째 view에서 예상되는 pixel
                            if (po < 0 || po > iFrmWidth) continue;
                            y_buf[0] = RaySpace[v-1][j][pixel1];
                            if (0 <= po && po < iFrmWidth)
                                y_buf[1] = RaySpace[v][j][po];

                            for (int y = 0; y < 2; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {
                                    temp_cost = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    temp_cost = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]) + (y_buf[y] - RaySpace[v][j][po]) * (y_buf[y] - RaySpace[v][j][po]);
                                    temp_cost = temp_cost / 2;
                                }
                                if (temp_cost < best_cost)
                                {
                                    best_cost = temp_cost;
                                    best_x = po;
                                    //EPI_position[v][po] = index_buf[0];

                                }
                            }
                        }
                    }
                    else
                    {
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = 2 * pixel1 - pixel0 +  x;
                            if (po < 0 || po > iFrmWidth) continue;
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[v - 1][s] = RaySpace[v - 1][j][pixel1 - window_size + s];
                            }
                            if (0 <= po && po < iFrmWidth) {
                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[v][s] = RaySpace[v][j][po - window_size + s];
                                }
                            }

                            for (int y = v-1; y < v+1; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {

                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    temp_cost = temp_cost / (window_size * 2 + 1);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]);
                                    }
                                    temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                                }
                                if (temp_cost < best_cost)
                                {
                                    best_cost = temp_cost;
                                    best_x = po;
                                }
                            }

                        }
                    }
                    bestbuf_pixel_sel[v][j][index_buf[0]] = best_x;
                    if(index_buf[0] < iFrmWidth)
                        EPI_position[v][best_x] = index_buf[0];
                    //printf("EPI_position[a][0] : %d/ %d %d \n", EPI_position[v][best_x], v, best_x);
                }
            }
            if ((!line_end[1]) && (best_line == 1))
            {
                // 0 번째 view에서 해당하는 위치
                pixel_sel0 = index_buf[1] + x_buf[1];
                if (line_count[0][pixel_sel0] != 0) {
                    //if (save_cost[1] < cost_infor[0][pixel_sel0][0]) {
                    if (save_cost[1] < cost_infor[0][pixel_sel0][0]) {
                        cost_infor[0][pixel_sel0][0] = save_cost[1];
                        int temp = bestbuf_pixel_sel[1][j][pixel_sel0];
                        line_count[1][temp]--;

                        if(pixel_sel0 < iFrmWidth) {
                            bestbuf_pixel_sel[0][j][pixel_sel0] = pixel_sel0;
                            bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                            EPI_position[1][index_buf[1]] = pixel_sel0;
                            line_count[1][index_buf[1]]++;
                        }
                        else
                        {
                            bestbuf_pixel_sel[0][j][pixel_sel0] = iFrmWidth - 1;
                            bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                            EPI_position[1][index_buf[1]] = iFrmWidth - 1;
                            line_count[1][index_buf[1]]++;
                        }
                    }

                }
                else
                {
                    inter_count++;
                    best_num[pixel_sel0] = 1;
                    line_count[0][pixel_sel0]++;
                    line_count[1][index_buf[1]]++;
                    cost_infor[0][pixel_sel0][0] = save_cost[1];
                    if (pixel_sel0 < iFrmWidth) {
                        bestbuf_pixel_sel[0][j][pixel_sel0] = pixel_sel0;
                        bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                        EPI_position[1][index_buf[1]] = pixel_sel0;
                    }
                    else
                    {
                        bestbuf_pixel_sel[0][j][pixel_sel0] = iFrmWidth - 1;
                        bestbuf_pixel_sel[1][j][pixel_sel0] = index_buf[1];
                        EPI_position[1][index_buf[1]] = iFrmWidth - 1;
                    }
                }
                for (int v = 2; v < iNumFrm; v++)
                {
                    temp_cost = INT32_MAX;
                    best_cost = INT32_MAX;
                    best_x = 0;
                    int pixel0 = bestbuf_pixel_sel[v - 2][j][pixel_sel0];
                    int pixel1 = bestbuf_pixel_sel[v - 1][j][pixel_sel0];
                    int po = 0;
                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = pixel1 + x; // 2번째 view에서 예상되는 pixel
                            if (po < 0 || po > iFrmWidth) continue;
                            y_buf[0] = RaySpace[v - 1][j][pixel1];
                            if (0 <= po && po < iFrmWidth)
                                y_buf[1] = RaySpace[v][j][po];

                            for (int y = 0; y < 2; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {
                                    temp_cost = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    temp_cost = (y_buf[y] - RaySpace[v - 1][j][pixel1]) * (y_buf[y] - RaySpace[v - 1][j][pixel1]) + (y_buf[y] - RaySpace[v][j][po]) * (y_buf[y] - RaySpace[v][j][po]);
                                    temp_cost = temp_cost / 2;
                                }
                                if (temp_cost < best_cost)
                                {
                                    best_cost = temp_cost;
                                    best_x = po;
                                    //EPI_position[v][po] = pixel_sel0;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                        {
                            po = 2 * pixel1 - pixel0 + x;
                            if (po < 0 || po > iFrmWidth) continue;
                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                y_buf_1D[v - 1][s] = RaySpace[v - 1][j][pixel1 - window_size + s];
                            }
                            if (0 <= po && po < iFrmWidth) {
                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[v][s] = RaySpace[v][j][po - window_size + s];
                                }
                            }

                            for (int y = v - 1; y < v + 1; y++)
                            {
                                if (po < 0 && y == 1) continue;

                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                {

                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    temp_cost = temp_cost / (window_size * 2 + 1);
                                }
                                else // 1view bound 넘어가지 않은 상황
                                {
                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[v - 1][j][pixel1]);
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        if (s == 0) continue;
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v - 1][j][pixel1 + s]);
                                    }
                                    for (int s = -window_size; s < window_size + 1; s++) {
                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][po + s]);
                                    }
                                    temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                                }
                                if (temp_cost < best_cost)
                                {
                                    best_cost = temp_cost;
                                    best_x = po;
                                    //EPI_position[v][po] = pixel_sel0;

                                }
                            }

                        }
                    }
                    bestbuf_pixel_sel[v][j][pixel_sel0] = best_x;
                    if (pixel_sel0 < iFrmWidth)
                        EPI_position[v][best_x] = pixel_sel0;
                }

            }

            index_buf[best_line]++;

            // end line check
            if (index_buf[0] > iFrmWidth - bound) {
                line_end[0] = 1;
            }
            if (index_buf[1] > iFrmWidth - bound) {
                line_end[1] = 1;
            }

            if ((line_end[0] == 1) && (line_end[1] == 1))
            {
                for (int b = bound; b < iFrmWidth - bound; b++)
                {
                    if (bestbuf_pixel_sel[0][j][b] < iFrmWidth - 1 && bestbuf_pixel_sel[1][j][b] < iFrmWidth - 1 && bestbuf_pixel_sel[2][j][b] < iFrmWidth - 1) {
                        if (!(bestbuf_pixel_sel[0][j][b] == 0 && bestbuf_pixel_sel[1][j][b] == 0 &&
                              bestbuf_pixel_sel[2][j][b] == 0 && bestbuf_pixel_sel[3][j][b] == 0)) {
                            line_count[1][bestbuf_pixel_sel[1][j][b]]++;
                            line_count[2][bestbuf_pixel_sel[2][j][b]]++;
                            line_count[3][bestbuf_pixel_sel[3][j][b]]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h * width + bestbuf_pixel_sel[0][j][b] + 8;
                            int position_num2 = ind_h * width + bestbuf_pixel_sel[1][j][b] + 8;
                            int position_num3 = ind_h * width + bestbuf_pixel_sel[2][j][b] + 8;
                            int position_num4 = ind_h * width + bestbuf_pixel_sel[3][j][b] + 8;
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                    }
                }
                // 빈영역 채우기
                for (int b = bound; b < iFrmWidth - bound; b+=3) {
                    for (int a = 1; a < iNumFrm; a++) {
                        if (line_count[a][b] == 0)
                        {
                            extra_count++;
                            //if(a==1&& ind_h==20) cout << b << endl;
                            int pixel0 = 0;
                            if (b == 0) pixel0 = EPI_position[a][0];
                            else pixel0 = EPI_position[a][b - 3]; //left pixel의 0번째 view에서의 위치
                            int pos[NumFrm];
                            int EPI[NumFrm];
                            EPI[a] = b; //b에 대한 EPI
                            for (int x = 0; x < iNumFrm; x++)
                            {
                                pos[x] = bestbuf_pixel_sel[x][j][pixel0]; //left pixel의 EPI를 가져옴

                            }

                            if (a < iNumFrm - 1) {
                                for (int p1 = a; p1 < iNumFrm - 1; p1++) //a번째 이상 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = EPI[p1] - (pos[p1] - pos[p1 + 1]); //예상위치
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p1][j][EPI[p1]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p1 + 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p1][j][EPI[p1]]) * (y_buf[y] - RaySpace[p1][j][EPI[p1]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p1][j][EPI[p1]]) * (y_buf[y] - RaySpace[p1][j][EPI[p1]]) + (y_buf[y] - RaySpace[p1 + 1][j][po]) * (y_buf[y] - RaySpace[p1 + 1][j][po]);
                                                    temp_cost = temp_cost / 2;
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p1 + 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p1 + 1] = best_x;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p1][s] = RaySpace[p1][j][EPI[p1] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p1 + 1][s] = RaySpace[p1 + 1][j][po - window_size + s];
                                                }
                                            }

                                            for (int y = p1; y < p1 + 2; y++)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]) * (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]);
                                                    }
                                                    temp_cost = temp_cost / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]) * (y_buf_1D[y][window_size] - RaySpace[p1][j][EPI[p1]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1][j][EPI[p1] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p1 + 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p1 + 1][j][po + s]);
                                                    }
                                                    temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p1 + 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p1 + 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    //EPI_position[p1][po] = pixel_sel0;

                                                }
                                            }

                                        }
                                    }
                                    //EPI[p1 + 1] = best_x;
                                }
                                //printf("%d %d\n", EPI_position[a][b - 1], b);
                                for (int p2 = a; p2 > 0; p2--) //a보다 이전 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = 0;
                                    pixel1 = EPI[p2] + (pos[p2 - 1] - pos[p2]); //예상위치
                                    //printf("pixel 1 : %d %d %d %d\n", pixel1, EPI[p2], pos[p2 - 1] , pos[p2]);
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p2][j][EPI[p2]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p2 - 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]) + (y_buf[y] - RaySpace[p2 - 1][j][po]) * (y_buf[y] - RaySpace[p2 - 1][j][po]);
                                                    temp_cost = temp_cost / 2;
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    //EPI[p2 - 1] = best_x;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p2][s] = RaySpace[p2][j][EPI[p2] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p2 - 1][s] = RaySpace[p2 - 1][j][po - window_size + s];
                                                }
                                            }
                                            for (int y = p2; y > p2 - 2; y--)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    temp_cost = temp_cost / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]);
                                                    }
                                                    temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    if (p2 == 1)
                                                        EPI_position[a][b] = EPI[0];
                                                }
                                            }

                                        }
                                    }
                                }
                                if (EPI[0] < iFrmWidth && EPI[1] < iFrmWidth && EPI[2] < iFrmWidth)
                                {
                                    line_count[1][EPI[1]]++;
                                    line_count[2][EPI[2]]++;
                                    line_count[3][EPI[3]]++;
                                    int k_r = (ind_h * (width+3000) + ind_j);
                                    int position_num1 = ind_h*width + EPI[0] + 8;
                                    int position_num2 = ind_h*width + EPI[1] + 8;
                                    int position_num3 = ind_h*width + EPI[2] + 8;
                                    int position_num4 = ind_h*width + EPI[3] + 8;
                                    patch_table1[k_r].clear();
                                    patch_table1[k_r].push_back(position_num1);
                                    patch_table1[k_r].push_back(position_num2);
                                    patch_table1[k_r].push_back(position_num3);
                                    patch_table1[k_r].push_back(position_num4);
                                    ind_j++;
                                }
                            }
                            else if (a == iNumFrm - 1)
                            {
                                for (int p2 = a; p2 > 0; p2--) //a보다 이전 view에 대해서
                                {
                                    temp_cost = INT32_MAX;
                                    best_cost = INT32_MAX;
                                    best_x = 0;
                                    int pixel1 = 0;
                                    pixel1 = EPI[p2] + (pos[p2 - 1] - pos[p2]); //예상위치
                                    //printf("pixel 1 : %d %d %d %d\n", pixel1, EPI[p2], pos[p2 - 1] , pos[p2]);
                                    if (pixel1 > iFrmWidth - window_size || pixel1 < window_size) { // 1D windo가 범위를 벗어나는 경우
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            if (po < 0 || po > iFrmWidth - 1) continue;
                                            y_buf[0] = RaySpace[p2][j][EPI[p2]]; // target pixel
                                            if (0 <= po && po < iFrmWidth)
                                                y_buf[1] = RaySpace[p2 - 1][j][po];
                                            for (int y = 0; y < 2; y++)
                                            {
                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf[y] - RaySpace[p2][j][EPI[p2]]) * (y_buf[y] - RaySpace[p2][j][EPI[p2]]) + (y_buf[y] - RaySpace[p2 - 1][j][po]) * (y_buf[y] - RaySpace[p2 - 1][j][po]);
                                                    temp_cost = temp_cost / 2;
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    //EPI[p2 - 1] = best_x;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    //EPI_position[p1][EPI[p1]] = po;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int x = -search_range2; x < search_range2; x++) //search range 10
                                        {
                                            int po = pixel1 + x;
                                            //printf("%d\n", po);
                                            if (po < window_size || po > iFrmWidth - 1) continue;
                                            for (int s = 0; s < 2 * window_size + 2; s++) {
                                                y_buf_1D[p2][s] = RaySpace[p2][j][EPI[p2] - window_size + s];
                                            }
                                            if (0 <= po && po < iFrmWidth) {
                                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                                    y_buf_1D[p2 - 1][s] = RaySpace[p2 - 1][j][po - window_size + s];
                                                }
                                            }
                                            for (int y = p2; y > p2 - 2; y--)
                                            {
                                                if (po < 0 && y == 1) continue;

                                                if (po < 0 || po >= iFrmWidth) // 1view bound 넘어간 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    temp_cost = temp_cost / (window_size * 2 + 1);
                                                }
                                                else // 1view bound 넘어가지 않은 상황
                                                {
                                                    temp_cost = (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]) * (y_buf_1D[y][window_size] - RaySpace[p2][j][EPI[p2]]);
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        if (s == 0) continue;
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2][j][EPI[p2] + s]);
                                                    }
                                                    for (int s = -window_size; s < window_size + 1; s++) {
                                                        temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]) * (y_buf_1D[y][s + window_size] - RaySpace[p2 - 1][j][po + s]);
                                                    }
                                                    temp_cost = temp_cost / 2 * (window_size * 2 + 1);
                                                }
                                                if (temp_cost < best_cost)
                                                {
                                                    best_cost = temp_cost;
                                                    best_x = po;
                                                    if (po >= iFrmWidth - 1)
                                                        EPI[p2 - 1] = iFrmWidth - 1;
                                                    else
                                                        EPI[p2 - 1] = best_x;
                                                    //printf("best position : %d\n", best_x);
                                                    if (p2 == 1)
                                                        EPI_position[a][b] = EPI[0];
                                                }
                                            }

                                        }
                                    }
                                    //EPI[p2 - 1] = best_x;
                                }
                                if (EPI[0] < iFrmWidth && EPI[1] < iFrmWidth && EPI[2] < iFrmWidth) {
                                    line_count[1][EPI[1]]++;
                                    line_count[2][EPI[2]]++;
                                    line_count[3][EPI[3]]++;
                                    int k_r = (ind_h * (width + 3000) + ind_j);
                                    patch_table1[k_r].clear();
                                    int position_num1 = ind_h*width + EPI[0] + 8;
                                    int position_num2 = ind_h*width + EPI[1] + 8;
                                    int position_num3 = ind_h*width + EPI[2] + 8;
                                    int position_num4 = ind_h*width + EPI[3] + 8;
                                    patch_table1[k_r].clear();
                                    patch_table1[k_r].push_back(position_num1);
                                    patch_table1[k_r].push_back(position_num2);
                                    patch_table1[k_r].push_back(position_num3);
                                    patch_table1[k_r].push_back(position_num4);
                                    ind_j++;
                                }
                            }

                        }
                    }

                }

                // 왼쪽 boundary처리
                for (int a = iNumFrm - 1; a > 0; a--)
                {
                    for (int b = 0; b < bound; b+=3) {
                        int pixel1 = b; //a번째 view
                        temp_cost = INT32_MAX;
                        best_cost = INT32_MAX;
                        best_x = 0;
                        bound_count++;
                        if (pixel1 < window_size) {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 + x; //a-1번째 view

                                for (int s = 0; s < 2*window_size; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[1][s] = RaySpace[a-1][j][pixel0 + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / 2*window_size;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = 0; s < 2*window_size; s++) {
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a-1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[a-1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 + x; //a-1번째 view

                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - window_size + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[1][s] = RaySpace[a-1][j][pixel0 - window_size + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {

                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / window_size * 2;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a-1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a-1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size ;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;

                                    }
                                }
                            }
                        } // 2view까지 그었음
                        int position[4];
                        for (int v = a-1; v > 0; v--)
                        {
                            temp_cost = INT32_MAX;
                            best_cost = INT32_MAX;
                            int pre = 2*best_x - pixel1; //예상 위치
                            if (pre < window_size) {
                                for (int x = 0; x < search_range2; x++) {
                                    int pixel0 = pre + x; //a-1번째 view

                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2*window_size; s++) {
                                            y_buf_1D[1][s] = RaySpace[v-1][j][pixel0 + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / 2*window_size;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = 0; s < 2*window_size; s++) {
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v-1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[v-1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v-1] = pixel0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int x = -search_range2; x < search_range2; x++) {
                                    int pixel0 = pixel1 + x; //a-1번째 view

                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x - window_size + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2 * window_size + 2; s++) {
                                            y_buf_1D[1][s] = RaySpace[v-1][j][pixel0 - window_size + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 && y == 1) continue;
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {

                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / window_size * 2;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v-1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v-1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size ;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v-1] = pixel0;

                                        }
                                    }
                                }
                            }
                        }

                        if (a == 1){
                            line_count[0][best_x]++;
                            line_count[1][pixel1]++;
                            //cout << "pixel1 " << best_x << " " << line_count[0][best_x] <<  endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + best_x + 8;
                            int position_num2 = ind_h*width + pixel1 + 8;
                            int position_num3 = -1;
                            int position_num4 = -1;
                            //cout << pixel1 + best_x << " " << pixel1 << endl;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                        else if(a == 2){
                            line_count[0][position[0]]++;
                            line_count[1][best_x]++;
                            line_count[2][pixel1]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + position[0] + 8;
                            int position_num2 = ind_h*width + best_x + 8;
                            int position_num3 = ind_h*width + pixel1 + 8;
                            int position_num4 = -1;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                        else if(a == 3){
                            line_count[0][position[0]]++;
                            line_count[1][position[1]]++;
                            line_count[2][best_x]++;
                            line_count[3][pixel1]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + position[0] + 8;
                            int position_num2 = ind_h*width + position[1] + 8;
                            int position_num3 = ind_h*width + best_x + 8;
                            int position_num4 = ind_h*width + pixel1 + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                    }
                }

                // 오른쪽 바운더리 처리
                for (int a = 0; a < iNumFrm - 1; a++)
                {
                    for (int b = iFrmWidth - bound; b < iFrmWidth; b+=3) {
                        int pixel1 = b; //a번째 view
                        temp_cost = INT32_MAX;
                        best_cost = INT32_MAX;
                        best_x = 0;
                        bound_count++;
                        if (pixel1 > iFrmWidth - window_size) {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 - x; //a+1번째 view

                                for (int s = 0; s < 2*window_size; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[1][s] = RaySpace[a+1][j][pixel0 - s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;

                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]);
                                        }
                                        temp_cost = temp_cost / 2*window_size ;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = 0; s < 2*window_size; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]) * (y_buf_1D[y][s] - RaySpace[a][j][pixel1 - s]);
                                        }
                                        for (int s = 0; s < 2*window_size; s++) {
                                            temp_cost += (y_buf_1D[y][s] - RaySpace[a+1][j][pixel0 - s]) * (y_buf_1D[y][s] - RaySpace[a+1][j][pixel0 - s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int x = 0; x < search_range; x++) {
                                int pixel0 = pixel1 - x; //a-1번째 view

                                for (int s = 0; s < 2 * window_size + 2; s++) {
                                    y_buf_1D[0][s] = RaySpace[a][j][pixel1 - window_size + s];
                                }
                                if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[1][s] = RaySpace[a+1][j][pixel0 - window_size + s];
                                    }
                                }

                                for (int y = 0; y < 2; y++)
                                {
                                    if (pixel0 < 0 && y == 1) continue;
                                    if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                    {

                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        temp_cost = temp_cost / window_size * 2;
                                    }
                                    else // 1view bound 넘어가지 않은 상황
                                    {
                                        temp_cost = (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]) * (y_buf_1D[y][window_size] - RaySpace[a][j][pixel1]);
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            if (s == 0) continue;
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a][j][pixel1 + s]);
                                        }
                                        for (int s = -window_size; s < window_size + 1; s++) {
                                            temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[a+1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[a+1][j][pixel0 + s]);
                                        }
                                        temp_cost = temp_cost / 2 * 2*window_size;
                                    }
                                    if (temp_cost < best_cost)
                                    {
                                        best_cost = temp_cost;
                                        best_x = pixel0;
                                    }
                                }
                            }
                        }
                        int position[4];
                        for (int v = 1; v < iNumFrm-1; v++)
                        {
                            temp_cost = INT32_MAX;
                            best_cost = INT32_MAX;
                            int pre = 2*best_x - pixel1; //예상 위치
                            if (pre < iFrmWidth - window_size) {
                                for (int x = 0; x < search_range2; x++) {
                                    int pixel0 = pre - x; //a-1번째 view

                                    for (int s = 0; s < 2*window_size; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2*window_size; s++) {
                                            y_buf_1D[1][s] = RaySpace[v+1][j][pixel0 + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / 2*window_size;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = 0; s < 2*window_size; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = 0; s < 2*window_size; s++) {
                                                temp_cost += (y_buf_1D[y][s] - RaySpace[v+1][j][pixel0 + s]) * (y_buf_1D[y][s] - RaySpace[v+1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v+1] = pixel0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int x = -search_range2; x < search_range2; x++) {
                                    int pixel0 = pixel1 + x; //a-1번째 view

                                    for (int s = 0; s < 2 * window_size + 2; s++) {
                                        y_buf_1D[0][s] = RaySpace[v][j][best_x - window_size + s];
                                    }
                                    if (0 <= pixel0 && pixel0 < iFrmWidth) {
                                        for (int s = 0; s < 2 * window_size + 2; s++) {
                                            y_buf_1D[1][s] = RaySpace[v+1][j][pixel0 - window_size + s];
                                        }
                                    }

                                    for (int y = 0; y < 2; y++)
                                    {
                                        if (pixel0 < 0 && y == 1) continue;
                                        if (pixel0 < 0 || pixel0 >= iFrmWidth) // 1view bound 넘어간 상황
                                        {

                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            temp_cost = temp_cost / window_size * 2;
                                        }
                                        else // 1view bound 넘어가지 않은 상황
                                        {
                                            temp_cost = (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]) * (y_buf_1D[y][window_size] - RaySpace[v][j][best_x]);
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                if (s == 0) continue;
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v][j][best_x + s]);
                                            }
                                            for (int s = -window_size; s < window_size + 1; s++) {
                                                temp_cost += (y_buf_1D[y][s + window_size] - RaySpace[v+1][j][pixel0 + s]) * (y_buf_1D[y][s + window_size] - RaySpace[v+1][j][pixel0 + s]);
                                            }
                                            temp_cost = temp_cost / 2 * 2*window_size ;
                                        }
                                        if (temp_cost < best_cost)
                                        {
                                            best_cost = temp_cost;
                                            position[v+1] = pixel0;

                                        }
                                    }
                                }
                            }
                        }


                        if (a == 0){
                            line_count[0][pixel1]++;
                            line_count[1][best_x]++;
                            line_count[2][position[2]]++;
                            line_count[3][position[3]]++;
                            //cout << "pixel1 " << pixel1 << endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = ind_h*width + pixel1 + 8;
                            int position_num2 = ind_h*width + best_x + 8;
                            int position_num3 = ind_h*width + position[2] + 8;
                            int position_num4 = ind_h*width + position[3] + 8;
                            //cout << pixel1 + best_x << " " << pixel1 << endl;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                        else if(a == 1){
                            line_count[1][pixel1]++;
                            line_count[2][best_x]++;
                            line_count[3][position[3]]++;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = -1;
                            int position_num2 = ind_h*width + pixel1 + 8;
                            int position_num3 = ind_h*width + best_x + 8;
                            int position_num4 = ind_h*width + position[3] + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                        else if(a == 2){
                            line_count[3][pixel1]++;
                            line_count[3][best_x]++;
                            //cout << "haight : " << ind_h << "line : " << best_x << endl;
                            int k_r = (ind_h * (width + 3000) + ind_j);
                            patch_table1[k_r].clear();
                            int position_num1 = -1;
                            int position_num2 = -1;
                            int position_num3 = ind_h*width + pixel1 + 8;
                            int position_num4 = ind_h*width + best_x + 8;
                            patch_table1[k_r].clear();
                            patch_table1[k_r].push_back(position_num1);
                            patch_table1[k_r].push_back(position_num2);
                            patch_table1[k_r].push_back(position_num3);
                            patch_table1[k_r].push_back(position_num4);
                            ind_j++;
                        }
                    }
                }

                patch_table2[ind_h].push_back(ind_j);
                int count = 0;
                int count1 = 0;
                for(int empty = 0; empty < bound; empty++)
                {
                    int k_r = ind_h * bound + count;
                    int k_r1 = ind_h * bound + count1;
                    if(line_count[0][empty] == 0)
                    {
                        patch_table3[k_r].clear();
                        int empty_num = ind_h*width + empty + 8;
                        patch_table3[k_r].push_back(empty_num);
                        count++;
                    }
                    if(line_count[3][iFrmWidth -1 - empty] == 0){
                        patch_table4[k_r].clear();
                        int empty_num = ind_h*width + iFrmWidth -1 - empty + 8;
                        patch_table4[k_r1].push_back(empty_num);
                        count1++;
                    }
                }
                patch_table2[ind_h].push_back(count);
                patch_table2[ind_h].push_back(count1);
                single_count += count;
                single_count += count1;
                break; // 모든 Row의 예측이 끝난 경우 for문 종료
            }
        } // infinite loop end  ( width ) , width에 대해 1view & 2view 끝
        ind_h++;
    }
    cout << "extra_count : " << extra_count << endl;
    cout << "inter_count : " << inter_count << endl;
    cout << "bound_count : " << bound_count << endl;
    cout << "single_count : " << single_count << endl;
}

void bm3d_1st_step_original_EPI(
        const float sigma
        ,   vector<float> const& noisy
        ,   vector<float> const& noisy2
        ,   vector<float> const& noisy3
        ,   vector<float> const& noisy4
        ,   vector<float> const& img_noisy // symetrized
        ,   vector<float> const& img_noisy2
        ,   vector<float> const& img_noisy3
        ,   vector<float> const& img_noisy4
        ,   vector<float> &img_basic
        ,   vector<float> &img_basic2
        ,   vector<float> &img_basic3
        ,   vector<float> &img_basic4
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
){
    clock_t EPI_time;
    clock_t trans_time;
    clock_t inv_trans_time;
    clock_t inv_trans_time1;
    clock_t inv_trans_time2;
    clock_t hard_time;
    clock_t hard_time1;
    clock_t hard_time2;
    clock_t hard_time3;
    clock_t hard_time4;
    clock_t aggreagation_time;
    clock_t aggreagation_time1;
    clock_t aggreagation_time2;
    double clock_trans;
    double clock_inv_trans;
    double clock_inv_trans1;
    double clock_inv_trans2;
    double clock_hard;
    double clock_hard1;
    double clock_hard2;
    double clock_hard3;
    double clock_hard4;
    double clock_EPI;
    double clock_aggreagation;
    double clock_aggreagation1;
    double clock_aggreagation2;
    double clock_result_trans = 0;
    double clock_result_inv_trans = 0;
    double clock_result_hard = 0;
    double clock_result_aggregation = 0;
    double clock_result_EPI = 0;
    int bound = 0;
    int num_group=0;
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if (img_basic2.size() != img_noisy2.size())
        img_basic2.resize(img_noisy2.size());
    if (img_basic3.size() != img_noisy3.size())
        img_basic3.resize(img_noisy3.size());
    if (img_basic4.size() != img_noisy4.size())
        img_basic4.resize(img_noisy4.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);

    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);
    vector<float> denominator2(width * height * chnls, 0.0f);
    vector<float> numerator2  (width * height * chnls, 0.0f);
    vector<float> denominator3(width * height * chnls, 0.0f);
    vector<float> numerator3  (width * height * chnls, 0.0f);
    vector<float> denominator4(width * height * chnls, 0.0f);
    vector<float> numerator4  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<int> > patch_table1;
    vector<vector<int> > patch_table2;
    vector<vector<int> > patch_table3;
    vector<vector<int> > patch_table4;// bound 영역에서 비어있는 부분
    cout << " start " << endl;
    //precompute(patch_table1, patch_table2, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);

    EPI_time = clock();
    precompute_original_EPI(patch_table1, patch_table2, patch_table3, patch_table4, noisy, noisy2 ,noisy3 ,noisy4 ,width ,height, tauMatch, pHard);
    clock_EPI = (double)(clock() - EPI_time) / CLOCKS_PER_SEC;
    clock_result_EPI += clock_EPI;

    //cout << " patch_table1 size : " << patch_table1.size() << " patch_table2 size : " << patch_table3.size() << endl;
    cout << "----------------EPI--------------"<< endl;
    //cout << "width : " << width << " height : " << height << endl;
    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D_1((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_2((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_3((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    vector<float> table_2D_4((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);
    int index = 0;
    int index2 = 0;
    //! Loop on i_r

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        //cout << "height : " << row_ind[ind_i] << endl;
        const unsigned i_r = row_ind[ind_i];
        //! Update of table_2D
        trans_time = clock();
        if (tau_2D == DCT) {
            dct_2d_process(table_2D_1, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_2, img_noisy2, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_3, img_noisy3, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_4, img_noisy4, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());

        }
        else if (tau_2D == BIOR){
            bior_2d_process(table_2D_1, img_noisy, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_2, img_noisy2, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_3, img_noisy3, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_4, img_noisy4, nHard, width, height, chnls,
                            kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);
        }
        clock_trans = (double)(clock() - trans_time) / CLOCKS_PER_SEC;
        clock_result_trans += clock_trans;

        //왼쪽 바운더리 채우기
        wx_r_table.clear();
        group_3D_table.clear();
        num_group += patch_table2[i_r][1];
        for(int b = 0; b < patch_table2[i_r][1]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r1 = 1;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r1 * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r1; n++)
                {
                    const unsigned ind = patch_table3[k_r][n] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r1 + c * kHard_2 * nSx_r1] =
                                table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }

            //! HT filtering of the 3D group
            hard_time1 = clock();
            vector<float> weight_table1(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r1, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table1, !useSD);
            clock_hard1 = (double) (clock() - hard_time1) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard1;

            const unsigned nSx_r = 2;
            group_3D.resize(chnls * nSx_r * kHard_2, 0.0f);
            //! Build of the 3D group
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table3[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }
            //! HT filtering of the 3D group
            hard_time2 = clock();
            vector<float> weight_table(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard2 = (double) (clock() - hard_time2) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard2;
            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++) {
                    //if (n == 0) continue;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);
                    }
                }

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        }

        inv_trans_time1 = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans1 = (double)(clock() - inv_trans_time1) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans1;

        //! Registration of the weighted estimation
        unsigned dec2 = 0;
        aggreagation_time1 = clock();

        for(int b = 0; b < patch_table2[i_r][1]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r = 1;
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table3[k_r][0] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {

                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kHard + q]
                                              * wx_r_table[c + b * chnls]
                                              * group_3D_table[p * kHard + q + n * kHard_2
                                                               + c * kHard_2 * nSx_r + dec2];
                            denominator[ind] += kaiser_window[p * kHard + q]
                                                * wx_r_table[c + b * chnls];
                        }
                }
            }
            dec2 += chnls * kHard_2;
        }
        clock_aggreagation1 = (double)(clock() - aggreagation_time1) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation1;
        //오른쪽 바운더리 채우기
        wx_r_table.clear();
        group_3D_table.clear();
        num_group += patch_table2[i_r][2];
        for(int b = 0; b < patch_table2[i_r][2]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r1 = 1;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r1 * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r1; n++)
                {
                    const unsigned ind = patch_table4[k_r][n] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r1 + c * kHard_2 * nSx_r1] =
                                table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }

            //! HT filtering of the 3D group
            hard_time3 = clock();
            vector<float> weight_table1(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r1, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table1, !useSD);
            clock_hard3 = (double) (clock() - hard_time3) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard3;

            const unsigned nSx_r = 2;
            group_3D.resize(chnls * nSx_r * kHard_2, 0.0f);
            //! Build of the 3D group
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table4[k_r][0] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }
                }
            //! HT filtering of the 3D group
            hard_time4 = clock();
            vector<float> weight_table(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard4 = (double) (clock() - hard_time4) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard4;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 1; n < nSx_r; n++) {
                    //if (n == 0) continue;
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);
                    }
                }

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        }

        inv_trans_time2 = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans2 = (double)(clock() - inv_trans_time2) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans2;

        //! Registration of the weighted estimation
        unsigned dec3 = 0;
        aggreagation_time2 = clock();
        for(int b = 0; b < patch_table2[i_r][2]; b++) //1st view에서 비어있는 영역의 양
        {
            const unsigned k_r = i_r * bound + b;
            const unsigned nSx_r = 1;
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table4[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator4[ind] += kaiser_window[p * kHard + q]
                                               * wx_r_table[c + b * chnls]
                                               * group_3D_table[p * kHard + q + n * kHard_2
                                                                + c * kHard_2 * nSx_r + dec3];
                            denominator4[ind] += kaiser_window[p * kHard + q]
                                                 * wx_r_table[c + b * chnls];
                        }
                }
            }
            dec3 += chnls * kHard_2;
        }
        clock_aggreagation2 = (double)(clock() - aggreagation_time2) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation2;
        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        //patch_table2[ind_i].size() : 해당 높이에 따른 EPI 수
        num_group += patch_table2[i_r][0];
        for (unsigned ind_j = 0; ind_j < patch_table2[i_r][0]; ind_j++) {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * (width + 3000) + ind_j;
            //cout << patch_table1[k_r][0] << " " << patch_table1[k_r][1] << " " << patch_table1[k_r][2] << " " << patch_table1[k_r][3] << endl;
            //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1) continue;
            //! Number of similar patches8
            unsigned nSx_r = patch_table1[k_r].size();
            for(int n = 0; n <patch_table1[k_r].size(); n++)
            {
                if(patch_table1[k_r][n] == -1)
                    nSx_r--;
            }
            if(nSx_r == 3)
                nSx_r = 4;
            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++) {
                if(patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1) {
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 3 && patch_table1[k_r][n] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][n] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //존재
                else if(patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
                {
                    //cout << ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][0] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][0] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //#01
                else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1)
                {
                    //cout << ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if(n == 0 && patch_table1[k_r][0] == -1){
                            for (unsigned c = 0; c < chnls; c++)
                                for (unsigned n = 0; n < 1; n++)
                                {
                                    const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                    for (unsigned k = 0; k < kHard_2; k++) {
                                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                                table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                                    }
                                }

                            //! HT filtering of the 3D group
                            vector<float> weight_table1(chnls);
                            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                                  lambdaHard3D, weight_table1, !useSD);
                        }
                        else if (n == 1 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][2] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                        else if (n == 3 && patch_table1[k_r][3] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][3] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //#02
                else if(patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] == -1)
                {
                    //cout <<"d "<< ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if(n == 0 && patch_table1[k_r][3] == -1){
                            for (unsigned c = 0; c < chnls; c++)
                                for (unsigned n = 0; n < 1; n++)
                                {
                                    const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                    for (unsigned k = 0; k < kHard_2; k++) {
                                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                                table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                                    }
                                }

                            //! HT filtering of the 3D group
                            vector<float> weight_table1(chnls);
                            ht_filtering_hadamard(group_3D, hadamard_tmp, 1, kHard, chnls, sigma_table,
                                                  lambdaHard3D, weight_table1, !useSD);
                        }
                        else if (n == 1 && patch_table1[k_r][0] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][0] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_1[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        } else if (n == 2 && patch_table1[k_r][1] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][1] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_2[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                        else if (n == 3 && patch_table1[k_r][2] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        }
                    }
                }
                    //존재
                else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1)
                {
                    //cout <<"dd "<< ind_j << endl;
                    for (unsigned n = 0; n < nSx_r; n++) {
                        if (n == 0 && patch_table1[k_r][2] != -1) {
                            if (ind_i == 0 && ind_j == 0)
                                cout << patch_table1[k_r][n] << endl;
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][2] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_3[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                            }
                        } else if (n == 1 && patch_table1[k_r][3] != -1) {
                            for (unsigned k = 0; k < kHard_2; k++) {
                                const unsigned ind = patch_table1[k_r][3] + (nHard - i_r) * width;
                                group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                                        table_2D_4[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];

                            }
                        }
                    }
                }
                //else if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
                //{
                //아무동작안함
                //}

            }
            //! HT filtering of the 3D group
            //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1)
            // {
            //아무동작안함
            //}
            //else {
            vector<float> weight_table(chnls);
            hard_time = clock();
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                  lambdaHard3D, weight_table, !useSD);
            clock_hard = (double) (clock() - hard_time) / CLOCKS_PER_SEC;
            clock_result_hard += clock_hard;

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++) {
                for (unsigned n = 0; n < nSx_r; n++) {
                    for (unsigned k = 0; k < kHard_2; k++) {
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                          c * kHard_2 * nSx_r]);
                    }
                }
            }

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);
            //}

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        inv_trans_time = clock();
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * patch_table2.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);
        clock_inv_trans = (double)(clock() - inv_trans_time) / CLOCKS_PER_SEC;
        clock_result_inv_trans += clock_inv_trans;
        //! Registration of the weighted estimation
        unsigned dec = 0;
        aggreagation_time = clock();
        for (unsigned ind_j = 0; ind_j < patch_table2[i_r][0]; ind_j++) {
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * (width + 3000) + ind_j;
            //if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 &&
            //    patch_table1[k_r][3] == -1) {
            //아무동작안함
            //} else{
            //if(patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] == -1 && patch_table1[k_r][3] == -1) continue;
            unsigned nSx_r = patch_table1[k_r].size();
            for (int n = 0; n < patch_table1[k_r].size(); n++) {
                if (patch_table1[k_r][n] == -1)
                    nSx_r--;
            }
            if (nSx_r == 3)
                nSx_r = 4;
            for (unsigned c = 0; c < chnls; c++) {
                for (unsigned n = 0; n < nSx_r; n++) {
                    if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                        patch_table1[k_r][3] != -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                //if(patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 && patch_table1[k_r][3] != -1) {
                                if (n == 0 && patch_table1[k_r][n] != -1) {

                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][n] != -1) {
                                    const unsigned k = patch_table1[k_r][n] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    } else if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] == -1 &&
                               patch_table1[k_r][3] == -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 0 && patch_table1[k_r][0] != -1) {
                                    const unsigned k = patch_table1[k_r][0] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }

                    }

                        //#01
                    else if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] != -1) {
                        if (n == 0) continue;
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 1 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][3] != -1) {
                                    const unsigned k = patch_table1[k_r][3] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    }

                        // #02
                    else if (patch_table1[k_r][0] != -1 && patch_table1[k_r][1] != -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] == -1) {
                        if (n == 0) continue;
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 1 && patch_table1[k_r][0] != -1) {
                                    const unsigned k = patch_table1[k_r][0] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator[ind] += kaiser_window[p * kHard + q]
                                                      * wx_r_table[c + ind_j * chnls]
                                                      * group_3D_table[p * kHard + q + n * kHard_2
                                                                       + c * kHard_2 * nSx_r + dec];
                                    denominator[ind] += kaiser_window[p * kHard + q]
                                                        * wx_r_table[c + ind_j * chnls];
                                } else if (n == 2 && patch_table1[k_r][1] != -1) {
                                    const unsigned k = patch_table1[k_r][1] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator2[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator2[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 3 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }
                    }
                        //
                    else if (patch_table1[k_r][0] == -1 && patch_table1[k_r][1] == -1 && patch_table1[k_r][2] != -1 &&
                             patch_table1[k_r][3] != -1) {
                        for (unsigned p = 0; p < kHard; p++) {
                            for (unsigned q = 0; q < kHard; q++) {
                                if (n == 0 && patch_table1[k_r][2] != -1) {
                                    const unsigned k = patch_table1[k_r][2] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator3[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator3[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                } else if (n == 1 && patch_table1[k_r][3] != -1) {
                                    const unsigned k = patch_table1[k_r][3] + c * width * height;
                                    const unsigned ind = k + p * width + q;
                                    numerator4[ind] += kaiser_window[p * kHard + q]
                                                       * wx_r_table[c + ind_j * chnls]
                                                       * group_3D_table[p * kHard + q + n * kHard_2
                                                                        + c * kHard_2 * nSx_r + dec];
                                    denominator4[ind] += kaiser_window[p * kHard + q]
                                                         * wx_r_table[c + ind_j * chnls];
                                }
                            }
                        }

                    }
                }
            }
            dec += nSx_r * chnls * kHard_2;
            //}
        }
        clock_aggreagation = (double)(clock() - aggreagation_time) / CLOCKS_PER_SEC;
        clock_result_aggregation += clock_aggreagation;

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++) {
        img_basic[k] = numerator[k] / denominator[k];
        img_basic2[k] = numerator2[k] / denominator2[k];
        img_basic3[k] = numerator3[k] / denominator3[k];
        img_basic4[k] = numerator4[k] / denominator4[k];
    }

    printf("\n< youna original >\npatch num : %d\n", num_group);
    printf("hardthresholding processing time : %5.2f\n", clock_result_hard);
    printf("EPI processing time : %5.2f\n", clock_result_EPI);
    printf("transform processing time : %5.2f\n", clock_result_trans);
    printf("inverse trans processing time : %5.2f\n", clock_result_inv_trans);
    printf("aggregation processing time : %5.2f\n", clock_result_aggregation);
}

void quick_sort(int *data, int start, int end){
    if(start >= end){
        // 원소가 1개인 경우
        return;
    }
int pivot = start;
    int i = pivot + 1; // 왼쪽 출발 지점
    int j = end; // 오른쪽 출발 지점
    int temp;

    while(i <= j){
        // 포인터가 엇갈릴때까지 반복
        while(i <= end && data[i] <= data[pivot]){
            i++;
        }
        while(j > start && data[j] >= data[pivot]){
            j--;
        }

        if(i > j){
            // 엇갈림
            temp = data[j];
            data[j] = data[pivot];
            data[pivot] = temp;
        }else{
            // i번째와 j번째를 스왑
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // 분할 계산
    quick_sort(data, start, j - 1);
    quick_sort(data, j + 1, end);
}