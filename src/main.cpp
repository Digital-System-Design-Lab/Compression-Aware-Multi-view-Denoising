#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string.h>
#include <time.h>
#include "bm3d.h"
#include "utilities.h"

#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6
#define NONE      7

using namespace std;

/**
 * @file   main.cpp
 * @brief  Main executable file. Do not use lib_fftw to
 *         process DCT.
 *
 * @author MARC LEBRUN  <marc.lebrun@cmla.ens-cachan.fr>
 */


int main(int argc, char **argv)
{
    clock_t before;
    double clock_result;
    //! Check if there is the right call for the algorithm

    // 16 multi-view
    //if (argc < 79)

    if (argc < 24)
    {
        cout << "usage: BM3D image sigma noisy basic denoised difference bias \
                 difference_bias computeBias tau_2d_hard useSD_hard \
                 tau_2d_wien useSD_wien color_space" << endl;
        return EXIT_FAILURE;
    }

    //! Declarations
    vector<float> img1, img2, img3, img4, img5, img6, img7, img_noisy1, img_noisy2, img_noisy3, img_noisy4, img_noisy5, img_noisy6, img_noisy7, img_basic1, img_basic2, img_basic3, img_basic4, img_original_basic1, img_original_basic2, img_original_basic3, img_original_basic4;
    vector<float> img_basic_bm3d1, img_basic_bm3d2, img_basic_bm3d3, img_basic_bm3d4;
    vector<float> img_denoised1, img_denoised2, img_denoised3, img_denoised4, img_bias1, img_bias2, img_bias3, img_bias4, img_diff1, img_diff2, img_diff3, img_diff4;
    vector<float> img_basic_bias1, img_basic_bias2, img_basic_bias3, img_basic_bias4;
    vector<float> img_diff_bias1, img_diff_bias2, img_diff_bias3, img_diff_bias4;
    unsigned width, height, chnls;


    //! Load image
    if(load_image(argv[1], img1, &width, &height, &chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (load_image(argv[2], img2, &width, &height, &chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (load_image(argv[3], img3, &width, &height, &chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (load_image(argv[4], img4, &width, &height, &chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    //! Variables initialization
    float          fSigma       = atof(argv[5]);
    const int          search_range       = atof(argv[6]);
    const int          search_range2       = atof(argv[7]);
    const int          window_size       = atof(argv[8]);
    const bool     useSD_1      = (bool) atof(argv[21]);
    const unsigned tau_2D_hard  = (strcmp(argv[22], "dct" ) == 0 ? DCT :
                                   (strcmp(argv[22], "bior") == 0 ? BIOR : NONE));
    if (tau_2D_hard == NONE)
    {
        cout << "tau_2d_hard is not known. Choice is :" << endl;
        cout << " -dct" << endl;
        cout << " -bior" << endl;
        return EXIT_FAILURE;
    }

    const unsigned color_space  = (strcmp(argv[23], "rgb"  ) == 0 ? RGB   :
                                   (strcmp(argv[23], "yuv"  ) == 0 ? YUV   :
                                    (strcmp(argv[23], "ycbcr") == 0 ? YCBCR :
                                     (strcmp(argv[23], "opp"  ) == 0 ? OPP   : NONE))));
    if (color_space == NONE)
    {
        cout << "color_space is not known. Choice is :" << endl;
        cout << " -rgb" << endl;
        cout << " -yuv" << endl;
        cout << " -opp" << endl;
        cout << " -ycbcr" << endl;
        return EXIT_FAILURE;
    };
    unsigned       wh           = (unsigned) width * height;
    unsigned       whc          = (unsigned) wh * chnls;
    bool           compute_bias = (bool) atof(argv[29]);

    img_noisy1.resize(whc);
    img_diff1.resize(whc);

    img_noisy2.resize(whc);
    img_diff2.resize(whc);

    img_noisy3.resize(whc);
    img_diff3.resize(whc);

    img_noisy4.resize(whc);
    img_diff4.resize(whc);

    if (compute_bias)
    {
        img_bias1.resize(whc);
        img_basic_bias1.resize(whc);
        img_diff_bias1.resize(whc);
    }
    if (compute_bias)
    {
        img_bias2.resize(whc);
        img_basic_bias2.resize(whc);
        img_diff_bias2.resize(whc);
    }
    if (compute_bias)
    {
        img_bias3.resize(whc);
        img_basic_bias3.resize(whc);
        img_diff_bias3.resize(whc);
    }
    if (compute_bias)
    {
        img_bias4.resize(whc);
        img_basic_bias4.resize(whc);
        img_diff_bias4.resize(whc);
    }

    //! Add noise
    cout << endl << "Add noise(imput1) [sigma = " << fSigma << "] ...";
    add_noise1(img1, img_noisy1, fSigma);
    cout << "done." << endl;
    cout << endl << "Add noise(imput2) [sigma = " << fSigma << "] ...";
    add_noise2(img2, img_noisy2, fSigma);
    cout << "done." << endl;
    cout << endl << "Add noise(imput3) [sigma = " << fSigma << "] ...";
    add_noise3(img3, img_noisy3, fSigma);
    cout << "done." << endl;
    cout << endl << "Add noise(imput4) [sigma = " << fSigma << "] ...";
    add_noise4(img4, img_noisy4, fSigma);

    cout << "done." << endl;

    if (save_image(argv[9], img_noisy1, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (save_image(argv[10], img_noisy2, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (save_image(argv[11], img_noisy3, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if (save_image(argv[12], img_noisy4, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    before = clock();
    //! Denoising
    if (run_bm3d(fSigma, img1, img2, img3, img4, img_noisy1, img_noisy2, img_noisy3, img_noisy4, img_basic1, img_basic2, img_basic3, img_basic4,
                 img_original_basic1, img_original_basic2, img_original_basic3, img_original_basic4, img_basic_bm3d1, img_basic_bm3d2, img_basic_bm3d3, img_basic_bm3d4,
            img_denoised1, img_denoised2, img_denoised3, img_denoised4, width, height, chnls, search_range, search_range2, window_size,
                 useSD_1, tau_2D_hard, color_space)
        != EXIT_SUCCESS)
        return EXIT_FAILURE;

    clock_result = (double)(clock() - before) / CLOCKS_PER_SEC;

    //outlier
    if (save_image(argv[13], img_basic1, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic, rmse_basic, psnr_basic_bias, rmse_basic_bias;
    if(compute_psnr(img1, img_basic1, &psnr_basic, &rmse_basic) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(youna input image1) :" << endl;
    cout << "PSNR: " << psnr_basic << endl;
    cout << "RMSE: " << rmse_basic << endl << endl;

    if (save_image(argv[14], img_basic2, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic2, rmse_basic2, psnr_basic_bias2, rmse_basic_bias2;
    if(compute_psnr(img2, img_basic2, &psnr_basic2, &rmse_basic2) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(youna input image2) :" << endl;
    cout << "PSNR: " << psnr_basic2 << endl;
    cout << "RMSE: " << rmse_basic2 << endl << endl;

    if (save_image(argv[15], img_basic3, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic3, rmse_basic3, psnr_basic_bias3, rmse_basic_bias3;
    if(compute_psnr(img3, img_basic3, &psnr_basic3, &rmse_basic3) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(youna input image3) :" << endl;
    cout << "PSNR: " << psnr_basic3 << endl;
    cout << "RMSE: " << rmse_basic3 << endl << endl;

    if (save_image(argv[16], img_basic4, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic4, rmse_basic4, psnr_basic_bias4, rmse_basic_bias4;
    if(compute_psnr(img4, img_basic4, &psnr_basic4, &rmse_basic4) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(youna input image4) :" << endl;
    cout << "PSNR: " << psnr_basic4 << endl;
    cout << "RMSE: " << rmse_basic4 << endl << endl;

    if (save_image(argv[17], img_basic_bm3d1, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic_bm3d, rmse_basic_bm3d, psnr_basic_bias_bm3d, rmse_basic_bias_bm3d;
    if(compute_psnr(img1, img_basic_bm3d1, &psnr_basic_bm3d, &rmse_basic_bm3d) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(BM3D input image1) :" << endl;
    cout << "PSNR: " << psnr_basic_bm3d << endl;
    cout << "RMSE: " << rmse_basic_bm3d << endl << endl;

    if (save_image(argv[18], img_basic_bm3d2, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic_bm3d2, rmse_basic_bm3d2, psnr_basic_bias_bm3d2, rmse_basic_bias_bm3d2;
    if(compute_psnr(img2, img_basic_bm3d2, &psnr_basic_bm3d2, &rmse_basic_bm3d2) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(BM3D input image2) :" << endl;
    cout << "PSNR: " << psnr_basic_bm3d2 << endl;
    cout << "RMSE: " << rmse_basic_bm3d2 << endl << endl;

    if (save_image(argv[19], img_basic_bm3d3, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic_bm3d3, rmse_basic_bm3d3, psnr_basic_bias_bm3d3, rmse_basic_bias_bm3d3;
    if(compute_psnr(img3, img_basic_bm3d3, &psnr_basic_bm3d3, &rmse_basic_bm3d3) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(BM3D input image3) :" << endl;
    cout << "PSNR: " << psnr_basic_bm3d3 << endl;
    cout << "RMSE: " << rmse_basic_bm3d3 << endl << endl;

    if (save_image(argv[20], img_basic_bm3d4, width, height, chnls) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    float psnr_basic_bm3d4, rmse_basic_bm3d4, psnr_basic_bias_bm3d4, rmse_basic_bias_bm3d4;
    if(compute_psnr(img4, img_basic_bm3d4, &psnr_basic_bm3d4, &rmse_basic_bm3d4) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    cout << "(BM3D input image4) :" << endl;
    cout << "PSNR: " << psnr_basic_bm3d4 << endl;
    cout << "RMSE: " << rmse_basic_bm3d4 << endl << endl;

    // 16 multi-view version


    //! Compute PSNR and RMSE
    float psnr, rmse, psnr_bias, rmse_bias;

    cout << "done." << endl;
    printf("\nprocessing time : %5.2f\n", clock_result);
    return EXIT_SUCCESS;
}
