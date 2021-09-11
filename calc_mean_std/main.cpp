#include <opencv.hpp>
#include <time.h>

typedef unsigned char UINT8_G;
typedef unsigned short UINT16_G;
typedef unsigned int UINT32_G;

using namespace std;
using namespace cv;

template<typename T>
void print_data(T* data, int h, int w)
{
    cout << endl;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            printf("%d\t", data[i * w + j]);
        }
        cout << endl;
    }
    cout << endl;
}

/*
*  Function: calculate a window's mean and std according to the integral image 
*  ref: https://en.wikipedia.org/wiki/Summed-area_table
*  patch_integ: s1 var in link.
*  patch_pow_integ: s2 var in link.
*  h_integ, w_integ: integral image h, w.
*  px, py: window center point pos in patch image.
*  win_r: window radius, define the range used to calc mean and std.
* 
*  a               b
*     -------------
*    |             |
*    |   (px, py)  |
*    |             |
*  c  ------------ d
*/
vector<double> CalcMeanStd(UINT32_G* patch_integ, UINT32_G* patch_pow_integ, int h_integ, int w_integ, int px, int py, int win_r)
{

    double ceoff = 1.0 / (((win_r << 1) + 1) * ((win_r << 1) + 1)); // 1/n

    // (px, py) is (px+1, py+1) in integral image
    int d_pos = (py + 1 + win_r) * w_integ + (px + 1 + win_r);
    int a_pos = (py - win_r) * w_integ + (px - win_r);
    int b_pos = (py - win_r) * w_integ + (px + 1 + win_r);
    int c_pos = (py + 1 + win_r) * w_integ + (px - win_r);

    long long s1 = patch_integ[d_pos]
        + patch_integ[a_pos]
        - patch_integ[b_pos]
        - patch_integ[c_pos];
    long long s2 = patch_pow_integ[d_pos]
        + patch_pow_integ[a_pos]
        - patch_pow_integ[b_pos]
        - patch_pow_integ[c_pos];

    double mean = s1 * ceoff;
    double std = std::sqrt(ceoff * (s2 - s1 * s1 * ceoff));
    return { mean, std };
}



/*
*  Function: Calclate a image patch's integral image.
*  img: full image
*  patch_integ: patch integral image, created before, size (patch_hight + 1, patch_width + 1)
*  x0: patch left-top x - 1
*  y0: patch left-top y - 1
*  h_integ: Integral image heigth, patch_hight + 1 
*  w_integ: patch_width + 1
*  w_full: full image width
*  Caution: May be overflow!
*/
template<typename T>
void IntegralPatch(T* img, UINT32_G* patch_integ, UINT32_G* patch_pow_integ, int x0, int y0, int h_integ, int w_integ, int w_full)
{
    // start from (x0+1, y0+1), this is the patch start point
    for (int i = 1; i < h_integ; ++i)
    {
        for (int j = 1; j < w_integ; ++j)
        { 
            int pos = (y0 + i) * w_full + x0 + j; // pos in full image

            patch_integ[i * w_integ + j] = img[pos]
                + patch_integ[i * w_integ + j - 1]
                + patch_integ[(i - 1) * w_integ + j]
                - patch_integ[(i - 1) * w_integ + j - 1];

            patch_pow_integ[i * w_integ + j] = img[pos] * img[pos]
                + patch_pow_integ[i * w_integ + j - 1]
                + patch_pow_integ[(i - 1) * w_integ + j]
                - patch_pow_integ[(i - 1) * w_integ + j - 1];
        }
    }
}


int main()
{
    
    Mat img = imread("src.JPG", IMREAD_GRAYSCALE);
    int img_h = img.rows;
    int img_w = img.cols;

    UINT8_G* data = (UINT8_G*)malloc(sizeof(UINT8_G*) * img_h * img_w);
    for (int i = 0; i < img_h; ++i)
    {
        for (int j = 0; j < img_w; ++j)
        {
            data[i * img_w + j] = img.at<uchar>(i, j);
        }
    }

    // patch_range
    int pat_x0 = 170;
    int pat_y0 = 150;
    int pat_h = 13;
    int pat_w = 13;

    // opencv result
    Rect patch_range(pat_x0, pat_y0, pat_w, pat_h);
    Mat img_patch = img(patch_range);
    Mat mean, std;
    clock_t t0 = clock();
    meanStdDev(img_patch, mean, std);
    clock_t t1 = clock();
    printf("cv   mean=%.10f, std=%.10f, time=%.3f\n", mean.at<double>(0, 0), std.at<double>(0, 0), (float)(t1 - t0));

    // self defined
    clock_t t2 = clock();
    // integral image is (pat_h + 1, pat_w + 1)
    int integ_h = pat_h + 1;
    int integ_w = pat_w + 1;
    UINT32_G* patch_integ = (UINT32_G*)malloc(sizeof(UINT32_G) * integ_h * integ_w);
    memset(patch_integ, 0, sizeof(UINT32_G) * integ_h * integ_w);
    UINT32_G* patch_pow_integ = (UINT32_G*)malloc(sizeof(UINT32_G) * integ_h * integ_w);
    memset(patch_pow_integ, 0, sizeof(UINT32_G) * integ_h * integ_w);

    IntegralPatch(data, patch_integ, patch_pow_integ, pat_x0-1, pat_y0-1, integ_h, integ_w, img_w);
    //print_data(patch_pow_integ, integ_h, integ_w);

    // pos in patch
    int px = 6;
    int py = 6;
    int win_r = 6;
    vector<double> res = CalcMeanStd(patch_integ, patch_pow_integ, integ_h, integ_w, px, py, win_r);
    clock_t t3 = clock();
    printf("self mean=%.10f, std=%.10f, time=%.3f\n", res[0], res[1], (float)(t3- t2));

    free(data);
    free(patch_integ);
    free(patch_pow_integ);
}