#include <opencv2/highgui.hpp>
#include <iostream>

cv::Mat claheGO(cv::Mat src,int step = 8) {
  cv::Mat CLAHE_GO = src.clone();
  int block        = step;          // pblock
  int width        = src.cols;
  int height       = src.rows;
  int width_block  = width / block; // the length and width of each small lattice
  int height_block = height / block;

  // Store each histogram  
  int tmp2[8*8][256] = {0};
  float C2[8*8][256] = {0.0};

  //Block 
  int total = width_block * height_block; 
  for (int i = 0; i < block; i++) {
    for (int j = 0; j < block; j++) {
      int start_x = i * width_block;
      int end_x   = start_x + width_block;
      int start_y = j * height_block;
      int end_y   = start_y + height_block;
      int num     = i + block * j;  
      
      // Traverse the small block, calculate the histogram
      for (int ii = start_x; ii < end_x; ii++) {
        for (int jj = start_y; jj < end_y; jj++) {
          int index =src.at<uchar>(jj,ii);
          tmp2[num][index]++;  
        }  
      } 

      // Crop and increase operations, that is, the cl part of the clahe
      // The parameters here correspond to "Gem" above fCliplimit = 4 , uiNrBins = 255
      int average = width_block * height_block / 255;  
      // How to choose the parameters, you need to discuss. Discuss different results
      // About the global time, how to calculate this cl here, need to discuss 
      int LIMIT = 40 * average;  
      int steal = 0;  
      for (int k = 0; k < 256; k++) {
        if (tmp2[num][k] > LIMIT) {
          steal       += tmp2[num][k] - LIMIT;  
          tmp2[num][k] = LIMIT;  
        }
      }

      int bonus = steal / 256;  
      
      // hand out the steals averagely  
      for (int k = 0; k < 256; k++) {
        tmp2[num][k] += bonus;  
      }
      
      // Calculate the cumulative distribution histogram  
      for (int k = 0; k < 256; k++) {
        if (k == 0)  
          C2[num][k] = 1.0f * tmp2[num][k] / total;  
        else  
          C2[num][k] = C2[num][k-1] + 1.0f * tmp2[num][k] / total;  
      }  
    }
  }

  // Calculate the transformed pixel value  
  // According to the location of the pixel, choose a different calculation method  
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      // four coners  
      if (i <= width_block / 2 && j <= height_block / 2) {
        int num = 0;  
        CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
      } else if (i <= width_block / 2 && j >= ((block - 1) * height_block + height_block / 2)) {  
        int num = block * (block - 1);  
        CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
      } else if (i >= ((block - 1) * width_block + width_block / 2) && j <= height_block / 2) {  
        int num = block - 1;  
        CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
      } else if (i >= ((block - 1) * width_block + width_block / 2) && j >= ((block - 1) * height_block + height_block / 2)) {
        int num = block * block - 1;  
        CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
      }  
      // four edges except coners  
      else if (i <= width_block / 2) {
        //linear interpolation  
        int num_i = 0;  
        int num_j = (j - height_block / 2) / height_block;  
        int num1  = num_j * block + num_i;  
        int num2  = num1 + block;  
        float p   = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);  
        float q   = 1 - p;  
        CLAHE_GO.at<uchar>(j,i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j,i)] + p * C2[num2][CLAHE_GO.at<uchar>(j,i)]) * 255);  
      } else if (i >= ((block - 1) * width_block + width_block / 2)) {  
        // linear interpolation  
        int num_i = block - 1;  
        int num_j = (j - height_block / 2) / height_block;  
        int num1  = num_j * block + num_i;  
        int num2  = num1 + block;  
        float p   = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);  
        float q   = 1 - p;  
        CLAHE_GO.at<uchar>(j,i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j,i)] + p * C2[num2][CLAHE_GO.at<uchar>(j,i)]) * 255);  
      } else if (j <= height_block / 2) {  
        // linear interpolation  
        int num_i = (i - width_block / 2) / width_block;  
        int num_j = 0;  
        int num1  = num_j * block + num_i;  
        int num2  = num1 + 1;  
        float p   = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);  
        float q   = 1 - p;  
        CLAHE_GO.at<uchar>(j,i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j,i)] + p * C2[num2][CLAHE_GO.at<uchar>(j,i)]) * 255);  
      } else if (j >= ((block - 1) * height_block + height_block / 2)) {  
        // linear interpolation  
        int num_i = (i - width_block / 2) / width_block;  
        int num_j = block - 1;  
        int num1  = num_j * block + num_i;  
        int num2  = num1 + 1;  
        float p   = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);  
        float q   = 1 - p;  
        CLAHE_GO.at<uchar>(j,i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j,i)] + p * C2[num2][CLAHE_GO.at<uchar>(j,i)]) * 255);  
      }
      // bilinear interpolation
      else {  
        int num_i = (i - width_block / 2) / width_block;  
        int num_j = (j - height_block / 2) / height_block;  
        int num1  = num_j * block + num_i;  
        int num2  = num1 + 1;  
        int num3  = num1 + block;  
        int num4  = num2 + block;  
        float u   = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);  
        float v   = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);  
        CLAHE_GO.at<uchar>(j,i) = (int)((   u  *    v  * C2[num4][CLAHE_GO.at<uchar>(j,i)] +   
                                         (1-u) * (1-v) * C2[num1][CLAHE_GO.at<uchar>(j,i)] +  
                                            u  * (1-v) * C2[num2][CLAHE_GO.at<uchar>(j,i)] +  
                                         (1-u) *    v  * C2[num3][CLAHE_GO.at<uchar>(j,i)]
					) * 255);  
      }  

      // The last step, like Gaussian smoothing
      CLAHE_GO.at<uchar>(j,i) = CLAHE_GO.at<uchar>(j,i) + (CLAHE_GO.at<uchar>(j,i) << 8) + (CLAHE_GO.at<uchar>(j,i) << 16);         
    }  
  }  
  return CLAHE_GO;
}

int main(int argc, char** argv) {
    // Read the grayscale image
    cv::Mat src = cv::imread(argv[1],0);
    cv::Mat dst = src.clone();
    cv::Mat CLAHE_GO;

    CLAHE_GO = claheGO(src);
 
    // The results show 
    imshow("original image", src);
    imshow("GOCLAHE", CLAHE_GO);
    cv::waitKey();

    return 0;
}
