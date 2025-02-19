/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupreproc_ref.cpp
Purpose:		Vitis HLS DPU pre-processor
Revision History:	23 July 2021
authors:		daniele.bagni@xilinx.com

**************************************************************************************

Copyright 2020 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*******************************************************************************/

#include "dpupreproc_defines.h"


#ifdef ARM_HOST

void arm_ref_dpupreproc(unsigned char *R, unsigned char *G, unsigned char *B,
		signed char *out_rgb_img, float norm_fact, float shift_fact, float scale_fact,
		unsigned short int height, unsigned short int width)
{

	unsigned short int rows = height;
	unsigned short int cols = width;

	float norm = 1.0f / norm_fact;
	float gain = scale_fact;

	printf("%s: Setting norm=%f shift=%f scale=%f\n", __func__, norm_fact,
			shift_fact, scale_fact);

	int k = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			unsigned char uB = B[r * PRE_MAX_WIDTH + c];
			unsigned char uG = G[r * PRE_MAX_WIDTH + c];
			unsigned char uR = R[r * PRE_MAX_WIDTH + c];
			// normalize
			float tmp_R = ((float) uR) * norm - shift_fact;
			float tmp_G = ((float) uG) * norm - shift_fact;
			float tmp_B = ((float) uB) * norm - shift_fact;
			//rescale
			float final_R = gain * tmp_R;
			float final_G = gain * tmp_G;
			float final_B = gain * tmp_B;
			//clip
			if (final_R > 127)
				final_R = 127;
			if (final_G > 127)
				final_G = 127;
			if (final_B > 127)
				final_B = 127;
			if (final_R < -128)
				final_R = -128;
			if (final_G < -128)
				final_G = -128;
			if (final_B < -128)
				final_B = -128;
			// final round
			//B is the LSB and R is the MSB, so first B then G and then R
			out_rgb_img[k + 0] = (int) final_B;
			out_rgb_img[k + 1] = (int) final_G;
			out_rgb_img[k + 2] = (int) final_R;
			k = k + 3;
		}
	}
}

/* **************************************************************************************************** */
/* just to check the reference output image and put it into 3 separate planes to write the BMP file later */

void format_ref_img(unsigned char *R, unsigned char *G, unsigned char *B,
		signed char *out_rgb_img, unsigned short int height, unsigned short int width)
{
	unsigned short int rows = height;
	unsigned short int cols = width;

	int k = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			unsigned char uB = (unsigned char) out_rgb_img[k + 0];
			unsigned char uG = (unsigned char) out_rgb_img[k + 1];
			unsigned char uR = (unsigned char) out_rgb_img[k + 2];
			B[r * PRE_MAX_WIDTH + c] = uB;
			G[r * PRE_MAX_WIDTH + c] = uG;
			R[r * PRE_MAX_WIDTH + c] = uR;
			k = k + 3;
		}
	}
}

#else
/* **************************************************************************************************** */
/* **************************************************************************************************** */


void ref_dpupreproc(uRGB_t *inp_img, iRGB_t *out_img,
		    float norm_fact, float shift_fact, float scale_fact,
		    unsigned short int height, unsigned short int width) {

  unsigned short int rows = height;
  unsigned short int cols = width;

  // copy the input image into the input array (RGB)
  // save reference output data in a text file
  FILE *fp1, *fp2, *fp3, *fp4, *fp5, *fp6;
  if ((fp1 = fopen("./data_pre/inp_ch1.txt", "w")) == NULL)
    printf("Error opening inp_ch1.txt");
  if ((fp2 = fopen("./data_pre/inp_ch2.txt", "w")) == NULL)
    printf("Error opening inp_ch2.txt");
  if ((fp3 = fopen("./data_pre/inp_ch3.txt", "w")) == NULL)
    printf("Error opening inp_ch3.txt");
  if ((fp4 = fopen("./data_pre/ref_ch1.txt", "w")) == NULL)
    printf("Error opening ref_ch1.txt");
  if ((fp5 = fopen("./data_pre/ref_ch2.txt", "w")) == NULL)
    printf("Error opening ref_ch2.txt");
  if ((fp6 = fopen("./data_pre/ref_ch3.txt", "w")) == NULL)
    printf("Error opening ref_ch3.txt");

  float norm = 1.0f / norm_fact;
  float gain = scale_fact;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      uRGB_t rgb_pix = inp_img[r * PRE_MAX_WIDTH + c];
      unsigned char B = rgb_pix.B;
      unsigned char G = rgb_pix.G;
      unsigned char R = rgb_pix.R;
      fprintf(fp1, "%18.14f  ", R);
      fprintf(fp2, "%18.14f  ", G);
      fprintf(fp3, "%18.14f  ", B);
      // normalize
      float tmp_R = ((float) R) * norm - shift_fact;
      float tmp_G = ((float) G) * norm - shift_fact;
      float tmp_B = ((float) B) * norm - shift_fact;
      //rescale
      float final_R = gain * tmp_R;
      float final_G = gain * tmp_G;
      float final_B = gain * tmp_B;
      //clip
      if (final_R >  127) final_R=127;
      if (final_G >  127) final_G=127;
      if (final_B >  127) final_B=127;
      if (final_R < -128) final_R=-128;
      if (final_G < -128) final_G=-128;
      if (final_B < -128) final_B=-128;
      //round
      iRGB_t i_rgb_pix;
      i_rgb_pix.R = (int) final_R;
      i_rgb_pix.G = (int) final_G;
      i_rgb_pix.B = (int) final_B;
		    
      fprintf(fp4, "%18.14f  ", (float) i_rgb_pix.R);
      fprintf(fp5, "%18.14f  ", (float) i_rgb_pix.G);
      fprintf(fp6, "%18.14f  ", (float) i_rgb_pix.B);
      //printf("G float=%18.14f int=%3d\n", final_G, i_rgb_pix.G);
      //G float=-28.36078262329102 int=-28
      //G float= -2.76078033447266 int= -2
      //G float=  7.78039550781250 int=  7
      //G float=  4.26667022705078 int=  4
      out_img[r * PRE_MAX_WIDTH + c] = i_rgb_pix;
    }
    fprintf(fp1, "\n");
    fprintf(fp2, "\n");
    fprintf(fp3, "\n");
    fprintf(fp4, "\n");
    fprintf(fp5, "\n");
    fprintf(fp6, "\n");
  }
  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  fclose(fp4);
  fclose(fp5);
  fclose(fp6);

}


/* **************************************************************************************************************** */

#endif
