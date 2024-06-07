/*******************************************************************************
* Copyright (C) 2019-2023 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// fruits360
// This file was @generated by ai8xize.py --test-dir synthed_net --prefix fruits360 --checkpoint-file trained/fruits360_trained-q.pth.tar --config-file networks/fruits360.yaml --sample-input ../ai8x-training/sample_fruits360.npy --device MAX78000 --board-name FTHR_RevA --compact-data --mexpress --timer 0 --display-checkpoint --verbose --fifo --overwrite

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// Data input: CHW 3x100x100 (30000 bytes total / 10000 bytes per channel):
static const uint32_t input_0[] = SAMPLE_INPUT_0;
static const uint32_t input_1[] = SAMPLE_INPUT_1;
static const uint32_t input_2[] = SAMPLE_INPUT_2;
void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  int i;
  const uint32_t *in0 = input_0;
  const uint32_t *in1 = input_1;
  const uint32_t *in2 = input_2;

  for (i = 0; i < 2500; i++) {
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while (((*((volatile uint32_t *) 0x50000004) & 2)) != 0); // Wait for FIFO 1
    *((volatile uint32_t *) 0x5000000c) = *in1++; // Write FIFO 1
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while (((*((volatile uint32_t *) 0x50000004) & 4)) != 0); // Wait for FIFO 2
    *((volatile uint32_t *) 0x50000010) = *in2++; // Write FIFO 2
  }
}

// Expected output of layer 7 for fruits360 given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

static int32_t ml_data32[(CNN_NUM_OUTPUTS + 3) / 4];

int main(void)
{
  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: APB (50 MHz) div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test fruits360 ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine
  cnn_start(); // Start CNN processing
  load_input(); // Load data input via FIFO

  while (cnn_time == 0)
    MXC_LP_EnterSleepMode(); // Wait for CNN

  if (check_output() != CNN_OK) fail();
  cnn_unload((uint32_t *) ml_data32);

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate data loading and inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral


  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 44,832,096 ops (44,196,800 macc; 626,080 comp; 9,216 add; 0 mul; 0 bitwise)
    Layer 0: 4,480,000 ops (4,320,000 macc; 160,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 5,960,000 ops (5,760,000 macc; 200,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 11,600,000 ops (11,520,000 macc; 80,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 5,860,000 ops (5,760,000 macc; 100,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 11,560,000 ops (11,520,000 macc; 40,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 5,354,496 ops (5,308,416 macc; 46,080 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 9,216 ops (0 macc; 0 comp; 9,216 add; 0 mul; 0 bitwise)
    Layer 7: 8,384 ops (8,384 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 80,240 bytes out of 442,368 bytes total (18.1%)
  Bias memory:   131 bytes out of 2,048 bytes total (6.4%)
*/
