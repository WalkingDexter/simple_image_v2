/*
Copyright © 2017 Andrey Tymchuk.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <Magick++.h>
#include <string.h>

// Object detection structure.
struct detection_structure {
  unsigned int x;
  unsigned int y;
  unsigned int size;
};

/**
 * Helper feature counter class.
 */
class FeatureCounter {
  public:
    // Set counter params.
    void setParams(float value, bool label, float weight);
    // Get counter value.
    float getValue();
    // Get counter label.
    float getLabel();
    // Get counter weight.
    float getWeight();
  protected:
    float value;
    bool label;
    float weight;
};

/**
 * Helper weights sum class.
 */
class WeightsSum {
  public:
    // Set weights sum params.
    void setParams(float positive, float negative);
    // Get positive sum.
    float getPositive();
    // Get negative sum.
    float getNegative();
  protected:
    // The positive weights sum.
    float positive;
    // The negative weights sum.
    float negative;
};

/**
 * Helper class for manage object detections on image.
 */
class DetectionManager {
  public:
    // Added detection to set.
    void addDetection(unsigned int x, unsigned int y, unsigned int size);
    // Load detections count.
    int count();
    // Show detections on image.
    void showDetections(Magick::Image image, std::string image_file_name);
  protected:
    // Object detections set.
    std::vector<detection_structure> detections;
};

// Check, that file is exist.
bool file_is_exist(std::string file_path);
// Give real random int from range.
int random_int(int min, int max);
// Give pixels shade in string.
std::string image_pixels_shade_to_string(Magick::Image image);
// Crop image by min size.
Magick::Image image_crop_by_min_size(Magick::Image image);
// Calculate false positive rate per step.
float* calculate_target_fpr(int cascade_steps, float common_fpr);
// Read sample string from file.
float* read_sample_from_string(std::string sample_string, int w, int h);
// Get normalize sample.
float* normalize_sample(float *sample, int w, int h);
// Mirroring samples by vertical.
void mirroring_samples(std::vector<float*> &samples, int w, int h);
// Compute integral image to sample.
float* compute_integral_image(float *sample, int w, int h, bool squared);
// Rotate sample to 90 degrees.
float* sample_rotate_90(float *sample, int w, int h);
// Calculate integral rectangle value.
float calculate_integral_rectangle(float *integral_image, int integral_width, int x, int y, int w, int h);
