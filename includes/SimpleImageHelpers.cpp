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

#include <fstream>
#include <random>
#include <sstream>
#include "SimpleImageHelpers.h"

/**
 * Set counter params.
 */
void FeatureCounter::setParams(float value, bool label, float weight) {
  this->value = value;
  this->label = label;
  this->weight = weight;
}

/**
 * Get counter value.
 */
float FeatureCounter::getValue() {
  return this->value;
}

/**
 * Get counter label.
 */
float FeatureCounter::getLabel() {
  return this->label;
}

/**
 * Get counter weight.
 */
float FeatureCounter::getWeight() {
  return this->weight;
}

/**
 * Set weights sum params.
 */
void WeightsSum::setParams(float positive, float negative) {
  this->positive = positive;
  this->negative = negative;
}

/**
 * Get positive sum.
 */
float WeightsSum::getPositive() {
  return this->positive;
}

/**
 * Get negative sum.
 */
float WeightsSum::getNegative() {
  return this->negative;
}

/**
 * Added detection to set.
 */
void DetectionManager::addDetection(unsigned int x, unsigned int y, unsigned int size) {
  detection_structure detection;
  detection.x = x;
  detection.y = y;
  detection.size = size;
  this->detections.push_back(detection);
}

/**
 * Load detections count.
 */
int DetectionManager::count() {
  int count = this->detections.size();
  return count;
}

/**
 * Show detections on image.
 */
void DetectionManager::showDetections(Magick::Image image, std::string image_file_name) {
  if (this->count() > 0) {
    std::vector<detection_structure>::iterator iterator;
    unsigned int stroke_width = image.rows();
    if (image.columns() > stroke_width) {
      stroke_width = image.columns();
    }
    stroke_width /= 100;
    // Construct drawing list.
    std::list<Magick::Drawable> drawList;
    drawList.push_back(Magick::DrawableStrokeColor("red"));
    drawList.push_back(Magick::DrawableStrokeWidth(stroke_width));
    drawList.push_back(Magick::DrawableFillColor("none"));

    for (iterator = this->detections.begin(); iterator != this->detections.end(); iterator++) {
      // Add a Rectangle to drawing list.
      drawList.push_back(Magick::DrawableRectangle((*iterator).x, (*iterator).y, (*iterator).x + (*iterator).size, (*iterator).y + (*iterator).size));
    }
    // Draw everything using completed drawing list.
    image.draw(drawList);
    image.write(image_file_name);
  }
}

/**
 * Check, that file is exist.
 */
bool file_is_exist(std::string file_path) {
  std::ifstream file(file_path);
  bool is_exist = file.is_open();
  file.close();
  return is_exist;
}

/**
 * Give real random int from range.
 */
int random_int(int min, int max) {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dis{min, max};
  return dis(gen);
}

/**
 * Give pixels shade in string.
 */
std::string image_pixels_shade_to_string(Magick::Image image) {
  std::string result = "";
  unsigned int rows = image.rows();
  unsigned int columns = image.columns();
  for (unsigned int x = 0; x < rows; x++) {
    for (unsigned int y = 0; y < columns; y++) {
      Magick::ColorGray color = image.pixelColor(x, y);
      result += std::to_string(color.shade());
      if (x != (rows - 1) && y != (columns - 1)) {
        result += " ";
      }
    }
  }
  return result;
}

/**
 * Crop image by min size.
 */
Magick::Image image_crop_by_min_size(Magick::Image image) {
  unsigned int width = image.size().width();
  unsigned int height = image.size().height();
  if (width < height) {
    image.crop(Magick::Geometry(width, width));
  }
  else {
    image.crop(Magick::Geometry(height, height));
  }
  return image;
}

/**
 *  Calculate false positive rate per step.
 */
float* calculate_target_fpr(int cascade_steps, float common_fpr) {
  float *result = new float[cascade_steps];
  if (cascade_steps == 1) {
    result[0] = common_fpr;
  }
  else {
    result[0] = 0.5;
    if (cascade_steps == 2) {
      result[1] = common_fpr / result[0];
    }
    else {
      result[1] = 0.25;
      float ifpr = result[0] * result[1];
      float fpr = pow((common_fpr / ifpr), 1 / float(cascade_steps - 2));
      for (int i = 2; i < cascade_steps; i++) {
        result[i] = fpr;
      }
    }
  }
  return result;
}

/**
 * Read sample string from file.
 */
float* read_sample_from_string(std::string sample_string, int w, int h) {
  float *sample = new float[w * h];
  int x = 0, y = 0;
  float value;
  std::istringstream stream(sample_string);

  while (stream >> value) {
    if (y == h) {
      return NULL;
    }
    sample[(y * w) + (x++)] = value;
    if (x == w) {
      x = 0;
      y++;
    }
  }
  if (x != 0 || y != h) {
    return NULL;
  }
  return sample;
}

/**
 * Get normalize sample.
 */
float* normalize_sample(float *sample, int w, int h) {
  float m, s;
  m = s = 0;
  int WxH = w * h;

  for (int i = 0; i < WxH; i++) {
    m += sample[i];
  }
  m = m / float(WxH);
  for (int i = 0; i < WxH; i++) {
    s += pow(sample[i] - m, 2);
  }
  s = s / (WxH);
  s = sqrt(s);
  if (s == 0) {
    s = 1;
  }
  for (int i = 0; i < WxH; i++) {
    sample[i] = (sample[i] - m) / s;
  }

  return sample;
}

/**
 * Mirroring samples by vertical.
 */
void mirroring_samples(std::vector<float*> &samples, int w, int h) {
  // TODO: Do it faster.
  float *mirror_sample;
  int WxH = w * h;
  unsigned int size = samples.size();
  for (unsigned int i = 0; i < size; i++) {
    mirror_sample = new float[WxH];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        mirror_sample[(y * w) + w - 1 - x] = samples[i][(y * w) + x];
      }
    }
    samples.push_back(mirror_sample);
  }
}

/**
 * Compute integral image to sample.
 */
float* compute_integral_image(float *sample, int w, int h, bool squared) {
  float* integral_image = new float[w*h];
  float* safety_val = new float[w*h];

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (x == 0) {
        safety_val[(y * w) + x] = squared ? pow(sample[(y * w) + x], 2) : sample[(y * w) + x];
      }
      else {
        safety_val[(y * w) + x] = squared ? safety_val[(y * w) + x - 1] + pow(sample[(y * w) + x], 2) : safety_val[(y * w) + x - 1] + sample[(y * w) + x];
      }
      if (y == 0) {
        integral_image[(y * w) + x] = safety_val[(y * w) + x];
      }
      else {
        integral_image[(y * w) + x] = integral_image[((y - 1) * w) + x] + safety_val[(y * w) + x];
      }
    }
  }
  return integral_image;
}

/**
 * Rotate sample to 90 degrees.
 */
float* sample_rotate_90(float *sample, int w, int h) {
  float *result = new float[w*h];
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      result[h - 1 - y + (x * h)] = sample[(y * w) + x];
    }
  }
  return result;
}

/**
 * Calculate integral rectangle value.
 */
float calculate_integral_rectangle(float *integral_image, int integral_width, int x, int y, int w, int h) {
  float result = integral_image[(y + h - 1) * integral_width + x + w - 1];
  if (x > 0) {
    result -= integral_image[(y + h - 1) * integral_width + x - 1];
  }
  if (y > 0) {
    result -= integral_image[(y - 1) * integral_width + x + w - 1];
  }
  if (x > 0 && y > 0) {
    result += integral_image[(y - 1) * integral_width + x - 1];
  }
  return result;
}
