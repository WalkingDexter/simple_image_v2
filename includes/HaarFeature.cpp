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

#include <phpcpp.h>
#include <string.h>
#include "HaarFeature.h"

/**
 * HaarFeature constructor method.
 */
HaarFeature::HaarFeature(int feature_type, int x, int y, int w, int h) {
  // Define feature values.
  this->feature_type = feature_type;
  this->x = x;
  this->y = y;
  this->w = w;
  this->h = h;
}

/**
 * Get feature type.
 */
int HaarFeature::type() {
  return this->feature_type;
}

/**
 * Get feature width.
 */
int HaarFeature::width() {
  return this->w;
}

/**
 * Get feature height.
 */
int HaarFeature::height() {
  return this->h;
}

/**
 * Scale feature by value.
 */
void HaarFeature::scaleByValue(float value) {
  this->x = (this->x)*value;
  this->y = (this->y)*value;
  this->w = (this->w)*value;
  this->h = (this->h)*value;
}

/**
 * Helper function for get feature value.
 */
float HaarFeature::valueHelper(float *image, int image_width, int x1, int y1, int x2, int y2, int iw, int ih) {
  int index;
  index = (y1 + y2 + ih - 1) * image_width + (x1 + x2 + iw - 1);
  float value = image[index];
  if ((x1 + x2) > 0) {
    index = (y1 + y2 + ih - 1) * image_width + (x1 + x2 - 1);
    value -= image[index];
  }
  if ((y1 + y2) > 0) {
    index = (y1 + y2 - 1) * image_width + (x1 + x2 + iw - 1);
    value -= image[index];
  }
  if ((x1 + x2) > 0 && (y1 + y2) > 0) {
    index = (y1 + y2 - 1) * image_width + (x1 + x2 - 1);
    value += image[index];
  }
  return value;
}

/**
 * Calculate feature value.
 */
float HaarFeature::value(float *image, int image_width, int x1, int y1) {
  float a, b, c;
  switch(this->feature_type) {
    case 0:
      a = this->valueHelper(image, image_width, x1, y1, this->x + (this->w / 2), this->y, this->w / 2, this->h);
      b = this->valueHelper(image, image_width, x1, y1, this->x, this->y, this->w/2, this->h);
      return a - b;
    case 1:
      a = this->valueHelper(image, image_width, x1, y1, this->x, this->y, this->w, this->h/2);
      b = this->valueHelper(image, image_width, x1, y1, this->x, this->y + (this->h / 2), this->w, this->h / 2);
      return a - b;
    case 2:
      a = this->valueHelper(image, image_width, x1, y1, this->x + (this->w / 3), this->y, this->w / 3, this->h);
      b = this->valueHelper(image, image_width, x1, y1, this->x, this->y, this->w / 3, this->h);
      c = this->valueHelper(image, image_width, x1, y1, this->x + (this->w * 2 / 3), this->y, this->w / 3, this->h);
      return a - b - c;
    case 3:
      a = this->valueHelper(image, image_width, x1, y1, this->x, this->y + (this->h / 3), this->w, this->h / 3);
      b = this->valueHelper(image, image_width, x1, y1, this->x, this->y, this->w, this->h / 3);
      c = this->valueHelper(image, image_width, x1, y1, this->x, this->y + (this->h * 2 / 3), this->w, this->h / 3);
      return a - b -c;
    default:
      throw Php::Exception("Simple Image: Feature type does not exist");
  }
}

/**
 * Transform feature to string representation.
 */
std::string HaarFeature::toString() {
  return std::to_string(this->feature_type) + " " + std::to_string(this->w) + " " + std::to_string(this->h) + " " + std::to_string(this->x) + " " + std::to_string(this->y);
}
