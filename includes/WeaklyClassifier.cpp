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

#include <string.h>
#include <math.h>
#include <algorithm>
#include "SimpleImageHelpers.h"
#include "HaarFeature.h"
#include "WeaklyClassifier.h"

/**
 * Compare counters function.
 */
static bool feature_counters_compare(FeatureCounter &a, FeatureCounter &b) {
  return a.getValue() < b.getValue();
}

/**
 * WeaklyClassifier simple constructor.
 */
WeaklyClassifier::WeaklyClassifier(HaarFeature *feature) {
  this->feature = feature;
}

/**
 * WeaklyClassifier complex constructor.
 */
WeaklyClassifier::WeaklyClassifier(HaarFeature *feature, float limit, bool state) {
  this->feature = feature;
  this->limit = limit;
  this->state = state;
}

/**
 * Get classifier feature.
 */
HaarFeature* WeaklyClassifier::getFeature() {
  return this->feature;
}

/**
 * Scale classifier feature by value.
 */
void WeaklyClassifier::scaleByValue(float value) {
  // Scale classifier limit.
  this->limit *= pow(value, 2);
  // Scale classifier feature.
  this->feature->scaleByValue(value);
}

/**
 * Calculate classifier limit.
 */
float WeaklyClassifier::calculateLimit(float *values, int size1, int size2, float *weights) {
  int sizes_sum = size1 + size2;
  FeatureCounter *counters = new FeatureCounter[sizes_sum];
  WeightsSum *weights_sum = new WeightsSum[sizes_sum];

  // Set counter params.
  for (int i = 0; i < size1; i++) {
    counters[i].setParams(values[i], true, weights[i]);
  }
  for (int i = 0; i < size2; i++) {
    counters[i + size1].setParams(values[i + size1], false, weights[i + size1]);
  }
  // Sort feature counters.
  std::sort(counters, counters + sizes_sum, feature_counters_compare);

  // Set weights sum.
  float tp, tn;
  tp = tn = 0;
  if (!counters[0].getLabel()) {
    tn = counters[0].getWeight();
    weights_sum[0].setParams(0, tn);
  }
  else {
    tp = counters[0].getWeight();
    weights_sum[0].setParams(tp, 0);
  }
  for (int i = 1; i < sizes_sum; i++) {
    if (!counters[i].getLabel()) {
      tn += counters[i].getWeight();
      weights_sum[i].setParams(weights_sum[i - 1].getPositive(), weights_sum[i - 1].getNegative() + counters[i].getWeight());
    }
    else {
      tp += counters[i].getWeight();
      weights_sum[i].setParams(weights_sum[i - 1].getPositive() + counters[i].getWeight(), weights_sum[i - 1].getNegative());
    }
  }

  // Calculate classifier limit.
  float min_error = 1, error1, error2;
  for (int i = 0; i < sizes_sum; i++) {
    error1 = weights_sum[i].getPositive() + tn - weights_sum[i].getNegative();
    error2 = weights_sum[i].getNegative() + tp - weights_sum[i].getPositive();
    if (error1 < error2) {
      if (error1 < min_error) {
        min_error = error1;
        this->limit = counters[i].getValue();
        this->state = false;
      }
    }
    else {
      if (error2 < min_error) {
        min_error = error2;
        this->limit = counters[i].getValue();
        this->state = true;
      }
    }
  }

  // Remove counters and weights sum from memory.
  delete[] counters;
  delete[] weights_sum;
  return min_error;
}

/**
 * Classify image by classifier.
 */
int WeaklyClassifier::classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2) {
  // Calculate feature value.
  HaarFeature *classifier_feature = this->feature;
  float feature_value = classifier_feature->value(image, image_width, x, y);
  int feature_type = classifier_feature->type();

  // If the number of rectangles is odd.
  if (feature_type == 2 || feature_type == 3) {
    feature_value += classifier_feature->width() * classifier_feature->height() * temp1 / 3;
  }
  if (temp2 != 0) {
    feature_value = feature_value / temp2;
  }
  if (feature_value < this->limit) {
    return this->state ? 1 : -1;
  }
  else {
    return this->state ? -1 : 1;
  }
}

/**
 * Transform classifier to string representation.
 */
std::string WeaklyClassifier::toString() {
  std::string result = this->feature->toString() + " " + std::to_string(this->limit);
  return this->state ? result + " 1" : result + " 0";
}
