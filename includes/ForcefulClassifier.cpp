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
#include <vector>
#include <algorithm>
#include "HaarFeature.h"
#include "WeaklyClassifier.h"
#include "ForcefulClassifier.h"

/**
 * ForcefulClassifier first constructor.
 */
ForcefulClassifier::ForcefulClassifier() {
  this->limit = 0;
}

/**
 * ForcefulClassifier second constructor.
 */
ForcefulClassifier::ForcefulClassifier(std::vector<WeaklyClassifier*> weakly_classifiers, float *weights) {
  this->weakly_classifiers = weakly_classifiers;
  for (unsigned int i = 0; i < weakly_classifiers.size(); i++) {
    this->weights.push_back(weights[i]);
  }
  this->limit = 0;
}

/**
 * ForcefulClassifier third constructor.
 */
ForcefulClassifier::ForcefulClassifier(std::vector<WeaklyClassifier*> weakly_classifiers, float *weights, float limit) {
  this->weakly_classifiers = weakly_classifiers;
  for (unsigned int i = 0; i < weakly_classifiers.size(); i++) {
    this->weights.push_back(weights[i]);
  }
  this->limit = limit;
}

/**
 * Get weakly classifiers set.
 */
std::vector<WeaklyClassifier*> ForcefulClassifier::getWeaklyClassifiers() {
  return this->weakly_classifiers;
}

/**
 * Scale each weakly classifier in set by value.
 */
void ForcefulClassifier::scaleByValue(float value) {
  std::vector<WeaklyClassifier*>::iterator iterator;
  for (iterator = this->weakly_classifiers.begin(); iterator != this->weakly_classifiers.end(); iterator++) {
    (*iterator)->scaleByValue(value);
  }
}

/**
 * Put new weakly classifier in set.
 */
void ForcefulClassifier::addClassifier(WeaklyClassifier* weakly_classifier, float weight) {
  this->weakly_classifiers.push_back(weakly_classifier);
  this->weights.push_back(weight);
}

/**
 * Calculate classifier limit.
 */
void ForcefulClassifier::calculateLimit(std::vector<float*> &positive_samples, int size, float maximum_fnr) {
  unsigned int positive_size = positive_samples.size(), temp = maximum_fnr * positive_size;
  int weights_index;
  float temp2, *counters = new float[positive_size];
  std::vector<WeaklyClassifier*>::iterator iterator;

  for (unsigned int i = 0; i < positive_size; i++) {
    counters[i] = 0;
    weights_index = 0;
    for (iterator = this->weakly_classifiers.begin(); iterator != this->weakly_classifiers.end(); iterator++) {
      counters[i] += this->weights[weights_index] * (*iterator)->classifyImage(positive_samples[i], size, 0, 0, 0, 1);
      weights_index++;
    }
  }
  std::sort(counters, counters + positive_size);
  if (temp >= 0 && temp < positive_size) {
    temp2 = counters[temp];
    while (temp > 0 && counters[temp] == temp2) {
      temp--;
    }
    this->limit = counters[temp];
  }
  delete[] counters;
}

/**
 * Calculate classifier FPR.
 */
float ForcefulClassifier::calculateFpr(std::vector<float*> &negative_samples, int size) {
  unsigned int negative_size = negative_samples.size();
  int classify_image_count = 0;
  for (unsigned int i = 0; i < negative_size; i++) {
    if (this->classifyImage(negative_samples[i], size, 0, 0, 0, 1)) {
      classify_image_count++;
    }
  }
  return float(classify_image_count) / float(negative_samples.size());
}

/**
 * Scale only limit by value.
 */
void ForcefulClassifier::scaleLimitByValue(float value) {
  this->limit *= value;
}

/**
 * Classify image by classifier.
 */
bool ForcefulClassifier::classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2) {
  float counter = 0;
  int weights_index = 0;
  std::vector<WeaklyClassifier*>::iterator iterator;
  for (iterator = this->weakly_classifiers.begin(); iterator != this->weakly_classifiers.end(); iterator++) {
    counter += this->weights[weights_index++] * (*iterator)->classifyImage(image, image_width, x, y, temp1, temp2);
  }
  return counter >= this->limit;
}

/**
 * Transform classifier to string representation.
 */
std::string ForcefulClassifier::toString() {
  std::vector<WeaklyClassifier*>::iterator iterator;
  std::string result = std::to_string(this->weakly_classifiers.size()) + " " + std::to_string(this->limit) + "\n";
  int weights_index = 0;
  for (iterator = this->weakly_classifiers.begin(); iterator != this->weakly_classifiers.end(); iterator++) {
    result += std::to_string(this->weights[weights_index++]) + " " + (*iterator)->toString() + "\n";
  }
  return result;
}
