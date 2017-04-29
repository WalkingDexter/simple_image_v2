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
#include <fstream>
#include "HaarFeature.h"
#include "WeaklyClassifier.h"
#include "ForcefulClassifier.h"
#include "CascadeClassifier.h"

/**
 * CascadeClassifier simple constructor.
 */
CascadeClassifier::CascadeClassifier(int size) {
  this->size = size;
}

/**
 * CascadeClassifier complex constructor.
 */
CascadeClassifier::CascadeClassifier(std::vector<ForcefulClassifier*> forceful_classifiers, int size) {
  this->forceful_classifiers = forceful_classifiers;
  this->size = size;
}

/**
 * Get classifier size property.
 */
int CascadeClassifier::getSize() {
  return this->size;
}

/**
 * Scale each forceful classifier and size in set by value.
 */
void CascadeClassifier::scaleByValue(float value) {
  this->size *= value;
  std::vector<ForcefulClassifier*>::iterator iterator;
  for (iterator = this->forceful_classifiers.begin(); iterator != this->forceful_classifiers.end(); iterator++) {
    (*iterator)->scaleByValue(value);
  }
}

/**
 * Scale forceful classifiers limit by value.
 */
void CascadeClassifier::scaleClassifiersLimitByValue(float value) {
  std::vector<ForcefulClassifier*>::iterator iterator;
  for (iterator = this->forceful_classifiers.begin(); iterator != this->forceful_classifiers.end(); iterator++) {
    (*iterator)->scaleLimitByValue(value);
  }
}

/**
 * Put new forceful classifier in set.
 */
void CascadeClassifier::addClassifier(ForcefulClassifier *forceful_classifier) {
  this->forceful_classifiers.push_back(forceful_classifier);
}

/**
 * Calculate classifier FPR.
 */
float CascadeClassifier::calculateFpr(std::vector<float*> &negative_samples) {
  unsigned int negative_size = negative_samples.size();
  int classify_image_count = 0;
  for (unsigned int i = 0; i < negative_size; i++) {
    if (this->classifyImage(negative_samples[i], this->size, 0, 0, 0, 1)) {
      classify_image_count++;
    }
  }
  return float(classify_image_count) / float(negative_size);
}

/**
 * Classify image by classifier.
 */
bool CascadeClassifier::classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2) {
  std::vector<ForcefulClassifier*>::iterator iterator;
  for (iterator = this->forceful_classifiers.begin(); iterator != this->forceful_classifiers.end(); iterator++) {
    if (!(*iterator)->classifyImage(image, image_width, x, y, temp1, temp2)) {
      return false;
    }
  }
  return true;
}

/**
 * Transform classifier to string representation.
 */
std::string CascadeClassifier::toString() {
  std::vector<ForcefulClassifier*>::iterator iterator;
  std::string result = std::to_string(this->size) + " " + std::to_string(this->forceful_classifiers.size()) + "\n";
  for (iterator = this->forceful_classifiers.begin(); iterator != this->forceful_classifiers.end(); iterator++) {
    result += (*iterator)->toString();
  }
  return result;
}

/**
 * Save classifier in text file.
 */
bool CascadeClassifier::save(std::string path) {
  try {
    std::ofstream file(path);
    file << this->toString();
    file.close();
    return true;
  }
  catch (std::exception e) {
    return false;
  }
}
