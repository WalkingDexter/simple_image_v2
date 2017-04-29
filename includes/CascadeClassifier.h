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

/**
 * Cascade classifier class.
 */
class CascadeClassifier {
  public:
    // Cascade classifier constructors.
    CascadeClassifier(int size);
    CascadeClassifier(std::vector<ForcefulClassifier*> forceful_classifiers, int size);
    // Get classifier size property.
    int getSize();
    // Scale forceful classifiers limit by value.
    void scaleByValue(float value);
    // Scale forceful classifiers limit by value.
    void scaleClassifiersLimitByValue(float value);
    // Put new forceful classifier in set.
    void addClassifier(ForcefulClassifier *forceful_classifier);
    // Calculate classifier FPR.
    float calculateFpr(std::vector<float*> &negative_samples);
    // Classify image by classifier.
    bool classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2);
    // Transform classifier to string representation.
    std::string toString();
    // Save classifier in text file.
    bool save(std::string path);
  protected:
    // Classifier basis variable.
    int size;
    // Forceful classifiers set.
    std::vector<ForcefulClassifier*> forceful_classifiers;
};
