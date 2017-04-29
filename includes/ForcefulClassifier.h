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
 * Forceful cascade classifier class.
 */
class ForcefulClassifier {
  public:
    // Forceful classifier constructors.
    ForcefulClassifier();
    ForcefulClassifier(std::vector<WeaklyClassifier*> weakly_classifiers, float *weights);
    ForcefulClassifier(std::vector<WeaklyClassifier*> weakly_classifiers, float *weights, float limit);
    // Get weakly classifiers set.
    std::vector<WeaklyClassifier*> getWeaklyClassifiers();
    // Scale weakly classifies by value.
    void scaleByValue(float value);
    // Scale limit by value.
    void scaleLimitByValue(float value);
    // Put new weakly classifier in set.
    void addClassifier(WeaklyClassifier* weakly_classifier, float weight);
    // Calculate classifier limit.
    void calculateLimit(std::vector<float*> &positive_samples, int size, float maximum_fnr);
    // Calculate classifier FPR.
    float calculateFpr(std::vector<float*> &negative_samples, int size);
    // Classify image by classifier.
    bool classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2);
    // Transform classifier to string representation.
    std::string toString();
  protected:
    // Classifier limit variable.
    float limit;
    // Classifier weights variable.
    std::vector<float> weights;
    // Weakly classifiers set.
    std::vector<WeaklyClassifier*> weakly_classifiers;
};
