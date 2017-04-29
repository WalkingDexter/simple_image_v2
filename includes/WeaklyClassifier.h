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
 * Class for weakly classifier.
 */
class WeaklyClassifier {
  public:
    // Weakly classifier constructors.
    WeaklyClassifier(HaarFeature *feature);
    WeaklyClassifier(HaarFeature *feature, float limit, bool state);
    // Get classifier feature.
    HaarFeature* getFeature();
    // Scale classifier feature by value.
    void scaleByValue(float value);
    // Calculate classifier limit.
    float calculateLimit(float *values, int size1, int size2, float *weights);
    // Classify image by classifier.
    int classifyImage(float *image, int image_width, int x, int y, float temp1, float temp2);
    // Transform classifier to string representation.
    std::string toString();
  protected:
    // Classifier state variable.
    bool state;
    // Classifier limit variable.
    float limit;
    // Classifier Haar feature.
    HaarFeature *feature;
};
