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
 * Class for Haar feature.
 */
class HaarFeature {
  public:
    // HaarFeature constructor method.
    HaarFeature(int feature_type, int x, int y, int w, int h);
    // Get feature type.
    int type();
    // Get feature width.
    int width();
    // Get feature height.
    int height();
    // Scale feature by value.
    void scaleByValue(float value);
    // Calculate feature value.
    float value(float *image, int image_width, int x1, int y1);
    // Transform feature to string representation.
    std::string toString();
  protected:
    // Feature type - exist 1-3 features.
    int feature_type;
    // Haar feature sizes.
    int w, h;
    // Top-left corner coordinates of Haar feature.
    int x, y;
    // Helper function for get feature value.
    float valueHelper(float *image, int image_width, int x1, int y1, int x2, int y2, int iw, int ih);
};
