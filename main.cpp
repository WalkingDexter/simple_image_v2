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

#include <phpcpp.h>      // Library PHP-CPP.
#include <Magick++.h>    // Library for work with images.
#include <string.h>      // Library for work with strings.
#include <vector>        // Library for work with vectors.
#include <fstream>       // Library for work with file streams.
#include <sstream>       // Library for work with string streams.
#include <stdlib.h>      // Standart C++ library.
#include <iostream>

#include "includes/SimpleImageHelpers.h"    // Simple Image helpers.
#include "includes/HaarFeature.h"             // HaarFeature class definition.
#include "includes/WeaklyClassifier.h"        // WeaklyClassifierr class definition.
#include "includes/ForcefulClassifier.h"      // ForcefulClassifier class definition.
#include "includes/CascadeClassifier.h"       // CascadeClassifier class definition.

using namespace std;     // C++ standard namespace.
using namespace Magick;  // Magick namespace.

// Helpful variable used with readdir function.
unsigned char is_file = 0x8;
// Define samples min/max sizes.
const int sample_min_size = 21;
const int sample_max_size = 500;

/**
 * Load cascade classifier from text file.
 */
CascadeClassifier* load_cascade_classifier_from_file(string file_name) {
  if (!file_is_exist(file_name)) {
    throw Php::Exception("Simple Image: Classifier file not exist");
  }

  ifstream file(file_name);
  HaarFeature *feature;
  vector<WeaklyClassifier*> *weakly_classifiers;
  vector<ForcefulClassifier*> forceful_classifiers;
  WeaklyClassifier *weakly_classifier;
  ForcefulClassifier *forceful_classifier;
  unsigned int forceful_count, weakly_count;
  int size, feature_type, x, y, w, h, state, index;
  float forceful_limit, weakly_limit, *weights;

  file >> size >> forceful_count;
  while (forceful_classifiers.size() < forceful_count) {
    file >> weakly_count >> forceful_limit;
    weights = new float[weakly_count];
    weakly_classifiers = new vector<WeaklyClassifier*>();
    index = 0;
    while (weakly_classifiers->size() < weakly_count) {
      file >> weights[index++] >> feature_type >> w >> h >> x >> y >> weakly_limit >> state;
      feature = new HaarFeature(feature_type, x, y, w, h);
      weakly_classifier = new WeaklyClassifier(feature, weakly_limit, (bool) state);
      weakly_classifiers->push_back(weakly_classifier);
    }
    forceful_classifier = new ForcefulClassifier(*weakly_classifiers, weights, forceful_limit);
    forceful_classifiers.push_back(forceful_classifier);
  }

  if (forceful_classifiers.empty() || size < sample_min_size || size > sample_max_size) {
    throw Php::Exception("Simple Image: Wrong classifier format");
  }
  return new CascadeClassifier(forceful_classifiers, size);
}

/**
 * AdaBoost algorithm function.
 */
ForcefulClassifier* ada_boost(CascadeClassifier *cascade_classifier, vector<HaarFeature*> &haar_features, vector<float*> &positive_samples, vector<float*> &negative_samples, float fpr, float fnr, int size) {
  WeaklyClassifier *prime_weakly_classifier, *weakly_classifier;
  ForcefulClassifier *forceful_classifier = new ForcefulClassifier();
  unsigned int positive_size = positive_samples.size(), negative_size = negative_samples.size(), sizes_sum = positive_size + negative_size;
  float weights_sum, minimal_error, weakly_classifier_error, classifier_fpr = 1.0, temp;
  float *weights = new float[sizes_sum];
  float *feature_values = new float[sizes_sum];

  for (unsigned int i = 0; i < positive_size; i++) {
    weights[i] = 1 / float(2 * positive_size);
  }
  for (unsigned int i = 0; i < negative_size; i++) {
    weights[positive_size + i] = 1 / float(2 * negative_size);
  }

  while (classifier_fpr > fpr) {
    // Stop adding new weakly classifiers after all negative samples are correctly classified.
    if (forceful_classifier->calculateFpr(negative_samples, size) == 0) {
      break;
    }

    // Normalize weights for current i.
    weights_sum = 0;
    for (unsigned int i = 0; i < sizes_sum; i++) {
      weights_sum += weights[i];
    }
    for (unsigned int i = 0; i < sizes_sum; i++) {
      weights[i] = weights[i] / weights_sum;
    }

    // Select prime weakly classifier.
    minimal_error = 1;
    prime_weakly_classifier = NULL;
    for (vector<HaarFeature*>::iterator feature_iterator = haar_features.begin(); feature_iterator != haar_features.end(); feature_iterator++) {
      weakly_classifier = new WeaklyClassifier(*feature_iterator);
      for (unsigned int i = 0; i < positive_size; i++) {
        feature_values[i] = (*feature_iterator)->value(positive_samples[i], size, 0, 0);
      }
      for (unsigned int i = 0; i < negative_size; i++) {
        feature_values[positive_size + i] = (*feature_iterator)->value(negative_samples[i], size, 0, 0);
      }
      weakly_classifier_error = weakly_classifier->calculateLimit(feature_values, positive_size, negative_size, weights);
      if (weakly_classifier_error < minimal_error) {
        delete prime_weakly_classifier;
        prime_weakly_classifier = weakly_classifier;
        minimal_error = weakly_classifier_error;
      }
      else {
        delete weakly_classifier;
      }
    }

    // Update weights array.
    temp = minimal_error / (1 - minimal_error);
    for (unsigned int i = 0; i < positive_size; i++) {
      if (prime_weakly_classifier->classifyImage(positive_samples[i], size, 0, 0, 0, 1) == 1) {
        weights[i] = weights[i] * temp;
      }
    }
    for (unsigned int i = 0; i < negative_size; i++) {
      if (prime_weakly_classifier->classifyImage(negative_samples[i], size, 0, 0, 0, 1) == -1) {
        weights[positive_size + i] = weights[positive_size + i] * temp;
      }
    }

    // Update current FPR.
    forceful_classifier->addClassifier(prime_weakly_classifier, log(1 / temp));
    forceful_classifier->calculateLimit(positive_samples, size, fnr);
    classifier_fpr = forceful_classifier->calculateFpr(negative_samples, size);
  }
  delete[] weights;
  delete[] feature_values;

  return forceful_classifier;
}

/**
 * Create Haar features set by samples sizes.
 */
vector<HaarFeature*> create_haar_features(int w, int h) {
  vector<HaarFeature*> result;
  HaarFeature *feature;
  int x, y, feature_min_w, feature_w, feature_h;

  // We have 4 features types.
  for (int feature_type = 0; feature_type < 4; feature_type++) {
    x = y = 0;
    if (feature_type != 2) {
      feature_min_w = feature_w = 4;
    }
    else {
      feature_min_w = feature_w = 3;
    }
    if (feature_type != 3) {
      feature_h = 4;
    }
    else {
      feature_h = 3;
    }
    while (feature_h <= h) {
      while (feature_w <= w) {
        while (y + feature_h <= h) {
          while (x + feature_w <= w) {
            feature = new HaarFeature(feature_type, x, y, feature_w, feature_h);
            result.push_back(feature);
            x++;
          }
          x = 0;
          y++;
        }
        y = 0;
        switch (feature_type) {
          case 0:
            feature_w += 2;
            break;
          case 2:
            feature_w += 3;
            break;
          default:
            feature_w++;
        }
      }
      feature_w = feature_min_w;
      switch (feature_type) {
        case 1:
          feature_h += 2;
          break;
        case 3:
          feature_h += 3;
          break;
        default:
          feature_h++;
      }
    }
  }
  return result;
}

/**
 * Train cascade by positive and negative samples.
 */
void simple_image_train_cascade(Php::Parameters &params) {
  // Try get train params from function params.
  // Model file path.
  string model_file_name = params[0];
  // Positive file path.
  string positive_file_name = params[1];
  if (!file_is_exist(positive_file_name)) {
    throw Php::Exception("Simple Image: Positive samples file not exist");
  }
  // Negative file path.
  string negative_file_name = params[2];
  if (!file_is_exist(negative_file_name)) {
    throw Php::Exception("Simple Image: Negative samples file not exist");
  }
  // Train samples size.
  int size = sample_min_size;
  if (params.size() > 3) {
    size = params[3];
  }
  if (size < sample_min_size || size > sample_max_size) {
    throw Php::Exception("Simple Image: Samples size must be >= " + std::to_string(sample_min_size) + " and <= " + std::to_string(sample_max_size));
  }
  // Cascade steps count.
  int cascade_steps = 10;
  if (params.size() > 4) {
    cascade_steps = params[4];
  }
  if (cascade_steps <= 0) {
    throw Php::Exception("Simple Image: Cascade steps count must be greater than zero");
  }
  // Use negative samples rotation.
  bool rotation = false;
  if (params.size() > 5) {
    rotation = params[5];
  }
  // Use positive samples mirroring.
  bool mirroring = false;
  if (params.size() > 6) {
    mirroring = params[6];
  }
  // Negative samples per training step.
  // Zero means all.
  int temp_int = 0;
  if (params.size() > 7) {
    temp_int = params[7];
  }
  if (temp_int < 0) {
    throw Php::Exception("Simple Image: Negative samples per step count must be greater than or equal to zero");
  }
  unsigned int negative_samples_per_step = temp_int;

  // Initialize train variables.
  // The maximum FNR.
  float maximum_fnr = 0.01;
  // Common target FPR.
  float common_fpr = 0.000001;
  // FPR for current classifier.
  float *current_fpr;
  current_fpr = calculate_target_fpr(cascade_steps, common_fpr);
  // Normalize sample flag.
  bool normalize = true;

  // Loading positive samples file.
  ifstream positive_file(positive_file_name);
  // Read positive samples in vector.
  string sample_line = "";
  float *sample, *sample_2;
  std::vector<float*> positive_samples, negative_samples;
  while (getline(positive_file, sample_line)) {
    if ((sample = read_sample_from_string(sample_line, size, size)) != NULL) {
      // Reading sample success.
      // Normalize sample, if needed.
      if (normalize) {
        sample = normalize_sample(sample, size, size);
      }
      // Save sample in positive_samples variable.
      positive_samples.push_back(sample);
    }
  }
  positive_file.close();
  if (positive_samples.empty()) {
    throw Php::Exception("Simple Image: Empty positive samples set");
  }

  // Create mirrors for positive samples, if needed.
  if (mirroring) {
    mirroring_samples(positive_samples, size, size);
  }
  // Set negative samples per step count to all (positive samples count),
  // if it's not specified.
  if (negative_samples_per_step == 0) {
    negative_samples_per_step = positive_samples.size();
  }
  // Load negative samples file.
  ifstream negative_file(negative_file_name);
  sample_line = "";

  // Compute integral images for positive samples.
  int positive_samples_count = positive_samples.size();
  for (int i = 0; i < positive_samples_count; i++) {
    positive_samples.push_back(compute_integral_image(positive_samples[i], size, size, false));
    positive_samples.erase(positive_samples.begin());
  }

  // Create features by sample sizes.
  vector<HaarFeature*> haar_features = create_haar_features(size, size);
  ForcefulClassifier *forceful_classifier;
  float maximum_fpr = 1.0;

  // Building cascade classifier.
  CascadeClassifier *cascade_classifier = new CascadeClassifier(size);
  for (int k = 0; k < cascade_steps; k++) {
    // Read negative sample from negative samples file.
    if (negative_samples.size() < negative_samples_per_step) {
      while (getline(negative_file, sample_line)) {
        if ((sample = read_sample_from_string(sample_line, size, size)) != NULL) {
          if (normalize) {
            sample = normalize_sample(sample, size, size);
          }

          if (rotation) {
            for (int rotation_index = 0; rotation_index < 4; rotation_index++) {
              sample_2 = compute_integral_image(sample, size, size, false);
              if (cascade_classifier->classifyImage(sample_2, size, 0, 0, 0, 1)) {
                negative_samples.push_back(sample_2);
                if (negative_samples.size() == negative_samples_per_step) {
                  break;
                }
              }
              if (rotation_index < 3) {
                sample = sample_rotate_90(sample, size, size);
              }
            }
            if (negative_samples.size() == negative_samples_per_step) {
              break;
            }
          }
          else {
            sample = compute_integral_image(sample, size, size, false);
            if (cascade_classifier->classifyImage(sample, size, 0, 0, 0, 1)) {
              negative_samples.push_back(sample);
              if (negative_samples.size() == negative_samples_per_step) {
                break;
              }
            }
          }
        }
      }
    }

    if (negative_samples.size() > 0) {
      // Run AdaBoost algorithm to select best Haar features.
      maximum_fpr = current_fpr[k];
      forceful_classifier = ada_boost(cascade_classifier, haar_features, positive_samples, negative_samples, maximum_fpr, maximum_fnr, size);
      cascade_classifier->addClassifier(forceful_classifier);
      // Remove false detections from training.
      for (unsigned int i = 0; i < negative_samples.size(); i++) {
        if (!forceful_classifier->classifyImage(negative_samples[i], size, 0, 0, 0, 1)) {
          negative_samples.erase(negative_samples.begin() + i);
          i--;
        }
      }
      for (unsigned int i = 0; i < positive_samples.size(); i++) {
        if (!forceful_classifier->classifyImage(positive_samples[i], size, 0, 0, 0, 1)) {
          positive_samples.erase(positive_samples.begin() + i);
          i--;
        }
      }
    }
    else {
      break;
    }
  }
  negative_file.close();
  cascade_classifier->save(model_file_name);
}

/**
 * Classify image by cascade classifier model.
 */
Php::Value simple_image_classify_image(Php::Parameters &params) {
  // Result variable definition.
  Php::Value result = 0;
  // Search image file name.
  string image_file_name = params[0];
  if (!file_is_exist(image_file_name)) {
    throw Php::Exception("Simple Image: Image file not exist");
  }
  // Classifier file name.
  string classifier_file_name = params[1];
  // Show detections in new file or not.
  bool show_detections = false;
  if (params.size() > 2) {
    show_detections = params[2];
  }
  // Scale step value.
  float scale_step = 1.25;
  double temp_double;
  if (params.size() > 3) {
    temp_double = params[3];
    scale_step = (float) temp_double;
  }
  // Slide step value.
  float slide_step = 0.1;
  if (params.size() > 4) {
    temp_double = params[4];
    slide_step = (float) temp_double;
  }
  // Classifiers limit scale value, defaults to 1.
  float scale_value = 1;
  if (params.size() > 5) {
    temp_double = params[5];
    scale_value = (float) temp_double;
  }

  // Load classifier from file.
  CascadeClassifier *cascade_classifier = load_cascade_classifier_from_file(classifier_file_name);
  cascade_classifier->scaleClassifiersLimitByValue(scale_value);

  // Initialize Magick++.
  InitializeMagick("");
  Image image, temp_image;
  try {
    // Load image file.
    image.read(image_file_name);
    temp_image = image;

    // Definition of helpful variables.
    DetectionManager *detection_manager = new DetectionManager();
    float *image_shade_pixels, *integral_image, *squared_integral_image;
    unsigned int rows = image.rows(), columns = image.columns(), size = cascade_classifier->getSize();
    float temp1, temp2;
    int slide;

    // Calculate image pixels shade array.
    image_shade_pixels = new float[rows * columns];
    temp_image.type(GrayscaleType);
    for (unsigned int x = 0; x < rows; x++) {
      for (unsigned int y = 0; y < columns; y++) {
        ColorGray color = temp_image.pixelColor(x, y);
        image_shade_pixels[x * columns + y] = color.shade();
      }
    }

    // Calculate integral and squared integral image and squared integral image
    integral_image = compute_integral_image(image_shade_pixels, columns, rows, false);
    squared_integral_image = compute_integral_image(image_shade_pixels, columns, rows, true);

    // Start object detection.
    while (size <= rows && size <= columns) {
      slide = size * slide_step;
      if (slide < 1) {
        slide = 1;
      }
      // Slide window over image
      for (unsigned int x = 0; (x + size) <= columns; x += slide) {
        for (unsigned int y = 0; (y + size) <= rows; y += slide) {
          // Calculate variables for current window.
          temp1 = calculate_integral_rectangle(integral_image, columns, x, y, size, size) / pow(size, 2);
          temp2 = sqrt((calculate_integral_rectangle(squared_integral_image, columns, x, y, size, size) / pow(size, 2)) - pow(temp1, 2));
          // Classify window by calculated values.
          if (cascade_classifier->classifyImage(integral_image, columns, x, y, temp1, temp2)) {
            detection_manager->addDetection(x, y, size);
          }
        }
      }
      // Scale cascade classifier.
      cascade_classifier->scaleByValue(scale_step);
      size = cascade_classifier->getSize();
    }

    // Load detections count.
    result = detection_manager->count();
    // Show object detections.
    if (show_detections) {
      size_t last_index = image_file_name.find_last_of(".");
      string file_without_extension = image_file_name;
      string file_extension = "";
      if (last_index != string::npos) {
        file_without_extension = image_file_name.substr(0, last_index);
        file_extension = image_file_name.substr(last_index);
      }
      string output_file_name = file_without_extension + ".simple_image_object_detections"  + file_extension;
      detection_manager->showDetections(image, output_file_name);
    }

    // Remove arrays from memory.
    delete[] image_shade_pixels;
    delete[] integral_image;
    delete[] squared_integral_image;
  }
  catch (Exception &error) {
    throw Php::Exception(error.what());
  }
  return result;
}

/**
 * Create samples for cascade training.
 */
void simple_image_create_samples(Php::Parameters &params) {
  // Output file name (file with positive samples for training).
  string positive_output_file_name = params[0];
  // Output file name (file with negative samples for training).
  string negative_output_file_name = params[1];
  // Search image file name.
  string image_file_name = params[2];
  if (!file_is_exist(image_file_name)) {
    throw Php::Exception("Simple Image: Search image file not exist");
  }
  // List of negative images.
  std::vector<string> negative_images_list = params[3];
  if (negative_images_list.empty()) {
    throw Php::Exception("Simple Image: Negative images list is empty array");
  }
  // Number of samples to generate.
  // Minimal count is one.
  int count;
  unsigned int min_count;
  count = min_count = 100;
  if (params.size() > 4) {
    count = params[4];
  }
  if (count <= 0) {
    throw Php::Exception("Simple Image: Samples count must be greater than zero");
  }
  // Samples images size in pixels WxH.
  // Minimal size is 21x21.
  int size;
  unsigned int min_size;
  size = min_size = sample_min_size;
  if (params.size() > 5) {
    size = params[5];
  }
  if (size < sample_min_size || size > sample_max_size) {
    throw Php::Exception("Simple Image: Samples size must be >= " + std::to_string(sample_min_size) + " and <= " + std::to_string(sample_max_size));
  }

  // Initialize Magick++.
  InitializeMagick("");
  // Define search image variable.
  Image search_image, temp_image;
  // Define background images vector.
  std::vector<Image> background_images;
  // Define file for output.
  fstream positive_file, negative_file;

  try {
    // Load search image file.
    search_image.read(image_file_name);
    // Check minimal image size.
    if (search_image.size().width() < min_size || search_image.size().height() < min_size) {
      throw Php::Exception("Simple Image: Too small size of search image");
    }

    // Open output files (if files now exists, they would be create).
    negative_file.open(negative_output_file_name, fstream::out);

    bool negative_files_exists = false;
    // Iterate negative images.
    for (auto const& file_path: negative_images_list) {
      // Check, that background image exist.
      if (file_is_exist(file_path)) {
        // Load background image file.
        temp_image = Image(file_path);
        // Check minimal image size.
        if (temp_image.size().width() >= min_size && temp_image.size().height() >= min_size) {
          // Crop negative image by min size.
          temp_image = image_crop_by_min_size(temp_image);
          negative_files_exists = true;
          // Put image object in vector.
          background_images.push_back(temp_image);
        }
      }
    }

    // Noises array variations.
    NoiseType noise_types[6] = {UniformNoise, GaussianNoise, MultiplicativeGaussianNoise, ImpulseNoise, LaplacianNoise, PoissonNoise};

    for (int i = 0; i < count * 2; i++) {
      temp_image = background_images[random_int(0, background_images.size() - 1)];
      // Try blur image.
      if ((bool) random_int(0, 1)) {
        temp_image.blur(random_int(0, 10), (float) 1 / random_int(1, 10));
      }
      if ((bool) random_int(0, 1)) {
        temp_image.shade(random_int(15, 70), random_int(15, 70), false);
      }
      // Try add noise to image.
      if ((bool) random_int(0, 1)) {
        temp_image.addNoise(noise_types[random_int(0, 5)]);
      }
      // Use gray-scale filter.
      // Scale search image to new size.
      temp_image.scale(Geometry(size, size));
      temp_image.type(GrayscaleType);

      // Output negative sample pixels values to file.
      negative_file << image_pixels_shade_to_string(temp_image);
      negative_file << endl;
    }

    // Close output files.
    negative_file.close();

    // If all background images not exists, throw exception.
    if (!negative_files_exists) {
      throw Php::Exception("Simple Image: All negative images not exists or have too small size");
    }

    // Open output files (if files now exists, they would be create).
    positive_file.open(positive_output_file_name, fstream::out);
    // Crop search image by min size.
    search_image = image_crop_by_min_size(search_image);
    // Generate positive samples.
    for (int i = 0; i < count; i++) {
      // Copy search image in temp variable.
      temp_image = search_image;

      // Rotate search image.
      temp_image.rotate(random_int(-25, 25));
      // Try blur image.
      if ((bool) random_int(0, 1)) {
        temp_image.blur(random_int(0, 10), (float) 1 / random_int(1, 10));
      }
      if ((bool) random_int(0, 1)) {
        temp_image.shade(random_int(15, 70), random_int(15, 70), false);
      }
      // Try add noise to image.
      if ((bool) random_int(0, 1)) {
        temp_image.addNoise(noise_types[random_int(0, 5)]);
      }
      // Use gray-scale filter.
      // Scale search image to new size.
      temp_image.scale(Geometry(size, size));
      temp_image.type(GrayscaleType);

      // Output positive sample pixels values to file.
      positive_file << image_pixels_shade_to_string(temp_image);
      if (i != (count - 1)) {
        positive_file << endl;
      }
    }
    // Close output files.
    positive_file.close();
  }
  catch (Exception &error) {
    throw Php::Exception(error.what());
  }
}

/**
 *  Tell the compiler that the get_module is a pure C function
 */
extern "C" {

  /**
   *  Function that is called by PHP right after the PHP process
   *  has started, and that returns an address of an internal PHP
   *  strucure with all the details and features of your extension
   *
   *  @return void*   a pointer to an address that is understood by PHP
   */
  PHPCPP_EXPORT void *get_module() {
    // static(!) Php::Extension object that should stay in memory
    // for the entire duration of the process (that's why it's static)
    static Php::Extension extension("simple_image", "1.0");

    //extension.add<simple_image_train_cascade>("simple_image_train_cascade");
    // Add create samples function to extension.
    extension.add<simple_image_create_samples>("simple_image_create_samples", {
      Php::ByVal("positive_output_file_name", Php::Type::String, true),
      Php::ByVal("negative_output_file_name", Php::Type::String, true),
      Php::ByVal("image_file_name", Php::Type::String, true),
      Php::ByVal("negative_images_list", Php::Type::Array, true),
      Php::ByVal("count", Php::Type::Numeric, false),
      Php::ByVal("size", Php::Type::Numeric, false)
    });

    // Add training function to extension.
    extension.add<simple_image_train_cascade>("simple_image_train_cascade", {
      Php::ByVal("model_file_name", Php::Type::String, true),
      Php::ByVal("positive_file_name", Php::Type::String, true),
      Php::ByVal("negative_file_name", Php::Type::String, true),
      Php::ByVal("size", Php::Type::Numeric, false),
      Php::ByVal("cascade_steps", Php::Type::Numeric, false),
      Php::ByVal("rotation", Php::Type::Bool, false),
      Php::ByVal("mirroring", Php::Type::Bool, false),
      Php::ByVal("negative_samples_per_step", Php::Type::Numeric, false)
    });

    // Add classify function to extension.
    extension.add<simple_image_classify_image>("simple_image_classify_image", {
      Php::ByVal("image_file_name", Php::Type::String, true),
      Php::ByVal("classifier_file_name", Php::Type::String, true),
      Php::ByVal("show_detections", Php::Type::Bool, false),
      Php::ByVal("scale_step", Php::Type::Float, false),
      Php::ByVal("slide_step", Php::Type::Float, false),
      Php::ByVal("scale_value", Php::Type::Float, false)
    });

    // Return the extension
    return extension;
  }
}
