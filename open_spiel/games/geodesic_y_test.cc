// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace geodesic_y_game {
namespace {

namespace testing = open_spiel::testing;

void BasicGeodesicYTests() {
  testing::LoadGameTest("geodesic_y(base_size=9)");
  testing::NoChanceOutcomesTest(*LoadGame("geodesic_y(base_size=9)"));

  testing::RandomSimTest(*LoadGame("geodesic_y"), 10);

  // All the sizes we care about.
  for (int i = 2; i <= 20; i++) {
    testing::RandomSimTest(*LoadGame(absl::StrCat("geodesic_y(base_size=", i, ")")),
                           10);
  }

  // Ansi colors!
  testing::RandomSimTest(
      *LoadGame("geodesic_y", {{"base_size", GameParameter(9)},
                      {"ansi_color_output", GameParameter(true)}}),
      1);
  testing::RandomSimTest(*LoadGame("geodesic_y(base_size=10,ansi_color_output=True)"),
                         3);
}

}  // namespace
}  // namespace geodesic_y_game
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::geodesic_y_game::BasicGeodesicYTests(); }
