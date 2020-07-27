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

#include <memory>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/geodesic_y.h"

ABSL_FLAG(int, base_size, 3, "The base size of the board.");

int main(int argc, char** argv) {

  absl::ParseCommandLine(argc, argv);
  const auto base_size = absl::GetFlag(FLAGS_base_size);

  open_spiel::GameParameters params;
  params["base_size"] = open_spiel::GameParameter(base_size);

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("geodesic_y", params);

  if (!game) {
    std::cerr << "problem with loading game, exiting..." << std::endl;
    return -1;
  }

  auto depth = open_spiel::geodesic_y_game::boardSize(base_size);

  std::cout << "Running alpha-beta on board with base size " << base_size << std::endl;

  auto p = open_spiel::algorithms::AlphaBetaSearch(*game, nullptr, nullptr, depth, open_spiel::geodesic_y_game::kPlayer1);
  std::cout << "value = " << p.first << std::endl;
  std::cout << "action = " << p.second << std::endl;
}
