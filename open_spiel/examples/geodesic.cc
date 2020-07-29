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
ABSL_FLAG(std::string, player, "black", "The starting player (black or white).");

int main(int argc, char** argv) {

  absl::ParseCommandLine(argc, argv);
  const auto base_size = absl::GetFlag(FLAGS_base_size);
  const auto player_str = absl::GetFlag(FLAGS_player);

  open_spiel::Player player;
  if (player_str == "black") {
    player = 0;
  } else if (player_str == "white") {
    player = 1;
  } else {
    std::cout << "Invalid player: " << player_str << std::endl;
    return 0;
  }

  open_spiel::GameParameters params;
  params["base_size"] = open_spiel::GameParameter(base_size);
  params["starting_player"] = open_spiel::GameParameter(player_str);

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("geodesic_y", params);

  if (!game) {
    std::cerr << "problem with loading game, exiting..." << std::endl;
    return -1;
  }

  auto depth = open_spiel::geodesic_y_game::boardSize(base_size);

  std::cout << "Running alpha-beta on board with base size " << base_size << std::endl;

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  std::vector<open_spiel::Action> actions = state->LegalActions();

  // Iterate over all available actions, and as the current player, play that action.
  // Then do an alpha-beta search from that position as the opponent. If they lose,
  // then that was a winning move for the first player.
  std::cout << "Winning moves: ";
  for (const auto action : actions) {
    state->ApplyAction(action);

    // Performs an alpha-beta search from the current state with the current player
    auto p = open_spiel::algorithms::AlphaBetaSearch(*game, state.get(), nullptr, depth, open_spiel::kInvalidPlayer);

    if (p.first == -1.0) {
      // The second player lost, so this is a winning move for the first player
      std::cout << action << ' ';
    }

    state->UndoAction(player, action);
  }
  std::cout << std::endl;
}
