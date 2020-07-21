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

#include "open_spiel/games/geodesic_y.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace geodesic_y_game {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"geodesic_y",
                         /*long_name=*/"Geodesic Y Connection Game",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                             {"board_size", GameParameter(kDefaultBoardSize)},
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GeodesicYGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// Neighbors for base3 geodesic Y (the default game)
static const Neighbors neighbors = {
  {1, 2, 3, 4, 8},
  {0, 2, 4, 5, 6},
  {0, 1, 6, 7, 8},
  {0, 4, 8},
  {0, 1, 3, 5},
  {1, 4, 6},
  {1, 2, 5, 7},
  {2, 6, 8},
  {0, 2, 3, 7},
};

// The board edges that each node touches.
static const std::vector<Edge> edges = {
  kNone, kNone, kNone,
  (Edge)(kRight | kLeft),
  kRight,
  (Edge)(kRight | kBottom),
  kBottom,
  (Edge)(kBottom | kLeft),
  kLeft,
};

std::string Move::ToString() const {
  return std::to_string(node);
}

GeodesicYState::GeodesicYState(std::shared_ptr<const Game> game, int board_size,
               bool ansi_color_output)
    : State(game),
      board_size_(board_size),
      neighbors_(neighbors),
      ansi_color_output_(ansi_color_output) {
  board_.resize(board_size_);
  for (Node i = 0; i < board_.size(); i++) {
    board_.at(i) = Cell(kPlayerNone, i, edges.at(i));
  }
}

Move GeodesicYState::ActionToMove(Action action_id) const {
  return Move(action_id);
}

std::vector<Action> GeodesicYState::LegalActions() const {
  // Can move in any empty cell.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  moves.reserve(board_.size() - moves_made_);
  for (Node cell = 0; cell < board_.size(); ++cell) {
    if (board_.at(cell).player == kPlayerNone) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string GeodesicYState::ActionToString(Player player, Action action_id) const {
  return ActionToMove(action_id).ToString();
}

#if 0
std::string GeodesicYState::ToString() const {
  // Generates something like:
  //  a b c d e f g h i j k
  // 1 O @ O O . @ @ O O @ O
  //  2 . O O . O @ @ . O O
  //   3 . O @ @ O @ O O @
  //    4 O O . @ . @ O O
  //     5 . . . @[@]@ O
  //      6 @ @ @ O O @
  //       7 @ . O @ O
  //        8 . @ @ O
  //         9 @ @ .
  //         10 O .
  //          11 @

  std::string white = "O";
  std::string black = "@";
  std::string empty = ".";
  std::string coord = "";
  std::string reset = "";
  if (ansi_color_output_) {
    std::string esc = "\033";
    reset = esc + "[0m";
    coord = esc + "[1;37m";  // bright white
    empty = reset + ".";
    white = esc + "[1;33m" + "@";  // bright yellow
    black = esc + "[1;34m" + "@";  // bright blue
  }

  std::ostringstream out;

  // Top x coords.
  out << ' ';
  for (int x = 0; x < board_size_; x++) {
    out << ' ' << coord << static_cast<char>('a' + x);
  }
  out << '\n';

  for (int y = 0; y < board_size_; y++) {
    out << std::string(y + ((y + 1) < 10), ' ');  // Leading space.
    out << coord << (y + 1);                      // Leading y coord.

    bool found_last = false;
    for (int x = 0; x < board_size_ - y; x++) {
      Move pos(x, y, board_size_);

      // Spacing and last-move highlight.
      if (found_last) {
        out << coord << ']';
        found_last = false;
      } else if (last_move_ == pos) {
        out << coord << '[';
        found_last = true;
      } else {
        out << ' ';
      }

      // Actual piece.
      Player p = board_[pos.xy].player;
      if (p == kPlayerNone) out << empty;
      if (p == kPlayer1) out << white;
      if (p == kPlayer2) out << black;
    }
    if (found_last) {
      out << coord << ']';
    }
    out << '\n';
  }
  out << reset;
  return out.str();
}
#endif

std::string GeodesicYState::ToString() const {
  std::ostringstream out{};

  out << "black: ";
  for (Node i = 0; i < board_.size(); ++i) {
    if (board_.at(i).player == kPlayer1) {
      out << i << ' ';
    }
  }
  out << '\n';

  out << "white: ";
  for (Node i = 0; i < board_.size(); ++i) {
    if (board_.at(i).player == kPlayer2) {
      out << i << ' ';
    }
  }
  out << '\n';

  return out.str();
}

std::vector<double> GeodesicYState::Returns() const {
  if (outcome_ == kPlayer1) return {1, -1};
  if (outcome_ == kPlayer2) return {-1, 1};
  return {0, 0};  // Unfinished
}

std::string GeodesicYState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string GeodesicYState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(GeodesicYPlayer state, Player current) {
  switch (state) {
    case kPlayer1:
      return current == 0 ? 0 : 1;
    case kPlayer2:
      return current == 1 ? 0 : 1;
    case kPlayerNone:
      return 2;
    default:
      SpielFatalError("Unknown player type.");
  }
}

void GeodesicYState::ObservationTensor(Player player,
                               std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (Node i = 0; i < board_.size(); ++i) {
    view[{PlayerRelative(board_.at(i).player, player), i}] = 1.0;
  }
}

void GeodesicYState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(board_.at(action).player, kPlayerNone);
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move = ActionToMove(action);

  last_move_ = move;
  board_.at(move.node).player = current_player_;
  moves_made_++;

  for (Node nhbr : neighbors_.at(move.node)) {
    if (current_player_ == board_.at(nhbr).player) {
      JoinGroups(move.node, nhbr);
    }
  }

  // Check if the current player has won the game
  if (board_.at(FindGroupLeader(move.node)).edge == (kRight | kBottom | kLeft)) {
    outcome_ = current_player_;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

Node GeodesicYState::FindGroupLeader(Node cell) {
  Node parent = board_.at(cell).parent;
  if (parent != cell) {
    do {  // Follow the parent chain up to the group leader.
      parent = board_.at(parent).parent;
    } while (parent != board_.at(parent).parent);
    // Do path compression, but only the current one to avoid recursion.
    board_.at(cell).parent = parent;
  }
  return parent;
}

bool GeodesicYState::JoinGroups(Node cell_a, Node cell_b) {
  Node leader_a = FindGroupLeader(cell_a);
  Node leader_b = FindGroupLeader(cell_b);

  if (leader_a == leader_b)  // Already the same group.
    return true;

  if (board_.at(leader_a).size < board_.at(leader_b).size) {
    // Force group a's subtree to be bigger.
    std::swap(leader_a, leader_b);
  }

  // Group b joins group a.
  board_.at(leader_b).parent = leader_a;
  board_.at(leader_a).size += board_.at(leader_b).size;
  board_.at(leader_a).edge |= board_.at(leader_b).edge;

  return false;
}

std::unique_ptr<State> GeodesicYState::Clone() const {
  return std::unique_ptr<State>(new GeodesicYState(*this));
}

GeodesicYGame::GeodesicYGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace geodesic_y_game
}  // namespace open_spiel
