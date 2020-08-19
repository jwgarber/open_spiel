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
                             {"base_size", GameParameter(kDefaultBaseSize)},
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GeodesicYGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// A cache of the neighbors for each geodesic Y board: [base_size]
std::vector<NeighborList> neighbor_list;

int TopCell(int base_size) {
  // The smallest node of the outer ring is equal to the size of the
  // base - 1 board. This even works when the base size is 2, where
  // the inner ring then has size 0.
  return BoardSize(base_size - 1);
}

int RightCell(int base_size) {
  return TopCell(base_size) + base_size - 1;
}

int LeftCell(int base_size) {
  return RightCell(base_size) + base_size - 1;
}

NeighborList GenNeighbors(int base_size) {

  int board_size = BoardSize(base_size);

  NeighborList neighbors{};
  neighbors.resize(board_size);

  neighbors.at(0).push_back(1);
  neighbors.at(1).push_back(2);
  neighbors.at(2).push_back(0);

  // This loop will be skipped if base_size == 2
  for (int ring = 3; ring <= base_size; ++ring) {

    int top = TopCell(ring);
    int right = RightCell(ring);
    int left = LeftCell(ring);

    int ring_size = TopCell(ring + 1);
    int last = ring_size - 1;

    for (int cell = top; cell < ring_size; ++cell) {

      // Add the next node clockwise in the ring
      if (cell == last) {
        neighbors.at(cell).push_back(top);
      } else {
        neighbors.at(cell).push_back(cell + 1);
      }

      // Now add the cells in the layer below
      int top_below = TopCell(ring - 1);
      int right_below = RightCell(ring - 1);
      int left_below = LeftCell(ring - 1);

      if (cell == top) {
        neighbors.at(cell).push_back(top_below);
      } else if ((top < cell) && (cell < right)) {
        int nhbr = top_below + cell - top;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == right) {
        neighbors.at(cell).push_back(right_below);
      } else if ((right < cell) && (cell < left)) {
        int nhbr = right_below + cell - right;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == left) {
        neighbors.at(cell).push_back(left_below);
      } else if (cell < last) {
        int nhbr = left_below + cell - left;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == last) {
        int nhbr = left_below + cell - left;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(top_below);
      } else {
        SpielFatalError("Unknown cell.");
      }
    }
  }

  // Make the graph symmetric
  NeighborList symmetric = neighbors;

  for (int i = 0; i < neighbors.size(); ++i) {
    for (Move m : neighbors.at(i)) {
      symmetric.at(m.cell).push_back(i);
    }
  }

  return symmetric;
}

const NeighborList& GetNeighbors(int base_size) {
  if (base_size >= neighbor_list.size()) {
    neighbor_list.resize(base_size + 1);
  }
  if (neighbor_list.at(base_size).empty()) {
    neighbor_list.at(base_size) = GenNeighbors(base_size);
  }
  return neighbor_list.at(base_size);
}

}  // namespace

BoardEdge Move::Edge(int base_size) const {

  int top = TopCell(base_size);
  int right = RightCell(base_size);
  int left = LeftCell(base_size);

  BoardEdge edge = kNone;
  if ((top <= cell) && (cell <= right)) {
    edge = (BoardEdge)(edge | kRight);
  }
  if ((right <= cell) && (cell <= left)) {
    edge = (BoardEdge)(edge | kBottom);
  }
  if ((left <= cell) || (cell == top)) {
    edge = (BoardEdge)(edge | kLeft);
  }

  return edge;
}

std::string Move::ToString() const {
  return absl::StrCat(cell);
}

GeodesicYState::GeodesicYState(std::shared_ptr<const Game> game, int base_size,
               bool ansi_color_output)
    : State(game),
      base_size_(base_size),
      neighbors_(GetNeighbors(base_size)),
      ansi_color_output_(ansi_color_output) {
  board_.resize(BoardSize(base_size));
  for (int i = 0; i < board_.size(); i++) {
    Move m = ActionToMove(i);
    board_.at(i) = Cell(kPlayerNone, i, m.Edge(base_size));
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
  for (int cell = 0; cell < board_.size(); ++cell) {
    if (board_.at(cell).player == kPlayerNone) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string GeodesicYState::ActionToString(Player player, Action action_id) const {
  return ActionToMove(action_id).ToString();
}

std::string GeodesicYState::ToString() const {
  std::ostringstream out{};

  out << "Player 1: ";
  for (int i = 0; i < board_.size(); ++i) {
    if (board_.at(i).player == kPlayer1) {
      out << i << ' ';
    }
  }
  out << '\n';

  out << "Player 2: ";
  for (int i = 0; i < board_.size(); ++i) {
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
                               absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (int i = 0; i < board_.size(); ++i) {
    view[{PlayerRelative(board_.at(i).player, player), i}] = 1.0;
  }
}

void GeodesicYState::ResetBoard() {
  current_player_ = kPlayer1;
  outcome_ = kPlayerNone;
  moves_made_ = 0;

  for (int i = 0; i < board_.size(); i++) {
    Move m = ActionToMove(i);
    board_.at(i) = Cell(kPlayerNone, i, m.Edge(base_size_));
  }
}

void GeodesicYState::UndoAction(Player player, Action move) {
  // Union-Find sets change when an action is played, so undoing
  // an action directly is tricky. Just reset the board and replay
  // the moves instead.
  history_.pop_back();
  ResetBoard();

  for (auto [_, action] : history_) {
    DoApplyAction(action);
  }
}

void GeodesicYState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(board_.at(action).player, kPlayerNone);
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move = ActionToMove(action);

  board_.at(move.cell).player = current_player_;
  moves_made_++;

  for (const Move& m : neighbors_.at(move.cell)) {
    if (current_player_ == board_.at(m.cell).player) {
      JoinGroups(move.cell, m.cell);
    }
  }

  if (board_.at(FindGroupLeader(move.cell)).edge == (kRight | kBottom | kLeft)) {
    outcome_ = current_player_;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

int GeodesicYState::FindGroupLeader(int cell) {
  int parent = board_.at(cell).parent;
  if (parent != cell) {
    do {  // Follow the parent chain up to the group leader.
      parent = board_.at(parent).parent;
    } while (parent != board_.at(parent).parent);
    // Do path compression, but only the current one to avoid recursion.
    board_.at(cell).parent = parent;
  }
  return parent;
}

bool GeodesicYState::JoinGroups(int cell_a, int cell_b) {
  int leader_a = FindGroupLeader(cell_a);
  int leader_b = FindGroupLeader(cell_b);

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
      base_size_(ParameterValue<int>("base_size")),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace geodesic_y_game
}  // namespace open_spiel
