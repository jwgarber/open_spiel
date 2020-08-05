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
                             {"starting_player", GameParameter(std::string("black"))},
                             {"starting_board", GameParameter(std::string(""))},
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GeodesicYGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// A cache of the neighbors for each geodesic Y board
static std::vector<Neighbors> neighbors_list;

static Node topNode(uint16_t base_size) {
  // The smallest node of the outer ring is equal to the size of the
  // base - 1 board. This even works when the base size is 2, where
  // the inner ring then has size 0.
  return boardSize(base_size - 1);
}

static Node rightNode(uint16_t base_size) {
  return topNode(base_size) + base_size - 1;
}

static Node leftNode(uint16_t base_size) {
  return rightNode(base_size) + base_size - 1;
}

static Neighbors generateNeighbors(uint16_t base_size) {

  Node board_size = boardSize(base_size);

  Neighbors neighbors{};
  neighbors.resize(board_size);

  neighbors.at(0).push_back(1);
  neighbors.at(1).push_back(2);
  neighbors.at(2).push_back(0);

  // This loop will be skipped if base_size == 2
  for (uint16_t ring = 3; ring <= base_size; ++ring) {

    Node top = topNode(ring);
    Node right = rightNode(ring);
    Node left = leftNode(ring);

    Node ring_size = topNode(ring + 1);
    Node last = ring_size - 1;

    for (Node cell = top; cell < ring_size; ++cell) {

      // Add the next node clockwise in the ring
      if (cell == last) {
        neighbors.at(cell).push_back(top);
      } else {
        neighbors.at(cell).push_back(cell + 1);
      }

      // Now add the cells in the layer below
      Node top_below = topNode(ring - 1);
      Node right_below = rightNode(ring - 1);
      Node left_below = leftNode(ring - 1);

      if (cell == top) {
        neighbors.at(cell).push_back(top_below);
      } else if ((top < cell) && (cell < right)) {
        Node nhbr = top_below + cell - top;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == right) {
        neighbors.at(cell).push_back(right_below);
      } else if ((right < cell) && (cell < left)) {
        Node nhbr = right_below + cell - right;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == left) {
        neighbors.at(cell).push_back(left_below);
      } else if (cell < last) {
        Node nhbr = left_below + cell - left;
        neighbors.at(cell).push_back(nhbr - 1);
        neighbors.at(cell).push_back(nhbr);
      } else if (cell == last) {
      	Node nhbr = left_below + cell - left;
      	neighbors.at(cell).push_back(nhbr - 1);
      	neighbors.at(cell).push_back(top_below);
      } else {
      	throw std::runtime_error("unknown cell");
      }
    }
  }

  // Make the graph symmetric
  Neighbors symmetric = neighbors;

  for (Node i = 0; i < neighbors.size(); ++i) {
    for (Node j : neighbors.at(i)) {
      symmetric.at(j).push_back(i);
    }
  }

  for (auto& vec : symmetric) {
  	std::sort(std::begin(vec), std::end(vec));
  }

  return symmetric;
}

static const Neighbors& getNeighbors(const uint16_t board_size) {
  // Extend the list if the board_size isn't present
  if (board_size >= neighbors_list.size()) {
    neighbors_list.resize(board_size + 1);
  }

  if (neighbors_list.at(board_size).empty()) {
    neighbors_list.at(board_size) = generateNeighbors(board_size);
  }
  return neighbors_list.at(board_size);
}

static Edge getEdge(Node node, uint16_t base_size) {

  Node top = topNode(base_size);
  Node right = rightNode(base_size);
  Node left = leftNode(base_size);

  Edge edge = kNone;
  if ((top <= node) && (node <= right)) {
    edge = (Edge)(edge | kRight);
  }
  if ((right <= node) && (node <= left)) {
    edge = (Edge)(edge | kBottom);
  }
  if ((left <= node) || (node == top)) {
    edge = (Edge)(edge | kLeft);
  }

  return edge;
}

std::string Move::ToString() const {
  return std::to_string(node);
}

static GeodesicYPlayer getStartingPlayer(const std::string& player_str) {
  if (player_str == "black") {
    return kPlayer1;
  }
  if (player_str == "white") {
    return kPlayer2;
  }
  SpielFatalError(absl::StrCat("Unknown player ", player_str));
}

static std::vector<GeodesicYPlayer> getStartingBoard(const uint16_t base_size,
                                                     const std::string& board_str) {

  const auto board_size = boardSize(base_size);

  std::vector<GeodesicYPlayer> board(board_size, kPlayerNone);

  const std::vector<std::string> split = absl::StrSplit(board_str, ' ', absl::SkipEmpty());
  for (const auto& str : split) {
    if (str.size() < 2) {
      SpielFatalError(absl::StrCat("Invalid configuration ", str));
    }

    const char player = str.at(0);

    const uint32_t pos = std::stoul(str.substr(1));
    if (pos >= board_size) {
      SpielFatalError(absl::StrCat("Invalid position ", std::to_string(pos)));
    }

    if (player == 'B') {
      board.at(pos) = kPlayer1;
    } else if (player == 'W') {
      board.at(pos) = kPlayer2;
    } else {
      SpielFatalError(absl::StrCat("Invalid player ", std::string(1, player)));
    }
  }

  return board;
}

GeodesicYState::GeodesicYState(std::shared_ptr<const Game> game, int base_size,
               const std::string& starting_player,
               const std::string& starting_board,
               bool ansi_color_output)
    : State(game),
      base_size_(base_size),
      starting_player_(getStartingPlayer(starting_player)),
      starting_board_(getStartingBoard(base_size, starting_board)),
      neighbors_(getNeighbors(base_size)),
      ansi_color_output_(ansi_color_output) {

  // Initialize an empty board
  board_.resize(boardSize(base_size));
  for (Node i = 0; i < board_.size(); i++) {
    board_.at(i) = Cell(kPlayerNone, i, getEdge(i, base_size));
  }

  // Then, place all the initial cells, without changing any state
  for (Node i = 0; i < board_.size(); i++) {
    GeodesicYPlayer player = starting_board_.at(i);
    if (player != kPlayerNone) {
      PlayCell(player, i);
    }
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
                               absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (Node i = 0; i < board_.size(); ++i) {
    view[{PlayerRelative(board_.at(i).player, player), i}] = 1.0;
  }
}

void GeodesicYState::ResetBoard() {

  current_player_ = starting_player_;
  outcome_ = kPlayerNone;
  moves_made_ = 0;
  last_move_ = Move(boardSize(base_size_));

  // Reset the board back to empty
  for (Node i = 0; i < board_.size(); i++) {
    board_.at(i) = Cell(kPlayerNone, i, getEdge(i, base_size_));
  }

  // Now set the starting board state
  for (Node i = 0; i < board_.size(); i++) {
    GeodesicYPlayer player = starting_board_.at(i);
    if (player != kPlayerNone) {
      PlayCell(player, i);
    }
  }
}

void GeodesicYState::UndoAction(Player player, Action move) {
  // UF groupings change when an action is played, so to undo that
  // action we also need to "undo-union" the groups. That's tricky,
  // just reset the board and replay the moves (like in Go).
  history_.pop_back();
  ResetBoard();

  for (auto [_, action] : history_) {
    DoApplyAction(action);
  }
}

void GeodesicYState::PlayCell(GeodesicYPlayer player, Node cell) {

  board_.at(cell).player = player;

  for (Node nhbr : neighbors_.at(cell)) {
    if (board_.at(nhbr).player == player) {
      JoinGroups(cell, nhbr);
    }
  }

  // The starting board should not win the game
  if (board_.at(FindGroupLeader(cell)).edge == (kRight | kBottom | kLeft)) {
    SpielFatalError(std::string("Starting board cannot be already won"));
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
      base_size_(ParameterValue<int>("base_size")),
      starting_player_(ParameterValue<std::string>("starting_player")),
      starting_board_(ParameterValue<std::string>("starting_board")),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace geodesic_y_game
}  // namespace open_spiel
