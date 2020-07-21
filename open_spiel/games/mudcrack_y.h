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

#ifndef OPEN_SPIEL_GAMES_MUDCRACK_Y_H_
#define OPEN_SPIEL_GAMES_MUDCRACK_Y_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// https://en.wikipedia.org/wiki/Y_(game)
// Does not implement pie rule to balance the game
//
// Parameters:
//   "board_size"        int     size of the board   (default = 9)
//   "ansi_color_output" bool    Whether to color the output for a terminal.

namespace open_spiel {
namespace mudcrack_y_game {

// Integer type that labels nodes in the graph
using Node = uint16_t;
// Adjacency list for the graph
using Neighbors = std::vector<std::vector<Node>>;

// Default board will be base 3 geodesic Y, which has 9 nodes

inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultBoardSize = 9;
inline constexpr int kCellStates = 1 + kNumPlayers;

enum MudcrackYPlayer : uint8_t {
  kPlayer1,
  kPlayer2,
  kPlayerNone,
};

enum Edge : uint8_t {
  kNone = 0x0,
  kRight = 0x1,
  kBottom = 0x2,
  kLeft = 0x4,
};

struct Move {
  Node node;

  inline constexpr Move(Node node_) : node(node_) {}

  std::string ToString() const;

  bool operator==(const Move& b) const { return node == b.node; }
  bool operator!=(const Move& b) const { return node != b.node; }
};

// State of an in-play game.
class MudcrackYState : public State {
  // Represents a single cell on the board, as well as the structures needed for
  // groups of cells. Groups of cells are defined by a union-find structure
  // embedded in the array of cells. Following the `parent` indices will lead to
  // the group leader which has the up to date size and edge
  // connectivity of that group. Size and edge are not valid for any
  // cell that is not a group leader.
  struct Cell {
    // Who controls this cell.
    MudcrackYPlayer player;

    // A parent index to allow finding the group leader. It is the leader of the
    // group if it points to itself. Allows path compression to shorten the path
    // from a direct parent to the leader.
    Node parent;

    // These three are only defined for the group leader's cell.
    uint16_t size;  // Size of this group of cells.
    uint8_t edge;   // A bitset of which edges this group is connected to.

    Cell() {}
    Cell(MudcrackYPlayer player_, Node parent_, Edge edge_)
        : player(player_), parent(parent_), size(1), edge(edge_) {}
  };

 public:
  MudcrackYState(std::shared_ptr<const Game> game, int board_size,
         bool ansi_color_output = false);

  MudcrackYState(const MudcrackYState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : static_cast<int>(current_player_);
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return outcome_ != kPlayerNone; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  // A 3d tensor, 3 player-relative one-hot 2d planes. The layers are: the
  // specified player, the other player, and empty.
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

  // Find the leader of the group. Not const due to union-find path compression.
  Node FindGroupLeader(Node cell);

  // Join the groups of two positions, propagating group size, and edge
  // connections. Returns true if they were already the same group.
  bool JoinGroups(Node cell_a, Node cell_b);

  // Turn an action id into a Move
  Move ActionToMove(Action action_id) const;

 private:
  std::vector<Cell> board_;
  MudcrackYPlayer current_player_ = kPlayer1;
  MudcrackYPlayer outcome_ = kPlayerNone;
  const int board_size_;
  uint16_t moves_made_ = 0;
  // The last move is initialized to the size of the graph
  // since at the beginning there are no previous moves.
  Move last_move_ = Move(board_size_);
  const Neighbors& neighbors_;
  const bool ansi_color_output_;
};

// Game object.
class MudcrackYGame : public Game {
 public:
  explicit MudcrackYGame(const GameParameters& params);

  int NumDistinctActions() const override {
    return board_size_;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new MudcrackYState(shared_from_this(), board_size_, ansi_color_output_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new MudcrackYGame(*this));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, board_size_, board_size_};
  }
  int MaxGameLength() const override {
    // The true number of playable cells on the board.
    // No stones are removed, and someone will win by filling the board.
    // Increase this by one if swap is ever implemented.
    return board_size_;
  }

 private:
  const int board_size_;
  const bool ansi_color_output_ = false;
};

}  // namespace mudcrack_y_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MUDCRACK_Y_H_
