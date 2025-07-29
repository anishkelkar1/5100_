class TicTacToeMinimax:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.nodes_examined = 0
    
    def evaluate_position(self, board):
        """
        Static evaluator from part (a)
        Returns: 1 if X wins, -1 if O wins, 0 for draw/ongoing
        """
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] != ' ':
                return 1 if row[0] == 'X' else -1
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != ' ':
                return 1 if board[0][col] == 'X' else -1
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return 1 if board[0][0] == 'X' else -1
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return 1 if board[0][2] == 'X' else -1
        
        return 0
    
    def print_board(self):
        """Print the current board state"""
        print("\n  0   1   2")
        for i in range(3):
            print(f"{i} {self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]}")
            if i < 2:
                print("  ---------")
    
    def get_available_moves(self):
        """Get all empty positions on the board"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def is_terminal(self):
        """Check if the game is over (win or draw)"""
        # Check for a winner
        if self.evaluate_position(self.board) != 0:
            return True
        # Check for draw (no empty spaces)
        return len(self.get_available_moves()) == 0
    
    def minimax(self, depth, is_maximizing):
        """
        Minimax algorithm implementation
        X is maximizing player (trying to get +1)
        O is minimizing player (trying to get -1)
        """
        self.nodes_examined += 1
        
        # Base case: terminal position
        if self.is_terminal():
            return self.evaluate_position(self.board)
        
        if is_maximizing:
            max_eval = float('-inf')
            for row, col in self.get_available_moves():
                # Make move
                self.board[row][col] = 'X'
                # Recursive call
                eval_score = self.minimax(depth + 1, False)
                # Undo move
                self.board[row][col] = ' '
                # Update max
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for row, col in self.get_available_moves():
                # Make move
                self.board[row][col] = 'O'
                # Recursive call
                eval_score = self.minimax(depth + 1, True)
                # Undo move
                self.board[row][col] = ' '
                # Update min
                min_eval = min(min_eval, eval_score)
            return min_eval
    
    def get_best_move(self, player):
        """
        Get the best move for the given player using minimax
        Returns: (best_move, nodes_examined)
        """
        self.nodes_examined = 0
        best_move = None
        
        if player == 'X':
            # X is maximizing
            best_value = float('-inf')
            for row, col in self.get_available_moves():
                self.board[row][col] = 'X'
                move_value = self.minimax(0, False)
                self.board[row][col] = ' '
                
                if move_value > best_value:
                    best_value = move_value
                    best_move = (row, col)
        else:
            # O is minimizing
            best_value = float('inf')
            for row, col in self.get_available_moves():
                self.board[row][col] = 'O'
                move_value = self.minimax(0, True)
                self.board[row][col] = ' '
                
                if move_value < best_value:
                    best_value = move_value
                    best_move = (row, col)
        
        return best_move, self.nodes_examined
    
    def make_move(self, row, col, player):
        """Make a move on the board"""
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False
    
    def play_game(self):
        """Play an interactive game against the computer"""
        print("Welcome to Tic-Tac-Toe with Minimax!")
        print("You are X, Computer is O")
        print("Enter moves as row,col (e.g., 1,2)")
        
        current_player = 'X'
        
        while not self.is_terminal():
            self.print_board()
            
            if current_player == 'X':
                # Human move
                while True:
                    try:
                        move_input = input(f"\nYour turn (X). Enter move: ")
                        row, col = map(int, move_input.split(','))
                        if 0 <= row <= 2 and 0 <= col <= 2 and self.board[row][col] == ' ':
                            self.make_move(row, col, 'X')
                            break
                        else:
                            print("Invalid move! Try again.")
                    except:
                        print("Invalid format! Use: row,col")
            else:
                # Computer move
                print("\nComputer's turn (O)...")
                move, nodes = self.get_best_move('O')
                print(f"Computer examined {nodes} nodes")
                print(f"Computer plays: {move[0]},{move[1]}")
                self.make_move(move[0], move[1], 'O')
            
            # Switch players
            current_player = 'O' if current_player == 'X' else 'X'
        
        # Game over
        self.print_board()
        result = self.evaluate_position(self.board)
        if result == 1:
            print("\nYou win! Congratulations!")
        elif result == -1:
            print("\nComputer wins!")
        else:
            print("\nIt's a draw!")


def analyze_performance():
    """Analyze minimax performance at different game states"""
    print("Minimax Performance Analysis")
    print("=" * 40)
    
    # Test 1: Empty board
    game1 = TicTacToeMinimax()
    print("\n1. Empty board - X to move:")
    move, nodes = game1.get_best_move('X')
    print(f"   Best move: {move}")
    print(f"   Nodes examined: {nodes:,}")
    
    # Test 2: After first move (center)
    game2 = TicTacToeMinimax()
    game2.board[1][1] = 'X'
    print("\n2. After X plays center - O to move:")
    game2.print_board()
    move, nodes = game2.get_best_move('O')
    print(f"   Best move: {move}")
    print(f"   Nodes examined: {nodes:,}")
    
    # Test 3: Mid-game position
    game3 = TicTacToeMinimax()
    game3.board = [['X', ' ', ' '],
                   [' ', 'O', ' '],
                   [' ', ' ', 'X']]
    print("\n3. Mid-game position - O to move:")
    game3.print_board()
    move, nodes = game3.get_best_move('O')
    print(f"   Best move: {move}")
    print(f"   Nodes examined: {nodes:,}")
    
    # Test 4: Near end-game
    game4 = TicTacToeMinimax()
    game4.board = [['X', 'O', 'X'],
                   ['X', 'O', ' '],
                   [s'O', ' ', ' ']]
    print("\n4. Near end-game - X to move:")
    game4.print_board()
    move, nodes = game4.get_best_move('X')
    print(f"   Best move: {move}")
    print(f"   Nodes examined: {nodes:,}")
    
    print("\n" + "=" * 40)
    print("Note: Node count decreases as game progresses")
    print("due to fewer available moves to explore")


def main():
    """Main function to run analysis and game"""
    # First, analyze performance
    analyze_performance()
    
    print("\n" + "=" * 50 + "\n")
    
    # Then offer to play
    play = input("Would you like to play a game? (y/n): ")
    if play.lower() == 'y':
        game = TicTacToeMinimax()
        game.play_game()


if __name__ == "__main__":
    main()