def evaluate_position(board):
    """
    Static evaluator for tic-tac-toe position
    
    Parameters:
    board: 3x3 list of lists containing 'X', 'O', or ' ' (empty)
    
    Returns:
    1 if X wins (crosses win)
    -1 if O wins (noughts win)
    0 for draw or ongoing game
    """
    
    # Check all rows
    for row in board:
        if row[0] == row[1] == row[2] != ' ':
            return 1 if row[0] == 'X' else -1
    
    # Check all columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return 1 if board[0][col] == 'X' else -1
    
    # Check main diagonal (top-left to bottom-right)
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return 1 if board[0][0] == 'X' else -1
    
    # Check anti-diagonal (top-right to bottom-left)
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return 1 if board[0][2] == 'X' else -1
    
    # No winner found, return 0 for draw or ongoing game
    return 0


def print_board(board):
    """Helper function to print the board nicely"""
    print("\n  0   1   2")
    for i in range(3):
        print(f"{i} {board[i][0]} | {board[i][1]} | {board[i][2]}")
        if i < 2:
            print("  ---------")


def test_evaluator():
    """Test cases for the static evaluator"""
    print("Testing Tic-Tac-Toe Static Evaluator")
    print("=" * 40)
    
    # Test Case 1: X wins horizontally (top row)
    board1 = [['X', 'X', 'X'],
              ['O', 'O', ' '],
              [' ', ' ', ' ']]
    print("\nTest 1 - X wins (horizontal):")
    print_board(board1)
    result = evaluate_position(board1)
    print(f"Evaluation: {result} (expected: 1)")
    assert result == 1, "Test 1 failed"
    
    # Test Case 2: O wins vertically (middle column)
    board2 = [['X', 'O', 'X'],
              ['X', 'O', ' '],
              [' ', 'O', ' ']]
    print("\nTest 2 - O wins (vertical):")
    print_board(board2)
    result = evaluate_position(board2)
    print(f"Evaluation: {result} (expected: -1)")
    assert result == -1, "Test 2 failed"
    
    # Test Case 3: X wins diagonally
    board3 = [['X', 'O', ' '],
              ['O', 'X', ' '],
              [' ', ' ', 'X']]
    print("\nTest 3 - X wins (diagonal):")
    print_board(board3)
    result = evaluate_position(board3)
    print(f"Evaluation: {result} (expected: 1)")
    assert result == 1, "Test 3 failed"
    
    # Test Case 4: O wins anti-diagonally
    board4 = [[' ', ' ', 'O'],
              ['X', 'O', 'X'],
              ['O', 'X', ' ']]
    print("\nTest 4 - O wins (anti-diagonal):")
    print_board(board4)
    result = evaluate_position(board4)
    print(f"Evaluation: {result} (expected: -1)")
    assert result == -1, "Test 4 failed"
    
    # Test Case 5: Draw (board full, no winner)
    board5 = [['X', 'O', 'X'],
              ['X', 'X', 'O'],
              ['O', 'X', 'O']]
    print("\nTest 5 - Draw (board full):")
    print_board(board5)
    result = evaluate_position(board5)
    print(f"Evaluation: {result} (expected: 0)")
    assert result == 0, "Test 5 failed"
    
    # Test Case 6: Ongoing game (no winner yet)
    board6 = [['X', 'O', ' '],
              [' ', 'X', ' '],
              [' ', ' ', 'O']]
    print("\nTest 6 - Ongoing game:")
    print_board(board6)
    result = evaluate_position(board6)
    print(f"Evaluation: {result} (expected: 0)")
    assert result == 0, "Test 6 failed"
    
    print("\n" + "=" * 40)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_evaluator()