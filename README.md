<<<<<<< HEAD
# Chess Analysis and Best-Move Finder

## Overview
This project is a chess analysis tool and best-move finder implemented in Python. It uses a trained machine learning model to evaluate chess positions and determine the optimal moves through a minimax algorithm. The project consists of two main components: the **Model** and the **Minimax Algorithm**.

---

## Model
The model is built using **Tensorflow** and **Keras** and follows a Sequential architecture. It's trained on [a dataset obtained from Kaggle](https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom), which contains chess games played by Grandmasters on chess.com. The training process involved:

- Extracting all chess positions from the games (converting the PGNs in the dataset to a collection of FENs).
- Associating each position with the outcome of the game to determine a win probability based on position.

The model acts as an evaluation system, providing win probabilities for different positions during the game.

---

## Minimax Algorithm
The main code integrates the trained model to analyze moves using a **minimax algorithm**. This algorithm evaluates the best move by simulating potential outcomes for both players and selecting the move with the highest evaluation. The code additionally consisted of an **iterative deepening** algorithm, which increases the depth of the minimax search only for moves it finds promising.

### Move Ordering
To enhance efficiency, moves are processed in the following order:
1. Checks
2. Promotions
3. Captures

This approach prioritises high-impact moves and ensures the search tree is optimized for critical positions.

---

## Future Prospects
1. **Rewrite in C++**:
   - To improve efficiency, the main code could be rewritten in C++, a language well-known for its speed and performance.

2. **Expand Training Dataset**:
   - With access to more resources, the training dataset size could be increased, improving the model's accuracy and generalization.

3. **Optimizing Move Searching**:
   - Explore faster algorithms or heuristics to optimize the minimax algorithm.
   - Expand the list of prioritized moves (checks, promotions, captures) to include other high-depth attacks, as complex middlegame and endgame positions often demand nuanced strategies beyond immediate tactical moves.

4. **Transposition Data**:
    - As the model currently evaluates standalone positions, a dictionary of previously evaluated positions would allow it to reuse the pre-existing best move and evaluation data.
    - If it's able to look through even 1 or 2 positions at a faster pace, its depth could be increased a lot without compromising as much on time.

---

## Acknowledgments
- **Dataset**: [Kaggle - GM Games on Chess.com](https://www.kaggle.com/datasets/dimitrioskourtikakis/gm-games-chesscom)
- **Libraries Used**: Tensorflow, Keras, Python Chess, Numpy, Pandas, Scikit Learn, Kagglehub, and other in-built libraries.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

=======

