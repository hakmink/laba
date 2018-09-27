import string
import torch

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
TRAIN_TEST_RATIO = 0.75
RANDOM_STATE = 10


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def surname_to_tensor(surname):
    tensor = torch.zeros(len(surname), 1, N_LETTERS)
    for li, letter in enumerate(surname):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor