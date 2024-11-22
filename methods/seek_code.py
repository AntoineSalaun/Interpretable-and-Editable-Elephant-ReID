import torch

import torch

class SEEK:
    # Class variables for attribute mappings, lengths, and attribute names
    mappings = {
        'sex': ['B', 'C', '_'],
        'age': ['00', '20'],
        'right_tusk': ['0', '_', '1'],
        'left_tusk': ['0', '_', '1'],
        'R_tear_1': ['0', '_', '7', '8', '9'],
        'R_hole_1': ['0', '_', '7', '8', '9'],
        'R_tear_2': ['0', '_', '7', '8', '9'],
        'R_hole_2': ['0', '_', '7', '8', '9'],
        'L_tear_1': ['0', '_', '3', '4', '5'],
        'L_hole_1': ['0', '_', '3', '4', '5'],
        'L_tear_2': ['0', '_', '3', '4', '5'],
        'L_hole_2': ['0', '_', '3', '4', '5'],
        'right_extreme': ['0', '_', '1'],
        'left_extreme': ['0', '_', '1'],
        'ear_special': ['0', '_', '1'],
        'body_special': ['0', '_', '1']
    }

    lengths = {k: len(v) for k, v in mappings.items()}

    attribute_names = [
        'sex', 'age', 'right_tusk', 'left_tusk', 'R_tear_1', 'R_hole_1',
        'R_tear_2', 'R_hole_2', 'L_tear_1', 'L_hole_1', 'L_tear_2', 'L_hole_2',
        'right_extreme', 'left_extreme', 'ear_special', 'body_special'
    ]

    def __init__(self, code):
        if isinstance(code, str):
            self.initialize_from_string(code)
        elif isinstance(code, torch.Tensor):
            self.initialize_from_one_hot(code)
        else:
            raise ValueError("code must be a string or a one-hot encoded torch.Tensor")

    def initialize_from_string(self, code):
        self.sex = code[0]
        self.age = code[1] + code[2]
        self.right_tusk = code[4]
        self.left_tusk = code[5]
        self.R_tear_1 = code[7]
        self.R_hole_1 = code[8]
        self.R_tear_2 = code[9]
        self.R_hole_2 = code[10]
        self.L_tear_1 = code[12]
        self.L_hole_1 = code[13]
        self.L_tear_2 = code[14]
        self.L_hole_2 = code[15]
        self.right_extreme = code[17]
        self.left_extreme = code[18]
        self.ear_special = code[20]
        self.body_special = code[21]

        # Store attributes for indexing
        self.attributes = [
            ('sex', self.sex), ('age', self.age), ('right_tusk', self.right_tusk), ('left_tusk', self.left_tusk),
            ('R_tear_1', self.R_tear_1), ('R_hole_1', self.R_hole_1), ('R_tear_2', self.R_tear_2), ('R_hole_2', self.R_hole_2),
            ('L_tear_1', self.L_tear_1), ('L_hole_1', self.L_hole_1), ('L_tear_2', self.L_tear_2), ('L_hole_2', self.L_hole_2),
            ('right_extreme', self.right_extreme), ('left_extreme', self.left_extreme),
            ('ear_special', self.ear_special), ('body_special', self.body_special)
        ]

    def initialize_from_one_hot(self, one_hot_vector):
        decoded_attributes = []
        index = 0
        for attr_name in SEEK.attribute_names:
            one_hot_slice = one_hot_vector[index:index + SEEK.lengths[attr_name]]
            decoded_value = SEEK.mappings[attr_name][torch.argmax(one_hot_slice).item()]
            decoded_attributes.append(decoded_value)
            index += SEEK.lengths[attr_name]

        seek_code = (
            f"{decoded_attributes[0]}{decoded_attributes[1]}T{decoded_attributes[2]}{decoded_attributes[3]}"
            f"E{decoded_attributes[4]}{decoded_attributes[5]}{decoded_attributes[6]}{decoded_attributes[7]}-"
            f"{decoded_attributes[8]}{decoded_attributes[9]}{decoded_attributes[10]}{decoded_attributes[11]}"
            f"X{decoded_attributes[12]}{decoded_attributes[13]}S{decoded_attributes[14]}{decoded_attributes[15]}"
        )
        self.initialize_from_string(seek_code)

    def __getitem__(self, index):
        attr_name, attr_value = self.attributes[index]
        mapping = SEEK.mappings[attr_name]
        one_hot_index = mapping.index(attr_value)
        return torch.eye(len(mapping))[one_hot_index]

    def one_hot_encode(self):
        one_hot_vector = [self[i] for i in range(len(self.attributes))]
        return torch.cat(one_hot_vector)

    def __str__(self):
        return (
            f"{self.sex}{self.age}T{self.right_tusk}{self.left_tusk}E{self.R_tear_1}{self.R_hole_1}{self.R_tear_2}{self.R_hole_2}-"
            f"{self.L_tear_1}{self.L_hole_1}{self.L_tear_2}{self.L_hole_2}X{self.right_extreme}{self.left_extreme}"
            f"S{self.ear_special}{self.body_special}"
        )
    
    def closest_valid_one_hot(prob_vector):
        """
        Converts a probability tensor into the closest valid one-hot encoded representation for each attribute.
        """
        closest_one_hot = []
        index = 0

        for group in SEEK.attribute_names:
            slice_length = SEEK.lengths[group]
            prob_slice = prob_vector[:, index:index + slice_length]
            one_hot_slice = torch.zeros_like(prob_slice)
            
            max_indices = torch.argmax(prob_slice, dim=1)
            one_hot_slice[torch.arange(prob_slice.size(0)), max_indices] = 1.0
            closest_one_hot.append(one_hot_slice)
            
            index += slice_length

        return torch.cat(closest_one_hot, dim=1)


# Updated test function
def test_seek():
    # Example SEEK code
    test_code = "B00T__E8000-0000X0_S00"
    
    # Test initializing from string
    seek_instance = SEEK(test_code)
    print("Original SEEK code from string:", seek_instance)
    
    # Test one-hot encoding
    one_hot_vector = seek_instance.one_hot_encode()
    print("\nOne-hot encoded vector:\n", one_hot_vector)
    
    # Test initializing from one-hot vector
    parsed_seek_instance = SEEK(one_hot_vector)
    print("\nParsed SEEK code from one-hot:", parsed_seek_instance)
    
    # Validate that the parsed code matches the original code
    assert str(seek_instance) == str(parsed_seek_instance), "Parsed SEEK code does not match original code"
    print("\nTest passed: Parsed code matches the original code.")

# Run the test function if the script is executed directly
if __name__ == "__main__":
    test_seek()
