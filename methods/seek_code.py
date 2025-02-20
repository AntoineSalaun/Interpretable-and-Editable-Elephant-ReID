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

    whole_indices = [0, 1, 2, 3, 12, 13, 14, 15]  # Indices for whole image attributes
    left_indices = [8, 9, 10, 11]     # Indices for left ear attributes
    right_indices = [4, 5, 6, 7]      # Indices for right ear attributes

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
                # Check if the vector has the correct length
        if one_hot_vector.shape[-1] != 63:
            raise ValueError(f"Input vector must have length 63, got {one_hot_vector.size(1)}")
        
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
            
        # Check if the vector has the correct length
        if prob_vector.size(1) != 63:
            raise ValueError(f"Input vector must have length 63, got {prob_vector.size(1)}")

        for group in SEEK.attribute_names:
            slice_length = SEEK.lengths[group]
            prob_slice = prob_vector[:, index:index + slice_length]
            one_hot_slice = torch.zeros_like(prob_slice)
            
            max_indices = torch.argmax(prob_slice, dim=1)
            one_hot_slice[torch.arange(prob_slice.size(0)), max_indices] = 1.0
            closest_one_hot.append(one_hot_slice)
            
            index += slice_length

        return torch.cat(closest_one_hot, dim=1)
    
    def categorical_tensor(self):
        """
        Converts the SEEK instance into a tensor of categorical values.
        """
        categorical_values = []
        for attr_name in SEEK.attribute_names:
            mapping = SEEK.mappings[attr_name]
            attr_value = getattr(self, attr_name)
            one_hot_index = mapping.index(attr_value)
            categorical_values.append(one_hot_index)

        cat_tensor = torch.tensor(categorical_values, dtype=torch.long)

        return cat_tensor, cat_tensor[self.whole_indices], cat_tensor[self.left_indices], cat_tensor[self.right_indices]
    
    def batch_categorical(labels):
        cat_labels, whole_labels, left_labels, right_labels = [],[],[],[]
        for x in labels:
            seek = SEEK(x)
            k, w, l, r = seek.categorical_tensor()
            cat_labels.append(k)
            whole_labels.append(w)
            left_labels.append(l)
            right_labels.append(r)

        return torch.stack(cat_labels).to('cuda'), torch.stack(whole_labels).to('cuda'), torch.stack(left_labels).to('cuda'), torch.stack(right_labels).to('cuda')
    
        
    @staticmethod
    def separate_one_hot(labels_1h):
        # Check if the input is a tensor        
        if labels_1h.dim() == 1:
            labels_1h = labels_1h.unsqueeze(0)  # Convert to batch format if single instance
        
        whole_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        right_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        left_indices = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        
        if labels_1h.size(1) != 63:
            raise ValueError(f"Input vector must have length 63, got {labels_1h.size(1)}")

        whole_vector = labels_1h[:, whole_indices]
        left_vector = labels_1h[:, left_indices]
        right_vector = labels_1h[:, right_indices]
        
        return whole_vector, left_vector, right_vector
    
    def rescontruct_from_separated_prob_vectors(whole_vector, left_vector, right_vector):
        if (whole_vector.size(1) != 23 and left_vector.size(1) != 20 and right_vector.size(1) != 20) and (whole_vector.size(1) != 6 and left_vector.size(1) != 5 and right_vector.size(1) != 5):
            raise ValueError(f"Input vector must have length 23, 20 and 20 OR 6, 5 amd 5 - got {whole_vector.size(1)}, {left_vector.size(1)} and {right_vector.size(1)}")
        
        if (whole_vector.size(1) != 23 and left_vector.size(1) != 20 and right_vector.size(1) != 20): 
            reconstructed_vector = whole_vector[:, 0:11]
            reconstructed_vector = torch.cat((reconstructed_vector, right_vector), dim=1)
            reconstructed_vector = torch.cat((reconstructed_vector, left_vector), dim=1)
            reconstructed_vector = torch.cat((reconstructed_vector, whole_vector[:, 11:]), dim=1)
        elif (whole_vector.size(1) != 6 and left_vector.size(1) != 5 and right_vector.size(1) != 5):
            reconstructed_vector = whole_vector[:, 0:4]
            reconstructed_vector = torch.cat((reconstructed_vector, right_vector), dim=1)
            reconstructed_vector = torch.cat((reconstructed_vector, left_vector), dim=1)
            reconstructed_vector = torch.cat((reconstructed_vector, whole_vector[:, 4:]), dim=1)

        if whole_vector.shape[0] == 1:
            reconstructed_vector = reconstructed_vector.squeeze(0)
        
        return reconstructed_vector


    # Test function for separate_one_hot and from_separated_prob_vectors
    def test_separate_and_reconstruct():
        test_code = "B00T__E8000-0000X0_S00"
        seek_instance = SEEK(test_code)
        print(seek_instance)
        one_hot_vector = seek_instance.one_hot_encode()
        print("Original one-hot :\n", one_hot_vector)

        whole_one_hot, left_one_hot, right_one_hot = SEEK.separate_one_hot(one_hot_vector)
        print("Whole one-hot vector:\n", whole_one_hot)
        print("Left one-hot vector:\n", left_one_hot)
        print("Right one-hot vector:\n", right_one_hot)
        
        reconstructed_one_hot = SEEK.rescontruct_from_separated_prob_vectors(whole_one_hot, left_one_hot, right_one_hot)
        print('reconstructed_one_hot',reconstructed_one_hot)

        print("Reconstructed SEEK:\n", SEEK(reconstructed_one_hot))
        assert str(seek_instance) == str(SEEK(reconstructed_one_hot)), "Reconstructed instance does not match original"
        print("Test passed: Reconstructed instance matches the original")
        

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
    #test_seek()
    test_separate_and_reconstruct()
