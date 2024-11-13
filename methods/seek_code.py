import torch

class SEEK:
    def __init__(self, code):
        # Initialize and validate attributes as before
        self.sex = code[0]
        self.age = code[1] + code[2]
        self.R_tusk = code[4]
        self.L_tusk = code[5]
        self.R_tear_1 = code[7]
        self.R_hole_1 = code[8]
        self.R_tear_2 = code[9]
        self.R_hole_2 = code[10]
        self.L_tear_1 = code[12]
        self.L_hole_1 = code[13]
        self.L_tear_2 = code[14]
        self.L_hole_2 = code[15]
        self.R_extreme = code[17]
        self.L_extreme = code[18]
        self.ear_special = code[20]
        self.body_special = code[21]
        
        # Define mappings for each attribute
        self.mappings = {
            'gender': ['B', 'C', '_'],
            'age': ['00', '20'],
            'tusk': ['0', '_', '1'],
            'left_ear': ['0', '_', '3', '4', '5'],
            'right_ear': ['0', '_', '7', '8', '9'],
            'extreme': ['0', '_', '1'],
            'special': ['0', '_', '1']
        }
        
        # Store attributes in a list to access by index
        self.attributes = [
            ('gender', self.sex), 
            ('age', self.age), 
            ('right_tusk', self.R_tusk), 
            ('left_tusk', self.L_tusk),
            ('right_ear', self.R_tear_1), 
            ('right_ear', self.R_hole_1), 
            ('right_ear', self.R_tear_2), 
            ('right_ear', self.R_hole_2),
            ('left_ear', self.L_tear_1), 
            ('left_ear', self.L_hole_1), 
            ('left_ear', self.L_tear_2), 
            ('left_ear', self.L_hole_2),
            ('right_ear_extreme', self.R_extreme), 
            ('left_ear_extreme', self.L_extreme), 
            ('ear_special', self.ear_special), 
            ('body_special', self.body_special)
        ]

    def __getitem__(self, index):
        """
        Returns the one-hot encoded representation of the specified attribute at the given index.
        
        Parameters:
            index (int): Index of the attribute in the SEEK code.
            
        Returns:
            torch.Tensor: One-hot encoded representation of the attribute.

        ### One-Hot Encoding Scheme for SEEK Code Attributes

        - **Gender**: 3 values
        - `[1, 0, 0]` for `B`
        - `[0, 1, 0]` for `C`
        - `[0, 0, 1]` for `_`

        - **Age**: 2 values
        - `[1, 0]` for `00`
        - `[0, 1]` for `20`

        - **Tusk (L and R)**: 3 values
        - `[1, 0, 0]` for `0`
        - `[0, 1, 0]` for `_`
        - `[0, 0, 1]` for `1`

        - **Left Ear (Tear/Hole)**: 5 values
        - `[1, 0, 0, 0, 0]` for `0`
        - `[0, 1, 0, 0, 0]` for `_`
        - `[0, 0, 1, 0, 0]` for `3`
        - `[0, 0, 0, 1, 0]` for `4`
        - `[0, 0, 0, 0, 1]` for `5`

        - **Right Ear (Tear/Hole)**: 5 values
        - `[1, 0, 0, 0, 0]` for `0`
        - `[0, 1, 0, 0, 0]` for `_`
        - `[0, 0, 1, 0, 0]` for `7`
        - `[0, 0, 0, 1, 0]` for `8`
        - `[0, 0, 0, 0, 1]` for `9`

        - **Extreme (L and R)**: 3 values
        - `[1, 0, 0]` for `0`
        - `[0, 1, 0]` for `_`
        - `[0, 0, 1]` for `1`

        - **Special (Ear and Body)**: 3 values
        - `[1, 0, 0]` for `0`
        - `[0, 1, 0]` for `_`
        - `[0, 0, 1]` for `1`

        """
        # Extract attribute name and value by index
        attr_name, attr_value = self.attributes[index]
        
        # Get the one-hot encoding for the attribute
        mapping = self.mappings[attr_name]
        one_hot_index = mapping.index(attr_value)
        one_hot_tensor = torch.eye(len(mapping))[one_hot_index]
        
        return one_hot_tensor

    def one_hot_encode(self):
        """
        Returns the full one-hot encoded representation of the SEEK code.
        
        Returns:
            torch.Tensor: A concatenated one-hot encoded vector for all attributes.
        """
        # Concatenate one-hot encodings of all attributes
        one_hot_vector = [self[i] for i in range(len(self.attributes))]
        return torch.cat(one_hot_vector)
    
    def __str__(self):
        """
        Returns the full SEEK code in its original string format.
        
        Returns:
            str: Reconstructed SEEK code string.
        """
        return f"{self.sex}{self.age}T{self.R_tusk}{self.L_tusk}E{self.R_tear_1}{self.R_hole_1}{self.R_tear_2}{self.R_hole_2}-" \
               f"{self.L_tear_1}{self.L_hole_1}{self.L_tear_2}{self.L_hole_2}X{self.R_extreme}{self.L_extreme}" \
               f"S{self.ear_special}{self.body_special}"




if __name__ == "__main__":   
    # Create SEEK instance
    print('creating instance by doing SEEK(B00T__E8000-0000X0_S00)')
    seek_instance = SEEK("B00T__E8000-0000X0_S00")
    
    # Test one-hot encoding of each attribute using __getitem__
    print("\nOne-hot encoding for each attribute:")
    for i in range(len(seek_instance.attributes)):
        print(f"Attribute {i} ({seek_instance.attributes[i][0]}): {seek_instance[i]}")
    
    # Test full one-hot encoding
    full_one_hot = seek_instance.one_hot_encode()
    print("\nFull one-hot encoded vector:")
    print(full_one_hot)
    print("\nLength of one-hot encoded vector:", len(full_one_hot))

    # Test string representation
    print("\nString representation of SEEK code is maintained when using print (seek_instance):")
    print(seek_instance)















'''
class SEEK:
    def __init__(self, code):
        """
        Initialize the class with a list of SEEK codes.
        """
        #'B00T__E8000-0000X0_S00'

        if len(code) != 22:
            raise ValueError("wrongly-sized SEEK code ! It is + ", len(code), " characters long, but should be 23 characters long")
        if code[3] != 'T':
            raise ValueError("wrongly formatted SEEK code, missing a T in position 4")
        if code[6] != 'E':    
            raise ValueError("wrongly formatted SEEK code, missing an E in position 7")
        if code[11] != '-':    
            raise ValueError("wrongly formatted SEEK code, missing a - in position 12")
        if code[16] != 'X':
            raise ValueError("wrongly formatted SEEK code, missing an X in position 17")
        if code[19] != 'S':
            raise ValueError("wrongly formatted SEEK code, missing an S in position 20")

        # General
        self.sex = code[0]
        self.age = code[1] + code[2] #To-do : make sure that they always join as strings and not add as number (if inputed as number)
        # Tusks
        self.R_tusk = code[4]
        self.L_tusk = code[5]
        # Ears
        self.R_tear_1 = code[7]
        self.R_hole_1 = code[8]
        self.R_tear_2 = code[9]
        self.R_hole_2 = code[10]
        #-
        self.L_tear_1 = code[12]
        self.L_hole_1 = code[13]
        self.L_tear_2 = code[14]
        self.L_hole_2 = code[15]
        # Extreme
        self.R_extreme = code[17]
        self.L_extreme = code[18]
        # Special
        self.ear_special = code[20]
        self.body_special = code[21]

        self.seek_list = [self.sex, self.age, self.R_tusk, self.L_tusk, self.R_tear_1, self.R_hole_1, self.R_tear_2, self.R_hole_2, self.L_tear_1, self.L_hole_1, self.L_tear_2, self.L_hole_2, self.R_extreme, self.L_extreme, self.ear_special, self.body_special]


    def __getitem__(self, index):
        """
        Access SEEK codes by index. Such that each index corresponds to a meaningful part of the SEEK code.
        """
        # Define possible values and mappings for each attribute
        mappings = {
            'gender': ['B', 'C', '_'],
            'age': ['00', '20'],
            'tusk': ['0', '_', '1'],
            'left_ear': ['0', '_', '3', '4', '5'],
            'right_ear': ['0', '_', '7', '8', '9'],
            'extreme': ['0', '_', '1'],
            'special': ['0', '_', '1']
        }
        
        # Initialize one-hot encoding list
        one_hot_vector = []
        
        # Map each attribute in the SEEK list to its one-hot encoding
        one_hot_vector += list(np.eye(len(mappings['gender']))[mappings['gender'].index(seek_list[0])])
        one_hot_vector += list(np.eye(len(mappings['age']))[mappings['age'].index(seek_list[1])])
        
        for i in [2, 3]:  # R_tusk, L_tusk
            one_hot_vector += list(np.eye(len(mappings['tusk']))[mappings['tusk'].index(seek_list[i])])

        for i in [4, 5, 6, 7]:  # Right ear tears and holes
            one_hot_vector += list(np.eye(len(mappings['right_ear']))[mappings['right_ear'].index(seek_list[i])])

        for i in [8, 9, 10, 11]:  # Left ear tears and holes
            one_hot_vector += list(np.eye(len(mappings['left_ear']))[mappings['left_ear'].index(seek_list[i])])

        for i in [12, 13]:  # R_extreme, L_extreme
            one_hot_vector += list(np.eye(len(mappings['extreme']))[mappings['extreme'].index(seek_list[i])])

        for i in [14, 15]:  # ear_special, body_special
            one_hot_vector += list(np.eye(len(mappings['special']))[mappings['special'].index(seek_list[i])])

        return one_hot_vector

    def __str__(self):
        """
        Print representation of the SEEK.
        """
        return f"{self.sex}{self.age}T{self.R_tusk}{self.L_tusk}E{self.R_tear_1}{self.R_hole_1}{self.R_tear_2}{self.R_hole_2}-{self.L_tear_1}{self.L_hole_1}{self.L_tear_2}{self.L_hole_2}X{self.R_extreme}{self.L_extreme}S{self.ear_special}{self.body_special}"
'''
