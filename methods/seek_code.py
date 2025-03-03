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
        
        reconstructed_vector = whole_vector[:, 0:11]
        reconstructed_vector = torch.cat((reconstructed_vector, right_vector), dim=1)
        reconstructed_vector = torch.cat((reconstructed_vector, left_vector), dim=1)
        reconstructed_vector = torch.cat((reconstructed_vector, whole_vector[:, 11:]), dim=1)

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
        print('parsed from one hot', SEEK(one_hot_vector))

        whole_one_hot, left_one_hot, right_one_hot = SEEK.separate_one_hot(one_hot_vector)
        print("Whole one-hot vector:\n", whole_one_hot)
        print("Left one-hot vector:\n", left_one_hot)
        print("Right one-hot vector:\n", right_one_hot)
        
        reconstructed_one_hot = SEEK.rescontruct_from_separated_prob_vectors(whole_one_hot, left_one_hot, right_one_hot)
        print('reconstructed_one_hot',reconstructed_one_hot)

        print("Reconstructed SEEK:\n", SEEK(reconstructed_one_hot))
        assert str(seek_instance) == str(SEEK(reconstructed_one_hot)), "Reconstructed instance does not match original"
        print("Test passed: Reconstructed instance matches the original")
    
    def aggregate_seek(SEEK_codes, ele_ids):
        def make_seek_df(SEEK_codes, ele_ids):
            data = []
            seeks = []

            for i in range(SEEK_codes.shape[0]):
                row = SEEK(SEEK_codes[i]).categorical_tensor()[0].tolist()
                data.append(row)
                seeks.append(SEEK(SEEK_codes[i]).__str__())
            import pandas as pd
            df = pd.DataFrame(data, columns=SEEK.attribute_names)

            df['subj_seek_str'] = seeks
            if isinstance(ele_ids, torch.Tensor):
                df['ele_id'] = ele_ids.cpu()
            else: 
                df['ele_id'] = ele_ids
                
            df['encounter_id'] = 1

            return df

        # Collect SEEK codes
        df = make_seek_df(SEEK_codes, ele_ids)

        # Copy DataFrame for mapped values
        df_mapped = df.copy()

        # Get mapping dictionary
        mapping = SEEK.mappings

        # Apply mapping correctly (avoiding index shift)
        for col in df.columns:
            if col in mapping:  # Ensure column exists in mapping
                max_idx = len(mapping[col]) - 1
                df_mapped[col] = df[col].apply(lambda x: mapping[col][x] if 0 <= x <= max_idx else None)


        ### HEURISTIC RULES (copied from the original code)
        def get_val(p_counts, u, most_likely, other, cutoff=.8, fraction=1):
            if u in p_counts and p_counts[u] > cutoff:
                return u
            if other in p_counts and most_likely in p_counts:
                if p_counts[other] > (p_counts[most_likely] * fraction):
                    return other
                else:
                    return most_likely
            if other in p_counts:
                return other
            if most_likely in p_counts:
                return most_likely
            return None

        def get_sex_val(p): return get_val(p, u='_', most_likely='B', other='C', fraction=2)
        def get_age_val(p): return get_val(p, u='_', most_likely='00', other='20')
        def get_tusk_val(p): return get_val(p, u='_', most_likely='1', other='0')

        def get_tear_hole_val(p, cutoff=.9, frac=2):
            u, z = '_', '0'
            if u in p and p[u] > cutoff: return u
            other_sum = sum(value for key, value in p.items() if key != z and key != u)
            if z in p and p[z] > (other_sum * frac): return z
            return max((k for k in p if k not in {u, z}), key=lambda k: p[k], default=None)

        def get_feature_val(p): return get_val(p, u='1', most_likely='0', other='_', fraction=1.5)

        ### FUNCTION TO GENERATE SEEK CODE STRING
        def generate_seek_code(row):
            """Generates a SEEK code string from DataFrame attributes for each row."""
            return (
                f"{row['sex']}{row['age']}T{row['right_tusk']}{row['left_tusk']}"
                f"E{row['R_tear_1']}{row['R_hole_1']}{row['R_tear_2']}{row['R_hole_2']}-"
                f"{row['L_tear_1']}{row['L_hole_1']}{row['L_tear_2']}{row['L_hole_2']}X"
                f"{row['right_extreme']}{row['left_extreme']}S{row['ear_special']}{row['body_special']}"
            )

        ### FUNCTION TO AGGREGATE SEEK CODES AT ELEPHANT LEVEL
        def aggregate_elephant_seek_codes(df):
            """Aggregates subject-level SEEK codes into elephant-level SEEK codes while keeping image order."""
            
            ### FUNCTION TO GENERATE SEEK CODE STRING
        
            elephant_seek_dict = {}

            for ele_id, group in df.groupby('ele_id'):
                if ele_id not in elephant_seek_dict:
                    elephant_seek_dict[ele_id] = {}

                for col in df.columns:
                    if col not in ['ele_id', 'encounter_id']:
                        counts = group[col].value_counts(normalize=True).to_dict()

                        if col == 'sex':
                            elephant_seek_dict[ele_id][col] = get_sex_val(counts)
                        elif col == 'age':
                            elephant_seek_dict[ele_id][col] = get_age_val(counts)
                        elif col in ['right_tusk', 'left_tusk']:
                            elephant_seek_dict[ele_id][col] = get_tusk_val(counts)
                        elif col in ['R_tear_1', 'R_hole_1', 'L_tear_1', 'L_hole_1']:
                            elephant_seek_dict[ele_id][col] = get_tear_hole_val(counts, cutoff=.9, frac=2)
                        elif col in ['R_tear_2', 'R_hole_2', 'L_tear_2', 'L_hole_2']:
                            elephant_seek_dict[ele_id][col] = get_tear_hole_val(counts, cutoff=.95, frac=2.5)
                        elif col in ['right_extreme', 'left_extreme', 'ear_special', 'body_special']:
                            elephant_seek_dict[ele_id][col] = get_feature_val(counts)
            import pandas as pd
            elephant_seek_df = pd.DataFrame.from_dict(elephant_seek_dict, orient='index')
            elephant_seek_df.index.name = 'ele_id'

            # Merge back the aggregated SEEK codes into the original DataFrame (keeping original order)
            merged_df = df.copy()
            for col in elephant_seek_df.columns:
                merged_df[col] = merged_df['ele_id'].map(elephant_seek_df[col])

            # Generate the SEEK code for each row
            merged_df["SEEK_code"] = merged_df.apply(generate_seek_code, axis=1)


            #print([SEEK(seek).one_hot_encode() for seek in merged_df["SEEK_code"]])
            ele_seek_list = [SEEK(seek).one_hot_encode() for seek in merged_df["SEEK_code"]]
            subject_seek_list = [SEEK(seek).one_hot_encode() for seek in df["subj_seek_str"]]
            ele_id = merged_df["ele_id"].tolist()

            ele_seek = torch.stack(ele_seek_list)
            subject_seek = torch.stack(subject_seek_list)

            return subject_seek, ele_seek, ele_id
        return aggregate_elephant_seek_codes(df_mapped)
    
    def perfect_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
        return subject_SEEK

    def oracle_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
        return elephant_SEEK


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
