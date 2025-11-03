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
    
        
    #@staticmethod
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
    
    def aggregate_seek(SEEK_codes, ele_ids, rules = None):

        if rules is None: rules = {
            'sex': {'cutoff': 0.9334337305774126, 'fraction': 0.7151849388533961},
            'age': {'cutoff': 0.9514760451994138, 'fraction': 0.5510405823660589},
            'tusks': {'cutoff': 0.4369025581005079, 'fraction': 1.8715870244928936},
            'ear_most_prominent':{'cutoff': 0.8346892883784236, 'fraction': 2.432201122842605},
            'ear_least_prominent': {'cutoff': 0.3902627093863492, 'fraction': 2.0680981200569604},
            'extremes': {'cutoff': 0.46972912427893604, 'fraction': 0.8622955318459603}
            }
        import pandas as pd

        def make_seek_df(SEEK_codes, ele_ids):
            data, seeks = [], []
            for i in range(SEEK_codes.shape[0]):
                row = SEEK(SEEK_codes[i]).categorical_tensor()[0].tolist()
                data.append(row)
                seeks.append(SEEK(SEEK_codes[i]).__str__())

            df = pd.DataFrame(data, columns=SEEK.attribute_names)
            df['subj_seek_str'] = seeks
            df['ele_id'] = ele_ids.cpu() if isinstance(ele_ids, torch.Tensor) else ele_ids
            df['encounter_id'] = 1
            return df

        df = make_seek_df(SEEK_codes, ele_ids)
        df_mapped = df.copy()
        mapping = SEEK.mappings
        for col in df.columns:
            if col in mapping:
                max_idx = len(mapping[col]) - 1
                df_mapped[col] = df[col].apply(lambda x: mapping[col][x] if 0 <= x <= max_idx else None)

        unknown_tokens = {'age': '__'}

        def _normalized_counts(series, attr):
            unknown = unknown_tokens.get(attr, '_')
            cleaned = series.fillna(unknown).replace({None: unknown})
            counts = cleaned.value_counts(normalize=True, dropna=False)
            return {str(key): round(float(value), 3) for key, value in counts.items()}

        def generate_seek_code(row):
            return (
                f"{row['sex']}{row['age']}T{row['right_tusk']}{row['left_tusk']}"
                f"E{row['R_tear_1']}{row['R_hole_1']}{row['R_tear_2']}{row['R_hole_2']}-"
                f"{row['L_tear_1']}{row['L_hole_1']}{row['L_tear_2']}{row['L_hole_2']}X"
                f"{row['right_extreme']}{row['left_extreme']}S{row['ear_special']}{row['body_special']}"
            )

        def aggregate_elephant_seek_codes(df):
            elephant_seek_dict = {}
            for ele_id, group in df.groupby('ele_id'):
                elephant_seek_dict[ele_id] = {}
                for attr in SEEK.attribute_names:
                    counts = _normalized_counts(group[attr], attr)
                    elephant_seek_dict[ele_id][attr] = SEEK.aggregation_heuristic(attr, counts, rules)

            elephant_seek_df = pd.DataFrame.from_dict(elephant_seek_dict, orient='index')
            elephant_seek_df.index.name = 'ele_id'

            merged_df = df.copy()
            for attr in SEEK.attribute_names:
                merged_df[attr] = merged_df['ele_id'].map(elephant_seek_df[attr])
            merged_df["SEEK_code"] = merged_df.apply(generate_seek_code, axis=1)

            ele_seek_list     = [SEEK(seek).one_hot_encode() for seek in merged_df["SEEK_code"]]
            subject_seek_list = [SEEK(seek).one_hot_encode() for seek in df["subj_seek_str"]]

            _device = ele_ids.device
            _dtype  = ele_ids.dtype  # usually torch.long
            subject_seek = torch.stack(subject_seek_list).to(_device)
            ele_seek     = torch.stack(ele_seek_list).to(_device)
            ele_id = torch.as_tensor(merged_df["ele_id"].to_numpy(), dtype=_dtype, device=_device)
            return subject_seek, ele_seek, ele_id
        return aggregate_elephant_seek_codes(df_mapped)

    @staticmethod
    def aggregation_heuristic(attr, counts, rules):
        attr_key = attr.lower()
        default = '__' if attr_key == 'age' else '_'
        if not counts:
            return default

        def get_val(u, most, other, cfg):
            cutoff, fraction = cfg.get('cutoff', 0.8), cfg.get('fraction', 1)
            if counts.get(u, 0) > cutoff:
                return u
            if counts.get(other, 0) and counts.get(most, 0):
                if counts[other] > counts[most] * fraction:
                    return other
                return most
            if counts.get(other, 0):
                return other
            if counts.get(most, 0):
                return most
            return u

        def get_tear(cfg):
            u, z = '_', '0'
            cutoff, fraction = cfg.get('cutoff', 0.9), cfg.get('fraction', 2)
            if counts.get(u, 0) > cutoff:
                return u
            other_sum = sum(val for key, val in counts.items() if key not in {u, z})
            if counts.get(z, 0) > other_sum * fraction:
                return z
            best = max((key for key in counts if key not in {u, z}), key=lambda k: counts[k], default=None)
            return best if best is not None else u

        if attr_key == 'sex':
            return get_val('_', 'B', 'C', rules['sex'])
        if attr_key == 'age':
            pick = get_val('__', '00', '20', rules['age'])
            return pick if pick in {'00', '20'} else '00'
        if attr_key in {'right_tusk', 'left_tusk', 'r_tusk', 'l_tusk'}:
            return get_val('_', '1', '0', rules['tusks'])
        if 'tear_1' in attr_key or 'hole_1' in attr_key:
            return get_tear(rules['ear_most_prominent'])
        if 'tear_2' in attr_key or 'hole_2' in attr_key:
            return get_tear(rules['ear_least_prominent'])
        if 'extreme' in attr_key:
            return get_val('1', '0', '_', rules['extremes'])
        if 'special' in attr_key:
            return get_val('1', '0', '_', {'cutoff': 0.8, 'fraction': 1.5})

        best = max(counts.items(), key=lambda item: item[1])[0]
        return best if best is not None else default

    @staticmethod
    def aggregate_from_vote_to_SEEK(
        seek_counts_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/SEEK_dict.json',
        rules=None,
        sub_ele_pairs_path='/archive/vision/beery/animal_reid/datasets/elephants_zooniverse/ele_id_project/data/out_apr2/sub_ele_pairs.json',
        orinal_image_dict_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_original.csv',
        export_path=None,
        drop_missing_ears=True
    ):
        import json
        import os
        import pandas as pd

        if rules is None:
                {
                'sex': {'cutoff': 0.7238117020284947, 'fraction': 3.2846153859098934},
                'age': {'cutoff': 0.6080374539126856, 'fraction': 3.030614746927617},
                'tusks': {'cutoff': 0.7864963346562002, 'fraction': 1.3312930166044628},
                'ear_most_prominent':{'cutoff': 0.2133936013358265, 'fraction': 1.788490949619698},
                'ear_least_prominent': {'cutoff': 0.1001741505264265, 'fraction': 1.6865733430829595},
                'extremes': {'cutoff': 0.9641272649686263, 'fraction': 0.9361114173246684}
                }

        with open(seek_counts_path, 'r', encoding='utf-8') as handle:
            seek_counts = json.load(handle)
        with open(sub_ele_pairs_path, 'r', encoding='utf-8') as handle:
            sub_ele_pairs = json.load(handle)

        aggregated = {}
        for subject_id, features in seek_counts.items():
            aggregated[subject_id] = {}
            for feature, counts in features.items():
                total = sum(counts.values())
                if not total:
                    aggregated[subject_id][feature] = None
                    continue
                percentages = {val: round(count / total, 3) for val, count in counts.items()}
                aggregated[subject_id][feature] = SEEK.aggregation_heuristic(feature, percentages, rules)

        image_seek_df = pd.DataFrame(aggregated).T
        image_seek_df.index.name = 'subject_id'

        def build_seek(row):
            return f"{row['Sex']}{row['Age']}T{row['R_tusk']}{row['L_tusk']}E{row['R_tear_1']}{row['R_hole_1']}{row['R_tear_2']}{row['R_hole_2']}-{row['L_tear_1']}{row['L_hole_1']}{row['L_tear_2']}{row['L_hole_2']}X{row['R_extreme']}{row['L_extreme']}S{row['Special_ear']}{row['Special_body']}"

        image_seek_df['SEEK'] = image_seek_df.apply(build_seek, axis=1).drop(image_seek_df.index[:3])

        image_seek_df['ele_id'] = image_seek_df.index.map(lambda idx: sub_ele_pairs.get(str(idx)))

        merged_df = pd.read_csv(orinal_image_dict_path)

        if 'subject-SEEK' in merged_df.columns and 'old-subject-SEEK' not in merged_df.columns:
            merged_df = merged_df.rename(columns={'subject-SEEK': 'old-subject-SEEK'})
        else:
            raise ValueError("Input DataFrame must contain 'subject-SEEK' column but contains 'old-subject-SEEK' column.")

        image_seek_df = image_seek_df.copy()
        image_seek_df.index = image_seek_df.index.astype(int)
        merged_df = merged_df.merge(image_seek_df[['SEEK']], left_on='subject_id', right_index=True, how='left')
        merged_df = merged_df.rename(columns={'SEEK': 'subject-SEEK'})

        if 'old-subject-SEEK' in merged_df.columns and 'subject-SEEK' in merged_df.columns:
            cols = list(merged_df.columns)
            old_idx, new_idx = cols.index('old-subject-SEEK'), cols.index('subject-SEEK')
            if old_idx > new_idx:
                cols[new_idx], cols[old_idx] = cols[old_idx], cols[new_idx]
                merged_df = merged_df[cols]

        if drop_missing_ears and {'right_ear_path', 'left_ear_path'}.issubset(merged_df.columns):
            def remove_ears(df):
                df = df.copy()
                right_missing = df['right_ear_path'].isna() | df['right_ear_path'].astype(str).str.lower().eq('nan')
                left_missing = df['left_ear_path'].isna() | df['left_ear_path'].astype(str).str.lower().eq('nan')
                valid_right = right_missing & df['subject-SEEK'].notna()
                valid_left = left_missing & df['subject-SEEK'].notna()
                df.loc[valid_right, 'subject-SEEK'] = (
                    df.loc[valid_right, 'subject-SEEK'].astype(str).str.slice_replace(7, 11, '____')
                )
                df.loc[valid_left, 'subject-SEEK'] = (
                    df.loc[valid_left, 'subject-SEEK'].astype(str).str.slice_replace(12, 16, '____')
                )
                return df

            merged_df = remove_ears(merged_df)

        if export_path:
            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            target_df = merged_df if merged_df is not None else image_seek_df
            target_df.to_csv(export_path, index=False if merged_df is not None else True)

        return merged_df, image_seek_df
    
    def perfect_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
        return subject_SEEK

    def oracle_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
        return elephant_SEEK

    def correct_or_soft(concept_logits, subject_SEEK, elephant_SEEK, p=0.5):
            # Randomly choose between oracle correction and perfect correction
            if torch.rand(1) < p:
                return subject_SEEK
            else:
                return concept_logits

    def correct_or_hard_at_prob(concept_logits, subject_SEEK, elephant_SEEK, p=0.5):
            # Randomly choose between oracle correction and perfect correction
            if torch.rand(1) < p:
                return subject_SEEK
            else:
                return SEEK.closest_valid_one_hot(concept_logits)
            

    def correct_or_hard(concept_logits, subject_SEEK, elephant_SEEK):
            # Randomly choose between oracle correction and perfect correction
            if torch.rand(1) < 0.5:
                return subject_SEEK
            else:
                return SEEK.closest_valid_one_hot(concept_logits)

    def hard(concept_logits, subject_SEEK, elephant_SEEK):
        return SEEK.closest_valid_one_hot(concept_logits)

    def perfect(concept_logits, subject_SEEK, elephant_SEEK):
        return elephant_SEEK
    
    def distance(query_elephant,galery_elephant):
        #print('query_elephant', query_elephant)
        #print('galery_elephant', galery_elephant)

        distance = 0
        for i in range(len(SEEK.attribute_names)):
            attr_name = SEEK.attribute_names[i]
            q_attr = getattr(query_elephant, attr_name)
            g_attr = getattr(galery_elephant, attr_name)
            
            if q_attr == g_attr:
                #print(f"Attribute {attr_name} is correct: query {q_attr}, galery {g_attr} -> +0 distance")
                distance += 0
            else: # There is an error !
                if q_attr == '_' or g_attr == '_': # One of the SEEK attribute is unknown, small error
                    #print(f"Attribute {attr_name} is a unknown in one of the elephants: query {q_attr}, galery {g_attr} -> +0.5 distance")
                    distance += 0.5
                else: # Both attributes are known but different, real error !
                    if (attr_name in ['R_tear_1', 'R_hole_1', 'L_tear_1', 'L_hole_1', 'R_tear_2', 'R_hole_2', 'L_tear_2', 'L_hole_2']): # There might be confusion on these attributes, the distance is proportional to the penalty
                        #print(f"Attribute {attr_name} is a in-ear distance error: query {q_attr}, galery {g_attr} -> + (a - b)/2 distance")
                        distance += abs(float(q_attr) - float(g_attr)) / 2
                    elif (attr_name in ['age', 'sex', 'right_tusk','left_tusk']): # We should be confident on these attributes, big error counted twice
                        #print(f"Attribute {attr_name} is an error on an easy attribute : query {q_attr}, galery {g_attr} -> +2 distance")
                        distance += 2
                    else: # Other errors count for 1
                        #print(f"Attribute {attr_name} is an error: query {q_attr}, galery {g_attr} -> +1 distance")
                        distance += 1
        #print('===========Total distance', distance)
        return distance
    
    def distance_2(query_elephant,galery_elephant):
        #print('query_elephant', query_elephant)
        #print('galery_elephant', galery_elephant)

        distance = 0
        for i in range(len(SEEK.attribute_names)):
            attr_name = SEEK.attribute_names[i]
            q_attr = getattr(query_elephant, attr_name)
            g_attr = getattr(galery_elephant, attr_name)
            
            if q_attr == g_attr:
                #print(f"Attribute {attr_name} is correct: query {q_attr}, galery {g_attr} -> +0 distance")
                distance += 0
            else: # There is an error !
                if q_attr == '_' or g_attr == '_': # One of the SEEK attribute is unknown, small error
                    #print(f"Attribute {attr_name} is a unknown in one of the elephants: query {q_attr}, galery {g_attr} -> +0.5 distance")
                    distance += 0
                else: # Both attributes are known but different, real error !
                    if (attr_name in ['R_tear_1', 'R_hole_1', 'L_tear_1', 'L_hole_1', 'R_tear_2', 'R_hole_2', 'L_tear_2', 'L_hole_2']): # There might be confusion on these attributes, the distance is proportional to the penalty
                        #print(f"Attribute {attr_name} is a in-ear distance error: query {q_attr}, galery {g_attr} -> + (a - b)/2 distance")
                        if q_attr == '0' or g_attr == '0':
                            distance += 1
                        else:
                            distance += abs(float(q_attr) - float(g_attr)) / 2
                    elif (attr_name in ['age', 'sex', 'right_tusk','left_tusk']): # We should be confident on these attributes, big error counted twice
                        #print(f"Attribute {attr_name} is an error on an easy attribute : query {q_attr}, galery {g_attr} -> +2 distance")
                        distance += 1
                    else: # Other errors count for 1
                        #print(f"Attribute {attr_name} is an error: query {q_attr}, galery {g_attr} -> +1 distance")
                        distance += 1
        #print('===========Total distance', distance)
        return distance

    def distance_3(query_elephant,galery_elephant):
        #print('query_elephant', query_elephant)
        #print('galery_elephant', galery_elephant)

        distance = 0
        for i in range(len(SEEK.attribute_names)):
            attr_name = SEEK.attribute_names[i]
            q_attr = getattr(query_elephant, attr_name)
            g_attr = getattr(galery_elephant, attr_name)
            
            if q_attr == g_attr:
                #print(f"Attribute {attr_name} is correct: query {q_attr}, galery {g_attr} -> +0 distance")
                distance += 0
            else: # There is an error !
                if q_attr == '_' and g_attr == '_': # Both attributes are unknown, no penalty
                    distance += 0
                elif q_attr == '_' or g_attr == '_': # One attribute is unknown, small penalty
                    distance += 0.05
                else: # Both attributes are known but different, real error !
                    if (attr_name in ['R_tear_1', 'R_hole_1', 'L_tear_1', 'L_hole_1', 'R_tear_2', 'R_hole_2', 'L_tear_2', 'L_hole_2']): # There might be confusion on these attributes, the distance is proportional to the penalty
                        #print(f"Attribute {attr_name} is a in-ear distance error: query {q_attr}, galery {g_attr} -> + (a - b)/2 distance")
                        if q_attr == '0' or g_attr == '0':
                            distance += 1
                        else:
                            distance += abs(float(q_attr) - float(g_attr))
                    elif (attr_name in ['age', 'sex', 'right_tusk','left_tusk']): # We should be confident on these attributes, big error counted twice
                        #print(f"Attribute {attr_name} is an error on an easy attribute : query {q_attr}, galery {g_attr} -> +2 distance")
                        distance += 1
                    else: # Other errors count for 1
                        #print(f"Attribute {attr_name} is an error: query {q_attr}, galery {g_attr} -> +1 distance")
                        distance += 1
        #print('===========Total distance', distance)
        return distance

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




def test_distance():
    elephant_a = SEEK("B00T__E6700-0000X0_S00")
    elephant_b = SEEK("B20T__E8000-0000X1_S01")
    
    distance = SEEK.distance(elephant_a, elephant_b)
    print("Distance between elephant_a and elephant_b:", distance)

    return 0

# Run the test function if the script is executed directly
if __name__ == "__main__":
    #test_seek()
    #test_separate_and_reconstruct()
    test_distance()
